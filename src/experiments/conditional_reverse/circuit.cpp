#include "riftco_transformer/experiments/conditional_reverse/circuit.hpp"

#include "riftco_transformer/lowering/cajal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::experiments::conditional_reverse {
namespace {

programmed::ProjectionState projection_state(Tensor::Shape weight_shape,
                                             std::vector<float> weight_values,
                                             Tensor::Shape bias_shape,
                                             ExecutionBackend backend) {
  return {
      Tensor(std::move(weight_shape), std::move(weight_values), backend),
      Tensor::zeros(std::move(bias_shape), backend),
  };
}

programmed::SequenceProjectionState
exact_projection_state(const Task &task, ExecutionBackend backend) {
  const std::size_t symbol_count = task.symbol_count();
  std::vector<float> condition_weights(2 * symbol_count, 0.0F);
  for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
    const std::size_t condition = task.reverses(symbol) ? 0 : 1;
    condition_weights[condition * symbol_count + symbol] = 1.0F;
  }

  std::vector<float> identity(symbol_count * symbol_count, 0.0F);
  for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
    identity[symbol * symbol_count + symbol] = 1.0F;
  }
  std::vector<float> output_identity = identity;

  programmed::SequenceProjectionState result{
      {
          projection_state({2, symbol_count}, std::move(condition_weights), {2},
                           backend),
          projection_state({symbol_count, symbol_count}, std::move(identity),
                           {symbol_count}, backend),
      },
      projection_state({symbol_count, symbol_count}, std::move(output_identity),
                       {symbol_count}, backend),
  };
  return result;
}

std::unique_ptr<programmed::ProgrammedSequenceAdapter>
make_adapter(const Task &task,
             const lowering::NeuralLoweringConfig &lowering_config,
             ResourceMetadata &resources) {
  lowering_config.validate();
  if (lowering_config.attention_query_axis.value_or(std::size_t{1}) != 1) {
    throw std::invalid_argument(
        "conditional-reverse circuit requires the source sequence as "
        "linear-attention query axis 1");
  }
  Program source_program = compile_program({
      .sequence_length = task.config().sequence_length,
      .symbol_count = task.symbol_count(),
      .max_coefficient_elements = lowering_config.max_coefficient_elements,
  });
  resources = source_program.resources;

  const lowering::LoweringAnalysis analysis = lowering::analyze_neural_lowering(
      source_program.compiled, lowering_config);
  if (!analysis.supported ||
      analysis.selected_strategy != lowering::kLinearAttentionStrategy) {
    throw std::invalid_argument(
        "conditional-reverse circuit requires the exact "
        "linear_attention lowering strategy");
  }
  auto lowered =
      lowering::lower_to_neural(source_program.compiled, lowering_config);
  programmed::SequencePlacementConfig placement{
      .source_offset = 0,
      .source_length = task.config().sequence_length,
      .output_length = task.config().sequence_length,
      .target_offset = task.config().sequence_length,
      .inputs =
          {
              {
                  .source = programmed::ProgramInputSource::SourcePosition,
                  .position = 0,
              },
              {
                  .source = programmed::ProgramInputSource::WholeSource,
                  .position = 0,
              },
          },
  };
  return std::make_unique<programmed::ProgrammedSequenceAdapter>(
      task.symbol_count(), std::move(placement), std::move(lowered),
      exact_projection_state(task, lowering_config.backend));
}

} // namespace

Circuit::Circuit(CircuitConfig config)
    : config_(std::move(config)), task_(config_.task),
      adapter_(make_adapter(task_, config_.lowering, resources_)) {
  register_module("programmed", *adapter_);
}

const CircuitConfig &Circuit::config() const noexcept { return config_; }

const Task &Circuit::task() const noexcept { return task_; }

const ResourceMetadata &Circuit::resources() const noexcept {
  return resources_;
}

ExecutionBackend Circuit::backend() const noexcept {
  return adapter_->program().backend();
}

Tensor
Circuit::encode_source_states(std::span<const SymbolId> flattened_sources,
                              std::size_t batch_size) const {
  if (batch_size == 0) {
    throw std::invalid_argument(
        "conditional-reverse forward needs a nonempty batch");
  }
  const std::size_t sequence_length = task_.config().sequence_length;
  if (batch_size > std::numeric_limits<std::size_t>::max() / sequence_length ||
      flattened_sources.size() != batch_size * sequence_length) {
    throw std::invalid_argument(
        "conditional-reverse flattened source batch has the wrong size");
  }
  const std::size_t symbol_count = task_.symbol_count();
  if (sequence_length > std::numeric_limits<std::size_t>::max() / 2 ||
      2 * sequence_length >
          std::numeric_limits<std::size_t>::max() / symbol_count) {
    throw std::overflow_error(
        "conditional-reverse state width exceeds addressable size");
  }
  const std::size_t state_elements_per_batch =
      2 * sequence_length * symbol_count;
  if (batch_size >
      std::numeric_limits<std::size_t>::max() / state_elements_per_batch) {
    throw std::overflow_error(
        "conditional-reverse state batch exceeds addressable size");
  }
  std::vector<float> values(batch_size * state_elements_per_batch, 0.0F);
  for (std::size_t row = 0; row < batch_size; ++row) {
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const SymbolId symbol =
          flattened_sources[row * sequence_length + position];
      if (symbol >= symbol_count) {
        throw std::out_of_range(
            "conditional-reverse source symbol is out of range");
      }
      values[(row * 2 * sequence_length + position) * symbol_count + symbol] =
          1.0F;
    }
  }
  return Tensor({batch_size, 2 * sequence_length, symbol_count},
                std::move(values), backend());
}

programmed::SequencePlacementResult
Circuit::forward(std::span<const SymbolId> flattened_sources,
                 std::size_t batch_size,
                 const programmed::SequenceForwardOptions &options) const {
  return adapter_->forward(
      Variable(encode_source_states(flattened_sources, batch_size), false),
      options);
}

std::vector<SymbolId>
Circuit::decode_target(const Tensor &sequence_states) const {
  const std::size_t sequence_length = task_.config().sequence_length;
  const std::size_t symbol_count = task_.symbol_count();
  if (sequence_states.rank() != 3 ||
      sequence_states.shape()[1] < 2 * sequence_length ||
      sequence_states.shape()[2] != symbol_count) {
    throw std::invalid_argument(
        "conditional-reverse decoded states have the wrong shape");
  }
  const Tensor host = sequence_states.backend() == ExecutionBackend::Cpu
                          ? Tensor(sequence_states)
                          : sequence_states.to(ExecutionBackend::Cpu);
  const std::size_t batch = host.shape()[0];
  std::vector<SymbolId> result;
  result.reserve(batch * sequence_length);
  for (std::size_t row = 0; row < batch; ++row) {
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const std::size_t base =
          (row * host.shape()[1] + sequence_length + position) * symbol_count;
      std::size_t best = 0;
      float best_value = host.flat(base);
      if (!std::isfinite(best_value)) {
        throw std::domain_error(
            "conditional-reverse target logits must be finite");
      }
      for (std::size_t symbol = 1; symbol < symbol_count; ++symbol) {
        const float value = host.flat(base + symbol);
        if (!std::isfinite(value)) {
          throw std::domain_error(
              "conditional-reverse target logits must be finite");
        }
        if (value > best_value) {
          best = symbol;
          best_value = value;
        }
      }
      result.push_back(best);
    }
  }
  return result;
}

EvaluationResult
Circuit::evaluate(std::span<const Example> examples,
                  const programmed::SequenceForwardOptions &options) const {
  if (examples.empty()) {
    throw std::invalid_argument(
        "conditional-reverse evaluation needs at least one example");
  }
  const std::size_t sequence_length = task_.config().sequence_length;
  if (examples.size() >
      std::numeric_limits<std::size_t>::max() / sequence_length) {
    throw std::overflow_error(
        "conditional-reverse evaluation batch exceeds addressable size");
  }
  std::vector<SymbolId> sources;
  sources.reserve(examples.size() * sequence_length);
  for (const auto &example : examples) {
    if (example.source.size() != sequence_length ||
        example.target.size() != sequence_length) {
      throw std::invalid_argument(
          "conditional-reverse evaluation example has the wrong length");
    }
    for (const SymbolId symbol : example.target) {
      if (symbol >= task_.symbol_count()) {
        throw std::out_of_range(
            "conditional-reverse target symbol is out of range");
      }
    }
    sources.insert(sources.end(), example.source.begin(), example.source.end());
  }

  programmed::SequencePlacementResult forward_result =
      forward(sources, examples.size(), options);
  EvaluationResult result;
  result.predictions = decode_target(forward_result.output.value());
  result.representations = std::move(forward_result.representations);
  result.metrics.example_count = examples.size();
  result.metrics.token_count = examples.size() * sequence_length;
  result.per_example_token_accuracy.reserve(examples.size());
  result.per_example_exact_accuracy.reserve(examples.size());

  for (std::size_t row = 0; row < examples.size(); ++row) {
    std::size_t correct = 0;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      if (result.predictions[row * sequence_length + position] ==
          examples[row].target[position]) {
        ++correct;
      }
    }
    result.metrics.correct_token_count += correct;
    const bool exact = correct == sequence_length;
    result.metrics.correct_sequence_count += exact ? 1 : 0;
    result.per_example_token_accuracy.push_back(
        static_cast<double>(correct) / static_cast<double>(sequence_length));
    result.per_example_exact_accuracy.push_back(exact ? 1.0 : 0.0);
  }
  result.metrics.token_accuracy =
      static_cast<double>(result.metrics.correct_token_count) /
      static_cast<double>(result.metrics.token_count);
  result.metrics.exact_sequence_accuracy =
      static_cast<double>(result.metrics.correct_sequence_count) /
      static_cast<double>(result.metrics.example_count);
  return result;
}

void Circuit::to(ExecutionBackend backend_value) {
  adapter_->to(backend_value);
}

programmed::ProgrammedSequenceAdapter &Circuit::adapter() noexcept {
  return *adapter_;
}

const programmed::ProgrammedSequenceAdapter &Circuit::adapter() const noexcept {
  return *adapter_;
}

ParameterList Circuit::parameters() { return Module::parameters(); }

} // namespace riftco_transformer::experiments::conditional_reverse
