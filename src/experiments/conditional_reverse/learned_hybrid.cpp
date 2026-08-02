#include "riftco_transformer/experiments/conditional_reverse/learned_hybrid.hpp"

#include "riftco_transformer/experiments/conditional_reverse/program.hpp"
#include "riftco_transformer/lowering/cajal.hpp"
#include "riftco_transformer/lowering/strategy.hpp"
#include "riftco_transformer/nn/loss.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace riftco_transformer::experiments::conditional_reverse {
namespace {

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right,
                                           const char *description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(description);
  }
  return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right,
                                      const char *description) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(description);
  }
  return left + right;
}

[[nodiscard]] LearnedHybridConfig checked_config(LearnedHybridConfig config) {
  config.validate();
  return config;
}

[[nodiscard]] std::unique_ptr<programmed::ProgrammedSequenceCore>
make_program_core(const LearnedHybridConfig &config, std::mt19937 &random) {
  if (config.variant == HybridVariant::I) {
    return nullptr;
  }

  ProgramConfig program_config{
      .sequence_length = config.protocol.sequence_length,
      .symbol_count = config.program_width(),
      .max_coefficient_elements = config.lowering.max_coefficient_elements,
  };
  lowering::NeuralLoweringConfig lowering_config = config.lowering;
  // The artifact configures the entire model from one experiment seed. Keep
  // T's randomized tensor on that same seed contract rather than exposing an
  // accidental second experiment seed through generic lowering policy.
  lowering_config.seed = config.seed;
  lowering_config.initialization =
      config.variant == HybridVariant::T
          ? lowering::CoefficientInitialization::RandomUniform
          : lowering::CoefficientInitialization::Compiled;
  lowering_config.trainable = config.variant == HybridVariant::T;

  std::unique_ptr<lowering::LoweredMultilinearModule> lowered;
  programmed::SequenceCoreConfig core_config;
  core_config.source_length = config.protocol.sequence_length;
  core_config.output_length = config.protocol.sequence_length;
  core_config.input_projection_bias = false;

  if (config.variant == HybridVariant::P) {
    const ReverseProgram program = compile_reverse_program(program_config);
    lowered = lowering::lower_to_neural(program.compiled, lowering_config);
    if (lowered->metadata().selected_strategy != lowering::kLinearStrategy) {
      throw std::invalid_argument(
          "learned P control requires exact linear program lowering");
    }
    core_config.inputs = {{
        .source = programmed::ProgramInputSource::WholeSource,
        .position = 0,
    }};
    core_config.input_projection_groups = {0};
  } else {
    lowering_config.attention_query_axis = 1;
    const TwoSequenceProgram program =
        compile_two_sequence_program(program_config);
    lowered = lowering::lower_to_neural(program.compiled, lowering_config);
    if (lowered->metadata().selected_strategy !=
        lowering::kLinearAttentionStrategy) {
      throw std::invalid_argument(
          "learned F/T control requires linear-attention program lowering");
    }
    core_config.inputs = {
        {
            .source = programmed::ProgramInputSource::WholeSource,
            .position = 0,
        },
        {
            .source = programmed::ProgramInputSource::WholeSource,
            .position = 0,
        },
    };
    core_config.input_projection_groups = {0, 0};
  }
  return std::make_unique<programmed::ProgrammedSequenceCore>(
      config.model_width, std::move(core_config), std::move(lowered), random);
}

[[nodiscard]] std::unique_ptr<Linear>
make_program_merge(const LearnedHybridConfig &config, std::mt19937 &random) {
  if (config.variant == HybridVariant::I) {
    return nullptr;
  }
  return std::make_unique<Linear>(
      checked_add(config.model_width, config.program_width(),
                  "learned conditional-reverse merge width overflows"),
      config.model_width, random, false);
}

[[nodiscard]] Variable roll_batch(const Variable &input, std::size_t shift) {
  if (input.value().rank() == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse ablation requires a batch axis");
  }
  const Tensor::Shape shape = input.value().shape();
  const std::size_t batch = shape.front();
  const std::size_t row_width = input.value().numel() / batch;
  const std::size_t normalized = shift % batch;
  std::vector<std::size_t> rows;
  rows.reserve(batch);
  for (std::size_t row = 0; row < batch; ++row) {
    rows.push_back(row >= batch - normalized ? row - (batch - normalized)
                                             : row + normalized);
  }
  return reshape(gather_rows(reshape(input, {batch, row_width}), rows, {batch}),
                 shape);
}

[[nodiscard]] Variable pad_program_output(const Variable &program_output) {
  const Tensor::Shape &shape = program_output.value().shape();
  if (shape.size() != 3) {
    throw std::invalid_argument(
        "learned program output must have shape [batch, length, width]");
  }
  const Variable transposed = permute(program_output, {0, 2, 1});
  const Variable zeros(
      Tensor::zeros(transposed.value().shape(), transposed.value().backend()),
      false);
  return permute(concatenate_last_axis(zeros, transposed), {0, 2, 1});
}

[[nodiscard]] Variable gather_time_prefix(const Variable &input,
                                          std::size_t length) {
  const Tensor::Shape &shape = input.value().shape();
  if (shape.size() != 3 || length == 0 || length > shape[1]) {
    throw std::invalid_argument(
        "learned program input prefix has incompatible dimensions");
  }
  std::vector<std::size_t> rows;
  rows.reserve(checked_multiply(shape[0], length,
                                "learned program input index count overflows"));
  for (std::size_t batch = 0; batch < shape[0]; ++batch) {
    for (std::size_t position = 0; position < length; ++position) {
      rows.push_back(batch * shape[1] + position);
    }
  }
  return gather_rows(reshape(input, {shape[0] * shape[1], shape[2]}), rows,
                     {shape[0], length});
}

void capture(analysis::RepresentationTrace &trace, std::string name,
             const Tensor &value) {
  if (value.backend() == ExecutionBackend::Cpu) {
    trace.capture(std::move(name), value.shape(), value.data());
    return;
  }
  const Tensor host = value.to(ExecutionBackend::Cpu);
  trace.capture(std::move(name), host.shape(), host.data());
}

void append_program_trace(analysis::RepresentationTrace &destination,
                          const analysis::RepresentationTrace &source) {
  for (const analysis::NamedRepresentation &entry : source.entries()) {
    std::vector<std::size_t> shape = entry.leading_shape;
    shape.push_back(entry.observations.columns);
    destination.capture("program." + entry.name, shape,
                        entry.observations.values);
  }
}

void capture_padded_representation(analysis::RepresentationTrace &destination,
                                   std::string name,
                                   const analysis::NamedRepresentation &source,
                                   std::size_t sequence_length) {
  if (source.leading_shape.size() != 2 ||
      source.leading_shape[1] != sequence_length) {
    throw std::logic_error(
        "learned program capture has an unexpected sequence shape");
  }
  const std::size_t batch = source.leading_shape[0];
  const std::size_t width = source.observations.columns;
  const std::size_t row_elements = checked_multiply(
      sequence_length, width, "learned program capture row size overflows");
  const std::size_t padded_row_elements = checked_multiply(
      2, row_elements, "learned padded program capture row size overflows");
  std::vector<float> values(
      checked_multiply(batch, padded_row_elements,
                       "learned padded program capture size overflows"),
      0.0F);
  for (std::size_t row = 0; row < batch; ++row) {
    std::copy_n(source.observations.values.data() + row * row_elements,
                row_elements,
                values.data() + row * padded_row_elements + row_elements);
  }
  const std::vector<std::size_t> shape{batch, 2 * sequence_length, width};
  destination.capture(std::move(name), shape, values);
}

[[nodiscard]] programmed::SequenceForwardOptions
program_options(const LearnedHybridConfig &config,
                const LearnedForwardOptions &options) {
  programmed::SequenceForwardOptions result;
  result.capture_representations = options.capture_representations;
  if (options.ablate_program_output) {
    result.ablation = programmed::BatchRollAblation{
        .program_input_indices = {},
        .program_output = true,
        .shift = options.batch_roll_shift,
    };
  }
  const std::size_t width = config.program_width();
  result.steering.reserve(options.steering.size());
  std::unordered_set<std::size_t> steered_positions;
  steered_positions.reserve(options.steering.size());
  for (const LearnedSteering &steering : options.steering) {
    if (steering.position >= config.protocol.sequence_length) {
      throw std::out_of_range(
          "learned conditional-reverse steering position is out of range");
    }
    if (!std::isfinite(steering.selected_coordinate_scale) ||
        !std::isfinite(steering.other_coordinate_scale)) {
      throw std::invalid_argument(
          "learned conditional-reverse steering scales must be finite");
    }
    if (!steered_positions.insert(steering.position).second) {
      throw std::invalid_argument(
          "learned conditional-reverse steering positions must be unique");
    }
    std::vector<float> scales(width, steering.other_coordinate_scale);
    scales.front() = steering.selected_coordinate_scale;
    result.steering.push_back({
        .input_index = 0,
        .positions = {steering.position},
        .scales = std::move(scales),
        .offsets = {},
    });
  }
  return result;
}

[[nodiscard]] Tensor host_tensor(const Tensor &value) {
  return value.backend() == ExecutionBackend::Cpu
             ? Tensor(value)
             : value.to(ExecutionBackend::Cpu);
}

} // namespace

void LearnedHybridConfig::validate() const {
  protocol.validate();
  lowering.validate();
  if (model_width == 0 || head_count == 0 || feed_forward_width == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse model dimensions must be positive");
  }
  if (model_width % head_count != 0 || model_width / head_count < 2) {
    throw std::invalid_argument(
        "learned conditional-reverse model width must provide at least two "
        "coordinates per head");
  }
  static_cast<void>(checked_multiply(
      model_width, 2,
      "learned conditional-reverse attention merge width overflows"));
  if (protocol.context_length() >
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::overflow_error(
        "learned conditional-reverse positions exceed TokenId");
  }
  switch (variant) {
  case HybridVariant::F:
  case HybridVariant::P:
  case HybridVariant::T:
  case HybridVariant::I:
    break;
  default:
    throw std::invalid_argument(
        "learned conditional-reverse hybrid variant is unknown");
  }
}

std::size_t LearnedHybridConfig::program_width() const {
  if (head_count == 0 || model_width % head_count != 0) {
    throw std::invalid_argument(
        "learned conditional-reverse head dimensions are invalid");
  }
  return model_width / head_count;
}

LearnedHybrid::LearnedHybrid(LearnedHybridConfig config)
    : config_(checked_config(std::move(config))),
      initialization_random_(config_.seed),
      token_embedding_(config_.protocol.vocabulary_size(), config_.model_width,
                       initialization_random_),
      position_embedding_(config_.protocol.context_length(),
                          config_.model_width, initialization_random_),
      feed_forward_(config_.model_width, config_.feed_forward_width,
                    initialization_random_, FeedForwardActivation::Relu),
      attention_first_(config_.model_width, config_.head_count,
                       initialization_random_, config_.attention_kind),
      attention_second_(config_.model_width, config_.head_count,
                        initialization_random_, config_.attention_kind),
      attention_merge_(2 * config_.model_width, config_.model_width,
                       initialization_random_),
      output_(config_.model_width, config_.protocol.vocabulary_size(),
              initialization_random_),
      program_core_(make_program_core(config_, initialization_random_)),
      program_merge_(make_program_merge(config_, initialization_random_)) {
  register_module("token_embedding", token_embedding_);
  register_module("position_embedding", position_embedding_);
  register_module("feed_forward", feed_forward_);
  register_module("attention_first", attention_first_);
  register_module("attention_second", attention_second_);
  register_module("attention_merge", attention_merge_);
  if (program_core_ != nullptr) {
    register_module("program", *program_core_);
    register_module("program_merge", *program_merge_);
  }
  register_module("output", output_);
  to(config_.lowering.backend);
}

const LearnedHybridConfig &LearnedHybrid::config() const noexcept {
  return config_;
}

HybridVariant LearnedHybrid::variant() const noexcept {
  return config_.variant;
}

ExecutionBackend LearnedHybrid::backend() const noexcept {
  return token_embedding_.weight().value().backend();
}

bool LearnedHybrid::has_program() const noexcept {
  return program_core_ != nullptr;
}

LearnedForwardResult
LearnedHybrid::forward(std::span<const TokenId> flattened_inputs,
                       std::size_t batch_size,
                       const LearnedForwardOptions &options) const {
  if (batch_size == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse forward requires a nonempty batch");
  }
  const std::size_t time = config_.protocol.context_length();
  const std::size_t expected = checked_multiply(
      batch_size, time,
      "learned conditional-reverse forward batch size overflows");
  if (flattened_inputs.size() != expected) {
    throw std::invalid_argument(
        "learned conditional-reverse input batch has the wrong size");
  }
  for (const TokenId token : flattened_inputs) {
    if (static_cast<std::size_t>(token) >= config_.protocol.vocabulary_size()) {
      throw std::out_of_range(
          "learned conditional-reverse input token is outside vocabulary");
    }
  }
  if ((options.ablate_learned_attention || options.ablate_program_output) &&
      options.batch_roll_shift == 0) {
    throw std::invalid_argument(
        "learned conditional-reverse batch-roll shift must be positive");
  }
  if (!has_program() &&
      (options.ablate_program_output || !options.steering.empty())) {
    throw std::invalid_argument(
        "learned I control has no program to ablate or steer");
  }

  std::vector<TokenId> positions;
  positions.reserve(expected);
  for (std::size_t row = 0; row < batch_size; ++row) {
    for (std::size_t position = 0; position < time; ++position) {
      positions.push_back(static_cast<TokenId>(position));
    }
  }
  const Variable token_state =
      token_embedding_.forward(flattened_inputs, {batch_size, time});
  const Variable position_state =
      position_embedding_.forward(positions, {batch_size, time});
  const Variable x1 = token_state + position_state;
  const Variable r1 = feed_forward_.forward(x1) + x1;

  const Variable attention_first = attention_first_.forward(r1);
  const Variable attention_second = attention_second_.forward(r1);
  Variable h1 = attention_merge_.forward(
      concatenate_last_axis(attention_first, attention_second));
  if (options.ablate_learned_attention) {
    h1 = roll_batch(h1, options.batch_roll_shift);
  }

  analysis::RepresentationTrace trace;
  if (options.capture_representations) {
    capture(trace, "x1", x1.value());
    capture(trace, "r1", r1.value());
    capture(trace, "h1", h1.value());
  }

  Variable r2 = h1 + r1;
  if (has_program()) {
    const Variable program_source =
        gather_time_prefix(r1, config_.protocol.sequence_length);
    programmed::SequenceCoreResult program_result = program_core_->forward(
        program_source, program_options(config_, options));
    const Variable padded_program = pad_program_output(program_result.output);
    r2 =
        program_merge_->forward(concatenate_last_axis(h1, padded_program)) + r1;
    if (options.capture_representations) {
      append_program_trace(trace, program_result.representations);
      const Variable unsteered_program_input =
          program_core_->input_projection_for_input(0).forward(program_source);
      capture(trace, "program_input",
              pad_program_output(unsteered_program_input).value());
      capture_padded_representation(
          trace, "program_input.selected",
          program_result.representations.at("program_input.0.projected"),
          config_.protocol.sequence_length);
      capture(trace, "program_output", padded_program.value());
    }
  }
  if (options.capture_representations) {
    capture(trace, "r2", r2.value());
  }
  const Variable logits = output_.forward(r2);
  if (options.capture_representations) {
    capture(trace, "logits", logits.value());
  }
  return {std::move(logits), std::move(trace)};
}

LearnedEvaluationResult
LearnedHybrid::evaluate(const LearnedBatch &batch,
                        const LearnedForwardOptions &options) const {
  if (batch.batch_size == 0 ||
      batch.context_length != config_.protocol.context_length()) {
    throw std::invalid_argument(
        "learned conditional-reverse evaluation batch has wrong dimensions");
  }
  const std::size_t expected = checked_multiply(
      batch.batch_size, batch.context_length,
      "learned conditional-reverse evaluation batch overflows");
  if (batch.inputs.size() != expected || batch.targets.size() != expected) {
    throw std::invalid_argument(
        "learned conditional-reverse evaluation data has the wrong size");
  }
  if (batch.reversed.size() != batch.batch_size) {
    throw std::invalid_argument(
        "learned conditional-reverse evaluation needs one branch flag per "
        "example");
  }
  for (const std::uint8_t reversed : batch.reversed) {
    if (reversed > 1) {
      throw std::invalid_argument(
          "learned conditional-reverse branch flags must be zero or one");
    }
  }
  const std::size_t sequence_length = config_.protocol.sequence_length;
  for (std::size_t row = 0; row < batch.batch_size; ++row) {
    const std::size_t base = row * batch.context_length;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      if (static_cast<std::size_t>(batch.inputs[base + position]) >=
              config_.protocol.alphabet.size() ||
          static_cast<std::size_t>(
              batch.targets[base + sequence_length + position]) >=
              config_.protocol.alphabet.size()) {
        throw std::invalid_argument(
            "learned conditional-reverse evaluation source/output must use "
            "alphabet symbols");
      }
    }
    if (batch.inputs[base + sequence_length] !=
        config_.protocol.delimiter_token()) {
      throw std::invalid_argument(
          "learned conditional-reverse evaluation input is missing its "
          "delimiter");
    }
    const TokenId first = batch.inputs[base];
    const bool expected_reversed =
        config_.protocol.reverse_when_first_is.find(
            config_.protocol.alphabet[first]) != std::string::npos;
    if ((batch.reversed[row] != 0) != expected_reversed) {
      throw std::invalid_argument(
          "learned conditional-reverse evaluation branch flag disagrees "
          "with source");
    }
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const std::size_t source_position =
          expected_reversed ? sequence_length - 1 - position : position;
      if (batch.targets[base + sequence_length + position] !=
          batch.inputs[base + source_position]) {
        throw std::invalid_argument(
            "learned conditional-reverse evaluation target violates task "
            "semantics");
      }
    }
  }

  LearnedForwardResult forward_result =
      forward(batch.inputs, batch.batch_size, options);
  const Variable loss = learned_target_half_loss(
      forward_result.logits, batch.targets, config_.protocol.sequence_length);
  const Tensor host_loss = host_tensor(loss.value());
  const Tensor host_logits = host_tensor(forward_result.logits.value());
  if (!std::isfinite(host_loss.flat(0))) {
    throw std::domain_error(
        "learned conditional-reverse evaluation loss must be finite");
  }

  LearnedEvaluationResult result;
  result.metrics.example_count = batch.batch_size;
  result.metrics.target_token_count =
      checked_multiply(batch.batch_size, config_.protocol.sequence_length,
                       "learned conditional-reverse target count overflows");
  result.metrics.loss = host_loss.flat(0);
  result.predictions.reserve(result.metrics.target_token_count);
  result.per_example_token_accuracy.reserve(batch.batch_size);
  result.per_example_exact_accuracy.reserve(batch.batch_size);
  result.representations = std::move(forward_result.representations);

  const std::size_t vocabulary = config_.protocol.vocabulary_size();
  for (std::size_t row = 0; row < batch.batch_size; ++row) {
    std::size_t correct = 0;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const std::size_t time_position = sequence_length + position;
      const std::size_t logits_base =
          (row * batch.context_length + time_position) * vocabulary;
      std::size_t best = 0;
      float best_value = host_logits.flat(logits_base);
      if (!std::isfinite(best_value)) {
        throw std::domain_error(
            "learned conditional-reverse logits must be finite");
      }
      for (std::size_t token = 1; token < vocabulary; ++token) {
        const float value = host_logits.flat(logits_base + token);
        if (!std::isfinite(value)) {
          throw std::domain_error(
              "learned conditional-reverse logits must be finite");
        }
        if (value > best_value) {
          best = token;
          best_value = value;
        }
      }
      const TokenId prediction = static_cast<TokenId>(best);
      result.predictions.push_back(prediction);
      const TokenId target =
          batch.targets[row * batch.context_length + time_position];
      correct += prediction == target ? 1 : 0;
    }
    result.metrics.correct_target_token_count += correct;
    const bool exact = correct == sequence_length;
    result.metrics.correct_sequence_count += exact ? 1 : 0;
    if (batch.reversed[row] != 0) {
      ++result.metrics.reverse_example_count;
      result.metrics.reverse_correct_target_token_count += correct;
      result.metrics.reverse_correct_sequence_count += exact ? 1 : 0;
    } else {
      ++result.metrics.copy_example_count;
      result.metrics.copy_correct_target_token_count += correct;
      result.metrics.copy_correct_sequence_count += exact ? 1 : 0;
    }
    result.per_example_token_accuracy.push_back(
        static_cast<double>(correct) / static_cast<double>(sequence_length));
    result.per_example_exact_accuracy.push_back(exact ? 1.0 : 0.0);
  }
  result.metrics.target_token_accuracy =
      static_cast<double>(result.metrics.correct_target_token_count) /
      static_cast<double>(result.metrics.target_token_count);
  result.metrics.exact_sequence_accuracy =
      static_cast<double>(result.metrics.correct_sequence_count) /
      static_cast<double>(result.metrics.example_count);
  if (result.metrics.reverse_example_count != 0) {
    result.metrics.reverse_target_token_accuracy =
        static_cast<double>(result.metrics.reverse_correct_target_token_count) /
        static_cast<double>(result.metrics.reverse_example_count *
                            sequence_length);
    result.metrics.reverse_exact_sequence_accuracy =
        static_cast<double>(result.metrics.reverse_correct_sequence_count) /
        static_cast<double>(result.metrics.reverse_example_count);
  }
  if (result.metrics.copy_example_count != 0) {
    result.metrics.copy_target_token_accuracy =
        static_cast<double>(result.metrics.copy_correct_target_token_count) /
        static_cast<double>(result.metrics.copy_example_count *
                            sequence_length);
    result.metrics.copy_exact_sequence_accuracy =
        static_cast<double>(result.metrics.copy_correct_sequence_count) /
        static_cast<double>(result.metrics.copy_example_count);
  }
  return result;
}

LearnedEvaluationResult
LearnedHybrid::evaluate(std::span<const LearnedExample> examples,
                        const LearnedForwardOptions &options) const {
  return evaluate(make_learned_batch(examples, config_.protocol), options);
}

void LearnedHybrid::to(ExecutionBackend backend_value) {
  if (program_core_ != nullptr) {
    program_core_->to(backend_value);
  }
  Module::to(backend_value);
}

programmed::ProgrammedSequenceCore *LearnedHybrid::program_core() noexcept {
  return program_core_.get();
}

const programmed::ProgrammedSequenceCore *
LearnedHybrid::program_core() const noexcept {
  return program_core_.get();
}

const Linear *LearnedHybrid::program_merge() const noexcept {
  return program_merge_.get();
}

ParameterList LearnedHybrid::parameters() { return Module::parameters(); }

Variable learned_target_half_loss(const Variable &logits,
                                  std::span<const TokenId> flattened_targets,
                                  std::size_t sequence_length) {
  if (sequence_length == 0 || logits.value().rank() != 3) {
    throw std::invalid_argument(
        "learned target-half loss needs [batch, time, vocabulary] logits");
  }
  const Tensor::Shape &shape = logits.value().shape();
  const std::size_t expected_time = checked_multiply(
      sequence_length, 2, "learned target-half context length overflows");
  if (shape[0] == 0 || shape[1] != expected_time || shape[2] == 0) {
    throw std::invalid_argument(
        "learned target-half logits have incompatible dimensions");
  }
  const std::size_t target_count = checked_multiply(
      shape[0], expected_time, "learned target-half target count overflows");
  if (flattened_targets.size() != target_count) {
    throw std::invalid_argument(
        "learned target-half targets have the wrong size");
  }

  std::vector<std::size_t> selected_rows;
  std::vector<TokenId> selected_targets;
  const std::size_t selected_count =
      checked_multiply(shape[0], sequence_length,
                       "learned target-half selected count overflows");
  selected_rows.reserve(selected_count);
  selected_targets.reserve(selected_count);
  for (std::size_t row = 0; row < shape[0]; ++row) {
    for (std::size_t position = sequence_length; position < expected_time;
         ++position) {
      const std::size_t flat_position = row * expected_time + position;
      selected_rows.push_back(flat_position);
      selected_targets.push_back(flattened_targets[flat_position]);
    }
  }
  const Variable selected_logits =
      gather_rows(reshape(logits, {shape[0] * expected_time, shape[2]}),
                  selected_rows, {shape[0], sequence_length});
  return cross_entropy(selected_logits, selected_targets);
}

LearnedHypothesisScores
score_learned_hypotheses(std::span<const LearnedExample> examples,
                         std::span<const TokenId> target_predictions,
                         const LearnedProtocolConfig &config) {
  config.validate();
  if (examples.empty()) {
    throw std::invalid_argument(
        "learned hypothesis scoring requires at least one example");
  }
  static_cast<void>(make_learned_batch(examples, config));
  const std::size_t expected =
      checked_multiply(examples.size(), config.sequence_length,
                       "learned hypothesis prediction count overflows");
  if (target_predictions.size() != expected) {
    throw std::invalid_argument(
        "learned hypothesis predictions have the wrong size");
  }
  for (const TokenId prediction : target_predictions) {
    if (static_cast<std::size_t>(prediction) >= config.vocabulary_size()) {
      throw std::out_of_range(
          "learned hypothesis prediction is outside the vocabulary");
    }
  }

  std::size_t copy_correct = 0;
  std::size_t reverse_correct = 0;
  std::size_t copy_exact = 0;
  std::size_t reverse_exact = 0;
  for (std::size_t row = 0; row < examples.size(); ++row) {
    const LearnedExample &example = examples[row];
    if (example.tokens.size() != config.context_length() + 1) {
      throw std::invalid_argument(
          "learned hypothesis example has the wrong protocol length");
    }
    std::size_t row_copy_correct = 0;
    std::size_t row_reverse_correct = 0;
    for (std::size_t position = 0; position < config.sequence_length;
         ++position) {
      const TokenId prediction =
          target_predictions[row * config.sequence_length + position];
      row_copy_correct += prediction == example.tokens[position] ? 1 : 0;
      row_reverse_correct +=
          prediction == example.tokens[config.sequence_length - 1 - position]
              ? 1
              : 0;
    }
    copy_correct += row_copy_correct;
    reverse_correct += row_reverse_correct;
    copy_exact += row_copy_correct == config.sequence_length ? 1 : 0;
    reverse_exact += row_reverse_correct == config.sequence_length ? 1 : 0;
  }

  return {
      .copy_target_token_accuracy =
          static_cast<double>(copy_correct) / static_cast<double>(expected),
      .copy_exact_sequence_accuracy = static_cast<double>(copy_exact) /
                                      static_cast<double>(examples.size()),
      .reverse_target_token_accuracy =
          static_cast<double>(reverse_correct) / static_cast<double>(expected),
      .reverse_exact_sequence_accuracy = static_cast<double>(reverse_exact) /
                                         static_cast<double>(examples.size()),
  };
}

} // namespace riftco_transformer::experiments::conditional_reverse
