#include "riftco_transformer/analysis/ablation.hpp"
#include "riftco_transformer/analysis/pca.hpp"
#include "riftco_transformer/compiler/cajal/multilinear_map.hpp"
#include "riftco_transformer/experiments/conditional_reverse/circuit.hpp"
#include "riftco_transformer/lowering/strategy.hpp"
#include "riftco_transformer/nn/loss.hpp"
#include "riftco_transformer/optim/adam.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace analysis = riftco_transformer::analysis;
namespace conditional_reverse =
    riftco_transformer::experiments::conditional_reverse;
namespace lowering = riftco_transformer::lowering;
namespace programmed = riftco_transformer::programmed;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Function>
void require_throws_as(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

conditional_reverse::CircuitConfig exact_config() {
  conditional_reverse::CircuitConfig config;
  config.task = {
      .sequence_length = 3,
      .alphabet = "abcd",
      .reverse_when_first_is = "a",
      .seed = 11,
  };
  config.lowering.max_coefficient_elements = 4096;
  return config;
}

std::vector<conditional_reverse::Example>
examples(const conditional_reverse::Task &task) {
  return {
      task.make_example("abc"),
      task.make_example("bca"),
  };
}

std::vector<conditional_reverse::SymbolId>
flattened_sources(const std::vector<conditional_reverse::Example> &batch) {
  std::vector<conditional_reverse::SymbolId> result;
  for (const auto &example : batch) {
    result.insert(result.end(), example.source.begin(), example.source.end());
  }
  return result;
}

double absolute_sum(const riftco_transformer::Tensor &value) {
  double result = 0.0;
  for (const float element : value.data()) {
    result += std::abs(static_cast<double>(element));
  }
  return result;
}

void test_exact_attention_circuit_and_capture() {
  conditional_reverse::Circuit circuit(exact_config());
  require(circuit.adapter().program().metadata().selected_strategy ==
                  lowering::kLinearAttentionStrategy &&
              circuit.adapter().program().metadata().preserves_compiled_map,
          "conditional reverse must use exact compiled linear attention");
  require(circuit.resources().input_dimensions[0] == 2 &&
              circuit.resources().input_dimensions[1] == 12 &&
              circuit.resources().output_dimension == 12,
          "conditional reverse circuit resources");

  const auto batch = examples(circuit.task());
  programmed::SequenceForwardOptions options;
  options.capture_representations = true;
  const auto evaluation = circuit.evaluate(batch, options);
  require(evaluation.metrics.token_accuracy == 1.0 &&
              evaluation.metrics.exact_sequence_accuracy == 1.0,
          "compiled circuit must solve both branches exactly");
  require(circuit.task().decode({
              evaluation.predictions.data(),
              circuit.task().config().sequence_length,
          }) == "cba" &&
              circuit.task().decode({
                  evaluation.predictions.data() + 3,
                  circuit.task().config().sequence_length,
              }) == "bca",
          "compiled circuit target predictions");

  const auto &trace = evaluation.representations;
  require(trace.contains("source") && trace.contains("program_input.0") &&
              trace.contains("program_input.1") &&
              trace.contains("program_output") &&
              trace.contains("programmed_branch") && trace.contains("output"),
          "sequence adapter must expose stable representation sites");
  const auto &program_output = trace.at("program_output");
  require(program_output.leading_shape == std::vector<std::size_t>({2, 3}) &&
              program_output.observations.rows == 6 &&
              program_output.observations.columns == 4,
          "captured program output shape");

  analysis::PcaOptions pca_options;
  pca_options.component_count = 2;
  const analysis::PcaFit pca =
      analysis::fit_pca(program_output.observations.view(), pca_options);
  require(pca.model.components.rows == 2 && pca.model.components.columns == 4 &&
              pca.scores.rows == 6 && pca.scores.columns == 2,
          "captured representations must feed the reusable PCA stage");

  const auto parameters = circuit.parameters();
  require(parameters.size() == 6,
          "frozen circuit exposes only learned projection parameters");
  for (const auto &named : parameters) {
    require(named.name.find("coefficients") == std::string::npos,
            "frozen compiled coefficients must stay out of Adam parameters");
  }
}

void test_ablation_and_steering() {
  conditional_reverse::Circuit circuit(exact_config());
  const auto batch = examples(circuit.task());
  const auto baseline = circuit.evaluate(batch);

  programmed::SequenceForwardOptions ablated_options;
  ablated_options.ablation = programmed::BatchRollAblation{
      .program_input_indices = {},
      .program_output = true,
      .shift = 1,
  };
  const auto ablated = circuit.evaluate(batch, ablated_options);
  require(ablated.metrics.exact_sequence_accuracy == 0.0,
          "rolling program outputs must break example association");
  const analysis::AblationSummary summary = analysis::summarize_ablation(
      baseline.per_example_token_accuracy, ablated.per_example_token_accuracy,
      analysis::MetricGoal::Maximize);
  require(summary.sample_count == 2 && summary.mean_degradation > 0.0 &&
              summary.fraction_degraded == 1.0,
          "paired ablation analysis must report causal degradation");

  programmed::SequenceForwardOptions force_copy;
  force_copy.steering.push_back({
      .input_index = 0,
      .positions = {0},
      .scales = {0.0F, 0.0F},
      .offsets = {0.0F, 1.0F},
  });
  const auto copied = circuit.evaluate(batch, force_copy);
  require(copied.metrics.exact_sequence_accuracy == 0.5,
          "forcing the copy coordinate must preserve copy and flip reverse");

  programmed::SequenceForwardOptions force_reverse;
  force_reverse.steering.push_back({
      .input_index = 0,
      .positions = {0},
      .scales = {0.0F, 0.0F},
      .offsets = {1.0F, 0.0F},
  });
  const auto reversed = circuit.evaluate(batch, force_reverse);
  require(reversed.metrics.exact_sequence_accuracy == 0.5,
          "forcing the reverse coordinate must preserve reverse and flip copy");

  const std::vector<conditional_reverse::Example> three_examples{
      circuit.task().make_example("abc"),
      circuit.task().make_example("bca"),
      circuit.task().make_example("cdb"),
  };
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  programmed::SequenceForwardOptions huge_identity_roll;
  huge_identity_roll.ablation = programmed::BatchRollAblation{
      .program_input_indices = {},
      .program_output = true,
      .shift = maximum - maximum % three_examples.size(),
  };
  require(circuit.evaluate(three_examples, huge_identity_roll)
                  .metrics.exact_sequence_accuracy == 1.0,
          "large batch-roll shifts must use overflow-safe modular arithmetic");
}

void test_gradients_controls_and_validation() {
  conditional_reverse::Circuit circuit(exact_config());
  const auto batch = examples(circuit.task());
  const auto sources = flattened_sources(batch);
  const auto result = circuit.forward(sources, batch.size());
  riftco_transformer::sum(result.output).backward();
  for (const auto &named : circuit.parameters()) {
    require(absolute_sum(named.parameter->gradient()) > 0.0,
            "gradient must reach learned projection " + named.name);
  }

  auto randomized_config = exact_config();
  randomized_config.lowering.initialization =
      lowering::CoefficientInitialization::RandomUniform;
  randomized_config.lowering.seed = 99;
  conditional_reverse::Circuit first_randomized(randomized_config);
  conditional_reverse::Circuit second_randomized(randomized_config);
  const auto first = first_randomized.forward(sources, batch.size());
  const auto second = second_randomized.forward(sources, batch.size());
  require(first.output.value().data().size() ==
              second.output.value().data().size(),
          "randomized control output size");
  bool differs_from_compiled = false;
  for (std::size_t index = 0; index < first.output.value().numel(); ++index) {
    require(first.output.value().flat(index) ==
                second.output.value().flat(index),
            "randomized control must be seeded");
    differs_from_compiled =
        differs_from_compiled ||
        first.output.value().flat(index) != result.output.value().flat(index);
  }
  require(differs_from_compiled && !first_randomized.adapter()
                                        .program()
                                        .metadata()
                                        .preserves_compiled_map,
          "randomized control must preserve shape but replace coefficients");

  auto trainable_config = exact_config();
  trainable_config.lowering.trainable = true;
  conditional_reverse::Circuit trainable(trainable_config);
  riftco_transformer::Parameter *coefficients = nullptr;
  const auto trainable_parameters = trainable.parameters();
  for (const auto &named : trainable_parameters) {
    if (named.name == "programmed.program.coefficients") {
      coefficients = named.parameter;
    }
  }
  require(coefficients != nullptr,
          "explicitly trainable circuit must expose its coefficients");
  std::vector<riftco_transformer::TokenId> task_targets;
  task_targets.reserve(batch.size() * circuit.task().config().sequence_length *
                       2);
  for (const auto &example : batch) {
    for (const auto symbol : example.source) {
      task_targets.push_back(static_cast<riftco_transformer::TokenId>(symbol));
    }
    for (const auto symbol : example.target) {
      task_targets.push_back(static_cast<riftco_transformer::TokenId>(symbol));
    }
  }
  const auto trainable_forward = trainable.forward(sources, batch.size());
  const auto loss =
      riftco_transformer::cross_entropy(trainable_forward.output, task_targets);
  loss.backward();
  require(absolute_sum(coefficients->gradient()) > 0.0,
          "task loss must reach explicitly trainable program coefficients");
  const std::vector<float> coefficients_before(
      coefficients->value().data().begin(), coefficients->value().data().end());
  riftco_transformer::Adam optimizer(trainable_parameters);
  const auto step = optimizer.step();
  require(step.step == 1 && !std::equal(coefficients_before.begin(),
                                        coefficients_before.end(),
                                        coefficients->value().data().begin(),
                                        coefficients->value().data().end()),
          "Adam must update explicitly trainable program coefficients");

  auto dense_config = exact_config();
  dense_config.lowering.strategy = lowering::kDenseContractionStrategy;
  require_throws_as<std::invalid_argument>(
      [&] { static_cast<void>(conditional_reverse::Circuit(dense_config)); },
      "conditional circuit must reject a non-attention strategy");

  auto wrong_query_axis_config = exact_config();
  wrong_query_axis_config.lowering.attention_query_axis = 0;
  require_throws_as<std::invalid_argument>(
      [&] {
        static_cast<void>(
            conditional_reverse::Circuit(wrong_query_axis_config));
      },
      "conditional circuit must reject attention query axis 0");
}

void test_evaluation_rejects_invalid_labels_and_logits() {
  conditional_reverse::Circuit invalid_label_circuit(exact_config());
  auto invalid_labels = examples(invalid_label_circuit.task());
  invalid_labels.front().target.front() =
      invalid_label_circuit.task().symbol_count();
  require_throws_as<std::out_of_range>(
      [&] {
        static_cast<void>(invalid_label_circuit.evaluate(invalid_labels));
      },
      "evaluation must reject target symbol IDs outside the alphabet");

  for (const float nonfinite : {
           std::numeric_limits<float>::quiet_NaN(),
           std::numeric_limits<float>::infinity(),
       }) {
    conditional_reverse::Circuit nonfinite_circuit(exact_config());
    bool replaced_bias = false;
    for (const auto &named :
         nonfinite_circuit.adapter().output_projection().parameters()) {
      if (named.name == "bias") {
        named.parameter->set_value(riftco_transformer::Tensor(
            {nonfinite_circuit.task().symbol_count()},
            std::vector<float>(nonfinite_circuit.task().symbol_count(),
                               nonfinite),
            nonfinite_circuit.backend()));
        replaced_bias = true;
      }
    }
    require(replaced_bias, "output projection bias must be addressable");
    const auto batch = examples(nonfinite_circuit.task());
    require_throws_as<std::domain_error>(
        [&] { static_cast<void>(nonfinite_circuit.evaluate(batch)); },
        "evaluation must reject every non-finite target logit");
  }
}

void test_constant_program_batch_and_packed_transfer() {
  namespace cajal = riftco_transformer::compiler::cajal;
  const cajal::MultilinearMap constant = cajal::MultilinearMap::constant(
      cajal::EncodedValue({1.0, 2.0, 3.0, 4.0}));
  auto lowered = lowering::lower_to_neural(constant);
  programmed::SequenceProjectionState projections{
      {},
      {
          riftco_transformer::Tensor(
              {2, 2}, std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F}),
          riftco_transformer::Tensor::zeros({2}),
      },
  };
  programmed::ProgrammedSequenceAdapter adapter(2,
                                                {
                                                    .source_offset = 0,
                                                    .source_length = 1,
                                                    .output_length = 2,
                                                    .target_offset = 1,
                                                    .inputs = {},
                                                },
                                                std::move(lowered),
                                                std::move(projections));
  const auto placed = adapter.forward(riftco_transformer::Variable(
      riftco_transformer::Tensor::zeros({2, 3, 2}), false));
  const std::vector<float> expected{
      0.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 0.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F,
  };
  require(std::equal(placed.output.value().data().begin(),
                     placed.output.value().data().end(), expected.begin(),
                     expected.end()),
          "zero-arity program must broadcast across every batch row");

  auto single_token_program = lowering::lower_to_neural(
      cajal::MultilinearMap::constant(cajal::EncodedValue({5.0, 6.0})));
  programmed::SequenceProjectionState single_token_projections{
      {},
      {
          riftco_transformer::Tensor(
              {2, 2}, std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F}),
          riftco_transformer::Tensor::zeros({2}),
      },
  };
  programmed::ProgrammedSequenceAdapter single_token_adapter(
      2,
      {
          .source_offset = 0,
          .source_length = 0,
          .output_length = 1,
          .target_offset = 0,
          .inputs = {},
      },
      std::move(single_token_program), std::move(single_token_projections));
  const auto single_token =
      single_token_adapter.forward(riftco_transformer::Variable(
          riftco_transformer::Tensor::zeros({2, 1, 2}), false));
  const std::vector<float> expected_single_token{5.0F, 6.0F, 5.0F, 6.0F};
  require(std::equal(single_token.output.value().data().begin(),
                     single_token.output.value().data().end(),
                     expected_single_token.begin(),
                     expected_single_token.end()),
          "zero-arity program must support a sole target token without a "
          "dummy source");

  auto unary_program =
      lowering::lower_to_neural(cajal::MultilinearMap::identity(2));
  programmed::SequenceProjectionState unary_projections{
      {
          {
              riftco_transformer::Tensor(
                  {2, 2}, std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F}),
              riftco_transformer::Tensor::zeros({2}),
          },
      },
      {
          riftco_transformer::Tensor(
              {2, 2}, std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F}),
          riftco_transformer::Tensor::zeros({2}),
      },
  };
  require_throws_as<std::invalid_argument>(
      [&] {
        static_cast<void>(programmed::ProgrammedSequenceAdapter(
            2,
            {
                .source_offset = 0,
                .source_length = 0,
                .output_length = 1,
                .target_offset = 0,
                .inputs =
                    {
                        {
                            .source =
                                programmed::ProgramInputSource::WholeSource,
                            .position = 0,
                        },
                    },
            },
            std::move(unary_program), std::move(unary_projections)));
      },
      "nonconstant programs must retain a source span");

  if (!riftco_transformer::execution_backend_available(
          riftco_transformer::ExecutionBackend::Metal)) {
    return;
  }
  conditional_reverse::Circuit circuit(exact_config());
  circuit.adapter().output_projection().quantize_weight_nf4(32);
  circuit.to(riftco_transformer::ExecutionBackend::Metal);
  require(
      circuit.backend() == riftco_transformer::ExecutionBackend::Metal &&
          circuit.adapter().output_projection().quantized_weight().backend() ==
              riftco_transformer::ExecutionBackend::Metal &&
          circuit.adapter().output_projection().bias().value().backend() ==
              riftco_transformer::ExecutionBackend::Metal,
      "adapter transfer must reach packed projection resources");
  const auto batch = examples(circuit.task());
  const auto metal_evaluation = circuit.evaluate(batch);
  require(metal_evaluation.metrics.token_accuracy == 1.0 &&
              metal_evaluation.metrics.exact_sequence_accuracy == 1.0,
          "packed Metal transfer must preserve circuit predictions");
}

void test_placement_graph_avoids_dense_routing_tensor() {
  namespace cajal = riftco_transformer::compiler::cajal;
  constexpr std::size_t batch_size = 2;
  constexpr std::size_t output_length = 64;
  constexpr std::size_t total_time = output_length + 1;

  const cajal::MultilinearMap constant = cajal::MultilinearMap::constant(
      cajal::EncodedValue(std::vector<double>(output_length, 1.0)));
  auto lowered = lowering::lower_to_neural(constant);
  programmed::SequenceProjectionState projections{
      {},
      {
          riftco_transformer::Tensor({1, 1}, std::vector<float>{1.0F}),
          riftco_transformer::Tensor::zeros({1}),
      },
  };
  programmed::ProgrammedSequenceAdapter adapter(
      1,
      {
          .source_offset = 0,
          .source_length = 1,
          .output_length = output_length,
          .target_offset = 1,
          .inputs = {},
      },
      std::move(lowered), std::move(projections));

  const auto placed = adapter.forward(riftco_transformer::Variable(
      riftco_transformer::Tensor::zeros({batch_size, total_time, 1}), false));
  require(placed.output.value().shape() ==
              riftco_transformer::Tensor::Shape{
                  batch_size,
                  total_time,
                  1,
              },
          "programmed placement output shape");
  for (std::size_t row = 0; row < batch_size; ++row) {
    require(placed.output.value().at({row, 0, 0}) == 0.0F,
            "programmed placement must preserve the source residual");
    for (std::size_t position = 1; position < total_time; ++position) {
      require(placed.output.value().at({row, position, 0}) == 1.0F,
              "programmed placement target value");
    }
  }

  const auto graph = placed.output.graph_statistics();
  const std::size_t dense_routing_node_elements =
      2 * batch_size * total_time * output_length;
  require(graph.node_tensor_elements < dense_routing_node_elements,
          "placement graph must not retain a dense [batch,time,output] "
          "routing Tensor and gradient");

  std::vector<float> upstream_values(batch_size * total_time, 0.0F);
  float expected_gradient = 0.0F;
  for (std::size_t row = 0; row < batch_size; ++row) {
    upstream_values[row * total_time] = 10000.0F;
    for (std::size_t position = 1; position < total_time; ++position) {
      const float value = static_cast<float>(row * 100 + position);
      upstream_values[row * total_time + position] = value;
      expected_gradient += value;
    }
  }
  placed.output.backward(riftco_transformer::Tensor(
      {batch_size, total_time, 1}, std::move(upstream_values)));
  require(adapter.output_projection().weight().gradient().flat(0) ==
                  expected_gradient &&
              adapter.output_projection().bias().gradient().flat(0) ==
                  expected_gradient,
          "placement custom gradient must return only the target-span VJP");
}

} // namespace

int main() {
  try {
    test_exact_attention_circuit_and_capture();
    test_ablation_and_steering();
    test_gradients_controls_and_validation();
    test_evaluation_rejects_invalid_labels_and_logits();
    test_constant_program_batch_and_packed_transfer();
    test_placement_graph_avoids_dense_routing_tensor();
    std::cout << "conditional reverse circuit tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "conditional reverse circuit test failure: " << error.what()
              << '\n';
    return 1;
  }
}
