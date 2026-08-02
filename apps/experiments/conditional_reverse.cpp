#include "riftco_transformer/analysis/ablation.hpp"
#include "riftco_transformer/analysis/pca.hpp"
#include "riftco_transformer/experiments/conditional_reverse/circuit.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace analysis = riftco_transformer::analysis;
namespace conditional_reverse =
    riftco_transformer::experiments::conditional_reverse;
namespace lowering = riftco_transformer::lowering;
namespace programmed = riftco_transformer::programmed;

constexpr std::size_t kBatchSize = 8;
constexpr std::size_t kSplitCount = 3;
constexpr std::size_t kPcaComponents = 2;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

conditional_reverse::CircuitConfig lab_config() {
  conditional_reverse::CircuitConfig config;
  config.task = {
      .sequence_length = 5,
      .alphabet = "abcde",
      .reverse_when_first_is = "a",
      .seed = 2026,
  };
  config.lowering.max_coefficient_elements = 4096;
  return config;
}

std::span<const conditional_reverse::SymbolId>
prediction_row(const conditional_reverse::Circuit &circuit,
               const conditional_reverse::EvaluationResult &evaluation,
               std::size_t row) {
  const std::size_t sequence_length = circuit.task().config().sequence_length;
  require(evaluation.predictions.size() ==
              evaluation.metrics.example_count * sequence_length,
          "evaluation returned an inconsistent prediction shape");
  require(row < evaluation.metrics.example_count,
          "prediction row is outside the batch");
  return {
      evaluation.predictions.data() + row * sequence_length,
      sequence_length,
  };
}

void validate_batch(const conditional_reverse::Circuit &circuit,
                    const std::vector<conditional_reverse::Example> &batch) {
  require(batch.size() == kBatchSize,
          "balanced generator returned the wrong batch size");
  std::size_t reverse_count = 0;
  for (std::size_t row = 0; row < batch.size(); ++row) {
    const auto &example = batch[row];
    require(example.reversed == (row % 2 == 0),
            "balanced generator did not alternate task branches");
    require(example.source.size() == circuit.task().config().sequence_length &&
                example.target.size() ==
                    circuit.task().config().sequence_length,
            "generated example has the wrong sequence length");
    reverse_count += example.reversed ? 1 : 0;
  }
  require(reverse_count * 2 == batch.size(),
          "generated batch is not branch-balanced");
}

void validate_disjoint_splits(
    const std::vector<conditional_reverse::Example> &examples) {
  std::set<std::vector<conditional_reverse::SymbolId>> unique_sources;
  for (const auto &example : examples) {
    require(unique_sources.insert(example.source).second,
            "deterministic analysis splits contain a duplicate source");
  }
}

void validate_baseline(const conditional_reverse::Circuit &circuit,
                       const std::vector<conditional_reverse::Example> &batch,
                       const conditional_reverse::EvaluationResult &baseline) {
  require(baseline.metrics.token_accuracy == 1.0,
          "compiled circuit lost exact token accuracy");
  require(baseline.metrics.exact_sequence_accuracy == 1.0,
          "compiled circuit lost exact sequence accuracy");
  for (std::size_t row = 0; row < batch.size(); ++row) {
    const auto prediction = prediction_row(circuit, baseline, row);
    require(std::equal(prediction.begin(), prediction.end(),
                       batch[row].target.begin(), batch[row].target.end()),
            "baseline prediction does not match its task target");
  }
}

void validate_steering(
    const conditional_reverse::Circuit &circuit,
    const std::vector<conditional_reverse::Example> &batch,
    const conditional_reverse::EvaluationResult &force_copy,
    const conditional_reverse::EvaluationResult &force_reverse) {
  for (std::size_t row = 0; row < batch.size(); ++row) {
    const auto copied = prediction_row(circuit, force_copy, row);
    require(std::equal(copied.begin(), copied.end(), batch[row].source.begin(),
                       batch[row].source.end()),
            "force-copy steering did not copy the source");

    const auto reversed = prediction_row(circuit, force_reverse, row);
    require(std::equal(reversed.begin(), reversed.end(),
                       batch[row].source.rbegin(), batch[row].source.rend()),
            "force-reverse steering did not reverse the source");
  }
}

void print_accuracy(std::string_view label,
                    const conditional_reverse::EvaluationMetrics &metrics) {
  std::cout << "  " << label << ": token=" << 100.0 * metrics.token_accuracy
            << "% (" << metrics.correct_token_count << '/'
            << metrics.token_count
            << "), exact=" << 100.0 * metrics.exact_sequence_accuracy << "% ("
            << metrics.correct_sequence_count << '/' << metrics.example_count
            << ")\n";
}

void print_steering_example(
    std::string_view label, std::size_t row,
    const conditional_reverse::Circuit &circuit,
    const std::vector<conditional_reverse::Example> &batch,
    const conditional_reverse::EvaluationResult &force_copy,
    const conditional_reverse::EvaluationResult &force_reverse) {
  const auto &task = circuit.task();
  std::cout << "  " << label << ": source="
            << task.decode(std::span<const conditional_reverse::SymbolId>(
                   batch[row].source))
            << ", target="
            << task.decode(std::span<const conditional_reverse::SymbolId>(
                   batch[row].target))
            << ", force-copy="
            << task.decode(prediction_row(circuit, force_copy, row))
            << ", force-reverse="
            << task.decode(prediction_row(circuit, force_reverse, row)) << '\n';
}

void run_lab() {
  conditional_reverse::Circuit circuit(lab_config());
  const auto &lowered = circuit.adapter().program().metadata();
  const auto &resources = circuit.resources();
  require(lowered.selected_strategy == lowering::kLinearAttentionStrategy &&
              lowered.preserves_compiled_map,
          "lab requires an exact compiled linear-attention circuit");

  const std::vector<conditional_reverse::Example> examples =
      circuit.task().generate_balanced(kSplitCount * kBatchSize);
  validate_disjoint_splits(examples);
  const std::vector<conditional_reverse::Example> training_batch(
      examples.begin(), examples.begin() + kBatchSize);
  const std::vector<conditional_reverse::Example> validation_batch(
      examples.begin() + kBatchSize, examples.begin() + 2 * kBatchSize);
  const std::vector<conditional_reverse::Example> test_batch(
      examples.begin() + 2 * kBatchSize, examples.end());
  validate_batch(circuit, training_batch);
  validate_batch(circuit, validation_batch);
  validate_batch(circuit, test_batch);

  programmed::SequenceForwardOptions capture;
  capture.capture_representations = true;
  const conditional_reverse::EvaluationResult training_baseline =
      circuit.evaluate(training_batch, capture);
  const conditional_reverse::EvaluationResult validation_baseline =
      circuit.evaluate(validation_batch, capture);
  const conditional_reverse::EvaluationResult baseline =
      circuit.evaluate(test_batch, capture);
  validate_baseline(circuit, training_batch, training_baseline);
  validate_baseline(circuit, validation_batch, validation_baseline);
  validate_baseline(circuit, test_batch, baseline);
  require(baseline.representations.contains("program_output"),
          "test trace is missing program_output");

  conditional_reverse::CircuitConfig randomized_config = lab_config();
  randomized_config.lowering.initialization =
      lowering::CoefficientInitialization::RandomUniform;
  randomized_config.lowering.seed = 2026;
  conditional_reverse::Circuit randomized(randomized_config);
  const conditional_reverse::EvaluationResult randomized_result =
      randomized.evaluate(test_batch);
  require(randomized.resources().input_dimensions ==
                  circuit.resources().input_dimensions &&
              randomized.resources().output_dimension ==
                  circuit.resources().output_dimension &&
              !randomized.adapter().program().metadata().preserves_compiled_map,
          "randomized control must preserve shape but replace coefficients");

  programmed::SequenceForwardOptions roll_output;
  roll_output.ablation = programmed::BatchRollAblation{
      .program_input_indices = {},
      .program_output = true,
      .shift = 1,
  };
  const conditional_reverse::EvaluationResult ablated =
      circuit.evaluate(test_batch, roll_output);
  const analysis::AblationSummary degradation = analysis::summarize_ablation(
      baseline.per_example_token_accuracy, ablated.per_example_token_accuracy,
      analysis::MetricGoal::Maximize);
  require(degradation.sample_count == test_batch.size() &&
              degradation.mean_degradation > 0.0 &&
              degradation.fraction_degraded > 0.0,
          "batch-roll ablation did not produce paired degradation");

  const auto &training_program_output =
      training_baseline.representations.at("program_output");
  const auto &validation_program_output =
      validation_baseline.representations.at("program_output");
  const auto &test_program_output =
      baseline.representations.at("program_output");
  require(training_program_output.leading_shape ==
                  std::vector<std::size_t>({
                      training_batch.size(),
                      circuit.task().config().sequence_length,
                  }) &&
              training_program_output.observations.rows ==
                  training_batch.size() *
                      circuit.task().config().sequence_length &&
              training_program_output.observations.columns ==
                  circuit.task().symbol_count(),
          "captured training program_output has an unexpected shape");
  analysis::PcaOptions pca_options;
  pca_options.component_count = kPcaComponents;
  const analysis::PcaFit pca = analysis::fit_pca(
      training_program_output.observations.view(), pca_options);
  const analysis::Matrix validation_scores = analysis::transform_pca(
      validation_program_output.observations.view(), pca.model);
  const analysis::Matrix test_scores = analysis::transform_pca(
      test_program_output.observations.view(), pca.model);
  require(pca.model.component_count == kPcaComponents &&
              pca.model.feature_count == circuit.task().symbol_count() &&
              pca.scores.rows == training_program_output.observations.rows &&
              pca.scores.columns == kPcaComponents &&
              validation_scores.rows ==
                  validation_program_output.observations.rows &&
              validation_scores.columns == kPcaComponents &&
              test_scores.rows == test_program_output.observations.rows &&
              test_scores.columns == kPcaComponents &&
              pca.model.explained_variance_ratio.size() == kPcaComponents &&
              std::isfinite(pca.off_diagonal_residual),
          "PCA fit returned inconsistent dimensions or diagnostics");
  for (const double ratio : pca.model.explained_variance_ratio) {
    require(std::isfinite(ratio),
            "PCA returned a non-finite explained-variance ratio");
  }

  programmed::SequenceForwardOptions force_copy_options;
  force_copy_options.steering.push_back({
      .input_index = 0,
      .positions = {0},
      .scales = {0.0F, 0.0F},
      .offsets = {0.0F, 1.0F},
  });
  const conditional_reverse::EvaluationResult force_copy =
      circuit.evaluate(test_batch, force_copy_options);

  programmed::SequenceForwardOptions force_reverse_options;
  force_reverse_options.steering.push_back({
      .input_index = 0,
      .positions = {0},
      .scales = {0.0F, 0.0F},
      .offsets = {1.0F, 0.0F},
  });
  const conditional_reverse::EvaluationResult force_reverse =
      circuit.evaluate(test_batch, force_reverse_options);
  validate_steering(circuit, test_batch, force_copy, force_reverse);

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Conditional reverse compiled-circuit lab\n"
            << "disjoint deterministic splits: train=" << training_batch.size()
            << ", validation=" << validation_batch.size()
            << ", test=" << test_batch.size()
            << " (each balanced 50/50 reverse/copy), sequence length "
            << circuit.task().config().sequence_length << "\n\n";

  std::cout << "Compiled program\n"
            << "  strategy: " << lowered.requested_strategy << " -> "
            << lowered.selected_strategy
            << " (exact=" << (lowered.preserves_compiled_map ? "yes" : "no")
            << ", fallback=" << (lowered.used_fallback ? "yes" : "no") << ")\n"
            << "  dimensions: condition=" << resources.condition_dimension
            << ", sequence=" << resources.sequence_dimension
            << ", output=" << resources.output_dimension << "\n"
            << "  coefficients: " << resources.coefficient_elements
            << " dense, " << resources.nonzero_coefficient_elements
            << " nonzero, " << resources.dense_coefficient_bytes
            << " bytes\n\n";

  std::cout << "Accuracy\n";
  print_accuracy("validation baseline", validation_baseline.metrics);
  print_accuracy("held-out test baseline", baseline.metrics);
  print_accuracy("held-out seeded randomized-head control",
                 randomized_result.metrics);
  print_accuracy("held-out batch-roll output ablation", ablated.metrics);
  std::cout << "  paired token degradation: mean="
            << 100.0 * degradation.mean_degradation << " percentage points, SE="
            << 100.0 * degradation.paired_standard_error
            << ", degraded examples=" << 100.0 * degradation.fraction_degraded
            << "%\n\n";

  std::cout << "PCA of captured program_output (fit on train only)\n"
            << "  fit observations: "
            << training_program_output.observations.rows << " x "
            << training_program_output.observations.columns
            << ", validation scores: " << validation_scores.rows << " x "
            << validation_scores.columns
            << ", test scores: " << test_scores.rows << " x "
            << test_scores.columns
            << ", components: " << pca.model.component_count << "\n"
            << "  explained variance:";
  for (std::size_t component = 0;
       component < pca.model.explained_variance_ratio.size(); ++component) {
    std::cout << " PC" << component + 1 << '='
              << 100.0 * pca.model.explained_variance_ratio[component] << '%';
  }
  std::cout << " (Jacobi residual=" << pca.off_diagonal_residual << ")\n\n";

  std::cout << "Condition steering\n";
  print_accuracy("force copy", force_copy.metrics);
  print_accuracy("force reverse", force_reverse.metrics);
  print_steering_example("reverse-branch example", 0, circuit, test_batch,
                         force_copy, force_reverse);
  print_steering_example("copy-branch example", 1, circuit, test_batch,
                         force_copy, force_reverse);
}

} // namespace

int main() {
  try {
    run_lab();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "conditional reverse lab failed: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "conditional reverse lab failed: unknown error\n";
    return 1;
  }
}
