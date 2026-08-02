#include "riftco_transformer/analysis/ablation.hpp"
#include "riftco_transformer/analysis/pca.hpp"
#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/experiments/conditional_reverse/learned.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace analysis = riftco_transformer::analysis;
namespace conditional_reverse =
    riftco_transformer::experiments::conditional_reverse;

struct LabOptions {
  conditional_reverse::HybridVariant variant =
      conditional_reverse::HybridVariant::F;
  riftco_transformer::ExecutionBackend backend =
      riftco_transformer::ExecutionBackend::Cpu;
  bool paper_scale = false;
  std::optional<std::size_t> epochs;
  std::optional<std::size_t> maximum_steps;
};

[[nodiscard]] std::size_t parse_positive_size(std::string_view text,
                                              std::string_view option) {
  std::size_t result = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), result);
  if (error != std::errc{} || end != text.data() + text.size() || result == 0) {
    throw std::invalid_argument(std::string(option) +
                                " requires a positive integer");
  }
  return result;
}

[[nodiscard]] conditional_reverse::HybridVariant
parse_variant(std::string_view value) {
  if (value == "f" || value == "F") {
    return conditional_reverse::HybridVariant::F;
  }
  if (value == "p" || value == "P") {
    return conditional_reverse::HybridVariant::P;
  }
  if (value == "t" || value == "T") {
    return conditional_reverse::HybridVariant::T;
  }
  if (value == "i" || value == "I") {
    return conditional_reverse::HybridVariant::I;
  }
  throw std::invalid_argument("--variant must be F, P, T, or I");
}

[[nodiscard]] riftco_transformer::ExecutionBackend
parse_backend(std::string_view value) {
  using riftco_transformer::ExecutionBackend;
  if (value == "cpu") {
    return ExecutionBackend::Cpu;
  }
  if (value == "metal") {
    return ExecutionBackend::Metal;
  }
  if (value == "cuda") {
    return ExecutionBackend::Cuda;
  }
  if (value == "tpu") {
    return ExecutionBackend::Tpu;
  }
  throw std::invalid_argument("--backend must be cpu, metal, cuda, or tpu");
}

void print_usage(std::ostream &output, std::string_view executable) {
  output
      << "Usage: " << executable
      << " [--variant F|P|T|I] [--backend NAME] [--epochs N] [--steps N]"
         " [--paper]\n\n"
      << "F: frozen compiled conditional program\n"
      << "P: frozen unconditional reverse program\n"
      << "T: randomized trainable program with F's shape\n"
      << "I: learned Transformer components only\n\n"
      << "The default is a quick deterministic smoke experiment. --paper uses"
         " the paper-scale L=15, d_model=20, 10k/5k/1k/1k protocol and can"
         " take substantially longer.\n";
}

[[nodiscard]] LabOptions parse_options(int argc, char **argv) {
  LabOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      print_usage(std::cout, argv[0]);
      std::exit(0);
    }
    if (argument == "--paper") {
      options.paper_scale = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument(std::string(argument) + " requires a value");
    }
    const std::string_view value(argv[++index]);
    if (argument == "--variant") {
      options.variant = parse_variant(value);
    } else if (argument == "--backend") {
      options.backend = parse_backend(value);
    } else if (argument == "--epochs") {
      options.epochs = parse_positive_size(value, argument);
    } else if (argument == "--steps") {
      options.maximum_steps = parse_positive_size(value, argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  return options;
}

[[nodiscard]] std::string_view
variant_name(conditional_reverse::HybridVariant variant) {
  switch (variant) {
  case conditional_reverse::HybridVariant::F:
    return "F (frozen compiled conditional program)";
  case conditional_reverse::HybridVariant::P:
    return "P (frozen unconditional reverse program)";
  case conditional_reverse::HybridVariant::T:
    return "T (randomized trainable program)";
  case conditional_reverse::HybridVariant::I:
    return "I (no program)";
  }
  throw std::invalid_argument("unknown hybrid variant");
}

struct LabConfiguration {
  conditional_reverse::LearnedHybridConfig model;
  conditional_reverse::LearnedSplitSizes splits;
  conditional_reverse::LearnedTrainingConfig training;
};

[[nodiscard]] LabConfiguration make_configuration(const LabOptions &options) {
  LabConfiguration result;
  result.model.variant = options.variant;
  result.model.lowering.backend = options.backend;
  result.model.lowering.seed = 42;
  result.model.seed = 42;
  result.training.seed = 42;
  result.training.adam.learning_rate = 1.0e-2F;

  if (options.paper_scale) {
    // Artifact-faithful dimensions and split sizes. The learned architecture
    // itself is the same in smoke mode; only the resource scale changes.
    result.model.protocol = {};
    result.model.model_width = 20;
    result.model.head_count = 2;
    result.model.feed_forward_width = 80;
    result.model.lowering.max_coefficient_elements = std::size_t{1} << 24U;
    result.splits = {};
    result.training.epochs = 10;
    result.training.batch_size = 128;
    result.training.evaluation_batch_size = 256;
  } else {
    result.model.protocol = {
        .sequence_length = 3,
        .alphabet = "abcdefghij",
        .reverse_when_first_is = "ae",
        .delimiter = '|',
        .seed = 42,
    };
    result.model.model_width = 8;
    result.model.head_count = 2;
    result.model.feed_forward_width = 24;
    result.model.lowering.max_coefficient_elements = 1U << 14U;
    result.splits = {
        .train = 128,
        .probe = 64,
        .validation = 64,
        .test = 64,
        .disjoint_sources = true,
    };
    result.training.epochs = 8;
    result.training.batch_size = 16;
    result.training.evaluation_batch_size = 64;
    result.training.maximum_steps = 64;
  }

  if (options.epochs.has_value()) {
    result.training.epochs = *options.epochs;
  }
  if (options.maximum_steps.has_value()) {
    result.training.maximum_steps = options.maximum_steps;
  }
  return result;
}

void print_metrics(
    std::string_view label,
    const conditional_reverse::LearnedEvaluationMetrics &metrics) {
  std::cout << "  " << label << ": loss=" << metrics.loss
            << ", token=" << 100.0 * metrics.target_token_accuracy
            << "%, exact=" << 100.0 * metrics.exact_sequence_accuracy
            << "%, reverse-token="
            << 100.0 * metrics.reverse_target_token_accuracy
            << "%, copy-token=" << 100.0 * metrics.copy_target_token_accuracy
            << "%\n";
}

void print_ablation(std::string_view label,
                    const analysis::AblationSummary &summary) {
  std::cout << "  " << label
            << ": paired token degradation=" << 100.0 * summary.mean_degradation
            << " percentage points, SE="
            << 100.0 * summary.paired_standard_error
            << ", degraded examples=" << 100.0 * summary.fraction_degraded
            << "%\n";
}

[[nodiscard]] std::string
token_label(riftco_transformer::TokenId token,
            const conditional_reverse::LearnedProtocolConfig &config) {
  if (token == config.delimiter_token()) {
    return std::string(1, config.delimiter);
  }
  if (static_cast<std::size_t>(token) >= config.alphabet.size()) {
    throw std::logic_error("PCA association token is outside vocabulary");
  }
  return std::string(1, config.alphabet[token]);
}

void print_pca_associations(const analysis::Matrix &scores,
                            const conditional_reverse::LearnedDataset &dataset,
                            std::size_t position_count, bool program_output) {
  if (scores.columns == 0 || scores.rows != dataset.size() * position_count) {
    throw std::logic_error("PCA score shape cannot be associated with data");
  }
  const auto &config = dataset.config();
  std::vector<double> position_sums(position_count, 0.0);
  std::vector<std::size_t> position_counts(position_count, 0);
  std::vector<double> token_sums(config.vocabulary_size(), 0.0);
  std::vector<std::size_t> token_counts(config.vocabulary_size(), 0);
  double reverse_sum = 0.0;
  double copy_sum = 0.0;
  std::size_t reverse_count = 0;
  std::size_t copy_count = 0;

  for (std::size_t row = 0; row < dataset.size(); ++row) {
    const auto &example = dataset[row];
    for (std::size_t position = 0; position < position_count; ++position) {
      const double score =
          scores.values[(row * position_count + position) * scores.columns];
      position_sums[position] += score;
      ++position_counts[position];
      if (example.reversed) {
        reverse_sum += score;
        ++reverse_count;
      } else {
        copy_sum += score;
        ++copy_count;
      }
      const riftco_transformer::TokenId token =
          program_output ? example.tokens[config.sequence_length + 1 + position]
                         : example.inputs[position];
      token_sums[token] += score;
      ++token_counts[token];
    }
  }
  if (reverse_count == 0 || copy_count == 0) {
    throw std::logic_error(
        "PCA branch association requires reverse and copy examples");
  }

  std::size_t minimum_token = 0;
  std::size_t maximum_token = 0;
  double minimum_mean = std::numeric_limits<double>::infinity();
  double maximum_mean = -std::numeric_limits<double>::infinity();
  for (std::size_t token = 0; token < token_counts.size(); ++token) {
    if (token_counts[token] == 0) {
      continue;
    }
    const double mean =
        token_sums[token] / static_cast<double>(token_counts[token]);
    if (mean < minimum_mean) {
      minimum_mean = mean;
      minimum_token = token;
    }
    if (mean > maximum_mean) {
      maximum_mean = mean;
      maximum_token = token;
    }
  }

  std::cout << "  held-out PC1 association: reverse-mean="
            << reverse_sum / static_cast<double>(reverse_count)
            << ", copy-mean=" << copy_sum / static_cast<double>(copy_count)
            << "\n  held-out PC1 mean by position:";
  for (std::size_t position = 0; position < position_count; ++position) {
    std::cout << ' ' << position << '='
              << position_sums[position] /
                     static_cast<double>(position_counts[position]);
  }
  std::cout
      << "\n  held-out PC1 token extremes: "
      << token_label(static_cast<riftco_transformer::TokenId>(minimum_token),
                     config)
      << '=' << minimum_mean << ", "
      << token_label(static_cast<riftco_transformer::TokenId>(maximum_token),
                     config)
      << '=' << maximum_mean << '\n';
}

void run_analysis(const conditional_reverse::LearnedHybrid &model,
                  const conditional_reverse::LearnedDatasetSplits &datasets,
                  std::size_t evaluation_batch_size, bool paper_scale) {
  conditional_reverse::LearnedForwardOptions capture;
  capture.capture_representations = true;
  const auto probe_result = conditional_reverse::evaluate_learned_dataset(
      model, datasets.probe, evaluation_batch_size, capture);
  const auto validation_result = conditional_reverse::evaluate_learned_dataset(
      model, datasets.validation, evaluation_batch_size, capture);
  const auto baseline = conditional_reverse::evaluate_learned_dataset(
      model, datasets.test, evaluation_batch_size, capture);

  const std::string representation_name =
      model.has_program() ? "program.program_output" : "h1";
  const auto &probe_representation =
      probe_result.representations.at(representation_name);
  const auto &validation_representation =
      validation_result.representations.at(representation_name);
  const auto &test_representation =
      baseline.representations.at(representation_name);

  analysis::PcaOptions pca_options;
  pca_options.component_count =
      std::min<std::size_t>(2, probe_representation.observations.columns);
  const analysis::PcaFit pca =
      analysis::fit_pca(probe_representation.observations.view(), pca_options);
  const analysis::Matrix validation_scores = analysis::transform_pca(
      validation_representation.observations.view(), pca.model);
  const analysis::Matrix test_scores = analysis::transform_pca(
      test_representation.observations.view(), pca.model);

  conditional_reverse::LearnedForwardOptions attention_ablation;
  attention_ablation.ablate_learned_attention = true;
  const auto without_attention = conditional_reverse::evaluate_learned_dataset(
      model, datasets.test, evaluation_batch_size, attention_ablation);
  const analysis::AblationSummary attention_summary =
      analysis::summarize_ablation(baseline.per_example_token_accuracy,
                                   without_attention.per_example_token_accuracy,
                                   analysis::MetricGoal::Maximize);

  std::cout << "\nInterpretation stage (held-out; PCA fit on probe only)\n"
            << "  representation: " << representation_name << " ("
            << probe_representation.observations.rows << " x "
            << probe_representation.observations.columns << ")\n"
            << "  validation/test PCA scores: " << validation_scores.rows
            << " x " << validation_scores.columns << " / " << test_scores.rows
            << " x " << test_scores.columns << "\n"
            << "  explained variance:";
  for (std::size_t component = 0;
       component < pca.model.explained_variance_ratio.size(); ++component) {
    std::cout << " PC" << component + 1 << '='
              << 100.0 * pca.model.explained_variance_ratio[component] << '%';
  }
  std::cout << '\n';
  if (test_representation.leading_shape.size() != 2 ||
      test_representation.leading_shape.front() != datasets.test.size()) {
    throw std::logic_error(
        "learned PCA representation has an unexpected sequence shape");
  }
  print_pca_associations(test_scores, datasets.test,
                         test_representation.leading_shape[1],
                         model.has_program());
  print_ablation("batch-roll learned-attention output", attention_summary);

  if (model.has_program()) {
    conditional_reverse::LearnedForwardOptions program_ablation;
    program_ablation.ablate_program_output = true;
    const auto without_program = conditional_reverse::evaluate_learned_dataset(
        model, datasets.test, evaluation_batch_size, program_ablation);
    const analysis::AblationSummary program_summary =
        analysis::summarize_ablation(baseline.per_example_token_accuracy,
                                     without_program.per_example_token_accuracy,
                                     analysis::MetricGoal::Maximize);
    print_ablation("batch-roll program output", program_summary);

    conditional_reverse::LearnedForwardOptions combined_ablation;
    combined_ablation.ablate_learned_attention = true;
    combined_ablation.ablate_program_output = true;
    const auto without_attention_or_program =
        conditional_reverse::evaluate_learned_dataset(
            model, datasets.test, evaluation_batch_size, combined_ablation);
    const analysis::AblationSummary combined_summary =
        analysis::summarize_ablation(
            baseline.per_example_token_accuracy,
            without_attention_or_program.per_example_token_accuracy,
            analysis::MetricGoal::Maximize);
    print_ablation("batch-roll learned attention + program output",
                   combined_summary);
  }

  if (model.variant() == conditional_reverse::HybridVariant::F) {
    const std::size_t steering_count = paper_scale ? 5'000 : 64;
    conditional_reverse::LearnedDataset steering_dataset(
        model.config().protocol,
        conditional_reverse::generate_balanced_learned_examples(
            model.config().protocol, steering_count, model.config().seed));
    const auto steering_baseline =
        conditional_reverse::evaluate_learned_dataset(model, steering_dataset,
                                                      evaluation_batch_size);
    conditional_reverse::LearnedForwardOptions suppress_selector;
    suppress_selector.steering.push_back({
        .position = 0,
        .selected_coordinate_scale = 0.0F,
        .other_coordinate_scale = 100.0F,
    });
    conditional_reverse::LearnedForwardOptions amplify_selector;
    amplify_selector.steering.push_back({
        .position = 0,
        .selected_coordinate_scale = 100.0F,
        .other_coordinate_scale = 0.0F,
    });
    const auto suppressed = conditional_reverse::evaluate_learned_dataset(
        model, steering_dataset, evaluation_batch_size, suppress_selector);
    const auto amplified = conditional_reverse::evaluate_learned_dataset(
        model, steering_dataset, evaluation_batch_size, amplify_selector);
    const auto baseline_hypotheses =
        conditional_reverse::score_learned_hypotheses(
            steering_dataset.examples(), steering_baseline.predictions,
            model.config().protocol);
    const auto suppressed_hypotheses =
        conditional_reverse::score_learned_hypotheses(
            steering_dataset.examples(), suppressed.predictions,
            model.config().protocol);
    const auto amplified_hypotheses =
        conditional_reverse::score_learned_hypotheses(
            steering_dataset.examples(), amplified.predictions,
            model.config().protocol);
    std::cout << "  F selector steering on " << steering_count
              << " balanced examples:\n"
              << "    baseline: copy-token="
              << 100.0 * baseline_hypotheses.copy_target_token_accuracy
              << "%, reverse-token="
              << 100.0 * baseline_hypotheses.reverse_target_token_accuracy
              << "%\n"
              << "    force copy (0,100): copy-token="
              << 100.0 * suppressed_hypotheses.copy_target_token_accuracy
              << "%, reverse-token="
              << 100.0 * suppressed_hypotheses.reverse_target_token_accuracy
              << "%\n"
              << "    force reverse (100,0): copy-token="
              << 100.0 * amplified_hypotheses.copy_target_token_accuracy
              << "%, reverse-token="
              << 100.0 * amplified_hypotheses.reverse_target_token_accuracy
              << "%\n";
  }
  std::cout << "  PCA is observational; paired ablations and steering are the"
               " causal checks.\n";
}

void run_lab(const LabOptions &options) {
  const LabConfiguration config = make_configuration(options);
  if (!riftco_transformer::execution_backend_available(options.backend)) {
    throw std::runtime_error(
        "requested backend is unavailable: " +
        std::string(riftco_transformer::execution_backend_unavailability_reason(
            options.backend)));
  }

  auto datasets = conditional_reverse::generate_learned_datasets(
      config.model.protocol, config.splits);
  conditional_reverse::LearnedHybrid model(config.model);
  conditional_reverse::LearnedHybridTrainer trainer(model, config.training);

  const auto initial_validation = conditional_reverse::evaluate_learned_dataset(
      model, datasets.validation, config.training.evaluation_batch_size);
  const conditional_reverse::LearnedTrainingHistory history =
      trainer.fit(datasets.train, datasets.validation);
  const auto test = conditional_reverse::evaluate_learned_dataset(
      model, datasets.test, config.training.evaluation_batch_size);

  std::cout << std::fixed << std::setprecision(2)
            << "Learned conditional-reverse hybrid lab\n"
            << "  variant: " << variant_name(options.variant) << "\n"
            << "  backend: "
            << riftco_transformer::execution_backend_name(options.backend)
            << "\n"
            << "  scale: " << (options.paper_scale ? "paper" : "smoke")
            << ", L=" << config.model.protocol.sequence_length
            << ", d_model=" << config.model.model_width
            << ", learned attention heads=" << 2 * config.model.head_count
            << "\n"
            << "  splits: train=" << datasets.train.size()
            << ", probe=" << datasets.probe.size()
            << ", validation=" << datasets.validation.size()
            << ", test=" << datasets.test.size() << "\n"
            << "  split sampling: "
            << (config.splits.disjoint_sources ? "source-disjoint"
                                               : "artifact random stream")
            << "\n"
            << "  Adam steps: " << trainer.optimizer().step_count() << "\n\n"
            << "Generalization\n";
  print_metrics("validation before training", initial_validation.metrics);
  if (!history.epochs.empty()) {
    print_metrics("validation after training",
                  history.epochs.back().validation);
  }
  print_metrics("held-out test", test.metrics);
  run_analysis(model, datasets, config.training.evaluation_batch_size,
               options.paper_scale);
}

} // namespace

int main(int argc, char **argv) {
  try {
    run_lab(parse_options(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "learned conditional-reverse lab failed: " << error.what()
              << '\n';
    return 1;
  } catch (...) {
    std::cerr << "learned conditional-reverse lab failed: unknown error\n";
    return 1;
  }
}
