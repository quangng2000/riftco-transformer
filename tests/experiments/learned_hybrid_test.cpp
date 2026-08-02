#include "riftco_transformer/experiments/conditional_reverse/learned_training.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace conditional_reverse =
    riftco_transformer::experiments::conditional_reverse;
namespace lowering = riftco_transformer::lowering;

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

conditional_reverse::LearnedProtocolConfig small_protocol() {
  return {
      .sequence_length = 2,
      .alphabet = "abcd",
      .reverse_when_first_is = "a",
      .delimiter = '|',
      .seed = 17,
  };
}

conditional_reverse::LearnedExample
make_example(const conditional_reverse::LearnedProtocolConfig &config,
             const std::string &source) {
  require(source.size() == config.sequence_length, "test source length");
  std::vector<riftco_transformer::TokenId> source_tokens;
  for (const char symbol : source) {
    source_tokens.push_back(
        static_cast<riftco_transformer::TokenId>(config.alphabet.find(symbol)));
  }
  const bool reversed =
      config.reverse_when_first_is.find(source.front()) != std::string::npos;
  std::vector<riftco_transformer::TokenId> output = source_tokens;
  if (reversed) {
    std::reverse(output.begin(), output.end());
  }
  conditional_reverse::LearnedExample result;
  result.reversed = reversed;
  result.tokens = source_tokens;
  result.tokens.push_back(config.delimiter_token());
  result.tokens.insert(result.tokens.end(), output.begin(), output.end());
  result.inputs.assign(result.tokens.begin(), result.tokens.end() - 1);
  result.targets.assign(result.tokens.begin() + 1, result.tokens.end());
  return result;
}

std::vector<conditional_reverse::LearnedExample> branch_examples() {
  const auto config = small_protocol();
  return {make_example(config, "ab"), make_example(config, "bc")};
}

conditional_reverse::LearnedHybridConfig
small_model_config(conditional_reverse::HybridVariant variant) {
  conditional_reverse::LearnedHybridConfig config;
  config.protocol = small_protocol();
  config.variant = variant;
  config.model_width = 4;
  config.head_count = 2;
  config.feed_forward_width = 8;
  config.seed = 23;
  config.lowering.max_coefficient_elements = 4096;
  return config;
}

bool exposes_coefficients(conditional_reverse::LearnedHybrid &model) {
  const auto parameters = model.parameters();
  return std::any_of(
      parameters.begin(), parameters.end(), [](const auto &named) {
        return named.name.find("coefficients") != std::string::npos;
      });
}

void test_dataset_protocol_and_validation() {
  const auto config = small_protocol();
  const conditional_reverse::LearnedSplitSizes sizes{
      .train = 4,
      .probe = 3,
      .validation = 2,
      .test = 2,
      .disjoint_sources = true,
  };
  const auto first =
      conditional_reverse::generate_learned_datasets(config, sizes);
  const auto second =
      conditional_reverse::generate_learned_datasets(config, sizes);
  require(first.train.size() == 4 && first.probe.size() == 3 &&
              first.validation.size() == 2 && first.test.size() == 2,
          "configurable learned split sizes");
  require(first.train[0].tokens == second.train[0].tokens &&
              first.test[1].tokens == second.test[1].tokens,
          "learned dataset generation must be seeded and deterministic");
  std::set<std::vector<riftco_transformer::TokenId>> unique_sources;
  const auto record_sources =
      [&](const conditional_reverse::LearnedDataset &split) {
        for (const auto &example : split.examples()) {
          const auto end = example.tokens.begin() +
                           static_cast<std::ptrdiff_t>(config.sequence_length);
          require(unique_sources.insert({example.tokens.begin(), end}).second,
                  "disjoint learned splits must not repeat a source");
        }
      };
  record_sources(first.train);
  record_sources(first.probe);
  record_sources(first.validation);
  record_sources(first.test);
  for (const auto &example : first.train.examples()) {
    require(example.tokens.size() == 5 && example.inputs.size() == 4 &&
                example.targets.size() == 4 &&
                example.tokens[2] == config.delimiter_token(),
            "teacher-forced source delimiter target protocol");
  }

  const auto balanced =
      conditional_reverse::generate_balanced_learned_examples(config, 8, 91);
  require(balanced.size() == 8, "balanced learned analysis generator size");
  for (std::size_t index = 0; index < balanced.size(); ++index) {
    require(balanced[index].reversed == (index % 2 == 0),
            "balanced learned analysis generator must alternate branches");
  }

  auto impossible_sizes = sizes;
  impossible_sizes.train = 6;
  impossible_sizes.probe = 6;
  impossible_sizes.validation = 3;
  impossible_sizes.test = 2;
  require_throws_as<std::invalid_argument>(
      [&] {
        static_cast<void>(conditional_reverse::generate_learned_datasets(
            config, impossible_sizes));
      },
      "disjoint splits must reject a request larger than the source space");

  auto invalid_flag = branch_examples().front();
  invalid_flag.reversed = !invalid_flag.reversed;
  require_throws_as<std::invalid_argument>(
      [&] {
        static_cast<void>(conditional_reverse::LearnedDataset(
            config, std::vector{invalid_flag}));
      },
      "dataset must reject a branch flag inconsistent with first symbol");

  auto invalid_label = branch_examples().front();
  invalid_label.tokens[3] = config.delimiter_token();
  invalid_label.targets.assign(invalid_label.tokens.begin() + 1,
                               invalid_label.tokens.end());
  require_throws_as<std::invalid_argument>(
      [&] {
        static_cast<void>(conditional_reverse::LearnedDataset(
            config, std::vector{invalid_label}));
      },
      "dataset must reject a delimiter in the output half");

  auto invalid_semantics = branch_examples().front();
  invalid_semantics.tokens[3] = 0;
  invalid_semantics.targets.assign(invalid_semantics.tokens.begin() + 1,
                                   invalid_semantics.tokens.end());
  require_throws_as<std::invalid_argument>(
      [&] {
        static_cast<void>(conditional_reverse::LearnedDataset(
            config, std::vector{invalid_semantics}));
      },
      "dataset must reject an output that violates reverse semantics");
}

void test_variants_parameter_sharing_and_captures() {
  conditional_reverse::LearnedHybrid model_f(
      small_model_config(conditional_reverse::HybridVariant::F));
  require(model_f.has_program() &&
              model_f.program_core()->logical_input_count() == 2 &&
              model_f.program_core()->input_projection_count() == 1 &&
              model_f.program_core()->input_projection_group(0) ==
                  model_f.program_core()->input_projection_group(1),
          "F must use a shared projection for xs1 and xs2");
  require(!model_f.program_core()->input_projection(0).has_bias() &&
              model_f.program_merge() != nullptr &&
              !model_f.program_merge()->has_bias(),
          "program input and residual merge projections must be bias-free");
  require(model_f.program_core()->program().metadata().selected_strategy ==
              lowering::kLinearAttentionStrategy,
          "F must use a compiled linear-attention head");
  require(!exposes_coefficients(model_f),
          "F compiled coefficients must remain frozen");

  const auto examples = branch_examples();
  conditional_reverse::LearnedForwardOptions capture;
  capture.capture_representations = true;
  const auto evaluation = model_f.evaluate(examples, capture);
  require(evaluation.metrics.example_count == 2 &&
              evaluation.metrics.reverse_example_count == 1 &&
              evaluation.metrics.copy_example_count == 1 &&
              evaluation.predictions.size() == 4,
          "evaluation must report both task branches separately");
  for (const std::string &name :
       {"x1", "r1", "h1", "program_input", "program_input.selected",
        "program_output", "r2", "logits"}) {
    require(evaluation.representations.contains(name),
            "missing learned representation " + name);
  }
  const auto &program_input = evaluation.representations.at("program_input");
  require(program_input.leading_shape == std::vector<std::size_t>({2, 4}) &&
              program_input.observations.columns == 2,
          "paper-facing program captures must be sequence aligned");

  conditional_reverse::LearnedHybrid model_p(
      small_model_config(conditional_reverse::HybridVariant::P));
  require(model_p.program_core()->logical_input_count() == 1 &&
              model_p.program_core()->program().metadata().selected_strategy ==
                  lowering::kLinearStrategy &&
              !model_p.program_core()->input_projection(0).has_bias(),
          "P must use one biasless projection and unary reverse program");
  require(!exposes_coefficients(model_p),
          "P compiled coefficients must remain frozen");

  conditional_reverse::LearnedHybrid model_t(
      small_model_config(conditional_reverse::HybridVariant::T));
  require(model_t.program_core()->program().metadata().initialization ==
                  lowering::CoefficientInitialization::RandomUniform &&
              !model_t.program_core()
                   ->program()
                   .metadata()
                   .preserves_compiled_map &&
              model_t.program_core()->program().metadata().trainable,
          "T must be a trainable seeded shape-matched randomized program");

  auto second_t_config =
      small_model_config(conditional_reverse::HybridVariant::T);
  second_t_config.seed += 1;
  // A stale lowering seed must not decouple T from the experiment seed.
  second_t_config.lowering.seed =
      small_model_config(conditional_reverse::HybridVariant::T).lowering.seed;
  conditional_reverse::LearnedHybrid second_model_t(second_t_config);
  const auto first_t_coefficients =
      model_t.program_core()->program().named_tensor("coefficients").data();
  const auto second_t_coefficients = second_model_t.program_core()
                                         ->program()
                                         .named_tensor("coefficients")
                                         .data();
  require(!std::equal(first_t_coefficients.begin(), first_t_coefficients.end(),
                      second_t_coefficients.begin(),
                      second_t_coefficients.end()),
          "T coefficients must follow the learned experiment seed");
  require(exposes_coefficients(model_t),
          "T randomized coefficients must be exposed to Adam");

  conditional_reverse::LearnedHybrid model_i(
      small_model_config(conditional_reverse::HybridVariant::I));
  require(!model_i.has_program() && model_i.program_core() == nullptr &&
              model_i.program_merge() == nullptr,
          "I must omit the program branch");
  require_throws_as<std::invalid_argument>(
      [&] {
        conditional_reverse::LearnedForwardOptions invalid;
        invalid.ablate_program_output = true;
        static_cast<void>(model_i.evaluate(examples, invalid));
      },
      "I must reject program-only interventions");

  require_throws_as<std::invalid_argument>(
      [&] {
        conditional_reverse::LearnedForwardOptions duplicate_steering;
        duplicate_steering.steering = {{.position = 0}, {.position = 0}};
        static_cast<void>(model_f.evaluate(examples, duplicate_steering));
      },
      "learned steering must define each position at most once");
}

void test_target_mask_hypotheses_and_adam() {
  riftco_transformer::Variable logits(
      riftco_transformer::Tensor::zeros({1, 4, 5}), true);
  const std::vector<riftco_transformer::TokenId> targets{99, 99, 1, 2};
  const auto loss =
      conditional_reverse::learned_target_half_loss(logits, targets, 2);
  loss.backward();
  for (std::size_t index = 0; index < 10; ++index) {
    require(logits.gradient().flat(index) == 0.0F,
            "source-side logits must be excluded from target-half loss");
  }
  double supervised_gradient = 0.0;
  for (std::size_t index = 10; index < 20; ++index) {
    supervised_gradient +=
        std::abs(static_cast<double>(logits.gradient().flat(index)));
  }
  require(supervised_gradient > 0.0,
          "target-half logits must receive loss gradients");

  const auto examples = branch_examples();
  const std::vector<riftco_transformer::TokenId> copy_predictions{0, 1, 1, 2};
  const auto copy_scores = conditional_reverse::score_learned_hypotheses(
      examples, copy_predictions, small_protocol());
  require(copy_scores.copy_exact_sequence_accuracy == 1.0 &&
              copy_scores.reverse_exact_sequence_accuracy == 0.0,
          "hypothesis scoring must identify unconditional copy outputs");
  const std::vector<riftco_transformer::TokenId> reverse_predictions{1, 0, 2,
                                                                     1};
  const auto reverse_scores = conditional_reverse::score_learned_hypotheses(
      examples, reverse_predictions, small_protocol());
  require(reverse_scores.reverse_exact_sequence_accuracy == 1.0 &&
              reverse_scores.copy_exact_sequence_accuracy == 0.0,
          "hypothesis scoring must identify unconditional reverse outputs");

  conditional_reverse::LearnedHybrid model(
      small_model_config(conditional_reverse::HybridVariant::I));
  const auto parameters = model.parameters();
  std::vector<std::vector<float>> before;
  for (const auto &named : parameters) {
    before.emplace_back(named.parameter->value().data().begin(),
                        named.parameter->value().data().end());
  }
  conditional_reverse::LearnedTrainingConfig training;
  training.epochs = 1;
  training.batch_size = 2;
  training.evaluation_batch_size = 1;
  training.maximum_steps = 1;
  require(training.adam.maximum_gradient_norm ==
              std::numeric_limits<float>::max(),
          "learned trainer must default to practical unclipped Adam");
  conditional_reverse::LearnedHybridTrainer trainer(model, training);
  const auto step = trainer.train_step(
      conditional_reverse::make_learned_batch(examples, small_protocol()), 1);
  require(step.step == 1 && std::isfinite(step.target_loss) &&
              std::isfinite(step.gradient_norm),
          "Adam trainer must complete a finite target-half update");
  bool changed = false;
  for (std::size_t parameter = 0; parameter < parameters.size(); ++parameter) {
    changed =
        changed ||
        !std::equal(before[parameter].begin(), before[parameter].end(),
                    parameters[parameter].parameter->value().data().begin(),
                    parameters[parameter].parameter->value().data().end());
  }
  require(changed, "Adam trainer must update learned model parameters");

  const conditional_reverse::LearnedDataset dataset(small_protocol(), examples);
  conditional_reverse::LearnedForwardOptions capture;
  capture.capture_representations = true;
  const auto evaluation =
      conditional_reverse::evaluate_learned_dataset(model, dataset, 1, capture);
  const auto metrics = evaluation.metrics;
  require(metrics.example_count == 2 && metrics.reverse_example_count == 1 &&
              metrics.copy_example_count == 1 && std::isfinite(metrics.loss),
          "batched dataset evaluation must preserve branch metrics");
  require(evaluation.representations.at("x1").leading_shape ==
              std::vector<std::size_t>({2, 4}),
          "batched dataset evaluation must aggregate every capture row");

  conditional_reverse::LearnedHybrid history_model(
      small_model_config(conditional_reverse::HybridVariant::I));
  auto history_config = training;
  history_config.epochs = 3;
  history_config.maximum_steps = 1;
  conditional_reverse::LearnedHybridTrainer history_trainer(history_model,
                                                            history_config);
  const auto history = history_trainer.fit(dataset, dataset);
  require(history.steps.size() == 1 && history.epochs.size() == 1,
          "step limits at epoch boundaries must not append an empty epoch");
}

void test_program_coefficient_training_contracts() {
  const auto examples = branch_examples();
  const auto batch =
      conditional_reverse::make_learned_batch(examples, small_protocol());
  for (const auto variant : {conditional_reverse::HybridVariant::F,
                             conditional_reverse::HybridVariant::P,
                             conditional_reverse::HybridVariant::T}) {
    conditional_reverse::LearnedHybrid model(small_model_config(variant));
    const auto coefficient_before =
        model.program_core()->program().named_tensor("coefficients").data();
    const std::vector<float> before(coefficient_before.begin(),
                                    coefficient_before.end());
    const auto parameters = model.parameters();
    const bool coefficient_registered = std::any_of(
        parameters.begin(), parameters.end(), [](const auto &named) {
          return named.name == "program.program.coefficients";
        });
    require(coefficient_registered ==
                (variant == conditional_reverse::HybridVariant::T),
            "only T may expose program coefficients to Adam");

    conditional_reverse::LearnedTrainingConfig training;
    training.epochs = 1;
    training.batch_size = 2;
    training.evaluation_batch_size = 2;
    training.maximum_steps = 1;
    conditional_reverse::LearnedHybridTrainer trainer(model, training);
    static_cast<void>(trainer.train_step(batch, 1));
    const auto coefficient_after =
        model.program_core()->program().named_tensor("coefficients").data();
    const bool changed =
        !std::equal(before.begin(), before.end(), coefficient_after.begin(),
                    coefficient_after.end());
    require(changed == (variant == conditional_reverse::HybridVariant::T),
            "Adam must update T coefficients and preserve frozen F/P maps");
  }
}

} // namespace

int main() {
  try {
    test_dataset_protocol_and_validation();
    test_variants_parameter_sharing_and_captures();
    test_target_mask_hypotheses_and_adam();
    test_program_coefficient_training_contracts();
  } catch (const std::exception &error) {
    std::cerr << "learned conditional-reverse test failed: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
