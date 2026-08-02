#include "riftco_transformer/compiler/cajal/cajal.hpp"
#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/lowering/lowering.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace cajal = riftco_transformer::compiler::cajal;
namespace lowering = riftco_transformer::lowering;
using riftco_transformer::ExecutionBackend;
using riftco_transformer::Tensor;
using riftco_transformer::Variable;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] const Tensor &
coefficient_tensor(const lowering::LoweredMultilinearModule &module) {
  return module.named_tensor("coefficients");
}

template <typename Exception, typename Function>
void require_throws_as(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  } catch (const std::exception &error) {
    throw std::runtime_error(message + ": wrong exception: " + error.what());
  }
  throw std::runtime_error(message + ": expected an exception");
}

template <typename Function>
void require_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message + ": expected an exception");
}

void require_close(float actual, double expected, const std::string &message,
                   double tolerance = 2.0e-5) {
  const double actual_double = static_cast<double>(actual);
  const double scale =
      std::max({1.0, std::fabs(actual_double), std::fabs(expected)});
  if (!std::isfinite(actual_double) || !std::isfinite(expected) ||
      std::fabs(actual_double - expected) > tolerance * scale) {
    throw std::runtime_error(message + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual_double));
  }
}

void require_tensor(const Tensor &actual, const Tensor::Shape &expected_shape,
                    std::span<const double> expected,
                    const std::string &message, double tolerance = 2.0e-5) {
  require(actual.shape() == expected_shape, message + ": shape mismatch");
  require(actual.numel() == expected.size(),
          message + ": element-count mismatch");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    require_close(actual.flat(index), expected[index],
                  message + " at flat index " + std::to_string(index),
                  tolerance);
  }
}

void require_encoded_close(const cajal::EncodedValue &actual,
                           const cajal::EncodedValue &expected,
                           const std::string &message) {
  require(cajal::approximately_equal(actual, expected, 1.0e-8), message);
}

[[nodiscard]] std::size_t element_count(const Tensor::Shape &shape) {
  std::size_t result = 1;
  for (const std::size_t dimension : shape) {
    require(dimension > 0, "test fixture contains a zero dimension");
    require(result <= std::numeric_limits<std::size_t>::max() / dimension,
            "test fixture shape overflows size_t");
    result *= dimension;
  }
  return result;
}

[[nodiscard]] std::vector<double>
expected_batched_values(const cajal::MultilinearMap &map,
                        const std::vector<Tensor> &inputs) {
  require(inputs.size() == map.arity(),
          "expected-value helper received the wrong arity");
  if (inputs.empty()) {
    return {};
  }

  const std::size_t sample_count =
      inputs.front().numel() / map.input_dimensions().front();
  std::vector<double> result;
  result.reserve(sample_count * map.output_dimension());
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    std::vector<cajal::EncodedValue> encoded;
    encoded.reserve(inputs.size());
    for (std::size_t input_index = 0; input_index < inputs.size();
         ++input_index) {
      const std::size_t dimension = map.input_dimensions()[input_index];
      std::vector<double> coordinates(dimension, 0.0);
      for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
        coordinates[coordinate] = static_cast<double>(
            inputs[input_index].flat(sample * dimension + coordinate));
      }
      encoded.emplace_back(std::move(coordinates));
    }
    const cajal::EncodedValue output = map.apply(encoded);
    result.insert(result.end(), output.coordinates().begin(),
                  output.coordinates().end());
  }
  return result;
}

void require_module_matches_map(lowering::LoweredMultilinearModule &module,
                                const cajal::MultilinearMap &map,
                                const std::vector<Tensor> &inputs,
                                const std::string &message) {
  require(!inputs.empty(),
          "nonconstant module helper requires at least one input");
  Tensor::Shape expected_shape = inputs.front().shape();
  expected_shape.back() = map.output_dimension();

  std::vector<Variable> variables;
  variables.reserve(inputs.size());
  for (const Tensor &input : inputs) {
    variables.emplace_back(input, false);
  }
  const Variable actual = module.forward(std::span<const Variable>(variables));
  const std::vector<double> expected = expected_batched_values(map, inputs);
  require_tensor(actual.value(), expected_shape, expected, message);
}

void require_constant_matches_map(lowering::LoweredMultilinearModule &module,
                                  const cajal::MultilinearMap &map,
                                  Tensor::Shape leading_shape,
                                  const std::string &message) {
  const cajal::EncodedValue expected_value =
      map.apply(std::vector<cajal::EncodedValue>{});
  const std::size_t copies = element_count(leading_shape);
  std::vector<double> expected;
  expected.reserve(copies * expected_value.size());
  for (std::size_t copy = 0; copy < copies; ++copy) {
    expected.insert(expected.end(), expected_value.coordinates().begin(),
                    expected_value.coordinates().end());
  }
  leading_shape.push_back(map.output_dimension());
  const Variable actual = module.forward_constant(
      Tensor::Shape(leading_shape.begin(), leading_shape.end() - 1));
  require_tensor(actual.value(), leading_shape, expected, message);
}

[[nodiscard]] cajal::MultilinearMap constant_map() {
  return cajal::MultilinearMap({}, 2, {2.0, -3.0});
}

[[nodiscard]] cajal::MultilinearMap unary_map() {
  return cajal::MultilinearMap({2}, 2, {1.0, -2.0, 0.5, 3.0});
}

[[nodiscard]] cajal::MultilinearMap bilinear_map() {
  return cajal::MultilinearMap({2, 3}, 2,
                               {
                                   1.0,
                                   -2.0,
                                   0.5,
                                   3.0,
                                   -1.0,
                                   2.0,
                                   -4.0,
                                   1.5,
                                   2.0,
                                   -0.5,
                                   3.0,
                                   -2.0,
                               });
}

[[nodiscard]] cajal::MultilinearMap ternary_map() {
  return cajal::MultilinearMap({2, 2, 2}, 2,
                               {
                                   1.0,
                                   -1.0,
                                   2.0,
                                   0.5,
                                   -2.0,
                                   3.0,
                                   1.5,
                                   -0.5,
                                   -1.0,
                                   2.0,
                                   -3.0,
                                   1.0,
                                   0.5,
                                   -1.5,
                                   2.5,
                                   4.0,
                               });
}

[[nodiscard]] lowering::NeuralLoweringConfig
strategy_config(std::string strategy) {
  lowering::NeuralLoweringConfig config;
  config.strategy = std::move(strategy);
  return config;
}

void test_config_validation() {
  lowering::NeuralLoweringConfig{}.validate();

  auto config = lowering::NeuralLoweringConfig{};
  config.strategy.clear();
  require_throws([&] { config.validate(); },
                 "empty strategy IDs must be rejected");

  config = lowering::NeuralLoweringConfig{};
  config.automatic_strategy_order.clear();
  require_throws([&] { config.validate(); },
                 "empty automatic strategy order must be rejected");

  config = lowering::NeuralLoweringConfig{};
  config.automatic_strategy_order = {
      lowering::kLinearStrategy,
      lowering::kLinearStrategy,
  };
  require_throws([&] { config.validate(); },
                 "duplicate automatic strategies must be rejected");

  config = lowering::NeuralLoweringConfig{};
  config.automatic_strategy_order = {lowering::kAutomaticStrategy};
  require_throws([&] { config.validate(); },
                 "automatic strategy order must not recursively contain auto");

  config = lowering::NeuralLoweringConfig{};
  config.unsupported_strategy =
      static_cast<lowering::UnsupportedStrategyPolicy>(UINT8_C(255));
  require_throws([&] { config.validate(); },
                 "unknown unsupported-strategy policy must be rejected");

  config = lowering::NeuralLoweringConfig{};
  config.precision = static_cast<lowering::CoefficientPrecision>(UINT8_C(255));
  require_throws([&] { config.validate(); },
                 "unknown coefficient precision must be rejected");

  config = lowering::NeuralLoweringConfig{};
  config.initialization =
      static_cast<lowering::CoefficientInitialization>(UINT8_C(255));
  require_throws([&] { config.validate(); },
                 "unknown coefficient initialization must be rejected");

  config = lowering::NeuralLoweringConfig{};
  config.backend = static_cast<ExecutionBackend>(UINT8_C(255));
  require_throws([&] { config.validate(); },
                 "unknown execution backend must be rejected");

  config = lowering::NeuralLoweringConfig{};
  config.max_coefficient_elements = 0;
  require_throws([&] { config.validate(); },
                 "zero coefficient resource limit must be rejected");

  config = lowering::NeuralLoweringConfig{};
  config.initialization = lowering::CoefficientInitialization::RandomUniform;
  config.random_scale = 0.0F;
  require_throws([&] { config.validate(); },
                 "zero random scale must be rejected");
  config.random_scale = std::numeric_limits<float>::infinity();
  require_throws([&] { config.validate(); },
                 "infinite random scale must be rejected");
  config.random_scale = std::numeric_limits<float>::quiet_NaN();
  require_throws([&] { config.validate(); },
                 "NaN random scale must be rejected");

  config = strategy_config(lowering::kLinearStrategy);
  config.automatic_strategy_order.clear();
  config.random_scale = 0.0F;
  config.validate();

  config = strategy_config(lowering::kLinearAttentionStrategy);
  config.attention_query_axis = 2;
  require_throws_as<std::invalid_argument>(
      [&] {
        static_cast<void>(
            lowering::analyze_neural_lowering(bilinear_map(), config));
      },
      "out-of-range attention query axis must be rejected");

  config = strategy_config(lowering::kDenseContractionStrategy);
  config.max_coefficient_elements = bilinear_map().coefficient_count() - 1;
  const lowering::LoweringAnalysis limited =
      lowering::analyze_neural_lowering(bilinear_map(), config);
  require(!limited.supported && !limited.reason.empty(),
          "resource-limited analysis must report unsupported");
  require_throws_as<lowering::LoweringError>(
      [&] {
        static_cast<void>(lowering::lower_to_neural(bilinear_map(), config));
      },
      "resource limit must prevent lowering");
}

class ProbeStrategy final : public lowering::MultilinearLoweringStrategy {
public:
  explicit ProbeStrategy(std::string strategy_id = "probe",
                         std::shared_ptr<bool> lower_called = {})
      : strategy_id_(std::move(strategy_id)),
        lower_called_(std::move(lower_called)) {}

  [[nodiscard]] std::string_view id() const noexcept override {
    return strategy_id_;
  }

  [[nodiscard]] lowering::LoweringAnalysis
  analyze(const cajal::MultilinearMap &map,
          const lowering::NeuralLoweringConfig &config) const override {
    return {
        true,
        true,
        false,
        config.strategy,
        std::string(id()),
        "custom strategy selected",
        map.coefficient_count(),
        0.0,
    };
  }

  [[nodiscard]] std::unique_ptr<lowering::LoweredMultilinearModule>
  lower(const cajal::MultilinearMap &, const lowering::NeuralLoweringConfig &,
        const lowering::LoweringAnalysis &) const override {
    if (lower_called_) {
      *lower_called_ = true;
    }
    throw lowering::LoweringError("probe custom lower() reached");
  }

private:
  std::string strategy_id_;
  std::shared_ptr<bool> lower_called_;
};

void test_registry_contract() {
  lowering::LoweringRegistry builtins;
  const std::vector<std::string> ids = builtins.strategy_ids();
  const std::set<std::string> id_set(ids.begin(), ids.end());
  require(id_set.contains(lowering::kDenseContractionStrategy),
          "registry is missing dense strategy");
  require(id_set.contains(lowering::kLinearStrategy),
          "registry is missing linear strategy");
  require(id_set.contains(lowering::kLinearAttentionStrategy),
          "registry is missing linear-attention strategy");
  require(id_set.contains(lowering::kMlpStrategy),
          "registry is missing honest MLP placeholder");

  auto unknown = lowering::NeuralLoweringConfig{};
  unknown.strategy = "does_not_exist";
  require_throws_as<lowering::LoweringError>(
      [&] { static_cast<void>(builtins.analyze(unary_map(), unknown)); },
      "unknown registry strategy must be rejected by analysis");
  require_throws_as<lowering::LoweringError>(
      [&] { static_cast<void>(builtins.lower(unary_map(), unknown)); },
      "unknown registry strategy must be rejected by lowering");

  lowering::LoweringRegistry custom(false);
  const auto lower_called = std::make_shared<bool>(false);
  const auto probe = std::make_shared<ProbeStrategy>("probe", lower_called);
  custom.register_strategy(probe);
  require(custom.strategy_ids() == std::vector<std::string>{"probe"},
          "custom registry ID lookup");
  auto probe_config = lowering::NeuralLoweringConfig{};
  probe_config.strategy = "probe";
  const lowering::LoweringAnalysis probe_analysis =
      custom.analyze(unary_map(), probe_config);
  require(probe_analysis.supported && probe_analysis.exact &&
              probe_analysis.selected_strategy == "probe",
          "custom registry strategy was not selected");
  require_throws_as<lowering::LoweringError>(
      [&] { static_cast<void>(custom.lower(unary_map(), probe_config)); },
      "custom registry lowering must dispatch to the selected strategy");
  require(
      *lower_called,
      "custom registry lowering did not invoke the strategy implementation");
  require_throws(
      [&] { custom.register_strategy(std::make_shared<ProbeStrategy>()); },
      "duplicate strategy IDs must be rejected");
  require_throws(
      [&] {
        custom.register_strategy(
            std::shared_ptr<const lowering::MultilinearLoweringStrategy>{});
      },
      "null strategies must be rejected");
  require_throws(
      [&] {
        custom.register_strategy(std::make_shared<ProbeStrategy>("auto"));
      },
      "the registry must reserve the automatic policy name");
  require_throws(
      [&] {
        custom.register_strategy(std::make_shared<ProbeStrategy>("bad id"));
      },
      "strategy IDs with invalid characters must be rejected");
}

void test_automatic_selection() {
  const std::array maps{
      constant_map(),
      unary_map(),
      bilinear_map(),
      ternary_map(),
  };
  const std::array<std::string_view, 4> expected{
      lowering::kDenseContractionStrategy,
      lowering::kLinearStrategy,
      lowering::kLinearAttentionStrategy,
      lowering::kDenseContractionStrategy,
  };

  for (std::size_t index = 0; index < maps.size(); ++index) {
    const lowering::LoweringAnalysis analysis =
        lowering::analyze_neural_lowering(maps[index]);
    require(analysis.supported && analysis.exact && !analysis.used_fallback,
            "automatic analysis must find an exact built-in strategy");
    require(analysis.requested_strategy == lowering::kAutomaticStrategy &&
                analysis.selected_strategy == expected[index],
            "automatic analysis selected the wrong strategy for arity " +
                std::to_string(index));

    auto module = lowering::lower_to_neural(maps[index]);
    const lowering::LoweringMetadata &metadata = module->metadata();
    require(metadata.requested_strategy == lowering::kAutomaticStrategy &&
                metadata.selected_strategy == expected[index] &&
                !metadata.used_fallback && metadata.preserves_compiled_map,
            "automatic module metadata does not record its exact selection");
    require(metadata.input_dimensions ==
                    std::vector<std::size_t>(
                        maps[index].input_dimensions().begin(),
                        maps[index].input_dimensions().end()) &&
                metadata.output_dimension == maps[index].output_dimension(),
            "automatic module metadata changed map dimensions");
    require(metadata.tensors.size() == 1 &&
                metadata.tensors.front().name == "coefficients" &&
                metadata.tensors.front().shape ==
                    metadata.logical_coefficient_shape &&
                !metadata.tensors.front().semantic_role.empty() &&
                !metadata.tensors.front().trainable,
            "automatic module metadata must describe its frozen tensor");
  }
}

void test_rejection_and_fallback() {
  auto automatic_reject = lowering::NeuralLoweringConfig{};
  automatic_reject.max_coefficient_elements =
      bilinear_map().coefficient_count() - 1;
  const lowering::LoweringAnalysis automatic_failure =
      lowering::analyze_neural_lowering(bilinear_map(), automatic_reject);
  require(!automatic_failure.supported &&
              automatic_failure.selected_strategy.empty() &&
              !automatic_failure.reason.empty(),
          "failed automatic analysis must not claim a selected strategy");

  auto reject = strategy_config(lowering::kLinearStrategy);
  const lowering::LoweringAnalysis rejected =
      lowering::analyze_neural_lowering(bilinear_map(), reject);
  require(!rejected.supported && !rejected.exact &&
              rejected.requested_strategy == lowering::kLinearStrategy &&
              !rejected.reason.empty(),
          "unsupported explicit linear analysis must be honest");
  require_throws_as<lowering::LoweringError>(
      [&] {
        static_cast<void>(lowering::lower_to_neural(bilinear_map(), reject));
      },
      "unsupported explicit strategy must reject lowering");

  auto fallback = reject;
  fallback.unsupported_strategy =
      lowering::UnsupportedStrategyPolicy::DenseFallback;
  const lowering::LoweringAnalysis fallback_analysis =
      lowering::analyze_neural_lowering(bilinear_map(), fallback);
  require(fallback_analysis.supported && fallback_analysis.exact &&
              fallback_analysis.used_fallback &&
              fallback_analysis.requested_strategy ==
                  lowering::kLinearStrategy &&
              fallback_analysis.selected_strategy ==
                  lowering::kDenseContractionStrategy &&
              !fallback_analysis.reason.empty(),
          "dense fallback analysis must expose the fallback decision");
  auto fallback_module = lowering::lower_to_neural(bilinear_map(), fallback);
  const auto &fallback_metadata = fallback_module->metadata();
  require(fallback_metadata.used_fallback &&
              fallback_metadata.requested_strategy ==
                  lowering::kLinearStrategy &&
              fallback_metadata.selected_strategy ==
                  lowering::kDenseContractionStrategy &&
              fallback_metadata.preserves_compiled_map &&
              !fallback_metadata.reason.empty(),
          "dense fallback metadata must remain auditable");

  auto mlp = strategy_config(lowering::kMlpStrategy);
  const lowering::LoweringAnalysis mlp_rejected =
      lowering::analyze_neural_lowering(unary_map(), mlp);
  require(!mlp_rejected.supported && !mlp_rejected.exact &&
              !mlp_rejected.reason.empty(),
          "unimplemented exact MLP strategy must reject honestly");
  require_throws_as<lowering::LoweringError>(
      [&] { static_cast<void>(lowering::lower_to_neural(unary_map(), mlp)); },
      "honest MLP rejection must prevent lowering");

  mlp.unsupported_strategy = lowering::UnsupportedStrategyPolicy::DenseFallback;
  const lowering::LoweringAnalysis mlp_fallback =
      lowering::analyze_neural_lowering(unary_map(), mlp);
  require(mlp_fallback.supported && mlp_fallback.exact &&
              mlp_fallback.used_fallback &&
              mlp_fallback.selected_strategy ==
                  lowering::kDenseContractionStrategy,
          "MLP fallback must select the dense exact implementation");
}

void test_precision_policy_and_source_immutability() {
  const cajal::MultilinearMap map({1}, 1, {0.1});
  const std::vector<std::size_t> original_dimensions(
      map.input_dimensions().begin(), map.input_dimensions().end());
  const std::vector<double> original_coefficients(map.coefficients().begin(),
                                                  map.coefficients().end());

  auto exact = strategy_config(lowering::kDenseContractionStrategy);
  const lowering::LoweringAnalysis exact_analysis =
      lowering::analyze_neural_lowering(map, exact);
  require(!exact_analysis.supported && !exact_analysis.exact &&
              exact_analysis.maximum_float_conversion_error > 0.0,
          "exact float policy must reject a non-round-trippable coefficient");
  require_throws_as<lowering::LoweringError>(
      [&] { static_cast<void>(lowering::lower_to_neural(map, exact)); },
      "exact float policy must prevent lossy lowering");

  auto rounded = exact;
  rounded.precision = lowering::CoefficientPrecision::AllowRoundedFloat32;
  const lowering::LoweringAnalysis rounded_analysis =
      lowering::analyze_neural_lowering(map, rounded);
  require(rounded_analysis.supported && !rounded_analysis.exact &&
              rounded_analysis.maximum_float_conversion_error > 0.0,
          "rounded float policy must report its loss of exactness");
  auto module = lowering::lower_to_neural(map, rounded);
  require(module->metadata().precision ==
                  lowering::CoefficientPrecision::AllowRoundedFloat32 &&
              !module->metadata().preserves_compiled_map &&
              module->metadata().maximum_float_conversion_error > 0.0,
          "rounded module metadata must expose coefficient conversion loss");
  require(coefficient_tensor(*module).shape() == Tensor::Shape({1, 1}),
          "rounded coefficient tensor shape");
  require(coefficient_tensor(*module).flat(0) ==
              static_cast<float>(map.coefficients().front()),
          "rounded coefficient tensor value");

  auto random = rounded;
  random.initialization = lowering::CoefficientInitialization::RandomUniform;
  random.seed = 9182U;
  random.random_scale = 0.25F;
  static_cast<void>(lowering::lower_to_neural(map, random));

  const std::span<const std::size_t> lowered_dimensions =
      map.input_dimensions();
  const std::span<const double> lowered_coefficients = map.coefficients();
  require(lowered_dimensions.size() == original_dimensions.size() &&
              std::equal(lowered_dimensions.begin(), lowered_dimensions.end(),
                         original_dimensions.begin()) &&
              lowered_coefficients.size() == original_coefficients.size() &&
              std::equal(lowered_coefficients.begin(),
                         lowered_coefficients.end(),
                         original_coefficients.begin()),
          "neural lowering must not mutate the source multilinear map");
}

void test_dense_exact_results() {
  auto dense = strategy_config(lowering::kDenseContractionStrategy);

  const cajal::MultilinearMap constant = constant_map();
  auto constant_module = lowering::lower_to_neural(constant, dense);
  require_constant_matches_map(*constant_module, constant, {},
                               "unbatched dense constant");
  require_constant_matches_map(*constant_module, constant, {2, 3},
                               "shared-leading dense constant");

  const cajal::MultilinearMap unary = unary_map();
  auto unary_module = lowering::lower_to_neural(unary, dense);
  require_module_matches_map(*unary_module, unary, {Tensor({2}, {-1.5F, 2.0F})},
                             "unbatched dense unary map");
  require_module_matches_map(*unary_module, unary,
                             {Tensor({2, 2}, {-1.5F, 2.0F, 0.25F, -3.0F})},
                             "shared-leading dense unary map");

  const cajal::MultilinearMap bilinear = bilinear_map();
  auto bilinear_module = lowering::lower_to_neural(bilinear, dense);
  require_module_matches_map(
      *bilinear_module, bilinear,
      {Tensor({2}, {-1.5F, 2.0F}), Tensor({3}, {0.5F, -2.0F, 3.0F})},
      "unbatched dense bilinear map");
  require_module_matches_map(
      *bilinear_module, bilinear,
      {
          Tensor({2, 2}, {-1.5F, 2.0F, 0.25F, -3.0F}),
          Tensor({2, 3}, {0.5F, -2.0F, 3.0F, -1.0F, 4.0F, 0.75F}),
      },
      "shared-leading dense bilinear map");

  const cajal::MultilinearMap ternary = ternary_map();
  auto ternary_module = lowering::lower_to_neural(ternary, dense);
  require_module_matches_map(*ternary_module, ternary,
                             {
                                 Tensor({2}, {-1.0F, 2.0F}),
                                 Tensor({2}, {0.5F, -3.0F}),
                                 Tensor({2}, {4.0F, -0.25F}),
                             },
                             "unbatched dense ternary map");
  require_module_matches_map(*ternary_module, ternary,
                             {
                                 Tensor({2, 2}, {-1.0F, 2.0F, 0.25F, -3.0F}),
                                 Tensor({2, 2}, {0.5F, -3.0F, -2.0F, 1.5F}),
                                 Tensor({2, 2}, {4.0F, -0.25F, -1.0F, 2.0F}),
                             },
                             "shared-leading dense ternary map");
}

void test_linear_attention_query_axes() {
  auto linear_config = strategy_config(lowering::kLinearStrategy);
  auto linear_module = lowering::lower_to_neural(unary_map(), linear_config);
  require_module_matches_map(
      *linear_module, unary_map(),
      {Tensor({2, 2, 2}, {-1.5F, 2.0F, 0.25F, -3.0F, 1.0F, 0.5F, -2.0F, 4.0F})},
      "specialized linear map with two shared leading axes");

  const cajal::MultilinearMap map = bilinear_map();
  const std::vector<Tensor> inputs{
      Tensor({2, 2, 2}, {-1.5F, 2.0F, 0.25F, -3.0F, 1.0F, 0.5F, -2.0F, 4.0F}),
      Tensor({2, 2, 3}, {0.5F, -2.0F, 3.0F, -1.0F, 4.0F, 0.75F, 2.0F, 1.0F,
                         -0.5F, -3.0F, 0.25F, 1.5F}),
  };

  for (const std::size_t query_axis : {std::size_t{0}, std::size_t{1}}) {
    auto config = strategy_config(lowering::kLinearAttentionStrategy);
    config.attention_query_axis = query_axis;
    const lowering::LoweringAnalysis analysis =
        lowering::analyze_neural_lowering(map, config);
    require(analysis.supported && analysis.exact &&
                analysis.selected_strategy ==
                    lowering::kLinearAttentionStrategy,
            "linear attention must support either bilinear query axis");
    auto module = lowering::lower_to_neural(map, config);
    require(module->metadata().attention_query_axis == query_axis,
            "linear-attention metadata lost the requested query axis");
    require_module_matches_map(
        *module, map, inputs,
        "signed non-softmax linear attention with query axis " +
            std::to_string(query_axis));
  }
}

void test_linear_attention_gradients() {
  const cajal::MultilinearMap map = bilinear_map();
  const std::array<float, 2> left_values{-1.5F, 2.0F};
  const std::array<float, 3> right_values{0.5F, -2.0F, 3.0F};
  const std::array<float, 2> seed_values{1.25F, -0.75F};

  std::vector<double> expected_left(left_values.size(), 0.0);
  std::vector<double> expected_right(right_values.size(), 0.0);
  std::vector<double> expected_coefficients(map.coefficient_count(), 0.0);
  for (std::size_t output = 0; output < map.output_dimension(); ++output) {
    for (std::size_t left = 0; left < left_values.size(); ++left) {
      for (std::size_t right = 0; right < right_values.size(); ++right) {
        const std::array indices{left, right};
        const double coefficient = map.coefficient_at(output, indices);
        const double seed = static_cast<double>(seed_values[output]);
        expected_left[left] +=
            seed * coefficient * static_cast<double>(right_values[right]);
        expected_right[right] +=
            seed * coefficient * static_cast<double>(left_values[left]);
        const std::size_t flat =
            (output * left_values.size() + left) * right_values.size() + right;
        expected_coefficients[flat] = seed *
                                      static_cast<double>(left_values[left]) *
                                      static_cast<double>(right_values[right]);
      }
    }
  }

  for (const std::size_t query_axis : {std::size_t{0}, std::size_t{1}}) {
    auto config = strategy_config(lowering::kLinearAttentionStrategy);
    config.attention_query_axis = query_axis;
    config.trainable = true;
    auto module = lowering::lower_to_neural(map, config);

    const Variable left(
        Tensor({left_values.size()},
               std::vector<float>(left_values.begin(), left_values.end())));
    const Variable right(
        Tensor({right_values.size()},
               std::vector<float>(right_values.begin(), right_values.end())));
    const std::array inputs{left, right};
    const Variable output = module->forward(inputs);
    output.backward(
        Tensor({seed_values.size()},
               std::vector<float>(seed_values.begin(), seed_values.end())));

    require_tensor(left.gradient(), {left_values.size()}, expected_left,
                   "linear-attention left-input gradient");
    require_tensor(right.gradient(), {right_values.size()}, expected_right,
                   "linear-attention right-input gradient");
    const auto parameters = module->parameters();
    require(parameters.size() == 1 && parameters.front().parameter != nullptr,
            "trainable linear attention must register coefficients");
    require_tensor(parameters.front().parameter->gradient(), {2, 2, 3},
                   expected_coefficients,
                   "linear-attention coefficient gradient");
  }
}

void test_input_validation() {
  auto dense = strategy_config(lowering::kDenseContractionStrategy);
  auto module = lowering::lower_to_neural(bilinear_map(), dense);

  require_throws_as<lowering::LoweringError>(
      [&] { static_cast<void>(module->named_tensor("missing")); },
      "named tensor lookup must reject names absent from metadata");

  require_throws_as<lowering::LoweringError>(
      [&] {
        const std::vector<Variable> inputs{
            Variable(Tensor({2}, {1.0F, 2.0F}), false),
        };
        static_cast<void>(module->forward(inputs));
      },
      "lowered module must reject the wrong arity");
  require_throws_as<lowering::LoweringError>(
      [&] {
        const std::vector<Variable> inputs{
            Variable(Tensor({3}, {1.0F, 2.0F, 3.0F}), false),
            Variable(Tensor({3}, {1.0F, 2.0F, 3.0F}), false),
        };
        static_cast<void>(module->forward(inputs));
      },
      "lowered module must reject an incorrect final dimension");
  require_throws_as<lowering::LoweringError>(
      [&] {
        const std::vector<Variable> inputs{
            Variable(Variable::scalar(2.0F, false)),
            Variable(Tensor({3}, {1.0F, 2.0F, 3.0F}), false),
        };
        static_cast<void>(module->forward(inputs));
      },
      "lowered module must reject rank-zero map inputs");
  require_throws_as<lowering::LoweringError>(
      [&] {
        const std::vector<Variable> inputs{
            Variable(Tensor({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}), false),
            Variable(Tensor({3, 3}, 1.0F), false),
        };
        static_cast<void>(module->forward(inputs));
      },
      "lowered module must reject mismatched leading dimensions");
  require_throws_as<lowering::LoweringError>(
      [&] { static_cast<void>(module->forward_constant({2})); },
      "nonconstant modules must reject forward_constant");

  auto constant = lowering::lower_to_neural(constant_map(), dense);
  const std::vector<Variable> no_inputs;
  const Variable natural_constant = constant->forward(no_inputs);
  const Variable explicit_constant = constant->forward_constant({});
  const std::span<const float> natural_data = natural_constant.value().data();
  const std::span<const float> explicit_data = explicit_constant.value().data();
  require(natural_constant.value().shape() ==
                  explicit_constant.value().shape() &&
              natural_data.size() == explicit_data.size() &&
              std::equal(natural_data.begin(), natural_data.end(),
                         explicit_data.begin()),
          "zero-arity forward must match unbatched forward_constant");
}

void test_frozen_and_trainable_gradients() {
  const cajal::MultilinearMap map = unary_map();
  auto frozen_config = strategy_config(lowering::kLinearStrategy);
  auto frozen = lowering::lower_to_neural(map, frozen_config);
  require(frozen->parameters().empty(),
          "frozen lowering must expose no optimizer parameters");

  const Variable frozen_input(Tensor({2}, {-2.0F, 3.0F}));
  const std::array<Variable, 1> frozen_inputs{frozen_input};
  const Variable frozen_output = frozen->forward(frozen_inputs);
  frozen_output.backward(Tensor({2}, {1.5F, -0.5F}));
  require_tensor(frozen_input.gradient(), {2},
                 std::array<double, 2>{1.25, -4.5},
                 "frozen map analytic input gradient");

  auto trainable_config = frozen_config;
  trainable_config.trainable = true;
  auto trainable = lowering::lower_to_neural(map, trainable_config);
  const auto trainable_parameters = trainable->parameters();
  require(trainable_parameters.size() == 1 &&
              trainable_parameters.front().name == "coefficients" &&
              trainable_parameters.front().parameter->value().shape() ==
                  Tensor::Shape({2, 2}),
          "trainable lowering must expose only its coefficient tensor");
  require(trainable->metadata().tensors.size() == 1 &&
              trainable->metadata().tensors.front().trainable,
          "trainable tensor metadata must match parameter registration");

  const Variable trainable_input(Tensor({2}, {2.0F, -1.0F}));
  const std::array<Variable, 1> trainable_inputs{trainable_input};
  const Variable trainable_output = trainable->forward(trainable_inputs);
  trainable_output.backward(Tensor({2}, {3.0F, -2.0F}));
  require_tensor(trainable_input.gradient(), {2},
                 std::array<double, 2>{2.0, -12.0},
                 "trainable map analytic input gradient");
  require_tensor(trainable_parameters.front().parameter->gradient(), {2, 2},
                 std::array<double, 4>{6.0, -3.0, -4.0, 2.0},
                 "trainable map analytic coefficient gradient");
}

void test_randomized_initialization() {
  const cajal::MultilinearMap map = bilinear_map();
  auto config = strategy_config(lowering::kDenseContractionStrategy);
  config.initialization = lowering::CoefficientInitialization::RandomUniform;
  config.seed = 1729U;
  config.random_scale = 0.125F;

  auto first = lowering::lower_to_neural(map, config);
  auto replay = lowering::lower_to_neural(map, config);
  config.seed = 1730U;
  auto changed = lowering::lower_to_neural(map, config);

  const std::span<const float> first_coefficients =
      coefficient_tensor(*first).data();
  const std::span<const float> replay_coefficients =
      coefficient_tensor(*replay).data();
  const std::span<const float> changed_coefficients =
      coefficient_tensor(*changed).data();
  require(coefficient_tensor(*first).shape() ==
                  coefficient_tensor(*replay).shape() &&
              first_coefficients.size() == replay_coefficients.size() &&
              std::equal(first_coefficients.begin(), first_coefficients.end(),
                         replay_coefficients.begin()),
          "randomized lowering must replay exactly for the same seed");
  require(first_coefficients.size() != changed_coefficients.size() ||
              !std::equal(first_coefficients.begin(), first_coefficients.end(),
                          changed_coefficients.begin()),
          "different randomized-lowering seeds must change coefficients");
  for (const float coefficient : coefficient_tensor(*first).data()) {
    require(std::isfinite(coefficient) && std::fabs(coefficient) <= 0.125F,
            "randomized coefficient lies outside configured bounds");
  }
  require(first->metadata().initialization ==
                  lowering::CoefficientInitialization::RandomUniform &&
              !first->metadata().preserves_compiled_map,
          "randomized lowering metadata must identify the ablation");

  const cajal::MultilinearMap shape_only_map(
      {1}, 1, {std::numeric_limits<double>::max()});
  auto shape_only_config = strategy_config(lowering::kDenseContractionStrategy);
  shape_only_config.initialization =
      lowering::CoefficientInitialization::RandomUniform;
  const lowering::LoweringAnalysis shape_only_analysis =
      lowering::analyze_neural_lowering(shape_only_map, shape_only_config);
  require(shape_only_analysis.supported && !shape_only_analysis.exact &&
              shape_only_analysis.maximum_float_conversion_error == 0.0,
          "random initialization must not convert unused program values");
  auto shape_only =
      lowering::lower_to_neural(shape_only_map, shape_only_config);
  require(std::isfinite(coefficient_tensor(*shape_only).flat(0)),
          "shape-only random lowering must produce finite coefficients");
}

[[nodiscard]] cajal::Type bit_type() {
  return cajal::Type::sum(cajal::Type::unit(), cajal::Type::unit());
}

[[nodiscard]] cajal::Expression bit_expression(bool one) {
  if (one) {
    return cajal::Expression::inject_right(cajal::Type::unit(),
                                           cajal::Expression::unit());
  }
  return cajal::Expression::inject_left(cajal::Expression::unit(),
                                        cajal::Type::unit());
}

[[nodiscard]] cajal::Value bit_value(bool one) {
  if (one) {
    return cajal::Value::inject_right(cajal::Type::unit(),
                                      cajal::Value::unit());
  }
  return cajal::Value::inject_left(cajal::Value::unit(), cajal::Type::unit());
}

[[nodiscard]] Tensor tensor_from_encoded(const cajal::EncodedValue &value) {
  std::vector<float> coordinates;
  coordinates.reserve(value.size());
  for (const double coordinate : value.coordinates()) {
    coordinates.push_back(static_cast<float>(coordinate));
  }
  return Tensor({coordinates.size()}, std::move(coordinates));
}

void test_compiled_program_three_way_equivalence() {
  const cajal::Expression expression = cajal::Expression::lookup(
      cajal::Expression::dictionary({
          {bit_expression(false), cajal::Expression::variable("value")},
          {bit_expression(true), cajal::Expression::variable("value")},
      }),
      cajal::Expression::variable("query"));
  const cajal::Context context{
      {"query", bit_type()},
      {"value", bit_type()},
  };
  const cajal::CompiledProgram program = cajal::compile(expression, context);
  auto module = lowering::lower_to_neural(program);
  require(module->metadata().selected_strategy ==
              lowering::kLinearAttentionStrategy,
          "compiled open lookup should auto-select linear attention");

  for (const bool query : {false, true}) {
    for (const bool value : {false, true}) {
      const cajal::Environment environment{
          {"query", bit_value(query)},
          {"value", bit_value(value)},
      };
      const cajal::EncodedValue interpreted =
          cajal::encode(cajal::evaluate(expression, environment));
      const cajal::EncodedValue compiled = program.apply(environment);
      require_encoded_close(compiled, interpreted,
                            "compiled map differs from interpreter encoding");

      const std::vector<cajal::EncodedValue> map_inputs{
          cajal::encode(environment[0].value),
          cajal::encode(environment[1].value),
      };
      require_encoded_close(program.map().apply(map_inputs), interpreted,
                            "direct multilinear map differs from interpreter");

      const std::vector<Variable> neural_inputs{
          Variable(tensor_from_encoded(map_inputs[0]), false),
          Variable(tensor_from_encoded(map_inputs[1]), false),
      };
      const Tensor neural = module->forward(neural_inputs).value();
      require_tensor(neural, {interpreted.size()}, interpreted.coordinates(),
                     "neural lowering differs from interpreter and map");
    }
  }
}

void test_cpu_and_optional_metal_transfer() {
  auto config = strategy_config(lowering::kDenseContractionStrategy);
  config.backend = ExecutionBackend::Cpu;
  auto module = lowering::lower_to_neural(unary_map(), config);
  require(module->backend() == ExecutionBackend::Cpu &&
              coefficient_tensor(*module).backend() == ExecutionBackend::Cpu,
          "CPU lowering must own CPU coefficients");
  require_module_matches_map(*module, unary_map(), {Tensor({2}, {-1.0F, 2.0F})},
                             "explicit CPU lowering");

  if (!riftco_transformer::execution_backend_available(
          ExecutionBackend::Metal)) {
    return;
  }

  module->to(ExecutionBackend::Metal);
  require(module->backend() == ExecutionBackend::Metal &&
              coefficient_tensor(*module).backend() == ExecutionBackend::Metal,
          "Metal transfer must include frozen coefficient storage");
  const Tensor metal_input =
      Tensor({2}, {-1.0F, 2.0F}).to(ExecutionBackend::Metal);
  const std::vector<Variable> metal_inputs{
      Variable(metal_input, false),
  };
  const Tensor metal_output = module->forward(metal_inputs).value();
  require(metal_output.backend() == ExecutionBackend::Metal,
          "Metal lowering must preserve output backend");
  const Tensor cpu_output = metal_output.to(ExecutionBackend::Cpu);
  require_tensor(cpu_output, {2}, std::array<double, 2>{-5.0, 5.5},
                 "Metal lowering parity");

  require_throws_as<lowering::LoweringError>(
      [&] {
        const std::vector<Variable> cpu_inputs{
            Variable(Tensor({2}, {-1.0F, 2.0F}), false),
        };
        static_cast<void>(module->forward(cpu_inputs));
      },
      "lowered module must reject inputs on a different backend");
}

} // namespace

int main() {
  try {
    test_config_validation();
    test_registry_contract();
    test_automatic_selection();
    test_rejection_and_fallback();
    test_precision_policy_and_source_immutability();
    test_dense_exact_results();
    test_linear_attention_query_axes();
    test_linear_attention_gradients();
    test_input_validation();
    test_frozen_and_trainable_gradients();
    test_randomized_initialization();
    test_compiled_program_three_way_equivalence();
    test_cpu_and_optional_metal_transfer();
    std::cout << "Cajal neural lowering tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Cajal neural lowering test failure: " << error.what() << '\n';
    return 1;
  }
}
