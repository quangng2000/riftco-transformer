#include "riftco_transformer/lowering/strategy.hpp"

#include "lowering/detail/module_factory.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace riftco_transformer::lowering {
namespace {

using StrategyPointer = std::shared_ptr<const MultilinearLoweringStrategy>;

struct FloatConversionAnalysis {
  double maximum_error = 0.0;
  bool all_finite = true;
};

[[nodiscard]] FloatConversionAnalysis
analyze_float_conversion(const compiler::cajal::MultilinearMap &map) noexcept {
  FloatConversionAnalysis result;
  const double maximum_float =
      static_cast<double>(std::numeric_limits<float>::max());

  for (const double coefficient : map.coefficients()) {
    if (coefficient > maximum_float || coefficient < -maximum_float) {
      result.maximum_error = std::numeric_limits<double>::infinity();
      result.all_finite = false;
      return result;
    }

    const float converted = static_cast<float>(coefficient);
    if (!std::isfinite(converted)) {
      result.maximum_error = std::numeric_limits<double>::infinity();
      result.all_finite = false;
      return result;
    }

    result.maximum_error =
        std::max(result.maximum_error,
                 std::abs(coefficient - static_cast<double>(converted)));
  }
  return result;
}

[[nodiscard]] std::string conversion_error_message(double maximum_error) {
  std::ostringstream message;
  message << "compiled coefficients do not round-trip through float32 "
             "exactly; maximum conversion error is "
          << maximum_error;
  return message.str();
}

[[nodiscard]] LoweringAnalysis
analyze_builtin(const compiler::cajal::MultilinearMap &map,
                const NeuralLoweringConfig &config,
                std::string_view strategy_id, bool structurally_supported,
                std::string structural_reason, std::string success_reason) {
  FloatConversionAnalysis conversion;
  if (config.initialization == CoefficientInitialization::Compiled) {
    conversion = analyze_float_conversion(map);
  }

  LoweringAnalysis analysis;
  analysis.requested_strategy = config.strategy;
  analysis.selected_strategy = std::string(strategy_id);
  analysis.logical_coefficient_elements = map.coefficient_count();
  analysis.maximum_float_conversion_error = conversion.maximum_error;

  if (!structurally_supported) {
    analysis.reason = std::move(structural_reason);
    return analysis;
  }

  if (map.coefficient_count() > config.max_coefficient_elements) {
    analysis.reason = "strategy '" + std::string(strategy_id) + "' requires " +
                      std::to_string(map.coefficient_count()) +
                      " coefficient elements, exceeding the configured "
                      "limit of " +
                      std::to_string(config.max_coefficient_elements);
    return analysis;
  }

  if (!conversion.all_finite) {
    analysis.reason =
        "a compiled coefficient cannot be represented as a finite float32";
    return analysis;
  }

  if (config.precision == CoefficientPrecision::RequireExactFloat32 &&
      conversion.maximum_error != 0.0) {
    analysis.reason = conversion_error_message(conversion.maximum_error);
    return analysis;
  }

  analysis.supported = true;
  analysis.exact =
      config.initialization == CoefficientInitialization::Compiled &&
      conversion.maximum_error == 0.0;
  analysis.reason = std::move(success_reason);
  return analysis;
}

class BuiltinLoweringStrategy : public MultilinearLoweringStrategy {
public:
  [[nodiscard]] std::unique_ptr<LoweredMultilinearModule>
  lower(const compiler::cajal::MultilinearMap &map,
        const NeuralLoweringConfig &config,
        const LoweringAnalysis &analysis) const final {
    const std::string strategy_id(id());
    if (!analysis.supported) {
      throw LoweringError("cannot lower with strategy '" + strategy_id +
                          "': " + analysis.reason);
    }
    if (analysis.selected_strategy != strategy_id) {
      throw LoweringError("strategy '" + strategy_id +
                          "' cannot construct analysis selected for '" +
                          analysis.selected_strategy + "'");
    }
    return detail::make_builtin_module(map, config, analysis);
  }
};

class DenseContractionStrategy final : public BuiltinLoweringStrategy {
public:
  [[nodiscard]] std::string_view id() const noexcept final {
    return kDenseContractionStrategy;
  }

  [[nodiscard]] LoweringAnalysis
  analyze(const compiler::cajal::MultilinearMap &map,
          const NeuralLoweringConfig &config) const final {
    return analyze_builtin(
        map, config, id(), true, {},
        "dense contraction supports multilinear maps of every arity");
  }
};

class LinearStrategy final : public BuiltinLoweringStrategy {
public:
  [[nodiscard]] std::string_view id() const noexcept final {
    return kLinearStrategy;
  }

  [[nodiscard]] LoweringAnalysis
  analyze(const compiler::cajal::MultilinearMap &map,
          const NeuralLoweringConfig &config) const final {
    const bool supported = map.arity() == 1U;
    return analyze_builtin(
        map, config, id(), supported,
        "linear lowering requires arity 1, but the map has arity " +
            std::to_string(map.arity()),
        "the map is unary and can be lowered as one linear projection");
  }
};

class LinearAttentionStrategy final : public BuiltinLoweringStrategy {
public:
  [[nodiscard]] std::string_view id() const noexcept final {
    return kLinearAttentionStrategy;
  }

  [[nodiscard]] LoweringAnalysis
  analyze(const compiler::cajal::MultilinearMap &map,
          const NeuralLoweringConfig &config) const final {
    if (map.arity() != 2U) {
      return analyze_builtin(
          map, config, id(), false,
          "linear-attention lowering requires arity 2, but the map has "
          "arity " +
              std::to_string(map.arity()),
          {});
    }

    if (config.attention_query_axis.has_value() &&
        *config.attention_query_axis > 1U) {
      return analyze_builtin(
          map, config, id(), false,
          "linear-attention query axis must be 0 or 1, but is " +
              std::to_string(*config.attention_query_axis),
          {});
    }

    return analyze_builtin(
        map, config, id(), true, {},
        "the bilinear map can be expressed as dynamic linear attention");
  }
};

class MlpStrategy final : public BuiltinLoweringStrategy {
public:
  [[nodiscard]] std::string_view id() const noexcept final {
    return kMlpStrategy;
  }

  [[nodiscard]] LoweringAnalysis
  analyze(const compiler::cajal::MultilinearMap &map,
          const NeuralLoweringConfig &config) const final {
    return analyze_builtin(
        map, config, id(), false,
        "the current GELU MLP cannot exactly preserve a general compiled "
        "multilinear map; exact MLP lowering is not implemented",
        {});
  }
};

[[nodiscard]] const MultilinearLoweringStrategy *
find_strategy(const std::vector<StrategyPointer> &strategies,
              std::string_view strategy_id) noexcept {
  for (const StrategyPointer &strategy : strategies) {
    if (strategy->id() == strategy_id) {
      return strategy.get();
    }
  }
  return nullptr;
}

[[nodiscard]] bool valid_strategy_id(std::string_view strategy_id) noexcept {
  return !strategy_id.empty() &&
         std::all_of(strategy_id.begin(), strategy_id.end(), [](char value) {
           const auto byte = static_cast<unsigned char>(value);
           return std::isalnum(byte) != 0 || value == '_' || value == '-' ||
                  value == '.';
         });
}

[[nodiscard]] std::string
registered_strategy_list(const std::vector<StrategyPointer> &strategies) {
  if (strategies.empty()) {
    return "none";
  }

  std::string result;
  for (const StrategyPointer &strategy : strategies) {
    if (!result.empty()) {
      result += ", ";
    }
    result += std::string(strategy->id());
  }
  return result;
}

[[nodiscard]] LoweringAnalysis
normalize_analysis(LoweringAnalysis analysis,
                   std::string_view requested_strategy,
                   std::string_view selected_strategy) {
  analysis.requested_strategy = std::string(requested_strategy);
  analysis.selected_strategy = std::string(selected_strategy);
  analysis.used_fallback = false;
  if (!analysis.supported) {
    analysis.exact = false;
    if (analysis.reason.empty()) {
      analysis.reason = "strategy '" + std::string(selected_strategy) +
                        "' does not support this map";
    }
  } else if (analysis.reason.empty()) {
    analysis.reason =
        "strategy '" + std::string(selected_strategy) + "' supports this map";
  }
  return analysis;
}

[[nodiscard]] LoweringAnalysis
analyze_strategy(const MultilinearLoweringStrategy &strategy,
                 const compiler::cajal::MultilinearMap &map,
                 const NeuralLoweringConfig &config,
                 std::string_view requested_strategy) {
  const std::string selected_strategy(strategy.id());
  return normalize_analysis(strategy.analyze(map, config), requested_strategy,
                            selected_strategy);
}

[[nodiscard]] LoweringAnalysis
apply_dense_fallback(const std::vector<StrategyPointer> &strategies,
                     const compiler::cajal::MultilinearMap &map,
                     const NeuralLoweringConfig &config,
                     const LoweringAnalysis &unsupported_analysis) {
  const MultilinearLoweringStrategy *const dense =
      find_strategy(strategies, kDenseContractionStrategy);
  if (dense == nullptr) {
    throw LoweringError("dense fallback was requested, but strategy '" +
                        std::string(kDenseContractionStrategy) +
                        "' is not registered");
  }

  LoweringAnalysis fallback =
      analyze_strategy(*dense, map, config, config.strategy);
  fallback.used_fallback = true;
  const std::string original_reason = unsupported_analysis.reason.empty()
                                          ? "no reason was provided"
                                          : unsupported_analysis.reason;
  if (fallback.supported) {
    fallback.reason = "requested strategy '" + config.strategy +
                      "' is unsupported: " + original_reason +
                      "; using dense fallback";
  } else {
    fallback.reason =
        "requested strategy '" + config.strategy +
        "' is unsupported: " + original_reason +
        "; dense fallback is also unsupported: " + fallback.reason;
  }
  return fallback;
}

[[nodiscard]] std::string automatic_failure_reason(
    const std::vector<std::pair<std::string, std::string>> &failures) {
  if (failures.empty()) {
    return "automatic strategy order is empty";
  }

  std::string result = "no automatic lowering strategy supports this map: ";
  for (std::size_t index = 0; index < failures.size(); ++index) {
    if (index != 0U) {
      result += "; ";
    }
    result += failures[index].first + ": " + failures[index].second;
  }
  return result;
}

} // namespace

LoweringRegistry::LoweringRegistry(bool register_builtins) {
  if (!register_builtins) {
    return;
  }

  register_strategy(std::make_shared<DenseContractionStrategy>());
  register_strategy(std::make_shared<LinearStrategy>());
  register_strategy(std::make_shared<LinearAttentionStrategy>());
  register_strategy(std::make_shared<MlpStrategy>());
}

void LoweringRegistry::register_strategy(
    std::shared_ptr<const MultilinearLoweringStrategy> strategy) {
  if (!strategy) {
    throw LoweringError("cannot register a null lowering strategy");
  }

  const std::string strategy_id(strategy->id());
  if (!valid_strategy_id(strategy_id)) {
    throw LoweringError(
        "lowering strategy id must be nonempty and contain only letters, "
        "digits, '_', '-', or '.'");
  }
  if (strategy_id == kAutomaticStrategy) {
    throw LoweringError("lowering strategy id 'auto' is reserved");
  }
  if (find_strategy(strategies_, strategy_id) != nullptr) {
    throw LoweringError("lowering strategy '" + strategy_id +
                        "' is already registered");
  }
  strategies_.push_back(std::move(strategy));
}

std::vector<std::string> LoweringRegistry::strategy_ids() const {
  std::vector<std::string> ids;
  ids.reserve(strategies_.size());
  for (const StrategyPointer &strategy : strategies_) {
    ids.emplace_back(strategy->id());
  }
  return ids;
}

LoweringAnalysis
LoweringRegistry::analyze(const compiler::cajal::MultilinearMap &map,
                          const NeuralLoweringConfig &config) const {
  config.validate();

  if (config.strategy != kAutomaticStrategy) {
    const MultilinearLoweringStrategy *const strategy =
        find_strategy(strategies_, config.strategy);
    if (strategy == nullptr) {
      throw LoweringError(
          "unknown lowering strategy '" + config.strategy +
          "'; registered strategies: " + registered_strategy_list(strategies_));
    }

    LoweringAnalysis analysis =
        analyze_strategy(*strategy, map, config, config.strategy);
    if (analysis.supported ||
        config.unsupported_strategy == UnsupportedStrategyPolicy::Reject ||
        config.strategy == kDenseContractionStrategy) {
      return analysis;
    }
    return apply_dense_fallback(strategies_, map, config, analysis);
  }

  std::vector<std::pair<std::string, std::string>> failures;
  LoweringAnalysis last_failure;
  last_failure.requested_strategy = config.strategy;
  last_failure.logical_coefficient_elements = map.coefficient_count();
  if (config.initialization == CoefficientInitialization::Compiled) {
    last_failure.maximum_float_conversion_error =
        analyze_float_conversion(map).maximum_error;
  }

  for (const std::string &strategy_id : config.automatic_strategy_order) {
    const MultilinearLoweringStrategy *const strategy =
        find_strategy(strategies_, strategy_id);
    if (strategy == nullptr) {
      throw LoweringError(
          "unknown automatic lowering strategy '" + strategy_id +
          "'; registered strategies: " + registered_strategy_list(strategies_));
    }

    LoweringAnalysis analysis =
        analyze_strategy(*strategy, map, config, config.strategy);
    if (analysis.supported) {
      return analysis;
    }
    failures.emplace_back(strategy_id, analysis.reason);
    last_failure = std::move(analysis);
  }

  last_failure.supported = false;
  last_failure.exact = false;
  last_failure.used_fallback = false;
  last_failure.requested_strategy = config.strategy;
  last_failure.selected_strategy.clear();
  last_failure.reason = automatic_failure_reason(failures);

  if (config.unsupported_strategy == UnsupportedStrategyPolicy::DenseFallback) {
    return apply_dense_fallback(strategies_, map, config, last_failure);
  }
  return last_failure;
}

std::unique_ptr<LoweredMultilinearModule>
LoweringRegistry::lower(const compiler::cajal::MultilinearMap &map,
                        const NeuralLoweringConfig &config) const {
  const LoweringAnalysis analysis = analyze(map, config);
  if (!analysis.supported) {
    throw LoweringError("cannot lower multilinear map: " + analysis.reason);
  }

  const MultilinearLoweringStrategy *const strategy =
      find_strategy(strategies_, analysis.selected_strategy);
  if (strategy == nullptr) {
    throw LoweringError("selected lowering strategy '" +
                        analysis.selected_strategy +
                        "' is no longer registered");
  }

  std::unique_ptr<LoweredMultilinearModule> module =
      strategy->lower(map, config, analysis);
  if (!module) {
    throw LoweringError("lowering strategy '" + analysis.selected_strategy +
                        "' returned a null module");
  }
  return module;
}

LoweringAnalysis
analyze_neural_lowering(const compiler::cajal::MultilinearMap &map,
                        const NeuralLoweringConfig &config) {
  return LoweringRegistry().analyze(map, config);
}

std::unique_ptr<LoweredMultilinearModule>
lower_to_neural(const compiler::cajal::MultilinearMap &map,
                const NeuralLoweringConfig &config) {
  return LoweringRegistry().lower(map, config);
}

} // namespace riftco_transformer::lowering
