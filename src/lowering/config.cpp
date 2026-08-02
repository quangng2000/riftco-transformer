#include "riftco_transformer/lowering/config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace riftco_transformer::lowering {
namespace {

void validate_strategy_id(std::string_view id, std::string_view field) {
  if (id.empty()) {
    throw std::invalid_argument(std::string(field) + " must not be empty");
  }
  const bool valid = std::all_of(id.begin(), id.end(), [](char character) {
    const auto byte = static_cast<unsigned char>(character);
    return std::isalnum(byte) != 0 || character == '_' || character == '-' ||
           character == '.';
  });
  if (!valid) {
    throw std::invalid_argument(std::string(field) +
                                " contains an invalid character");
  }
}

void validate_backend(ExecutionBackend backend) {
  switch (backend) {
  case ExecutionBackend::Cpu:
  case ExecutionBackend::Metal:
  case ExecutionBackend::Cuda:
  case ExecutionBackend::Tpu:
    break;
  default:
    throw std::invalid_argument("neural lowering backend is not recognized");
  }
  if (!execution_backend_available(backend)) {
    throw std::invalid_argument("neural lowering backend is unavailable: " +
                                std::string(execution_backend_name(backend)));
  }
}

} // namespace

void NeuralLoweringConfig::validate() const {
  validate_strategy_id(strategy, "neural lowering strategy");

  std::unordered_set<std::string> ordered_strategies;
  if (strategy == kAutomaticStrategy) {
    if (automatic_strategy_order.empty()) {
      throw std::invalid_argument(
          "neural lowering automatic strategy order must not be empty");
    }
    ordered_strategies.reserve(automatic_strategy_order.size());
    for (const std::string &candidate : automatic_strategy_order) {
      validate_strategy_id(candidate,
                           "neural lowering automatic strategy identifier");
      if (candidate == kAutomaticStrategy) {
        throw std::invalid_argument(
            "neural lowering automatic strategy order cannot contain auto");
      }
      if (!ordered_strategies.insert(candidate).second) {
        throw std::invalid_argument(
            "neural lowering automatic strategy order contains a duplicate");
      }
    }
  }

  switch (unsupported_strategy) {
  case UnsupportedStrategyPolicy::Reject:
  case UnsupportedStrategyPolicy::DenseFallback:
    break;
  default:
    throw std::invalid_argument(
        "neural lowering unsupported-strategy policy is not recognized");
  }

  switch (precision) {
  case CoefficientPrecision::RequireExactFloat32:
  case CoefficientPrecision::AllowRoundedFloat32:
    break;
  default:
    throw std::invalid_argument(
        "neural lowering coefficient precision is not recognized");
  }

  switch (initialization) {
  case CoefficientInitialization::Compiled:
  case CoefficientInitialization::RandomUniform:
    break;
  default:
    throw std::invalid_argument(
        "neural lowering coefficient initialization is not recognized");
  }

  validate_backend(backend);

  if (initialization == CoefficientInitialization::RandomUniform &&
      (!std::isfinite(random_scale) || random_scale <= 0.0F)) {
    throw std::invalid_argument(
        "neural lowering random scale must be finite and positive");
  }
  if (max_coefficient_elements == 0) {
    throw std::invalid_argument(
        "neural lowering coefficient limit must be greater than zero");
  }

  if (attention_query_axis.has_value()) {
    if (*attention_query_axis >= 2) {
      throw std::invalid_argument(
          "neural lowering attention query axis must be zero or one");
    }
    if (strategy != kAutomaticStrategy &&
        strategy != kLinearAttentionStrategy) {
      throw std::invalid_argument(
          "neural lowering attention query axis requires auto or "
          "linear_attention strategy");
    }
    if (strategy == kAutomaticStrategy &&
        !ordered_strategies.contains(kLinearAttentionStrategy)) {
      throw std::invalid_argument(
          "neural lowering attention query axis requires linear_attention "
          "in the automatic strategy order");
    }
  }
}

} // namespace riftco_transformer::lowering
