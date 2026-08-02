#pragma once

#include "riftco_transformer/core/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace riftco_transformer::lowering {

inline constexpr char kAutomaticStrategy[] = "auto";
inline constexpr char kDenseContractionStrategy[] = "dense";
inline constexpr char kLinearStrategy[] = "linear";
inline constexpr char kLinearAttentionStrategy[] = "linear_attention";
inline constexpr char kMlpStrategy[] = "mlp";

enum class UnsupportedStrategyPolicy : std::uint8_t {
  Reject,
  DenseFallback,
};

enum class CoefficientPrecision : std::uint8_t {
  RequireExactFloat32,
  AllowRoundedFloat32,
};

// Compiled preserves the program coefficients. RandomUniform provides a
// seeded, shape-preserving ablation without changing the module contract.
enum class CoefficientInitialization : std::uint8_t {
  Compiled,
  RandomUniform,
};

// This configuration belongs to the optional neural lowering target, not to
// the backend-neutral Cajal frontend or the global language-model Config.
struct NeuralLoweringConfig {
  std::string strategy = kAutomaticStrategy;
  std::vector<std::string> automatic_strategy_order = {
      kLinearStrategy,
      kLinearAttentionStrategy,
      kDenseContractionStrategy,
  };
  UnsupportedStrategyPolicy unsupported_strategy =
      UnsupportedStrategyPolicy::Reject;
  CoefficientPrecision precision = CoefficientPrecision::RequireExactFloat32;
  CoefficientInitialization initialization =
      CoefficientInitialization::Compiled;
  bool trainable = false;
  ExecutionBackend backend = ExecutionBackend::Cpu;
  std::uint32_t seed = 42;
  float random_scale = 0.02F;
  std::size_t max_coefficient_elements = 1U << 24U;
  // For a bilinear map, selects which input is the query q in D(x)q.
  // The other input generates the dynamic identity-kernel attention matrix.
  std::optional<std::size_t> attention_query_axis;

  void validate() const;
};

} // namespace riftco_transformer::lowering
