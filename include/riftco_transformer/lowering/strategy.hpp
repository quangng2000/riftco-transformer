#pragma once

#include "riftco_transformer/compiler/cajal/multilinear_map.hpp"
#include "riftco_transformer/lowering/module.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer::lowering {

class LoweringError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct LoweringAnalysis {
  bool supported = false;
  bool exact = false;
  bool used_fallback = false;
  std::string requested_strategy;
  std::string selected_strategy;
  std::string reason;
  std::size_t logical_coefficient_elements = 0;
  double maximum_float_conversion_error = 0.0;
};

class MultilinearLoweringStrategy {
public:
  virtual ~MultilinearLoweringStrategy() = default;

  [[nodiscard]] virtual std::string_view id() const noexcept = 0;
  [[nodiscard]] virtual LoweringAnalysis
  analyze(const compiler::cajal::MultilinearMap &map,
          const NeuralLoweringConfig &config) const = 0;
  [[nodiscard]] virtual std::unique_ptr<LoweredMultilinearModule>
  lower(const compiler::cajal::MultilinearMap &map,
        const NeuralLoweringConfig &config,
        const LoweringAnalysis &analysis) const = 0;
};

// Runtime registry keeps the built-in policy configurable and lets later
// exact MLP, sparse, fused, or hardware strategies be added without changing
// the symbolic compiler or the module interface.
class LoweringRegistry {
public:
  explicit LoweringRegistry(bool register_builtins = true);

  void register_strategy(
      std::shared_ptr<const MultilinearLoweringStrategy> strategy);
  [[nodiscard]] std::vector<std::string> strategy_ids() const;

  [[nodiscard]] LoweringAnalysis
  analyze(const compiler::cajal::MultilinearMap &map,
          const NeuralLoweringConfig &config = {}) const;
  [[nodiscard]] std::unique_ptr<LoweredMultilinearModule>
  lower(const compiler::cajal::MultilinearMap &map,
        const NeuralLoweringConfig &config = {}) const;

private:
  std::vector<std::shared_ptr<const MultilinearLoweringStrategy>> strategies_;
};

[[nodiscard]] LoweringAnalysis
analyze_neural_lowering(const compiler::cajal::MultilinearMap &map,
                        const NeuralLoweringConfig &config = {});

[[nodiscard]] std::unique_ptr<LoweredMultilinearModule>
lower_to_neural(const compiler::cajal::MultilinearMap &map,
                const NeuralLoweringConfig &config = {});

} // namespace riftco_transformer::lowering
