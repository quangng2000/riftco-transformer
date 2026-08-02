#pragma once

#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/lowering/config.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer::lowering {

struct NamedTensorShape {
  std::string name;
  std::string semantic_role;
  Tensor::Shape shape;
  bool trainable = false;
};

struct LoweringMetadata {
  std::string requested_strategy;
  std::string selected_strategy;
  bool used_fallback = false;
  bool preserves_compiled_map = false;
  std::string reason;
  std::vector<std::size_t> input_dimensions;
  std::size_t output_dimension = 0;
  // Logical dense map shape [output, input_0, ...]. Physical strategies may
  // expose different or multiple shapes through tensors.
  Tensor::Shape logical_coefficient_shape;
  std::vector<NamedTensorShape> tensors;
  CoefficientPrecision precision = CoefficientPrecision::RequireExactFloat32;
  CoefficientInitialization initialization =
      CoefficientInitialization::Compiled;
  bool trainable = false;
  double maximum_float_conversion_error = 0.0;
  std::optional<std::size_t> attention_query_axis;
};

// A differentiable neural module implementing one lowered multilinear map.
// Input tensors share arbitrary leading dimensions and use their final axis as
// the corresponding map input: [..., d_i] -> [..., d_out].
class LoweredMultilinearModule : public Module {
public:
  ~LoweredMultilinearModule() override = default;

  [[nodiscard]] virtual Variable
  forward(std::span<const Variable> inputs) const = 0;

  // A zero-arity map has no input from which to infer batch/time dimensions.
  // This broadcasts its constant output over an explicit leading shape;
  // forward({}) is the unbatched shorthand.
  [[nodiscard]] virtual Variable
  forward_constant(Tensor::Shape leading_shape = {}) const = 0;

  [[nodiscard]] virtual const LoweringMetadata &metadata() const noexcept = 0;
  // Names come from metadata().tensors. This avoids assuming that every future
  // sparse, factored, or fused strategy owns exactly one coefficient tensor.
  [[nodiscard]] virtual const Tensor &
  named_tensor(std::string_view name) const = 0;
  [[nodiscard]] virtual ExecutionBackend backend() const noexcept = 0;

  // The inherited Module::parameters() is the single optimizer-facing
  // contract: frozen circuits return no parameters, while trainable circuits
  // expose their coefficient tensor without a lowering-specific adapter.
};

} // namespace riftco_transformer::lowering
