#pragma once

#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/core/quantized_weight.hpp"

namespace riftco_transformer {

// Applies an immutable packed [output,input] weight to the final dimension of
// input. The autograd edge contains input only: backward computes dInput while
// the packed weight never receives a gradient or optimizer state.
[[nodiscard]] Variable quantized_linear(
    const Variable& input,
    const QuantizedWeight& weight
);

}  // namespace riftco_transformer
