#pragma once

#include "riftco_transformer/core/autograd.hpp"

#include <cstddef>

namespace riftco_transformer {

[[nodiscard]] Variable gelu(const Variable& input);
// Exact rectified linear unit, max(0, x), with derivative zero at x == 0.
[[nodiscard]] Variable relu(const Variable &input);
// Sigmoid linear unit used by SwiGLU: x * sigmoid(x).
[[nodiscard]] Variable silu(const Variable& input);
[[nodiscard]] Variable softmax(
    const Variable& input,
    std::size_t axis
);

}  // namespace riftco_transformer
