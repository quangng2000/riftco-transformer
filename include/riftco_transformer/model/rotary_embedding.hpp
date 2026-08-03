#pragma once

#include "riftco_transformer/core/autograd.hpp"

#include <cstddef>

namespace riftco_transformer {

// Applies split-half rotary position embedding to
// [batch, head, time, head_width]. Positions begin at position_offset. The
// operation is backend-neutral and differentiable with respect to input.
[[nodiscard]] Variable apply_rotary_position_embedding(
    const Variable& input,
    std::size_t position_offset = 0,
    float theta = 10000.0F
);

}  // namespace riftco_transformer
