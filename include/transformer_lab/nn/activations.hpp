#pragma once

#include "transformer_lab/core/autograd.hpp"

#include <cstddef>

namespace transformer_lab {

[[nodiscard]] Variable gelu(const Variable& input);
[[nodiscard]] Variable softmax(
    const Variable& input,
    std::size_t axis
);

}  // namespace transformer_lab
