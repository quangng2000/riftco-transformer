#pragma once

#include "riftco_transformer/core/autograd.hpp"

#include <cstddef>

namespace riftco_transformer {

[[nodiscard]] Variable gelu(const Variable& input);
[[nodiscard]] Variable softmax(
    const Variable& input,
    std::size_t axis
);

}  // namespace riftco_transformer
