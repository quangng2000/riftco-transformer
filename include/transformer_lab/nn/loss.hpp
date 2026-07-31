#pragma once

#include "transformer_lab/core/autograd.hpp"
#include "transformer_lab/data/tokenizer.hpp"

#include <span>

namespace transformer_lab {

// Targets correspond to the flattened leading positions of logits. The final
// logits dimension is the vocabulary.
[[nodiscard]] Variable cross_entropy(
    const Variable& logits,
    std::span<const TokenId> targets
);

}  // namespace transformer_lab
