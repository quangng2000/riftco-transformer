#pragma once

#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/data/tokenizer.hpp"

#include <span>

namespace riftco_transformer {

// Targets correspond to the flattened leading positions of logits. The final
// logits dimension is the vocabulary.
[[nodiscard]] Variable cross_entropy(
    const Variable& logits,
    std::span<const TokenId> targets
);

}  // namespace riftco_transformer
