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

// Selects [time_offset, time_offset + time_count) independently in each batch
// row of [batch, time, vocabulary] logits and flattened [batch, time] targets.
[[nodiscard]] Variable cross_entropy_time_range(
    const Variable& logits,
    std::span<const TokenId> targets,
    std::size_t time_offset,
    std::size_t time_count
);

}  // namespace riftco_transformer
