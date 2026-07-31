#pragma once

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"

#include <cstddef>

namespace riftco_transformer::stages::serving::cache_detail {

[[nodiscard]] TransformerDimensions checked_dimensions(
    TransformerDimensions dimensions
);

[[nodiscard]] ExecutionBackend checked_backend(
    ExecutionBackend backend
);

[[nodiscard]] std::size_t checked_block_size(
    std::size_t block_size
);

[[nodiscard]] std::size_t divide_rounding_up(
    std::size_t value,
    std::size_t divisor
) noexcept;

[[nodiscard]] std::size_t checked_block_count(
    std::size_t block_count,
    std::size_t blocks_per_cache
);

}  // namespace riftco_transformer::stages::serving::cache_detail
