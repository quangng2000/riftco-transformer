#pragma once

#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/model/decoder_only_transformer.hpp"

#include <cstddef>

namespace transformer_lab::stages::serving::cache_detail {

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

}  // namespace transformer_lab::stages::serving::cache_detail
