#include "stages/serving/cache/detail/validation.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace transformer_lab::stages::serving::cache_detail {

TransformerDimensions checked_dimensions(
    TransformerDimensions dimensions
) {
    if (dimensions.vocabulary_size == 0 ||
        dimensions.maximum_context == 0 ||
        dimensions.model_width == 0 ||
        dimensions.head_count == 0 ||
        dimensions.block_count == 0 ||
        dimensions.feed_forward_width == 0) {
        throw std::invalid_argument(
            "KV cache transformer dimensions must be greater than zero"
        );
    }
    if (dimensions.model_width % dimensions.head_count != 0) {
        throw std::invalid_argument(
            "KV cache model width must be divisible by head count"
        );
    }
    return dimensions;
}

ExecutionBackend checked_backend(ExecutionBackend backend) {
    switch (backend) {
        case ExecutionBackend::Cpu:
        case ExecutionBackend::Metal:
            break;
        default:
            throw std::invalid_argument(
                "KV cache backend is not recognized"
            );
    }
    if (!execution_backend_available(backend)) {
        throw std::invalid_argument("KV cache backend is unavailable");
    }
    return backend;
}

std::size_t checked_block_size(std::size_t block_size) {
    if (block_size == 0) {
        throw std::invalid_argument(
            "KV cache block size must be greater than zero"
        );
    }
    return block_size;
}

std::size_t divide_rounding_up(
    std::size_t value,
    std::size_t divisor
) noexcept {
    return 1 + (value - 1) / divisor;
}

std::size_t checked_block_count(
    std::size_t block_count,
    std::size_t blocks_per_cache
) {
    const std::size_t resolved =
        block_count == 0 ? blocks_per_cache : block_count;
    if (resolved < blocks_per_cache) {
        throw std::invalid_argument(
            "paged KV cache block count cannot hold one full context"
        );
    }
    if (resolved - 1 >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::overflow_error(
            "paged KV cache block IDs exceed uint32_t"
        );
    }
    return resolved;
}

}  // namespace transformer_lab::stages::serving::cache_detail
