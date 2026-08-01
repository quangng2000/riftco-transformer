#include "riftco_transformer/stages/serving/config.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace riftco_transformer::stages::serving {
namespace {

void validate_backend(ExecutionBackend backend) {
    switch (backend) {
        case ExecutionBackend::Cpu:
        case ExecutionBackend::Metal:
        case ExecutionBackend::Cuda:
        case ExecutionBackend::Tpu:
            break;
        default:
            throw std::invalid_argument(
                "serving backend is not recognized"
            );
    }
    if (!execution_backend_available(backend)) {
        throw std::invalid_argument(
            "serving backend is unavailable"
        );
    }
}

}  // namespace

void ServingConfig::validate() const {
    validate_backend(backend);
    if (maximum_new_tokens == 0) {
        throw std::invalid_argument(
            "serving maximum_new_tokens must be greater than zero"
        );
    }
    switch (kv_cache_kind) {
        case KvCacheKind::Contiguous:
        case KvCacheKind::Paged:
            break;
        default:
            throw std::invalid_argument(
                "serving KV cache kind is not recognized"
            );
    }
    if (kv_cache_block_size == 0) {
        throw std::invalid_argument(
            "serving KV cache block size must be greater than zero"
        );
    }
    if (kv_cache_block_count != 0 &&
        kv_cache_block_count - 1 >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            )) {
        throw std::overflow_error(
            "serving KV cache block IDs exceed uint32_t"
        );
    }
}

}  // namespace riftco_transformer::stages::serving
