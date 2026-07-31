#pragma once

#include "riftco_transformer/core/backend.hpp"

#include <cstddef>

namespace riftco_transformer::stages::serving {

enum class KvCacheKind {
    Contiguous,
    Paged,
};

// Serving owns inference limits and device placement. It intentionally has no
// optimizer, training-loop, or gradient configuration.
struct ServingConfig {
    ExecutionBackend backend = ExecutionBackend::Cpu;
    std::size_t maximum_new_tokens = 256;
    KvCacheKind kv_cache_kind = KvCacheKind::Paged;
    std::size_t kv_cache_block_size = 16;
    // Zero allocates exactly enough pages for one maximum-length context.
    std::size_t kv_cache_block_count = 0;

    void validate() const;
};

}  // namespace riftco_transformer::stages::serving
