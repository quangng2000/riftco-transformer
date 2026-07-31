#pragma once

#include "core/backend/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace riftco_transformer::backend_detail {

// Baseline full-sequence training/autograd attention. This contract
// deliberately exposes its materialized probability tensor; the sibling
// Flash contract below saves only row statistics.
struct MaterializedCausalAttentionDimensions {
    std::size_t batch;
    std::size_t heads;
    std::size_t time;
    std::size_t head_width;
};

struct MaterializedCausalAttentionForwardRequest {
    const TensorStorage& queries;
    const TensorStorage& keys;
    const TensorStorage& values;
    TensorStorage& probabilities;
    TensorStorage& context;
    MaterializedCausalAttentionDimensions dimensions;
};

struct MaterializedCausalAttentionContextBackwardRequest {
    const TensorStorage& queries;
    const TensorStorage& keys;
    const TensorStorage& values;
    const TensorStorage& probabilities;
    const TensorStorage& upstream_context;
    TensorStorage& query_gradient;
    TensorStorage& key_gradient;
    TensorStorage& value_gradient;
    MaterializedCausalAttentionDimensions dimensions;
};

struct MaterializedCausalAttentionProbabilitiesBackwardRequest {
    const TensorStorage& queries;
    const TensorStorage& keys;
    const TensorStorage& probabilities;
    const TensorStorage& upstream_probabilities;
    TensorStorage& query_gradient;
    TensorStorage& key_gradient;
    MaterializedCausalAttentionDimensions dimensions;
};

// Full-sequence exact attention using tiled online softmax. Unlike the
// materialized contract, the Flash contract stores only two row statistics
// with shape [B,H,T] and never exposes or allocates a [B,H,T,T] probability
// tensor.
struct FlashCausalAttentionDimensions {
    std::size_t batch;
    std::size_t heads;
    std::size_t time;
    std::size_t head_width;
};

struct FlashCausalAttentionForwardRequest {
    const TensorStorage& queries;
    const TensorStorage& keys;
    const TensorStorage& values;
    TensorStorage& row_maxima;
    TensorStorage& row_exp_sums;
    TensorStorage& context;
    FlashCausalAttentionDimensions dimensions;
};

// Backward rematerializes probability tiles from Q, K, the saved row maxima,
// and the saved exponential sums. Its temporary storage is linear in T.
struct FlashCausalAttentionBackwardRequest {
    const TensorStorage& queries;
    const TensorStorage& keys;
    const TensorStorage& values;
    const TensorStorage& row_maxima;
    const TensorStorage& row_exp_sums;
    const TensorStorage& upstream_context;
    TensorStorage& query_gradient;
    TensorStorage& key_gradient;
    TensorStorage& value_gradient;
    FlashCausalAttentionDimensions dimensions;
};

// One-query serving attention over block-addressed K/V storage. Queries and
// context contain [head, head_width]. K/V pages use
// [physical_block, head, block_offset, head_width].
struct PagedDecodeAttentionDimensions {
    std::size_t heads;
    std::size_t head_width;
    std::size_t block_size;
    std::size_t physical_block_count;
    std::size_t sequence_length;
};

struct PagedDecodeAttentionForwardRequest {
    const TensorStorage& queries;
    const TensorStorage& key_pages;
    const TensorStorage& value_pages;
    std::span<const std::uint32_t> block_table;
    TensorStorage& context;
    PagedDecodeAttentionDimensions dimensions;
};

}  // namespace riftco_transformer::backend_detail
