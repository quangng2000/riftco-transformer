#pragma once

#include "core/backend/attention/contracts.hpp"

namespace riftco_transformer::backend_detail {

// Objective-C++ execution boundary for the native Metal attention kernels.
// Implementations share the Metal runtime with the small NN kernels without
// making either capability own the other's public contract.
void metal_attention_materialized_causal_forward(
    const MaterializedCausalAttentionForwardRequest& request);
void metal_attention_materialized_causal_context_backward(
    const MaterializedCausalAttentionContextBackwardRequest& request);
void metal_attention_materialized_causal_probabilities_backward(
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request);
void metal_attention_flash_causal_forward(
    const FlashCausalAttentionForwardRequest& request);
void metal_attention_flash_causal_backward(
    const FlashCausalAttentionBackwardRequest& request);
void metal_attention_paged_decode_forward(
    const PagedDecodeAttentionForwardRequest& request);

} // namespace riftco_transformer::backend_detail
