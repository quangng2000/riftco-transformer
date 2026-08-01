#pragma once

#include "core/backend/attention/contracts.hpp"

namespace riftco_transformer::backend_detail {

void cuda_materialized_causal_attention_forward(
    const MaterializedCausalAttentionForwardRequest& request);
void cuda_materialized_causal_attention_context_backward(
    const MaterializedCausalAttentionContextBackwardRequest& request);
void cuda_materialized_causal_attention_probabilities_backward(
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request);
void cuda_flash_causal_attention_forward(
    const FlashCausalAttentionForwardRequest& request);
void cuda_flash_causal_attention_backward(
    const FlashCausalAttentionBackwardRequest& request);
void cuda_paged_decode_attention_forward(
    const PagedDecodeAttentionForwardRequest& request);

}  // namespace riftco_transformer::backend_detail
