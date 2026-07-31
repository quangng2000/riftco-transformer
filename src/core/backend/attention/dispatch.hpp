#pragma once

#include "core/backend/attention/contracts.hpp"
#include "riftco_transformer/core/backend.hpp"

namespace riftco_transformer::backend_detail {

void dispatch_materialized_causal_attention_forward(
    ExecutionBackend backend,
    const MaterializedCausalAttentionForwardRequest& request
);
void dispatch_materialized_causal_attention_context_backward(
    ExecutionBackend backend,
    const MaterializedCausalAttentionContextBackwardRequest& request
);
void dispatch_materialized_causal_attention_probabilities_backward(
    ExecutionBackend backend,
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
);
void dispatch_flash_causal_attention_forward(
    ExecutionBackend backend,
    const FlashCausalAttentionForwardRequest& request
);
void dispatch_flash_causal_attention_backward(
    ExecutionBackend backend,
    const FlashCausalAttentionBackwardRequest& request
);
void dispatch_paged_decode_attention_forward(
    ExecutionBackend backend,
    const PagedDecodeAttentionForwardRequest& request
);

}  // namespace riftco_transformer::backend_detail
