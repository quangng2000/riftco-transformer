#pragma once

#include "core/backend/attention/contracts.hpp"

namespace transformer_lab::backend_detail {

// Attention is an independent backend capability so new algorithms can be
// added without changing storage, optimizer, or serving-cache policy.
class AttentionCapability {
public:
    virtual ~AttentionCapability() = default;

    virtual void materialized_causal_attention_forward(
        const MaterializedCausalAttentionForwardRequest& request
    ) const = 0;
    virtual void materialized_causal_attention_context_backward(
        const MaterializedCausalAttentionContextBackwardRequest& request
    ) const = 0;
    virtual void materialized_causal_attention_probabilities_backward(
        const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
    ) const = 0;
    virtual void flash_causal_attention_forward(
        const FlashCausalAttentionForwardRequest& request
    ) const = 0;
    virtual void flash_causal_attention_backward(
        const FlashCausalAttentionBackwardRequest& request
    ) const = 0;
    virtual void paged_decode_attention_forward(
        const PagedDecodeAttentionForwardRequest& request
    ) const = 0;
};

}  // namespace transformer_lab::backend_detail
