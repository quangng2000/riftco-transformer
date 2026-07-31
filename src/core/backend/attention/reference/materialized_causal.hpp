#pragma once

#include "core/backend/attention/contracts.hpp"

namespace transformer_lab::backend_detail {

void reference_materialized_causal_attention_forward(
    const MaterializedCausalAttentionForwardRequest& request
);
void reference_materialized_causal_attention_context_backward(
    const MaterializedCausalAttentionContextBackwardRequest& request
);
void reference_materialized_causal_attention_probabilities_backward(
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
);

}  // namespace transformer_lab::backend_detail
