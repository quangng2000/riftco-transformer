#pragma once

#include "core/backend/attention/contracts.hpp"

namespace riftco_transformer::backend_detail {

void tpu_materialized_causal_attention_forward(
    const MaterializedCausalAttentionForwardRequest& request);
void tpu_materialized_causal_attention_context_backward(
    const MaterializedCausalAttentionContextBackwardRequest& request);
void tpu_materialized_causal_attention_probabilities_backward(
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request);

}  // namespace riftco_transformer::backend_detail
