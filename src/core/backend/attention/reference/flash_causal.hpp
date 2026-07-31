#pragma once

#include "core/backend/attention/contracts.hpp"

namespace riftco_transformer::backend_detail {

void reference_flash_causal_attention_forward(
    const FlashCausalAttentionForwardRequest& request
);
void reference_flash_causal_attention_backward(
    const FlashCausalAttentionBackwardRequest& request
);

}  // namespace riftco_transformer::backend_detail
