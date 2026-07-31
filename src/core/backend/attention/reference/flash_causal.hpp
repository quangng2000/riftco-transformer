#pragma once

#include "core/backend/attention/contracts.hpp"

namespace transformer_lab::backend_detail {

void reference_flash_causal_attention_forward(
    const FlashCausalAttentionForwardRequest& request
);
void reference_flash_causal_attention_backward(
    const FlashCausalAttentionBackwardRequest& request
);

}  // namespace transformer_lab::backend_detail
