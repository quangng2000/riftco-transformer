#pragma once

#include "core/backend/attention/contracts.hpp"

namespace transformer_lab::backend_detail {

void reference_paged_decode_attention_forward(
    const PagedDecodeAttentionForwardRequest& request
);

}  // namespace transformer_lab::backend_detail
