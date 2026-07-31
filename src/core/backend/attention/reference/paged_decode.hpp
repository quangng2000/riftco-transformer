#pragma once

#include "core/backend/attention/contracts.hpp"

namespace riftco_transformer::backend_detail {

void reference_paged_decode_attention_forward(
    const PagedDecodeAttentionForwardRequest& request
);

}  // namespace riftco_transformer::backend_detail
