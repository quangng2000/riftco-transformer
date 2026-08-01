#pragma once

#include "core/backend/attention/contracts.hpp"

namespace riftco_transformer::backend_detail {

void tpu_paged_decode_attention_forward(
    const PagedDecodeAttentionForwardRequest& request);

}  // namespace riftco_transformer::backend_detail
