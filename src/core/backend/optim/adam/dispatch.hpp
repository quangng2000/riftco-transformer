#pragma once

#include "core/backend/optim/adam/contracts.hpp"
#include "riftco_transformer/core/backend.hpp"

namespace riftco_transformer::backend_detail {

void dispatch_adam_update(ExecutionBackend backend,
                          const AdamUpdateRequest& request);

} // namespace riftco_transformer::backend_detail
