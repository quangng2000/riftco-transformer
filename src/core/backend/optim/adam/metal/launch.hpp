#pragma once

#include "core/backend/optim/adam/contracts.hpp"

namespace riftco_transformer::backend_detail {

// Objective-C++ boundary for the fused Metal Adam candidate-state update.
// The implementation may retry the complete batch through the wide reference
// path before returning; live optimizer state is committed only by the caller.
void metal_adam_update(const AdamUpdateRequest& request);

} // namespace riftco_transformer::backend_detail
