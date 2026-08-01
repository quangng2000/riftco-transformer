#pragma once

#include "core/backend/optim/adam/contracts.hpp"

namespace riftco_transformer::backend_detail {

// Launches one grid-stride update kernel per parameter tensor and synchronizes
// once after the complete out-of-place candidate batch has been encoded.
void cuda_adam_update(const AdamUpdateRequest& request);

} // namespace riftco_transformer::backend_detail
