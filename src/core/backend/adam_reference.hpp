#pragma once

#include "adapter.hpp"

namespace riftco_transformer::backend_detail {

// Portable semantic reference used by the CPU adapter and by accelerated
// backends when native float arithmetic cannot safely represent an
// intermediate. It writes only the request's out-of-place candidate storage.
void adam_reference_update(const AdamUpdateRequest& request);

}  // namespace riftco_transformer::backend_detail
