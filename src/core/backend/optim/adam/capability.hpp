#pragma once

#include "core/backend/optim/adam/contracts.hpp"

namespace riftco_transformer::backend_detail {

class AdamCapability {
  public:
    virtual ~AdamCapability() = default;

    // Writes only next-state buffers. A backend must complete the entire batch
    // or throw before returning; callers commit only after success.
    virtual void adam_update(const AdamUpdateRequest& request) const = 0;
};

} // namespace riftco_transformer::backend_detail
