#include "core/backend/optim/adam/metal/diagnostics.hpp"
#include "core/backend/unavailable_adapter.hpp"

namespace riftco_transformer::backend_detail {

const BackendAdapter& metal_backend_adapter() noexcept {
    static const UnavailableBackendAdapter adapter("metal");
    return adapter;
}

void reset_metal_adam_path_counts() noexcept {}

MetalAdamPathCounts metal_adam_path_counts() noexcept {
    return {0, 0};
}

}  // namespace riftco_transformer::backend_detail
