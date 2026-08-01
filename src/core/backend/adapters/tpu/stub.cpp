#include "core/backend/unavailable_adapter.hpp"

namespace riftco_transformer::backend_detail {

const BackendAdapter& tpu_backend_adapter() noexcept {
    static const UnavailableBackendAdapter adapter("tpu");
    return adapter;
}

}  // namespace riftco_transformer::backend_detail
