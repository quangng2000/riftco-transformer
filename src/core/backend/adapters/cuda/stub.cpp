#include "core/backend/unavailable_adapter.hpp"

namespace riftco_transformer::backend_detail {

const BackendAdapter& cuda_backend_adapter() noexcept {
    static const UnavailableBackendAdapter adapter("cuda");
    return adapter;
}

}  // namespace riftco_transformer::backend_detail
