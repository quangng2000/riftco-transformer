#include "core/backend/optim/adam/dispatch.hpp"

#include "core/backend/adapter.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace riftco_transformer::backend_detail {
namespace {

const BackendAdapter& require_backend_adapter(ExecutionBackend backend) {
    const auto* adapter = find_backend_adapter(backend);
    if (adapter == nullptr) {
        throw std::invalid_argument("unknown execution backend");
    }
    return *adapter;
}

void require_available(const BackendAdapter& adapter) {
    if (!adapter.is_available()) {
        std::string message =
            std::string(adapter.name()) + " execution backend is unavailable";
        const std::string_view reason = adapter.unavailability_reason();
        if (!reason.empty()) {
            message += ": ";
            message += reason;
        }
        throw std::runtime_error(message);
    }
}

void require_out_of_place(std::span<const AdamTensorUpdate> tensors) {
    std::unordered_set<const TensorStorage*> live_objects;
    std::unordered_set<const void*> live_handles;
    for (const auto& tensor : tensors) {
        const TensorStorage* const entries[]{
            &tensor.value,
            &tensor.gradient,
            &tensor.first_moment,
            &tensor.second_moment,
        };
        for (const auto* entry : entries) {
            live_objects.insert(entry);
            if (const void* handle = entry->native_handle(); handle != nullptr) {
                live_handles.insert(handle);
            }
        }
    }

    std::unordered_set<const TensorStorage*> candidate_objects;
    std::unordered_set<const void*> candidate_handles;
    for (const auto& tensor : tensors) {
        const TensorStorage* const entries[]{
            &tensor.next_value,
            &tensor.next_first_moment,
            &tensor.next_second_moment,
        };
        for (const auto* entry : entries) {
            const void* const handle = entry->native_handle();
            if (live_objects.contains(entry) ||
                (handle != nullptr && live_handles.contains(handle))) {
                throw std::invalid_argument(
                    "Adam candidate outputs must not alias live state");
            }
            if (!candidate_objects.insert(entry).second ||
                (handle != nullptr &&
                 !candidate_handles.insert(handle).second)) {
                throw std::invalid_argument(
                    "Adam candidate outputs must not alias each other");
            }
        }
    }
}

} // namespace

void dispatch_adam_update(ExecutionBackend backend,
                          const AdamUpdateRequest& request) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.tensors.empty()) {
        throw std::invalid_argument("Adam update requires at least one tensor");
    }
    if (!std::isfinite(request.learning_rate) ||
        request.learning_rate <= 0.0F || !std::isfinite(request.beta1) ||
        request.beta1 <= 0.0F || request.beta1 >= 1.0F ||
        !std::isfinite(request.beta2) || request.beta2 <= 0.0F ||
        request.beta2 >= 1.0F || !std::isfinite(request.epsilon) ||
        request.epsilon <= 0.0F || !std::isfinite(request.clip_scale) ||
        request.clip_scale <= 0.0 || request.clip_scale > 1.0 ||
        !std::isfinite(request.first_correction) ||
        request.first_correction <= 0.0 ||
        !std::isfinite(request.second_correction) ||
        request.second_correction <= 0.0) {
        throw std::invalid_argument(
            "Adam update received invalid scalar state");
    }
    for (const auto& tensor : request.tensors) {
        const auto element_count = tensor.value.size();
        if (element_count == 0 || tensor.value.backend() != backend ||
            tensor.gradient.backend() != backend ||
            tensor.first_moment.backend() != backend ||
            tensor.second_moment.backend() != backend ||
            tensor.next_value.backend() != backend ||
            tensor.next_first_moment.backend() != backend ||
            tensor.next_second_moment.backend() != backend) {
            throw std::invalid_argument(
                "Adam update tensors must share its backend");
        }
        if (tensor.gradient.size() != element_count ||
            tensor.first_moment.size() != element_count ||
            tensor.second_moment.size() != element_count ||
            tensor.next_value.size() != element_count ||
            tensor.next_first_moment.size() != element_count ||
            tensor.next_second_moment.size() != element_count) {
            throw std::logic_error(
                "Adam update tensor storage sizes do not match");
        }
    }
    require_out_of_place(request.tensors);
    adapter.adam_update(request);
}

} // namespace riftco_transformer::backend_detail
