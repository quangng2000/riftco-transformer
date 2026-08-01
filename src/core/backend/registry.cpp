#include "riftco_transformer/core/backend.hpp"

#include "adapter.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer {
namespace {

thread_local ExecutionBackend active_backend =
    ExecutionBackend::Cpu;

const backend_detail::BackendAdapter& require_backend_adapter(
    ExecutionBackend backend
) {
    const auto* adapter =
        backend_detail::find_backend_adapter(backend);
    if (adapter == nullptr) {
        throw std::invalid_argument("unknown execution backend");
    }
    return *adapter;
}

void require_available(
    const backend_detail::BackendAdapter& adapter
) {
    if (!adapter.is_available()) {
        std::string message = std::string(adapter.name()) +
            " execution backend is unavailable";
        const std::string_view reason = adapter.unavailability_reason();
        if (!reason.empty()) {
            message += ": ";
            message += reason;
        }
        throw std::runtime_error(message);
    }
}

std::size_t checked_product(
    std::initializer_list<std::size_t> factors
) {
    std::size_t result = 1;
    for (const auto factor : factors) {
        if (factor != 0 &&
            result >
                std::numeric_limits<std::size_t>::max() / factor) {
            throw std::overflow_error(
                "backend operation size exceeds addressable storage"
            );
        }
        result *= factor;
    }
    return result;
}

void require_storage_contract(
    const backend_detail::TensorStorage& storage,
    ExecutionBackend backend,
    std::size_t expected_size
) {
    if (storage.backend() != backend ||
        storage.size() != expected_size ||
        storage.data().size() != expected_size) {
        throw std::logic_error(
            "backend adapter returned invalid tensor storage"
        );
    }
}

}  // namespace

bool execution_backend_available(
    ExecutionBackend backend
) noexcept {
    const auto* adapter =
        backend_detail::find_backend_adapter(backend);
    return adapter != nullptr && adapter->is_available();
}

std::string_view execution_backend_unavailability_reason(
    ExecutionBackend backend
) noexcept {
    const auto* adapter =
        backend_detail::find_backend_adapter(backend);
    if (adapter == nullptr || adapter->is_available()) {
        return {};
    }
    return adapter->unavailability_reason();
}

std::string_view execution_backend_name(
    ExecutionBackend backend
) {
    return require_backend_adapter(backend).name();
}

ExecutionBackend execution_backend() noexcept {
    return active_backend;
}

void set_execution_backend(ExecutionBackend backend) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    active_backend = backend;
}

ScopedExecutionBackend::ScopedExecutionBackend(
    ExecutionBackend backend
)
    : previous_(execution_backend()) {
    set_execution_backend(backend);
}

ScopedExecutionBackend::~ScopedExecutionBackend() noexcept {
    active_backend = previous_;
}

namespace backend_detail {

const BackendAdapter* find_backend_adapter(
    ExecutionBackend backend
) noexcept {
    switch (backend) {
        case ExecutionBackend::Cpu:
            return &cpu_backend_adapter();
        case ExecutionBackend::Metal:
            return &metal_backend_adapter();
        case ExecutionBackend::Cuda:
            return &cuda_backend_adapter();
        case ExecutionBackend::Tpu:
            return &tpu_backend_adapter();
    }
    return nullptr;
}

std::unique_ptr<TensorStorage> make_tensor_storage(
    ExecutionBackend backend,
    std::size_t element_count,
    float fill_value
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    auto storage = adapter.make_storage(element_count, fill_value);
    if (storage == nullptr) {
        throw std::logic_error(
            "backend adapter returned null tensor storage"
        );
    }
    require_storage_contract(*storage, backend, element_count);
    return storage;
}

std::unique_ptr<TensorStorage> make_tensor_storage(
    ExecutionBackend backend,
    std::vector<float> values
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    const auto element_count = values.size();
    auto storage = adapter.make_storage(std::move(values));
    if (storage == nullptr) {
        throw std::logic_error(
            "backend adapter returned null tensor storage"
        );
    }
    require_storage_contract(*storage, backend, element_count);
    return storage;
}

void dispatch_matmul(
    ExecutionBackend backend,
    const TensorStorage& left,
    const TensorStorage& right,
    TensorStorage& output,
    MatmulDimensions dimensions
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (left.backend() != right.backend() ||
        left.backend() != output.backend()) {
        throw std::invalid_argument(
            "matmul storage backends must match"
        );
    }
    const auto expected_left = checked_product({
        dimensions.batch_count,
        dimensions.rows,
        dimensions.shared,
    });
    const auto expected_right = checked_product({
        dimensions.batch_count,
        dimensions.shared,
        dimensions.columns,
    });
    const auto expected_output = checked_product({
        dimensions.batch_count,
        dimensions.rows,
        dimensions.columns,
    });
    if (left.size() != expected_left ||
        right.size() != expected_right ||
        output.size() != expected_output) {
        throw std::logic_error(
            "matmul storage size does not match its dimensions"
        );
    }
    adapter.matmul({left, right, output, dimensions});
}

}  // namespace backend_detail
}  // namespace riftco_transformer
