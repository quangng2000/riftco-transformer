#pragma once

#include "core/backend/attention/capability.hpp"
#include "core/backend/nn/capability.hpp"
#include "core/backend/optim/adam/capability.hpp"
#include "storage.hpp"
#include "riftco_transformer/core/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace riftco_transformer::backend_detail {

struct MatmulDimensions {
    std::size_t batch_count;
    std::size_t rows;
    std::size_t shared;
    std::size_t columns;
};

struct MatmulRequest {
    const TensorStorage& left;
    const TensorStorage& right;
    TensorStorage& output;
    MatmulDimensions dimensions;
};

class StorageCapability {
public:
    virtual ~StorageCapability() = default;

    [[nodiscard]] virtual std::unique_ptr<TensorStorage> make_storage(
        std::size_t element_count,
        float fill_value
    ) const = 0;
    [[nodiscard]] virtual std::unique_ptr<TensorStorage> make_storage(
        std::vector<float> values
    ) const = 0;
};

class MatmulCapability {
public:
    virtual ~MatmulCapability() = default;

    // Shape validation belongs to tensor_ops. Adapters receive contiguous,
    // correctly sized storage and must finish writing output before returning.
    virtual void matmul(const MatmulRequest& request) const = 0;
};

// Internal Adapter facade. Capability-specific base interfaces keep storage,
// linear algebra, and optimizer growth independent.
class BackendAdapter
    : public StorageCapability,
      public MatmulCapability,
      public ElementwiseCapability,
      public ReductionCapability,
      public LayoutCapability,
      public SoftmaxCapability,
      public IndexingCapability,
      public NormalizationCapability,
      public LossCapability,
      public AttentionCapability,
      public AdamCapability {
public:
    ~BackendAdapter() override = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool is_available() const noexcept = 0;
    [[nodiscard]] virtual std::string_view unavailability_reason()
        const noexcept {
        return {};
    }
};

// Implemented by the platform-neutral CPU adapter.
[[nodiscard]] const BackendAdapter& cpu_backend_adapter() noexcept;

// Implemented by adapters/metal/adapter.mm on Apple and by
// adapters/metal/stub.cpp on other platforms.
[[nodiscard]] const BackendAdapter& metal_backend_adapter() noexcept;

// Implemented by adapters/cuda/adapter.cu when explicitly enabled and by
// adapters/cuda/stub.cpp in builds without CUDA and in standard wheels.
[[nodiscard]] const BackendAdapter& cuda_backend_adapter() noexcept;

// Implemented by the PJRT/libtpu adapter when explicitly enabled and by the
// unavailable stub in ordinary builds and standard wheels.
[[nodiscard]] const BackendAdapter& tpu_backend_adapter() noexcept;

// The registry is deliberately closed and internal for now. Public callers
// select a stable ExecutionBackend value rather than owning adapter objects.
[[nodiscard]] const BackendAdapter* find_backend_adapter(
    ExecutionBackend backend
) noexcept;

[[nodiscard]] std::unique_ptr<TensorStorage> make_tensor_storage(
    ExecutionBackend backend,
    std::size_t element_count,
    float fill_value
);
[[nodiscard]] std::unique_ptr<TensorStorage> make_tensor_storage(
    ExecutionBackend backend,
    std::vector<float> values
);

void dispatch_matmul(
    ExecutionBackend backend,
    const TensorStorage& left,
    const TensorStorage& right,
    TensorStorage& output,
    MatmulDimensions dimensions
);

}  // namespace riftco_transformer::backend_detail
