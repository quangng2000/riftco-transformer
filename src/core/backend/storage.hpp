#pragma once

#include "riftco_transformer/core/backend.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace riftco_transformer::backend_detail {

// Private, backend-owned contiguous storage. Metal and CUDA expose host-visible
// accelerator allocations; TPU storage keeps a host mirror and stages selected
// native PJRT programs. Host spans keep the reference capabilities usable while
// each platform grows native kernels independently.
class TensorStorage {
public:
    virtual ~TensorStorage() = default;

    [[nodiscard]] virtual ExecutionBackend backend() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual std::span<float> data() noexcept = 0;
    [[nodiscard]] virtual std::span<const float> data() const noexcept = 0;

    // Non-null only when the backend has a native persistent buffer.
    [[nodiscard]] virtual void* native_handle() noexcept = 0;
    [[nodiscard]] virtual const void* native_handle() const noexcept = 0;

    [[nodiscard]] virtual std::unique_ptr<TensorStorage> clone() const = 0;
};

}  // namespace riftco_transformer::backend_detail
