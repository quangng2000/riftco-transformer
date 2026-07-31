#pragma once

#include "transformer_lab/core/backend.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace transformer_lab::backend_detail {

// Private, backend-owned contiguous storage. Metal uses a persistent shared
// MTLBuffer so both GPU kernels and the existing host reference operations can
// access the same allocation without per-operation copies.
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

}  // namespace transformer_lab::backend_detail
