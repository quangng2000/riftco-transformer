#pragma once

#include <cstdint>
#include <string_view>

namespace transformer_lab {

// Identifies both tensor storage and the built-in implementation used by
// dispatched operations. CPU tensors use host storage; Metal tensors own
// persistent shared MTLBuffers.
enum class ExecutionBackend : std::uint8_t {
    Cpu = 0,
    Metal = 1,
};

// Returns false for both recognized-but-unavailable and unknown values.
[[nodiscard]] bool execution_backend_available(
    ExecutionBackend backend
) noexcept;

// Throws std::invalid_argument for an unknown value.
[[nodiscard]] std::string_view execution_backend_name(
    ExecutionBackend backend
);

// Returns the calling thread's default for newly constructed tensors and
// modules. Operations derived from existing tensors preserve their backend.
[[nodiscard]] ExecutionBackend execution_backend() noexcept;

// Changes the calling thread's construction default. Rejects unavailable or
// unknown backends without changing the current selection.
void set_execution_backend(ExecutionBackend backend);

// Exception-safe temporary override for the calling thread. Scoped overrides
// must be properly nested and destroyed on the thread that constructed them.
class [[nodiscard]] ScopedExecutionBackend final {
public:
    explicit ScopedExecutionBackend(ExecutionBackend backend);

    ScopedExecutionBackend(const ScopedExecutionBackend&) = delete;
    ScopedExecutionBackend& operator=(
        const ScopedExecutionBackend&
    ) = delete;
    ScopedExecutionBackend(ScopedExecutionBackend&&) = delete;
    ScopedExecutionBackend& operator=(
        ScopedExecutionBackend&&
    ) = delete;

    ~ScopedExecutionBackend() noexcept;

private:
    ExecutionBackend previous_;
};

}  // namespace transformer_lab
