#pragma once

#include "core/backend/adapter.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace riftco_transformer::backend_detail {

enum class TpuProgramKind : std::uint8_t {
    Matmul,
    MaterializedCausalAttentionForward,
    MaterializedCausalAttentionContextBackward,
    MaterializedCausalAttentionProbabilitiesBackward,
    PagedDecodeAttentionForward,
};

enum class TpuElementType : std::uint8_t {
    F32,
    S32,
};

struct TpuProgramKey {
    TpuProgramKind kind;
    std::array<std::size_t, 6> dimensions{};
    std::size_t dimension_count = 0;

    [[nodiscard]] bool
    operator==(const TpuProgramKey&) const noexcept = default;
};

struct TpuProgram {
    TpuProgramKey key;
    std::string stablehlo;
    std::string operation_name;
};

struct TpuHostInput {
    const void* data;
    std::size_t byte_size;
    TpuElementType element_type;
    std::span<const std::int64_t> dimensions;
};

struct TpuHostOutput {
    void* data;
    std::size_t byte_size;
    TpuElementType element_type;
    std::span<const std::int64_t> dimensions;
};

// The TPU runtime is isolated from the Adapter facade so the PJRT ABI and
// libtpu lifecycle do not leak into shared backend code.
[[nodiscard]] bool tpu_runtime_available() noexcept;
[[nodiscard]] std::string_view tpu_runtime_unavailability_reason() noexcept;
void tpu_runtime_execute(
    const TpuProgram& program,
    std::span<const TpuHostInput> inputs,
    std::span<const TpuHostOutput> outputs);
void tpu_runtime_matmul(const MatmulRequest& request);

}  // namespace riftco_transformer::backend_detail
