#pragma once

#include "riftco_transformer/core/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace riftco_transformer::backend_detail {

enum class Nf4ScaleEncoding : std::uint8_t {
    Float32 = 0,
    DoubleQuantizedUInt8 = 1,
};

// Owning transfer object used only while constructing immutable backend
// storage. Exactly one of fp32_scales or quantized_scales is populated.
struct Nf4ScaleStorageData {
    Nf4ScaleEncoding encoding = Nf4ScaleEncoding::Float32;
    std::vector<float> fp32_scales;
    std::vector<std::uint8_t> quantized_scales;
    std::vector<float> second_level_scales;
    std::size_t second_level_block_size = 0;
    float offset = 0.0F;
};

// Non-owning view consumed directly by validation and kernels. Double-
// quantized storage never exposes a resolved FP32 first-level scale vector.
struct Nf4ScaleStorageView {
    Nf4ScaleEncoding encoding = Nf4ScaleEncoding::Float32;
    std::span<const float> fp32_scales;
    std::span<const std::uint8_t> quantized_scales;
    std::span<const float> second_level_scales;
    std::size_t second_level_block_size = 0;
    float offset = 0.0F;

    [[nodiscard]] std::size_t scale_count() const noexcept {
        return encoding == Nf4ScaleEncoding::Float32
                   ? fp32_scales.size()
                   : quantized_scales.size();
    }
};

[[nodiscard]] inline float decode_nf4_block_scale(
    const Nf4ScaleStorageView& scales,
    std::size_t scale_index
) noexcept {
    if (scales.encoding == Nf4ScaleEncoding::Float32) {
        return scales.fp32_scales[scale_index];
    }
    const int centered_code =
        static_cast<int>(scales.quantized_scales[scale_index]) - 128;
    const float factor = static_cast<float>(centered_code) / 127.0F;
    const float delta = scales.second_level_scales[
                            scale_index / scales.second_level_block_size
                        ] * factor;
    if (delta > 0.0F &&
        scales.offset > std::numeric_limits<float>::max() - delta) {
        return std::numeric_limits<float>::max();
    }
    const float reconstructed = scales.offset + delta;
    return reconstructed > 0.0F ? reconstructed : 0.0F;
}

[[nodiscard]] inline std::size_t nf4_scale_payload_bytes(
    const Nf4ScaleStorageView& scales
) noexcept {
    if (scales.encoding == Nf4ScaleEncoding::Float32) {
        return scales.fp32_scales.size() * sizeof(float);
    }
    return scales.quantized_scales.size() +
        scales.second_level_scales.size() * sizeof(float) + sizeof(float);
}

// Immutable backend-owned storage for one packed NF4 weight. Quantized weights
// intentionally do not inherit TensorStorage: TensorStorage's float spans are
// part of its contract and representing nibbles through them would erase the
// memory benefit that quantized linear is meant to preserve.
class QuantizedWeightStorage {
public:
    virtual ~QuantizedWeightStorage() = default;

    [[nodiscard]] virtual ExecutionBackend backend() const noexcept = 0;
    [[nodiscard]] virtual std::span<const std::uint8_t>
    packed_codes() const noexcept = 0;
    [[nodiscard]] virtual Nf4ScaleStorageView
    scale_storage() const noexcept = 0;

    // Native handles name persistent packed-code and scale allocations. They
    // are null for the CPU reference storage.
    [[nodiscard]] virtual const void*
    packed_codes_native_handle() const noexcept = 0;
    // Primary names FP32 first-level scales for legacy storage or uint8 scale
    // codes for double-quantized storage. Secondary is null for legacy storage
    // and names FP32 second-level scales for double-quantized storage.
    [[nodiscard]] virtual const void*
    primary_scales_native_handle() const noexcept = 0;
    [[nodiscard]] virtual const void*
    secondary_scales_native_handle() const noexcept = 0;

    // Counts packed-code and scale bytes across every persistent backend
    // allocation or mirror, excluding allocator/device metadata. It is at
    // least the logical payload size; a unified allocation counts once.
    [[nodiscard]] virtual std::size_t
    resident_payload_bytes() const noexcept = 0;
};

}  // namespace riftco_transformer::backend_detail
