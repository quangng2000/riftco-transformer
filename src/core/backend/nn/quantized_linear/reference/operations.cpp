#include "core/backend/nn/quantized_linear/reference/operations.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace riftco_transformer::backend_detail {
namespace {

// The normalized-float-4 levels used by QLoRA/bitsandbytes. Keeping this table
// identical across reference and accelerator paths makes packed checkpoints
// portable between backends.
inline constexpr std::array<float, 16> kNf4Codebook{
    -1.0F,
    -0.6961928009986877F,
    -0.5250730514526367F,
    -0.39491748809814453F,
    -0.28444138169288635F,
    -0.18477343022823334F,
    -0.09105003625154495F,
    0.0F,
    0.07958029955625534F,
    0.16093020141124725F,
    0.24611230194568634F,
    0.33791524171829224F,
    0.44070982933044434F,
    0.5626170039176941F,
    0.7229568362236023F,
    1.0F,
};

[[nodiscard]] float decode_nf4(
    std::span<const std::uint8_t> packed_codes,
    const Nf4ScaleStorageView& scales,
    std::size_t block_size,
    std::size_t flat_index
) noexcept {
    const std::uint8_t packed = packed_codes[flat_index / 2];
    const std::uint8_t code =
        (flat_index & 1U) == 0U
            ? static_cast<std::uint8_t>(packed & 0x0FU)
            : static_cast<std::uint8_t>(packed >> 4U);
    return kNf4Codebook[code] *
        decode_nf4_block_scale(scales, flat_index / block_size);
}

}  // namespace

void reference_quantized_linear_forward(
    const QuantizedLinearForwardRequest& request
) {
    const auto input = request.input.data();
    const auto packed_codes = request.weight.packed_codes();
    const auto scales = request.weight.scale_storage();
    auto output = request.output.data();
    const auto& dimensions = request.dimensions;

    for (std::size_t row = 0; row < dimensions.rows; ++row) {
        const std::size_t input_offset = row * dimensions.input_width;
        const std::size_t output_offset = row * dimensions.output_width;
        for (std::size_t output_column = 0;
             output_column < dimensions.output_width;
             ++output_column) {
            const std::size_t weight_offset =
                output_column * dimensions.input_width;
            float total = 0.0F;
            for (std::size_t input_column = 0;
                 input_column < dimensions.input_width;
                 ++input_column) {
                total += input[input_offset + input_column] *
                    decode_nf4(
                        packed_codes,
                        scales,
                        dimensions.block_size,
                        weight_offset + input_column
                    );
            }
            output[output_offset + output_column] = total;
        }
    }
}

void reference_quantized_linear_input_backward(
    const QuantizedLinearInputBackwardRequest& request
) {
    const auto upstream = request.upstream.data();
    const auto packed_codes = request.weight.packed_codes();
    const auto scales = request.weight.scale_storage();
    auto input_gradient = request.input_gradient.data();
    const auto& dimensions = request.dimensions;

    for (std::size_t row = 0; row < dimensions.rows; ++row) {
        const std::size_t upstream_offset = row * dimensions.output_width;
        const std::size_t gradient_offset = row * dimensions.input_width;
        for (std::size_t input_column = 0;
             input_column < dimensions.input_width;
             ++input_column) {
            float total = 0.0F;
            for (std::size_t output_column = 0;
                 output_column < dimensions.output_width;
                 ++output_column) {
                const std::size_t weight_index =
                    output_column * dimensions.input_width + input_column;
                total += upstream[upstream_offset + output_column] *
                    decode_nf4(
                        packed_codes,
                        scales,
                        dimensions.block_size,
                        weight_index
                    );
            }
            input_gradient[gradient_offset + input_column] = total;
        }
    }
}

}  // namespace riftco_transformer::backend_detail
