#pragma once

namespace riftco_transformer::backend_detail::quantized_linear_metal_detail {

inline constexpr char kQuantizedLinearKernelSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float rt_nf4_codebook[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

inline float rt_decode_nf4(
    device const uchar* packed_codes,
    device const float* block_scales,
    ulong block_size,
    ulong flat_index
) {
    const uchar packed = packed_codes[flat_index >> 1];
    const uint code =
        (flat_index & 1ul) == 0ul
            ? uint(packed & uchar(0x0f))
            : uint(packed >> 4);
    return rt_nf4_codebook[code] * block_scales[flat_index / block_size];
}

kernel void rt_nf4_linear_forward(
    device const float* input [[buffer(0)]],
    device const uchar* packed_codes [[buffer(1)]],
    device const float* block_scales [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ulong& rows [[buffer(4)]],
    constant ulong& input_width [[buffer(5)]],
    constant ulong& output_width [[buffer(6)]],
    constant ulong& block_size [[buffer(7)]],
    uint position [[thread_position_in_grid]]
) {
    const ulong output_count = rows * output_width;
    const ulong output_index = ulong(position);
    if (output_index >= output_count) {
        return;
    }

    const ulong row = output_index / output_width;
    const ulong output_column = output_index % output_width;
    const ulong input_offset = row * input_width;
    const ulong weight_offset = output_column * input_width;
    float total = 0.0f;
    for (ulong input_column = 0;
         input_column < input_width;
         ++input_column) {
        total += input[input_offset + input_column] *
            rt_decode_nf4(
                packed_codes,
                block_scales,
                block_size,
                weight_offset + input_column
            );
    }
    output[output_index] = total;
}

kernel void rt_nf4_linear_input_backward(
    device const float* upstream [[buffer(0)]],
    device const uchar* packed_codes [[buffer(1)]],
    device const float* block_scales [[buffer(2)]],
    device float* input_gradient [[buffer(3)]],
    constant ulong& rows [[buffer(4)]],
    constant ulong& input_width [[buffer(5)]],
    constant ulong& output_width [[buffer(6)]],
    constant ulong& block_size [[buffer(7)]],
    uint position [[thread_position_in_grid]]
) {
    const ulong gradient_count = rows * input_width;
    const ulong gradient_index = ulong(position);
    if (gradient_index >= gradient_count) {
        return;
    }

    const ulong row = gradient_index / input_width;
    const ulong input_column = gradient_index % input_width;
    const ulong upstream_offset = row * output_width;
    float total = 0.0f;
    for (ulong output_column = 0;
         output_column < output_width;
         ++output_column) {
        const ulong weight_index =
            output_column * input_width + input_column;
        total += upstream[upstream_offset + output_column] *
            rt_decode_nf4(
                packed_codes,
                block_scales,
                block_size,
                weight_index
            );
    }
    input_gradient[gradient_index] = total;
}

inline float rt_decode_nf4_double_quantized(
    device const uchar* packed_codes,
    device const uchar* scale_codes,
    device const float* second_level_scales,
    ulong block_size,
    ulong scale_block_size,
    float scale_offset,
    ulong flat_index
) {
    const uchar packed = packed_codes[flat_index >> 1];
    const uint code =
        (flat_index & 1ul) == 0ul
            ? uint(packed & uchar(0x0f))
            : uint(packed >> 4);
    const ulong scale_index = flat_index / block_size;
    const int centered_code = int(scale_codes[scale_index]) - 128;
    const float delta =
        second_level_scales[scale_index / scale_block_size] *
        (float(centered_code) / 127.0f);
    const float block_scale = clamp(
        scale_offset + delta,
        0.0f,
        0x1.fffffep+127f
    );
    return rt_nf4_codebook[code] * block_scale;
}

kernel void rt_nf4_double_quantized_linear_forward(
    device const float* input [[buffer(0)]],
    device const uchar* packed_codes [[buffer(1)]],
    device const uchar* scale_codes [[buffer(2)]],
    device const float* second_level_scales [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant ulong& rows [[buffer(5)]],
    constant ulong& input_width [[buffer(6)]],
    constant ulong& output_width [[buffer(7)]],
    constant ulong& block_size [[buffer(8)]],
    constant ulong& scale_block_size [[buffer(9)]],
    constant float& scale_offset [[buffer(10)]],
    uint position [[thread_position_in_grid]]
) {
    const ulong output_count = rows * output_width;
    const ulong output_index = ulong(position);
    if (output_index >= output_count) {
        return;
    }

    const ulong row = output_index / output_width;
    const ulong output_column = output_index % output_width;
    const ulong input_offset = row * input_width;
    const ulong weight_offset = output_column * input_width;
    float total = 0.0f;
    for (ulong input_column = 0;
         input_column < input_width;
         ++input_column) {
        total += input[input_offset + input_column] *
            rt_decode_nf4_double_quantized(
                packed_codes,
                scale_codes,
                second_level_scales,
                block_size,
                scale_block_size,
                scale_offset,
                weight_offset + input_column
            );
    }
    output[output_index] = total;
}

kernel void rt_nf4_double_quantized_linear_input_backward(
    device const float* upstream [[buffer(0)]],
    device const uchar* packed_codes [[buffer(1)]],
    device const uchar* scale_codes [[buffer(2)]],
    device const float* second_level_scales [[buffer(3)]],
    device float* input_gradient [[buffer(4)]],
    constant ulong& rows [[buffer(5)]],
    constant ulong& input_width [[buffer(6)]],
    constant ulong& output_width [[buffer(7)]],
    constant ulong& block_size [[buffer(8)]],
    constant ulong& scale_block_size [[buffer(9)]],
    constant float& scale_offset [[buffer(10)]],
    uint position [[thread_position_in_grid]]
) {
    const ulong gradient_count = rows * input_width;
    const ulong gradient_index = ulong(position);
    if (gradient_index >= gradient_count) {
        return;
    }

    const ulong row = gradient_index / input_width;
    const ulong input_column = gradient_index % input_width;
    const ulong upstream_offset = row * output_width;
    float total = 0.0f;
    for (ulong output_column = 0;
         output_column < output_width;
         ++output_column) {
        const ulong weight_index =
            output_column * input_width + input_column;
        total += upstream[upstream_offset + output_column] *
            rt_decode_nf4_double_quantized(
                packed_codes,
                scale_codes,
                second_level_scales,
                block_size,
                scale_block_size,
                scale_offset,
                weight_index
            );
    }
    input_gradient[gradient_index] = total;
}
)METAL";

}  // namespace riftco_transformer::backend_detail::quantized_linear_metal_detail
