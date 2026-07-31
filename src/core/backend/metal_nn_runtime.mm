#include "metal_nn_runtime.hpp"

#include "core/backend/attention/metal/flash_causal_kernels.hpp"
#include "core/backend/attention/metal/materialized_causal_kernels.hpp"
#include "core/backend/attention/metal/paged_decode_kernels.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

constexpr const char* kNeuralKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint RT_STATUS_DOMAIN = 1u;
constant uint RT_STATUS_INVALID_ROW = 2u;
constant uint RT_STATUS_OVERFLOW = 4u;
constant float RT_INV_SQRT_TWO = 0.7071067811865475244f;
constant float RT_INV_SQRT_TWO_PI = 0.3989422804014326779f;

inline void rt_flag(
    device atomic_uint* status,
    uint value
) {
    atomic_fetch_or_explicit(
        status,
        value,
        memory_order_relaxed
    );
}

// Metal Shading Language does not expose erf on all supported SDK targets.
// This Abramowitz-Stegun approximation has a maximum absolute error around
// 1.5e-7, which is below the framework's float32 parity tolerance.
inline float rt_erf(float value) {
    if (value == 0.0f) {
        return value;
    }
    const float magnitude = fabs(value);
    const float t = 1.0f / (1.0f + 0.3275911f * magnitude);
    const float polynomial =
        ((((
            1.061405429f * t -
            1.453152027f
        ) * t +
            1.421413741f
        ) * t -
            0.284496736f
        ) * t +
            0.254829592f
        ) * t;
    const float approximation =
        1.0f - polynomial * exp(-magnitude * magnitude);
    return value < 0.0f ? -approximation : approximation;
}

inline void rt_compensated_add(
    float value,
    thread float& sum,
    thread float& correction
) {
    const float combined = sum + value;
    if (fabs(sum) >= fabs(value)) {
        correction += (sum - combined) + value;
    } else {
        correction += (value - combined) + sum;
    }
    sum = combined;
}

inline ulong rt_axis_offset(
    ulong slice,
    ulong coordinate,
    ulong width,
    ulong inner
) {
    const ulong outer_coordinate = slice / inner;
    const ulong inner_coordinate = slice % inner;
    return
        (outer_coordinate * width + coordinate) * inner +
        inner_coordinate;
}

kernel void rt_unary_elementwise(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant ulong& count [[buffer(3)]],
    constant uint& operation [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }
    const float value = input[index];
    switch (operation) {
        case 0u:
            output[index] = -value;
            break;
        case 1u:
            output[index] = exp(value);
            break;
        case 2u:
            if (value <= 0.0f) {
                rt_flag(status, RT_STATUS_DOMAIN);
                output[index] = 0.0f;
            } else {
                output[index] = log(value);
            }
            break;
        case 3u:
            if (value < 0.0f) {
                rt_flag(status, RT_STATUS_DOMAIN);
                output[index] = 0.0f;
            } else {
                output[index] = sqrt(value);
            }
            break;
        default:
            output[index] = rt_erf(value);
            break;
    }
}

kernel void rt_binary_elementwise(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ulong& count [[buffer(4)]],
    constant uint& operation [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }
    switch (operation) {
        case 0u:
            output[index] = left[index] + right[index];
            break;
        case 1u:
            output[index] = left[index] - right[index];
            break;
        case 2u:
            output[index] = left[index] * right[index];
            break;
        default:
            if (right[index] == 0.0f) {
                rt_flag(status, RT_STATUS_DOMAIN);
                output[index] = 0.0f;
            } else {
                output[index] = left[index] / right[index];
            }
            break;
    }
}

kernel void rt_scale(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong& count [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        output[index] = input[index] * scale;
    }
}

kernel void rt_gelu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }
    const float value = input[index];
    output[index] =
        0.5f * value *
        (1.0f + rt_erf(value * RT_INV_SQRT_TWO));
}

kernel void rt_gelu_backward(
    device const float* input [[buffer(0)]],
    device const float* upstream [[buffer(1)]],
    device float* input_gradient [[buffer(2)]],
    constant ulong& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }
    const float value = input[index];
    const float derivative =
        0.5f *
            (1.0f + rt_erf(value * RT_INV_SQRT_TWO)) +
        value * exp(-0.5f * value * value) *
            RT_INV_SQRT_TWO_PI;
    input_gradient[index] = upstream[index] * derivative;
}

kernel void rt_reduce_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong& outer [[buffer(2)]],
    constant ulong& width [[buffer(3)]],
    constant ulong& inner [[buffer(4)]],
    constant uint& mean [[buffer(5)]],
    uint slice_index [[thread_position_in_grid]]
) {
    const ulong slice_count = outer * inner;
    if (slice_index >= slice_count) {
        return;
    }
    float total = 0.0f;
    for (ulong coordinate = 0; coordinate < width; ++coordinate) {
        total += input[
            rt_axis_offset(
                slice_index,
                coordinate,
                width,
                inner
            )
        ];
    }
    output[slice_index] =
        mean == 0u ? total : total / float(width);
}

kernel void rt_copy(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) {
        output[index] = input[index];
    }
}

kernel void rt_permute(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const ulong* input_strides [[buffer(2)]],
    device const ulong* output_strides [[buffer(3)]],
    device const ulong* axes [[buffer(4)]],
    constant ulong& rank [[buffer(5)]],
    constant ulong& output_count [[buffer(6)]],
    uint output_index [[thread_position_in_grid]]
) {
    if (output_index >= output_count) {
        return;
    }
    ulong remainder = output_index;
    ulong input_index = 0;
    for (ulong dimension = 0; dimension < rank; ++dimension) {
        const ulong coordinate =
            remainder / output_strides[dimension];
        remainder %= output_strides[dimension];
        input_index +=
            coordinate * input_strides[axes[dimension]];
    }
    output[output_index] = input[input_index];
}

kernel void rt_broadcast(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const ulong* input_shape [[buffer(2)]],
    device const ulong* input_strides [[buffer(3)]],
    device const ulong* output_strides [[buffer(4)]],
    constant ulong& input_rank [[buffer(5)]],
    constant ulong& output_rank [[buffer(6)]],
    constant ulong& output_count [[buffer(7)]],
    uint output_index [[thread_position_in_grid]]
) {
    if (output_index >= output_count) {
        return;
    }
    const ulong rank_offset = output_rank - input_rank;
    ulong remainder = output_index;
    ulong input_index = 0;
    for (
        ulong output_dimension = 0;
        output_dimension < output_rank;
        ++output_dimension
    ) {
        const ulong coordinate =
            remainder / output_strides[output_dimension];
        remainder %= output_strides[output_dimension];
        if (output_dimension >= rank_offset) {
            const ulong input_dimension =
                output_dimension - rank_offset;
            if (input_shape[input_dimension] != 1u) {
                input_index +=
                    coordinate *
                    input_strides[input_dimension];
            }
        }
    }
    output[output_index] = input[input_index];
}

kernel void rt_sum_to_shape(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const ulong* input_strides [[buffer(2)]],
    device const ulong* output_shape [[buffer(3)]],
    device const ulong* output_strides [[buffer(4)]],
    device const ulong* reduction_axes [[buffer(5)]],
    device const ulong* reduction_strides [[buffer(6)]],
    constant ulong& input_rank [[buffer(7)]],
    constant ulong& output_rank [[buffer(8)]],
    constant ulong& reduction_rank [[buffer(9)]],
    constant ulong& reduction_count [[buffer(10)]],
    constant ulong& output_count [[buffer(11)]],
    uint output_index [[thread_position_in_grid]]
) {
    if (output_index >= output_count) {
        return;
    }

    // Build the fixed part of the input coordinate from dimensions that
    // survive the broadcast reduction.
    const ulong rank_offset = input_rank - output_rank;
    ulong output_remainder = output_index;
    ulong input_base = 0;
    for (
        ulong output_dimension = 0;
        output_dimension < output_rank;
        ++output_dimension
    ) {
        const ulong coordinate =
            output_remainder / output_strides[output_dimension];
        output_remainder %= output_strides[output_dimension];
        if (output_shape[output_dimension] != 1u) {
            input_base +=
                coordinate *
                input_strides[rank_offset + output_dimension];
        }
    }

    // reduction_axes are sorted in input-dimension order and their compact
    // strides are row-major. Iterating this coordinate therefore visits the
    // matching input elements in increasing flat-index order, exactly like
    // the CPU reference, without testing every unrelated input element.
    float total = 0.0f;
    for (
        ulong reduction_index = 0;
        reduction_index < reduction_count;
        ++reduction_index
    ) {
        ulong reduction_remainder = reduction_index;
        ulong input_index = input_base;
        for (
            ulong reduction_dimension = 0;
            reduction_dimension < reduction_rank;
            ++reduction_dimension
        ) {
            const ulong coordinate =
                reduction_remainder /
                reduction_strides[reduction_dimension];
            reduction_remainder %=
                reduction_strides[reduction_dimension];
            input_index +=
                coordinate *
                input_strides[
                    reduction_axes[reduction_dimension]
                ];
        }
        total += input[input_index];
    }
    output[output_index] = total;
}

kernel void rt_softmax_forward(
    device const float* input [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant ulong& outer [[buffer(3)]],
    constant ulong& width [[buffer(4)]],
    constant ulong& inner [[buffer(5)]],
    uint slice_index [[thread_position_in_grid]]
) {
    const ulong slice_count = outer * inner;
    if (slice_index >= slice_count) {
        return;
    }
    float maximum = -INFINITY;
    bool valid = true;
    for (ulong coordinate = 0; coordinate < width; ++coordinate) {
        const float value = input[
            rt_axis_offset(
                slice_index,
                coordinate,
                width,
                inner
            )
        ];
        if (isnan(value) || value == INFINITY) {
            valid = false;
        }
        maximum = max(maximum, value);
    }
    if (!valid || maximum == -INFINITY) {
        rt_flag(status, RT_STATUS_INVALID_ROW);
        for (ulong coordinate = 0; coordinate < width; ++coordinate) {
            probabilities[
                rt_axis_offset(
                    slice_index,
                    coordinate,
                    width,
                    inner
                )
            ] = 0.0f;
        }
        return;
    }
    float denominator_sum = 0.0f;
    float denominator_correction = 0.0f;
    for (ulong coordinate = 0; coordinate < width; ++coordinate) {
        rt_compensated_add(
            exp(
                input[
                    rt_axis_offset(
                        slice_index,
                        coordinate,
                        width,
                        inner
                    )
                ] - maximum
            ),
            denominator_sum,
            denominator_correction
        );
    }
    const float denominator =
        denominator_sum + denominator_correction;
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        rt_flag(status, RT_STATUS_INVALID_ROW);
        return;
    }
    for (ulong coordinate = 0; coordinate < width; ++coordinate) {
        const ulong offset = rt_axis_offset(
            slice_index,
            coordinate,
            width,
            inner
        );
        probabilities[offset] =
            exp(input[offset] - maximum) / denominator;
    }
}

kernel void rt_softmax_backward(
    device const float* probabilities [[buffer(0)]],
    device const float* upstream [[buffer(1)]],
    device float* input_gradient [[buffer(2)]],
    constant ulong& outer [[buffer(3)]],
    constant ulong& width [[buffer(4)]],
    constant ulong& inner [[buffer(5)]],
    uint slice_index [[thread_position_in_grid]]
) {
    const ulong slice_count = outer * inner;
    if (slice_index >= slice_count) {
        return;
    }
    float dot_sum = 0.0f;
    float dot_correction = 0.0f;
    for (ulong coordinate = 0; coordinate < width; ++coordinate) {
        const ulong offset = rt_axis_offset(
            slice_index,
            coordinate,
            width,
            inner
        );
        rt_compensated_add(
            probabilities[offset] * upstream[offset],
            dot_sum,
            dot_correction
        );
    }
    const float dot = dot_sum + dot_correction;
    for (ulong coordinate = 0; coordinate < width; ++coordinate) {
        const ulong offset = rt_axis_offset(
            slice_index,
            coordinate,
            width,
            inner
        );
        input_gradient[offset] =
            probabilities[offset] * (upstream[offset] - dot);
    }
}

kernel void rt_causal_softmax_forward(
    device const float* scores [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant ulong& row_count [[buffer(3)]],
    constant ulong& time [[buffer(4)]],
    constant float& score_scale [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= row_count) {
        return;
    }
    const ulong query = row % time;
    const ulong base = row * time;
    float maximum = -INFINITY;
    bool valid = true;
    for (ulong key = 0; key <= query; ++key) {
        const float value = scores[base + key] * score_scale;
        if (isnan(value) || value == INFINITY) {
            valid = false;
        }
        maximum = max(maximum, value);
    }
    for (ulong key = query + 1; key < time; ++key) {
        probabilities[base + key] = 0.0f;
    }
    if (!valid || maximum == -INFINITY) {
        rt_flag(status, RT_STATUS_INVALID_ROW);
        for (ulong key = 0; key <= query; ++key) {
            probabilities[base + key] = 0.0f;
        }
        return;
    }
    float denominator_sum = 0.0f;
    float denominator_correction = 0.0f;
    for (ulong key = 0; key <= query; ++key) {
        rt_compensated_add(
            exp(
                scores[base + key] * score_scale -
                maximum
            ),
            denominator_sum,
            denominator_correction
        );
    }
    const float denominator =
        denominator_sum + denominator_correction;
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        rt_flag(status, RT_STATUS_INVALID_ROW);
        return;
    }
    for (ulong key = 0; key <= query; ++key) {
        probabilities[base + key] =
            exp(scores[base + key] * score_scale - maximum) /
            denominator;
    }
}

kernel void rt_causal_softmax_backward(
    device const float* probabilities [[buffer(0)]],
    device const float* upstream [[buffer(1)]],
    device float* score_gradient [[buffer(2)]],
    constant ulong& row_count [[buffer(3)]],
    constant ulong& time [[buffer(4)]],
    constant float& score_scale [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= row_count) {
        return;
    }
    const ulong query = row % time;
    const ulong base = row * time;
    float dot_sum = 0.0f;
    float dot_correction = 0.0f;
    for (ulong key = 0; key <= query; ++key) {
        rt_compensated_add(
            probabilities[base + key] * upstream[base + key],
            dot_sum,
            dot_correction
        );
    }
    const float dot = dot_sum + dot_correction;
    for (ulong key = 0; key <= query; ++key) {
        score_gradient[base + key] =
            score_scale * probabilities[base + key] *
            (upstream[base + key] - dot);
    }
    for (ulong key = query + 1; key < time; ++key) {
        score_gradient[base + key] = 0.0f;
    }
}

kernel void rt_gather_rows(
    device const float* table [[buffer(0)]],
    device const uint* row_indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& position_count [[buffer(3)]],
    constant ulong& width [[buffer(4)]],
    uint output_index [[thread_position_in_grid]]
) {
    const ulong output_count = position_count * width;
    if (output_index >= output_count) {
        return;
    }
    const ulong position = output_index / width;
    const ulong column = output_index % width;
    output[output_index] =
        table[ulong(row_indices[position]) * width + column];
}

kernel void rt_scatter_add_rows(
    device const float* upstream [[buffer(0)]],
    device const ulong* row_offsets [[buffer(1)]],
    device const ulong* grouped_positions [[buffer(2)]],
    device float* table_gradient [[buffer(3)]],
    constant ulong& row_count [[buffer(4)]],
    constant ulong& width [[buffer(5)]],
    uint table_index [[thread_position_in_grid]]
) {
    const ulong table_count = row_count * width;
    if (table_index >= table_count) {
        return;
    }
    const ulong row = table_index / width;
    const ulong column = table_index % width;
    float total = 0.0f;
    for (
        ulong grouped_index = row_offsets[row];
        grouped_index < row_offsets[row + 1];
        ++grouped_index
    ) {
        const ulong position = grouped_positions[grouped_index];
        total += upstream[position * width + column];
    }
    table_gradient[table_index] = total;
}

kernel void rt_layer_norm_forward(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* mean [[buffer(4)]],
    device float* inverse_standard_deviation [[buffer(5)]],
    constant ulong& rows [[buffer(6)]],
    constant ulong& width [[buffer(7)]],
    constant float& epsilon [[buffer(8)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    const ulong base = ulong(row) * width;
    float row_mean = 0.0f;
    for (ulong column = 0; column < width; ++column) {
        row_mean += input[base + column];
    }
    row_mean /= float(width);
    float variance = 0.0f;
    for (ulong column = 0; column < width; ++column) {
        const float centered = input[base + column] - row_mean;
        variance += centered * centered;
    }
    variance /= float(width);
    const float inverse = rsqrt(variance + epsilon);
    mean[row] = row_mean;
    inverse_standard_deviation[row] = inverse;
    for (ulong column = 0; column < width; ++column) {
        const float normalized =
            (input[base + column] - row_mean) * inverse;
        output[base + column] =
            normalized * scale[column] + bias[column];
    }
}

kernel void rt_layer_norm_input_backward(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* mean [[buffer(2)]],
    device const float* inverse_standard_deviation [[buffer(3)]],
    device const float* upstream [[buffer(4)]],
    device float* input_gradient [[buffer(5)]],
    constant ulong& rows [[buffer(6)]],
    constant ulong& width [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }
    const ulong base = ulong(row) * width;
    const float row_mean = mean[row];
    const float inverse = inverse_standard_deviation[row];
    float gradient_sum = 0.0f;
    float normalized_gradient_sum = 0.0f;
    for (ulong column = 0; column < width; ++column) {
        const float normalized =
            (input[base + column] - row_mean) * inverse;
        const float gradient =
            upstream[base + column] * scale[column];
        gradient_sum += gradient;
        normalized_gradient_sum += gradient * normalized;
    }
    for (ulong column = 0; column < width; ++column) {
        const float normalized =
            (input[base + column] - row_mean) * inverse;
        const float gradient =
            upstream[base + column] * scale[column];
        input_gradient[base + column] =
            inverse / float(width) *
            (
                float(width) * gradient -
                gradient_sum -
                normalized * normalized_gradient_sum
            );
    }
}

kernel void rt_layer_norm_parameter_backward(
    device const float* input [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* inverse_standard_deviation [[buffer(2)]],
    device const float* upstream [[buffer(3)]],
    device float* scale_gradient [[buffer(4)]],
    device float* bias_gradient [[buffer(5)]],
    constant ulong& rows [[buffer(6)]],
    constant ulong& width [[buffer(7)]],
    uint column [[thread_position_in_grid]]
) {
    if (column >= width) {
        return;
    }
    float scale_total = 0.0f;
    float bias_total = 0.0f;
    for (ulong row = 0; row < rows; ++row) {
        const ulong offset = row * width + column;
        const float normalized =
            (input[offset] - mean[row]) *
            inverse_standard_deviation[row];
        scale_total += upstream[offset] * normalized;
        bias_total += upstream[offset];
    }
    scale_gradient[column] = scale_total;
    bias_gradient[column] = bias_total;
}

kernel void rt_cross_entropy_rows(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* base_gradient [[buffer(2)]],
    device float* row_losses [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant ulong& positions [[buffer(5)]],
    constant ulong& classes [[buffer(6)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= positions) {
        return;
    }
    const ulong base = ulong(position) * classes;
    const uint target = targets[position];
    float maximum = -INFINITY;
    bool valid = true;
    for (ulong column = 0; column < classes; ++column) {
        const float value = logits[base + column];
        if (isnan(value) || value == INFINITY) {
            valid = false;
        }
        maximum = max(maximum, value);
    }
    if (
        !valid ||
        maximum == -INFINITY ||
        !isfinite(logits[base + ulong(target)])
    ) {
        rt_flag(status, RT_STATUS_INVALID_ROW);
        row_losses[position] = 0.0f;
        for (ulong column = 0; column < classes; ++column) {
            base_gradient[base + column] = 0.0f;
        }
        return;
    }
    float denominator_sum = 0.0f;
    float denominator_correction = 0.0f;
    for (ulong column = 0; column < classes; ++column) {
        rt_compensated_add(
            exp(logits[base + column] - maximum),
            denominator_sum,
            denominator_correction
        );
    }
    const float denominator =
        denominator_sum + denominator_correction;
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        rt_flag(status, RT_STATUS_INVALID_ROW);
        return;
    }
    const float inverse_positions = 1.0f / float(positions);
    for (ulong column = 0; column < classes; ++column) {
        float gradient =
            exp(logits[base + column] - maximum) /
            denominator;
        if (column == ulong(target)) {
            gradient -= 1.0f;
        }
        base_gradient[base + column] =
            gradient * inverse_positions;
    }
    const float loss =
        log(denominator) +
        maximum -
        logits[base + ulong(target)];
    if (!isfinite(loss)) {
        rt_flag(status, RT_STATUS_OVERFLOW);
        row_losses[position] = 0.0f;
    } else {
        row_losses[position] = loss;
    }
}

kernel void rt_cross_entropy_reduce(
    device const float* row_losses [[buffer(0)]],
    device float* loss [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant ulong& positions [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0u) {
        return;
    }
    float maximum = 0.0f;
    for (ulong position = 0; position < positions; ++position) {
        maximum = max(maximum, row_losses[position]);
    }
    if (maximum == 0.0f) {
        loss[0] = 0.0f;
        return;
    }
    float ratio_sum = 0.0f;
    float ratio_correction = 0.0f;
    for (ulong position = 0; position < positions; ++position) {
        rt_compensated_add(
            row_losses[position] / maximum,
            ratio_sum,
            ratio_correction
        );
    }
    const float mean =
        maximum *
        (
            (ratio_sum + ratio_correction) /
            float(positions)
        );
    if (!isfinite(mean)) {
        rt_flag(status, RT_STATUS_OVERFLOW);
        loss[0] = 0.0f;
    } else {
        loss[0] = mean;
    }
}

)METAL";

std::string error_description(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const char* description =
        error.localizedDescription.UTF8String;
    return description == nullptr
               ? "unknown Metal error"
               : std::string(description);
}

template <typename Value>
NSUInteger checked_bytes(std::size_t count) {
    if (count >
        std::numeric_limits<NSUInteger>::max() /
            sizeof(Value)) {
        throw std::overflow_error(
            "Metal neural buffer size exceeds NSUInteger"
        );
    }
    return static_cast<NSUInteger>(count * sizeof(Value));
}

std::vector<std::uint64_t> contiguous_strides(
    std::span<const std::size_t> shape
) {
    std::vector<std::uint64_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index > 1; --index) {
        const auto current = index - 2;
        strides[current] =
            strides[current + 1] *
            static_cast<std::uint64_t>(shape[current + 1]);
    }
    return strides;
}

std::vector<std::uint64_t> as_u64(
    std::span<const std::size_t> values
) {
    std::vector<std::uint64_t> converted;
    converted.reserve(values.size());
    for (const auto value : values) {
        converted.push_back(
            static_cast<std::uint64_t>(value)
        );
    }
    return converted;
}

struct SumToShapeMetadata {
    std::vector<std::uint64_t> reduction_axes;
    std::vector<std::uint64_t> reduction_strides;
    std::uint64_t reduction_count;
};

SumToShapeMetadata make_sum_to_shape_metadata(
    const SumToShapeRequest& request
) {
    const auto rank_offset =
        request.input_shape.size() - request.output_shape.size();
    std::vector<std::size_t> reduction_axes;
    std::vector<std::size_t> reduction_shape;
    reduction_axes.reserve(request.input_shape.size());
    reduction_shape.reserve(request.input_shape.size());

    for (std::size_t input_dimension = 0;
         input_dimension < request.input_shape.size();
         ++input_dimension) {
        const bool leading_dimension =
            input_dimension < rank_offset;
        const bool collapsed_dimension =
            !leading_dimension &&
            request.output_shape[
                input_dimension - rank_offset
            ] == 1;
        if (leading_dimension || collapsed_dimension) {
            reduction_axes.push_back(input_dimension);
            reduction_shape.push_back(
                request.input_shape[input_dimension]
            );
        }
    }

    if (request.output.size() == 0 ||
        request.input.size() % request.output.size() != 0) {
        throw std::logic_error(
            "sum-to-shape storage sizes are not reduction-compatible"
        );
    }
    return {
        as_u64(reduction_axes),
        contiguous_strides(reduction_shape),
        static_cast<std::uint64_t>(
            request.input.size() / request.output.size()
        ),
    };
}

struct ScatterGrouping {
    std::vector<std::uint64_t> row_offsets;
    std::vector<std::uint64_t> grouped_positions;
};

ScatterGrouping make_scatter_grouping(
    const ScatterAddRowsRequest& request
) {
    if (request.row_count ==
        std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(
            "Metal scatter row-offset table exceeds addressable storage"
        );
    }

    ScatterGrouping grouping{
        std::vector<std::uint64_t>(request.row_count + 1, 0),
        std::vector<std::uint64_t>(request.row_indices.size(), 0),
    };
    for (const auto row : request.row_indices) {
        ++grouping.row_offsets[
            static_cast<std::size_t>(row) + 1
        ];
    }
    for (std::size_t row = 0; row < request.row_count; ++row) {
        grouping.row_offsets[row + 1] +=
            grouping.row_offsets[row];
    }

    // Fill each row's group backwards while scanning positions backwards.
    // The resulting positions inside every group remain in original ascending
    // order, which preserves the CPU reference's deterministic addition order.
    for (std::size_t position = request.row_indices.size();
         position > 0;
         --position) {
        const auto row = static_cast<std::size_t>(
            request.row_indices[position - 1]
        );
        const auto destination =
            --grouping.row_offsets[row + 1];
        grouping.grouped_positions[destination] = position - 1;
    }

    // The backwards fill changed each end offset into that row's start offset.
    // Shift those starts into canonical CSR offsets and restore the final end.
    for (std::size_t row = 0; row < request.row_count; ++row) {
        grouping.row_offsets[row] =
            grouping.row_offsets[row + 1];
    }
    grouping.row_offsets[request.row_count] =
        request.row_indices.size();
    return grouping;
}

class MetalNeuralRuntime {
public:
    MetalNeuralRuntime()
        : device_(MTLCreateSystemDefaultDevice()),
          pipelines_([NSMutableDictionary dictionary]) {
        if (device_ == nil) {
            throw std::runtime_error(
                "no Metal device is available"
            );
        }
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) {
            throw std::runtime_error(
                "could not create a Metal neural command queue"
            );
        }
    }

    void unary_elementwise(
        const UnaryElementwiseRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_unary_elementwise",
                    "unary elementwise"
                );
            auto command = command_encoder(
                "unary elementwise"
            );
            const auto status = make_status_buffer();
            const std::uint64_t count = request.element_count;
            const std::uint32_t operation =
                static_cast<std::uint32_t>(request.operation);
            [command.encoder setComputePipelineState:pipeline];
            [command.encoder
                setBuffer:require_buffer(request.input)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.output)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:status
                   offset:0
                  atIndex:2];
            [command.encoder
                setBytes:&count length:sizeof(count) atIndex:3];
            [command.encoder
                setBytes:&operation
                   length:sizeof(operation)
                  atIndex:4];
            dispatch(command.encoder, pipeline, request.element_count);
            complete(command, "unary elementwise");
            require_status_clear(
                status,
                "unary elementwise domain error"
            );
        }
    }

    void binary_elementwise(
        const BinaryElementwiseRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_binary_elementwise",
                    "binary elementwise"
                );
            auto command = command_encoder(
                "binary elementwise"
            );
            const auto status = make_status_buffer();
            const std::uint64_t count = request.element_count;
            const std::uint32_t operation =
                static_cast<std::uint32_t>(request.operation);
            [command.encoder setComputePipelineState:pipeline];
            set_three_buffers(
                command.encoder,
                request.left,
                request.right,
                request.output
            );
            [command.encoder setBuffer:status offset:0 atIndex:3];
            [command.encoder
                setBytes:&count length:sizeof(count) atIndex:4];
            [command.encoder
                setBytes:&operation
                   length:sizeof(operation)
                  atIndex:5];
            dispatch(command.encoder, pipeline, request.element_count);
            complete(command, "binary elementwise");
            require_status_clear(
                status,
                "division by zero in tensor operation"
            );
        }
    }

    void scale(const ScaleRequest& request) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(@"rt_scale", "scale");
            auto command = command_encoder("scale");
            const std::uint64_t count = request.element_count;
            [command.encoder setComputePipelineState:pipeline];
            set_two_buffers(
                command.encoder,
                request.input,
                request.output
            );
            [command.encoder
                setBytes:&count length:sizeof(count) atIndex:2];
            [command.encoder
                setBytes:&request.scale
                   length:sizeof(request.scale)
                  atIndex:3];
            dispatch(command.encoder, pipeline, request.element_count);
            complete(command, "scale");
        }
    }

    void gelu_forward(const GeluForwardRequest& request) {
        unary_count_operation(
            @"rt_gelu_forward",
            "GELU forward",
            request.input,
            request.output,
            request.element_count
        );
    }

    void gelu_backward(const GeluBackwardRequest& request) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_gelu_backward",
                    "GELU backward"
                );
            auto command = command_encoder("GELU backward");
            const std::uint64_t count = request.element_count;
            [command.encoder setComputePipelineState:pipeline];
            set_three_buffers(
                command.encoder,
                request.input,
                request.upstream,
                request.input_gradient
            );
            [command.encoder
                setBytes:&count length:sizeof(count) atIndex:3];
            dispatch(command.encoder, pipeline, request.element_count);
            complete(command, "GELU backward");
        }
    }

    void reduce(const ReductionRequest& request) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_reduce_axis",
                    "axis reduction"
                );
            auto command = command_encoder("axis reduction");
            const std::uint64_t outer =
                request.dimensions.outer;
            const std::uint64_t width =
                request.dimensions.width;
            const std::uint64_t inner =
                request.dimensions.inner;
            const std::uint32_t mean =
                request.operation ==
                        ReductionOperation::Mean
                    ? 1U
                    : 0U;
            [command.encoder setComputePipelineState:pipeline];
            set_two_buffers(
                command.encoder,
                request.input,
                request.output
            );
            [command.encoder
                setBytes:&outer length:sizeof(outer) atIndex:2];
            [command.encoder
                setBytes:&width length:sizeof(width) atIndex:3];
            [command.encoder
                setBytes:&inner length:sizeof(inner) atIndex:4];
            [command.encoder
                setBytes:&mean length:sizeof(mean) atIndex:5];
            dispatch(
                command.encoder,
                pipeline,
                request.dimensions.outer *
                    request.dimensions.inner
            );
            complete(command, "axis reduction");
        }
    }

    void copy(const CopyRequest& request) {
        unary_count_operation(
            @"rt_copy",
            "copy",
            request.input,
            request.output,
            request.element_count
        );
    }

    void permute(const PermuteRequest& request) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(@"rt_permute", "permute");
            const auto input_strides =
                contiguous_strides(request.input_shape);
            std::vector<std::size_t> output_shape;
            output_shape.reserve(request.axes.size());
            for (const auto axis : request.axes) {
                output_shape.push_back(
                    request.input_shape[axis]
                );
            }
            const auto output_strides =
                contiguous_strides(output_shape);
            const auto axes = as_u64(request.axes);
            const auto input_stride_buffer =
                make_vector_buffer(input_strides);
            const auto output_stride_buffer =
                make_vector_buffer(output_strides);
            const auto axes_buffer = make_vector_buffer(axes);
            auto command = command_encoder("permute");
            const std::uint64_t rank =
                request.input_shape.size();
            const std::uint64_t output_count =
                request.output.size();
            [command.encoder setComputePipelineState:pipeline];
            set_two_buffers(
                command.encoder,
                request.input,
                request.output
            );
            [command.encoder
                setBuffer:input_stride_buffer
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:output_stride_buffer
                   offset:0
                  atIndex:3];
            [command.encoder
                setBuffer:axes_buffer offset:0 atIndex:4];
            [command.encoder
                setBytes:&rank length:sizeof(rank) atIndex:5];
            [command.encoder
                setBytes:&output_count
                   length:sizeof(output_count)
                  atIndex:6];
            dispatch(command.encoder, pipeline, request.output.size());
            complete(command, "permute");
        }
    }

    void broadcast(const BroadcastRequest& request) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_broadcast",
                    "broadcast"
                );
            const auto input_shape = as_u64(request.input_shape);
            const auto input_strides =
                contiguous_strides(request.input_shape);
            const auto output_strides =
                contiguous_strides(request.output_shape);
            const auto input_shape_buffer =
                make_vector_buffer(input_shape);
            const auto input_stride_buffer =
                make_vector_buffer(input_strides);
            const auto output_stride_buffer =
                make_vector_buffer(output_strides);
            auto command = command_encoder("broadcast");
            const std::uint64_t input_rank =
                request.input_shape.size();
            const std::uint64_t output_rank =
                request.output_shape.size();
            const std::uint64_t output_count =
                request.output.size();
            [command.encoder setComputePipelineState:pipeline];
            set_two_buffers(
                command.encoder,
                request.input,
                request.output
            );
            [command.encoder
                setBuffer:input_shape_buffer
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:input_stride_buffer
                   offset:0
                  atIndex:3];
            [command.encoder
                setBuffer:output_stride_buffer
                   offset:0
                  atIndex:4];
            [command.encoder
                setBytes:&input_rank
                   length:sizeof(input_rank)
                  atIndex:5];
            [command.encoder
                setBytes:&output_rank
                   length:sizeof(output_rank)
                  atIndex:6];
            [command.encoder
                setBytes:&output_count
                   length:sizeof(output_count)
                  atIndex:7];
            dispatch(command.encoder, pipeline, request.output.size());
            complete(command, "broadcast");
        }
    }

    void sum_to_shape(const SumToShapeRequest& request) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_sum_to_shape",
                    "sum-to-shape"
                );
            const auto input_strides =
                contiguous_strides(request.input_shape);
            const auto output_shape = as_u64(request.output_shape);
            const auto output_strides =
                contiguous_strides(request.output_shape);
            const auto reduction =
                make_sum_to_shape_metadata(request);
            const auto input_stride_buffer =
                make_vector_buffer(input_strides);
            const auto output_shape_buffer =
                make_vector_buffer(output_shape);
            const auto output_stride_buffer =
                make_vector_buffer(output_strides);
            const auto reduction_axis_buffer =
                make_vector_buffer(reduction.reduction_axes);
            const auto reduction_stride_buffer =
                make_vector_buffer(reduction.reduction_strides);
            auto command = command_encoder("sum-to-shape");
            const std::uint64_t input_rank =
                request.input_shape.size();
            const std::uint64_t output_rank =
                request.output_shape.size();
            const std::uint64_t reduction_rank =
                reduction.reduction_axes.size();
            const std::uint64_t output_count =
                request.output.size();
            [command.encoder setComputePipelineState:pipeline];
            set_two_buffers(
                command.encoder,
                request.input,
                request.output
            );
            [command.encoder
                setBuffer:input_stride_buffer
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:output_shape_buffer
                   offset:0
                  atIndex:3];
            [command.encoder
                setBuffer:output_stride_buffer
                   offset:0
                  atIndex:4];
            [command.encoder
                setBuffer:reduction_axis_buffer
                   offset:0
                  atIndex:5];
            [command.encoder
                setBuffer:reduction_stride_buffer
                   offset:0
                  atIndex:6];
            [command.encoder
                setBytes:&input_rank
                   length:sizeof(input_rank)
                   atIndex:7];
            [command.encoder
                setBytes:&output_rank
                   length:sizeof(output_rank)
                   atIndex:8];
            [command.encoder
                setBytes:&reduction_rank
                   length:sizeof(reduction_rank)
                  atIndex:9];
            [command.encoder
                setBytes:&reduction.reduction_count
                   length:sizeof(reduction.reduction_count)
                  atIndex:10];
            [command.encoder
                setBytes:&output_count
                   length:sizeof(output_count)
                  atIndex:11];
            dispatch(command.encoder, pipeline, request.output.size());
            complete(command, "sum-to-shape");
        }
    }

    void softmax_forward(
        const SoftmaxForwardRequest& request
    ) {
        softmax_forward_impl(
            @"rt_softmax_forward",
            "softmax forward",
            request.input,
            request.probabilities,
            request.dimensions
        );
    }

    void softmax_backward(
        const SoftmaxBackwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_softmax_backward",
                    "softmax backward"
                );
            auto command = command_encoder("softmax backward");
            [command.encoder setComputePipelineState:pipeline];
            set_three_buffers(
                command.encoder,
                request.probabilities,
                request.upstream,
                request.input_gradient
            );
            set_axis_dimensions(
                command.encoder,
                request.dimensions,
                3
            );
            dispatch(
                command.encoder,
                pipeline,
                request.dimensions.outer *
                    request.dimensions.inner
            );
            complete(command, "softmax backward");
        }
    }

    void causal_softmax_forward(
        const CausalSoftmaxForwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_causal_softmax_forward",
                    "causal softmax forward"
                );
            auto command = command_encoder(
                "causal softmax forward"
            );
            const auto status = make_status_buffer();
            const std::uint64_t row_count =
                request.batch * request.heads * request.time;
            const std::uint64_t time = request.time;
            [command.encoder setComputePipelineState:pipeline];
            set_two_buffers(
                command.encoder,
                request.scores,
                request.probabilities
            );
            [command.encoder setBuffer:status offset:0 atIndex:2];
            [command.encoder
                setBytes:&row_count
                   length:sizeof(row_count)
                  atIndex:3];
            [command.encoder
                setBytes:&time length:sizeof(time) atIndex:4];
            [command.encoder
                setBytes:&request.score_scale
                   length:sizeof(request.score_scale)
                  atIndex:5];
            dispatch(command.encoder, pipeline, row_count);
            complete(command, "causal softmax forward");
            require_status_clear(
                status,
                "causal softmax received a non-finite row"
            );
        }
    }

    void causal_softmax_backward(
        const CausalSoftmaxBackwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_causal_softmax_backward",
                    "causal softmax backward"
                );
            auto command = command_encoder(
                "causal softmax backward"
            );
            const std::uint64_t row_count =
                request.batch * request.heads * request.time;
            const std::uint64_t time = request.time;
            [command.encoder setComputePipelineState:pipeline];
            set_three_buffers(
                command.encoder,
                request.probabilities,
                request.upstream,
                request.score_gradient
            );
            [command.encoder
                setBytes:&row_count
                   length:sizeof(row_count)
                  atIndex:3];
            [command.encoder
                setBytes:&time length:sizeof(time) atIndex:4];
            [command.encoder
                setBytes:&request.score_scale
                   length:sizeof(request.score_scale)
                  atIndex:5];
            dispatch(command.encoder, pipeline, row_count);
            complete(command, "causal softmax backward");
        }
    }

    void gather_rows(const GatherRowsRequest& request) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_gather_rows",
                    "embedding gather"
                );
            const auto indices =
                make_span_buffer(request.row_indices);
            auto command = command_encoder("embedding gather");
            const std::uint64_t positions =
                request.row_indices.size();
            const std::uint64_t width = request.width;
            [command.encoder setComputePipelineState:pipeline];
            [command.encoder
                setBuffer:require_buffer(request.table)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:indices offset:0 atIndex:1];
            [command.encoder
                setBuffer:require_buffer(request.output)
                   offset:0
                  atIndex:2];
            [command.encoder
                setBytes:&positions
                   length:sizeof(positions)
                  atIndex:3];
            [command.encoder
                setBytes:&width length:sizeof(width) atIndex:4];
            dispatch(command.encoder, pipeline, request.output.size());
            complete(command, "embedding gather");
        }
    }

    void scatter_add_rows(
        const ScatterAddRowsRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_scatter_add_rows",
                    "embedding scatter-add"
                );
            const auto grouping = make_scatter_grouping(request);
            const auto row_offsets =
                make_vector_buffer(grouping.row_offsets);
            const auto grouped_positions =
                make_vector_buffer(grouping.grouped_positions);
            auto command = command_encoder(
                "embedding scatter-add"
            );
            const std::uint64_t rows = request.row_count;
            const std::uint64_t width = request.width;
            [command.encoder setComputePipelineState:pipeline];
            [command.encoder
                setBuffer:require_buffer(request.upstream)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:row_offsets offset:0 atIndex:1];
            [command.encoder
                setBuffer:grouped_positions
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:require_buffer(
                              request.table_gradient
                          )
                   offset:0
                  atIndex:3];
            [command.encoder
                setBytes:&rows length:sizeof(rows) atIndex:4];
            [command.encoder
                setBytes:&width length:sizeof(width) atIndex:5];
            dispatch(
                command.encoder,
                pipeline,
                request.table_gradient.size()
            );
            complete(command, "embedding scatter-add");
        }
    }

    void layer_norm_forward(
        const LayerNormForwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(
                    @"rt_layer_norm_forward",
                    "LayerNorm forward"
                );
            auto command = command_encoder("LayerNorm forward");
            const std::uint64_t rows = request.rows;
            const std::uint64_t width = request.width;
            [command.encoder setComputePipelineState:pipeline];
            [command.encoder
                setBuffer:require_buffer(request.input)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.scale)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:require_buffer(request.bias)
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:require_buffer(request.output)
                   offset:0
                  atIndex:3];
            [command.encoder
                setBuffer:require_buffer(request.mean)
                   offset:0
                  atIndex:4];
            [command.encoder
                setBuffer:require_buffer(
                              request.inverse_standard_deviation
                          )
                   offset:0
                  atIndex:5];
            [command.encoder
                setBytes:&rows length:sizeof(rows) atIndex:6];
            [command.encoder
                setBytes:&width length:sizeof(width) atIndex:7];
            [command.encoder
                setBytes:&request.epsilon
                   length:sizeof(request.epsilon)
                  atIndex:8];
            dispatch(command.encoder, pipeline, request.rows);
            complete(command, "LayerNorm forward");
        }
    }

    void layer_norm_backward(
        const LayerNormBackwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto input_pipeline =
                require_pipeline(
                    @"rt_layer_norm_input_backward",
                    "LayerNorm input backward"
                );
            const auto parameter_pipeline =
                require_pipeline(
                    @"rt_layer_norm_parameter_backward",
                    "LayerNorm parameter backward"
                );
            auto command = command_encoder("LayerNorm backward");
            const std::uint64_t rows = request.rows;
            const std::uint64_t width = request.width;
            [command.encoder setComputePipelineState:input_pipeline];
            [command.encoder
                setBuffer:require_buffer(request.input)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.scale)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:require_buffer(request.mean)
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:require_buffer(
                              request.inverse_standard_deviation
                          )
                   offset:0
                  atIndex:3];
            [command.encoder
                setBuffer:require_buffer(request.upstream)
                   offset:0
                  atIndex:4];
            [command.encoder
                setBuffer:require_buffer(
                              request.input_gradient
                          )
                   offset:0
                  atIndex:5];
            [command.encoder
                setBytes:&rows length:sizeof(rows) atIndex:6];
            [command.encoder
                setBytes:&width length:sizeof(width) atIndex:7];
            dispatch(command.encoder, input_pipeline, request.rows);

            [command.encoder setComputePipelineState:parameter_pipeline];
            [command.encoder
                setBuffer:require_buffer(request.input)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.mean)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:require_buffer(
                              request.inverse_standard_deviation
                          )
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:require_buffer(request.upstream)
                   offset:0
                  atIndex:3];
            [command.encoder
                setBuffer:require_buffer(
                              request.scale_gradient
                          )
                   offset:0
                  atIndex:4];
            [command.encoder
                setBuffer:require_buffer(
                              request.bias_gradient
                          )
                   offset:0
                  atIndex:5];
            [command.encoder
                setBytes:&rows length:sizeof(rows) atIndex:6];
            [command.encoder
                setBytes:&width length:sizeof(width) atIndex:7];
            dispatch(command.encoder, parameter_pipeline, request.width);
            complete(command, "LayerNorm backward");
        }
    }

    void cross_entropy_forward(
        const CrossEntropyForwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto row_pipeline =
                require_pipeline(
                    @"rt_cross_entropy_rows",
                    "cross-entropy rows"
                );
            const auto reduce_pipeline =
                require_pipeline(
                    @"rt_cross_entropy_reduce",
                    "cross-entropy reduction"
                );
            const auto targets = make_span_buffer(request.targets);
            const auto row_losses =
                make_float_buffer(request.positions);
            const auto status = make_status_buffer();
            auto command = command_encoder("cross-entropy");
            const std::uint64_t positions = request.positions;
            const std::uint64_t classes = request.classes;
            [command.encoder setComputePipelineState:row_pipeline];
            [command.encoder
                setBuffer:require_buffer(request.logits)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:targets offset:0 atIndex:1];
            [command.encoder
                setBuffer:require_buffer(
                              request.base_gradient
                          )
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:row_losses offset:0 atIndex:3];
            [command.encoder
                setBuffer:status offset:0 atIndex:4];
            [command.encoder
                setBytes:&positions
                   length:sizeof(positions)
                  atIndex:5];
            [command.encoder
                setBytes:&classes
                   length:sizeof(classes)
                  atIndex:6];
            dispatch(
                command.encoder,
                row_pipeline,
                request.positions
            );

            [command.encoder setComputePipelineState:reduce_pipeline];
            [command.encoder
                setBuffer:row_losses offset:0 atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.loss)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:status offset:0 atIndex:2];
            [command.encoder
                setBytes:&positions
                   length:sizeof(positions)
                  atIndex:3];
            dispatch(command.encoder, reduce_pipeline, 1);
            complete(command, "cross-entropy");
            require_cross_entropy_status(status);
        }
    }

    void materialized_causal_attention_forward(
        const MaterializedCausalAttentionForwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto probability_pipeline =
                require_pipeline(
                    @"rt_materialized_causal_attention_probabilities",
                    "causal attention probabilities"
                );
            const auto context_pipeline =
                require_pipeline(
                    @"rt_materialized_causal_attention_context",
                    "causal attention context"
                );
            const auto status = make_status_buffer();
            auto command = command_encoder(
                "causal attention forward"
            );
            const auto dimensions = attention_constants(
                request.dimensions
            );
            [command.encoder
                setComputePipelineState:probability_pipeline];
            [command.encoder
                setBuffer:require_buffer(request.queries)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.keys)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:require_buffer(
                              request.probabilities
                          )
                   offset:0
                  atIndex:2];
            [command.encoder setBuffer:status offset:0 atIndex:3];
            set_attention_probability_constants(
                command.encoder,
                dimensions,
                4
            );
            dispatch(
                command.encoder,
                probability_pipeline,
                dimensions.row_count
            );

            [command.encoder
                setComputePipelineState:context_pipeline];
            [command.encoder
                setBuffer:require_buffer(
                              request.probabilities
                          )
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.values)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:require_buffer(request.context)
                   offset:0
                  atIndex:2];
            [command.encoder
                setBytes:&dimensions.vector_count
                   length:sizeof(dimensions.vector_count)
                  atIndex:3];
            [command.encoder
                setBytes:&dimensions.time
                   length:sizeof(dimensions.time)
                  atIndex:4];
            [command.encoder
                setBytes:&dimensions.head_width
                   length:sizeof(dimensions.head_width)
                  atIndex:5];
            dispatch(
                command.encoder,
                context_pipeline,
                dimensions.vector_count
            );
            complete(command, "causal attention forward");
            require_status_clear(
                status,
                "causal attention received non-finite scores"
            );
        }
    }

    void flash_causal_attention_forward(
        const FlashCausalAttentionForwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline = require_pipeline(
                @"rt_flash_causal_attention_forward",
                "Flash causal attention forward"
            );
            const auto delta_pipeline = require_pipeline(
                @"rt_flash_causal_attention_delta",
                "Flash causal attention delta"
            );
            const auto query_pipeline = require_pipeline(
                @"rt_flash_causal_attention_query_backward",
                "Flash causal attention query backward"
            );
            const auto key_value_pipeline = require_pipeline(
                @"rt_flash_causal_attention_key_value_backward",
                "Flash causal attention key/value backward"
            );
            const auto constants =
                flash_attention_constants(request.dimensions);
            const NSUInteger scratch_bytes =
                flash_scratch_bytes(
                    request.dimensions.head_width,
                    32,
                    96
                );
            const NSUInteger delta_scratch_bytes =
                flash_scratch_bytes(
                    request.dimensions.head_width,
                    32,
                    136
                );
            const NSUInteger query_scratch_bytes =
                flash_scratch_bytes(
                    request.dimensions.head_width,
                    40,
                    64
                );
            const NSUInteger key_value_scratch_bytes =
                flash_scratch_bytes(
                    request.dimensions.head_width,
                    48,
                    128
                );
            require_flash_resources(
                pipeline,
                scratch_bytes,
                "Flash causal attention forward"
            );
            // A model forward creates an autograd graph. Preflight every
            // backward kernel now so a device-specific head-width limit
            // cannot surface only after the caller has computed a loss.
            require_flash_resources(
                delta_pipeline,
                delta_scratch_bytes,
                "Flash causal attention delta"
            );
            require_flash_resources(
                query_pipeline,
                query_scratch_bytes,
                "Flash causal attention query backward"
            );
            require_flash_resources(
                key_value_pipeline,
                key_value_scratch_bytes,
                "Flash causal attention key/value backward"
            );
            const auto status = make_status_buffer();
            auto command = command_encoder(
                "Flash causal attention forward"
            );
            [command.encoder setComputePipelineState:pipeline];
            [command.encoder
                setBuffer:require_buffer(request.queries)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.keys)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:require_buffer(request.values)
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:require_buffer(request.row_maxima)
                   offset:0
                  atIndex:3];
            [command.encoder
                setBuffer:require_buffer(request.row_exp_sums)
                   offset:0
                  atIndex:4];
            [command.encoder
                setBuffer:require_buffer(request.context)
                   offset:0
                  atIndex:5];
            [command.encoder setBuffer:status offset:0 atIndex:6];
            set_flash_attention_constants(
                command.encoder,
                constants,
                7
            );
            dispatch_flash_threadgroups(
                command.encoder,
                pipeline,
                constants.query_tile_count,
                constants.batch_heads,
                scratch_bytes
            );
            complete(command, "Flash causal attention forward");
            require_status_clear(
                status,
                "Flash causal attention received non-finite scores"
            );
        }
    }

    void flash_causal_attention_backward(
        const FlashCausalAttentionBackwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto delta_pipeline = require_pipeline(
                @"rt_flash_causal_attention_delta",
                "Flash causal attention delta"
            );
            const auto query_pipeline = require_pipeline(
                @"rt_flash_causal_attention_query_backward",
                "Flash causal attention query backward"
            );
            const auto key_value_pipeline = require_pipeline(
                @"rt_flash_causal_attention_key_value_backward",
                "Flash causal attention key/value backward"
            );
            const auto constants =
                flash_attention_constants(request.dimensions);
            const NSUInteger delta_scratch_bytes =
                flash_scratch_bytes(
                    request.dimensions.head_width,
                    32,
                    136
                );
            const NSUInteger query_scratch_bytes =
                flash_scratch_bytes(
                    request.dimensions.head_width,
                    40,
                    64
                );
            const NSUInteger key_value_scratch_bytes =
                flash_scratch_bytes(
                    request.dimensions.head_width,
                    48,
                    128
                );
            require_flash_resources(
                delta_pipeline,
                delta_scratch_bytes,
                "Flash causal attention delta"
            );
            require_flash_resources(
                query_pipeline,
                query_scratch_bytes,
                "Flash causal attention query backward"
            );
            require_flash_resources(
                key_value_pipeline,
                key_value_scratch_bytes,
                "Flash causal attention key/value backward"
            );

            const auto delta =
                make_float_buffer(request.row_maxima.size());
            auto command = command_encoder(
                "Flash causal attention backward"
            );

            [command.encoder
                setComputePipelineState:delta_pipeline];
            set_flash_backward_inputs(
                command.encoder,
                request,
                delta
            );
            set_flash_attention_constants(
                command.encoder,
                constants,
                7
            );
            dispatch_flash_threadgroups(
                command.encoder,
                delta_pipeline,
                constants.query_tile_count,
                constants.batch_heads,
                delta_scratch_bytes
            );

            [command.encoder
                setComputePipelineState:query_pipeline];
            set_flash_backward_inputs(
                command.encoder,
                request,
                delta
            );
            [command.encoder
                setBuffer:require_buffer(request.query_gradient)
                   offset:0
                  atIndex:7];
            set_flash_attention_constants(
                command.encoder,
                constants,
                8
            );
            dispatch_flash_threadgroups(
                command.encoder,
                query_pipeline,
                constants.query_tile_count,
                constants.batch_heads,
                query_scratch_bytes
            );

            [command.encoder
                setComputePipelineState:key_value_pipeline];
            set_flash_backward_inputs(
                command.encoder,
                request,
                delta
            );
            [command.encoder
                setBuffer:require_buffer(request.key_gradient)
                   offset:0
                  atIndex:7];
            [command.encoder
                setBuffer:require_buffer(request.value_gradient)
                   offset:0
                  atIndex:8];
            set_flash_attention_constants(
                command.encoder,
                constants,
                9
            );
            dispatch_flash_threadgroups(
                command.encoder,
                key_value_pipeline,
                constants.query_tile_count,
                constants.batch_heads,
                key_value_scratch_bytes
            );
            complete(command, "Flash causal attention backward");
        }
    }

    void paged_decode_attention_forward(
        const PagedDecodeAttentionForwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto probability_pipeline =
                require_pipeline(
                    @"rt_paged_decode_attention_probabilities",
                    "paged decode attention probabilities"
                );
            const auto context_pipeline =
                require_pipeline(
                    @"rt_paged_decode_attention_context",
                    "paged decode attention context"
                );
            const auto block_table =
                reusable_paged_block_table(request.block_table);
            const auto probabilities = reusable_paged_probabilities(
                request.dimensions.heads *
                request.dimensions.sequence_length
            );
            const auto status = reusable_paged_status();
            auto command = command_encoder(
                "paged decode attention forward"
            );

            const std::uint64_t heads =
                request.dimensions.heads;
            const std::uint64_t head_width =
                request.dimensions.head_width;
            const std::uint64_t block_size =
                request.dimensions.block_size;
            const std::uint64_t sequence_length =
                request.dimensions.sequence_length;
            const float score_scale =
                1.0F /
                std::sqrt(
                    static_cast<float>(
                        request.dimensions.head_width
                    )
                );

            [command.encoder
                setComputePipelineState:probability_pipeline];
            [command.encoder
                setBuffer:require_buffer(request.queries)
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.key_pages)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:block_table offset:0 atIndex:2];
            [command.encoder
                setBuffer:probabilities offset:0 atIndex:3];
            [command.encoder setBuffer:status offset:0 atIndex:4];
            [command.encoder
                setBytes:&heads length:sizeof(heads) atIndex:5];
            [command.encoder
                setBytes:&head_width
                   length:sizeof(head_width)
                  atIndex:6];
            [command.encoder
                setBytes:&block_size
                   length:sizeof(block_size)
                  atIndex:7];
            [command.encoder
                setBytes:&sequence_length
                   length:sizeof(sequence_length)
                  atIndex:8];
            [command.encoder
                setBytes:&score_scale
                   length:sizeof(score_scale)
                  atIndex:9];
            dispatch(
                command.encoder,
                probability_pipeline,
                request.dimensions.heads
            );

            [command.encoder
                setComputePipelineState:context_pipeline];
            [command.encoder
                setBuffer:probabilities offset:0 atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.value_pages)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:block_table offset:0 atIndex:2];
            [command.encoder
                setBuffer:require_buffer(request.context)
                   offset:0
                  atIndex:3];
            [command.encoder
                setBytes:&heads length:sizeof(heads) atIndex:4];
            [command.encoder
                setBytes:&head_width
                   length:sizeof(head_width)
                  atIndex:5];
            [command.encoder
                setBytes:&block_size
                   length:sizeof(block_size)
                  atIndex:6];
            [command.encoder
                setBytes:&sequence_length
                   length:sizeof(sequence_length)
                  atIndex:7];
            dispatch(
                command.encoder,
                context_pipeline,
                request.dimensions.heads *
                    request.dimensions.head_width
            );

            complete(command, "paged decode attention forward");
            require_status_clear(
                status,
                "paged decode attention received non-finite scores"
            );
        }
    }

    void materialized_causal_attention_context_backward(
        const MaterializedCausalAttentionContextBackwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto score_pipeline =
                require_pipeline(
                    @"rt_materialized_causal_attention_context_score_backward",
                    "causal attention context score backward"
                );
            const auto query_pipeline =
                require_pipeline(
                    @"rt_materialized_causal_attention_query_backward",
                    "causal attention query backward"
                );
            const auto key_pipeline =
                require_pipeline(
                    @"rt_materialized_causal_attention_key_backward",
                    "causal attention key backward"
                );
            const auto value_pipeline =
                require_pipeline(
                    @"rt_materialized_causal_attention_value_backward",
                    "causal attention value backward"
                );
            const auto dimensions =
                attention_constants(request.dimensions);
            const auto score_gradient =
                make_float_buffer(
                    dimensions.row_count * dimensions.time
                );
            auto command = command_encoder(
                "causal attention context backward"
            );
            [command.encoder setComputePipelineState:score_pipeline];
            [command.encoder
                setBuffer:require_buffer(
                              request.probabilities
                          )
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(request.values)
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:require_buffer(
                              request.upstream_context
                          )
                   offset:0
                  atIndex:2];
            [command.encoder
                setBuffer:score_gradient offset:0 atIndex:3];
            set_attention_probability_constants(
                command.encoder,
                dimensions,
                4
            );
            dispatch(
                command.encoder,
                score_pipeline,
                dimensions.row_count
            );

            encode_query_key_gradients(
                command.encoder,
                query_pipeline,
                key_pipeline,
                request.queries,
                request.keys,
                score_gradient,
                request.query_gradient,
                request.key_gradient,
                dimensions
            );

            [command.encoder setComputePipelineState:value_pipeline];
            [command.encoder
                setBuffer:require_buffer(
                              request.probabilities
                          )
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(
                              request.upstream_context
                          )
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:require_buffer(
                              request.value_gradient
                          )
                   offset:0
                  atIndex:2];
            set_attention_vector_constants(
                command.encoder,
                dimensions,
                3
            );
            dispatch(
                command.encoder,
                value_pipeline,
                dimensions.vector_count
            );
            complete(
                command,
                "causal attention context backward"
            );
        }
    }

    void materialized_causal_attention_probabilities_backward(
        const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto score_pipeline =
                require_pipeline(
                    @"rt_materialized_causal_attention_probability_score_backward",
                    "causal attention probability backward"
                );
            const auto query_pipeline =
                require_pipeline(
                    @"rt_materialized_causal_attention_query_backward",
                    "causal attention query backward"
                );
            const auto key_pipeline =
                require_pipeline(
                    @"rt_materialized_causal_attention_key_backward",
                    "causal attention key backward"
                );
            const auto dimensions =
                attention_constants(request.dimensions);
            const auto score_gradient =
                make_float_buffer(
                    dimensions.row_count * dimensions.time
                );
            auto command = command_encoder(
                "causal attention probabilities backward"
            );
            [command.encoder setComputePipelineState:score_pipeline];
            [command.encoder
                setBuffer:require_buffer(
                              request.probabilities
                          )
                   offset:0
                  atIndex:0];
            [command.encoder
                setBuffer:require_buffer(
                              request.upstream_probabilities
                          )
                   offset:0
                  atIndex:1];
            [command.encoder
                setBuffer:score_gradient offset:0 atIndex:2];
            [command.encoder
                setBytes:&dimensions.row_count
                   length:sizeof(dimensions.row_count)
                  atIndex:3];
            [command.encoder
                setBytes:&dimensions.time
                   length:sizeof(dimensions.time)
                  atIndex:4];
            [command.encoder
                setBytes:&dimensions.score_scale
                   length:sizeof(dimensions.score_scale)
                  atIndex:5];
            dispatch(
                command.encoder,
                score_pipeline,
                dimensions.row_count
            );

            encode_query_key_gradients(
                command.encoder,
                query_pipeline,
                key_pipeline,
                request.queries,
                request.keys,
                score_gradient,
                request.query_gradient,
                request.key_gradient,
                dimensions
            );
            complete(
                command,
                "causal attention probabilities backward"
            );
        }
    }

private:
    struct Command {
        id<MTLCommandBuffer> buffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
    };

    struct AttentionConstants {
        std::uint64_t row_count;
        std::uint64_t vector_count;
        std::uint64_t time;
        std::uint64_t head_width;
        float score_scale;
    };

    struct FlashAttentionConstants {
        std::uint64_t batch_heads;
        std::uint64_t time;
        std::uint64_t head_width;
        float score_scale;
        std::size_t query_tile_count;
    };

    [[nodiscard]] id<MTLLibrary> neural_library() {
        if (library_error_ != nullptr) {
            std::rethrow_exception(library_error_);
        }
        if (library_ != nil) {
            return library_;
        }
        try {
            const std::string source_text =
                std::string(kNeuralKernelSource) +
                attention_metal_detail::
                    kMaterializedCausalAttentionKernelSource +
                attention_metal_detail::
                    kFlashCausalAttentionKernelSource +
                attention_metal_detail::
                    kPagedDecodeAttentionKernelSource;
            NSString* source = [NSString
                stringWithUTF8String:source_text.c_str()];
            MTLCompileOptions* options = [MTLCompileOptions new];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
                options.mathFloatingPointFunctions =
                    MTLMathFloatingPointFunctionsPrecise;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
#else
            options.fastMathEnabled = NO;
#endif
            NSError* error = nil;
            library_ =
                [device_ newLibraryWithSource:source
                                      options:options
                                        error:&error];
            if (library_ == nil) {
                throw std::runtime_error(
                    "could not compile Metal neural kernels: " +
                    error_description(error)
                );
            }
        } catch (...) {
            library_error_ = std::current_exception();
            throw;
        }
        return library_;
    }

    [[nodiscard]] id<MTLComputePipelineState> require_pipeline(
        NSString* function_name,
        const char* description
    ) {
        const std::string key(
            function_name.UTF8String == nullptr
                ? description
                : function_name.UTF8String
        );
        const auto error = pipeline_errors_.find(key);
        if (error != pipeline_errors_.end()) {
            std::rethrow_exception(error->second);
        }
        id<MTLComputePipelineState> existing =
            [pipelines_ objectForKey:function_name];
        if (existing != nil) {
            return existing;
        }
        try {
            id<MTLFunction> function =
                [neural_library()
                    newFunctionWithName:function_name];
            if (function == nil) {
                throw std::runtime_error(
                    std::string(
                        "compiled Metal library is missing "
                    ) +
                    description + " kernel"
                );
            }
            NSError* error_value = nil;
            id<MTLComputePipelineState> pipeline =
                [device_
                    newComputePipelineStateWithFunction:function
                                                   error:&error_value];
            if (pipeline == nil) {
                throw std::runtime_error(
                    std::string(
                        "could not create Metal "
                    ) +
                    description + " pipeline: " +
                    error_description(error_value)
                );
            }
            [pipelines_ setObject:pipeline forKey:function_name];
            return pipeline;
        } catch (...) {
            pipeline_errors_.emplace(
                key,
                std::current_exception()
            );
            throw;
        }
    }

    [[nodiscard]] Command command_encoder(
        const char* description
    ) const {
        id<MTLCommandBuffer> buffer = [queue_ commandBuffer];
        if (buffer == nil) {
            throw std::runtime_error(
                std::string("could not create Metal ") +
                description + " command buffer"
            );
        }
        id<MTLComputeCommandEncoder> encoder =
            [buffer computeCommandEncoder];
        if (encoder == nil) {
            throw std::runtime_error(
                std::string("could not create Metal ") +
                description + " command encoder"
            );
        }
        return {buffer, encoder};
    }

    static void complete(
        const Command& command,
        const char* description
    ) {
        [command.encoder endEncoding];
        [command.buffer commit];
        [command.buffer waitUntilCompleted];
        if (
            command.buffer.status !=
            MTLCommandBufferStatusCompleted
        ) {
            throw std::runtime_error(
                std::string("Metal ") + description +
                " command failed: " +
                error_description(command.buffer.error)
            );
        }
    }

    static void dispatch(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> pipeline,
        std::size_t count
    ) {
        if (
            count == 0 ||
            count >
                std::numeric_limits<std::uint32_t>::max()
        ) {
            throw std::overflow_error(
                "Metal neural grid size is outside uint32 range"
            );
        }
        const NSUInteger group_width =
            std::max<NSUInteger>(
                1,
                std::min<NSUInteger>(
                    pipeline.maxTotalThreadsPerThreadgroup,
                    pipeline.threadExecutionWidth
                )
            );
        [encoder
            dispatchThreads:MTLSizeMake(
                                static_cast<NSUInteger>(count),
                                1,
                                1
                            )
            threadsPerThreadgroup:
                MTLSizeMake(group_width, 1, 1)];
    }

    static void dispatch_flash_threadgroups(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> pipeline,
        std::size_t tile_count,
        std::uint64_t batch_heads,
        NSUInteger scratch_bytes
    ) {
        constexpr NSUInteger thread_count = 64;
        if (
            tile_count == 0 ||
            tile_count >
                std::numeric_limits<std::uint32_t>::max() ||
            batch_heads == 0 ||
            batch_heads >
                std::numeric_limits<std::uint32_t>::max()
        ) {
            throw std::overflow_error(
                "Metal Flash attention grid is outside uint32 range"
            );
        }
        if (
            thread_count >
            pipeline.maxTotalThreadsPerThreadgroup
        ) {
            throw std::runtime_error(
                "Metal Flash attention requires a 64-thread group"
            );
        }
        [encoder
            setThreadgroupMemoryLength:scratch_bytes
                               atIndex:0];
        [encoder
            dispatchThreadgroups:
                MTLSizeMake(
                    static_cast<NSUInteger>(tile_count),
                    static_cast<NSUInteger>(batch_heads),
                    1
                )
            threadsPerThreadgroup:
                MTLSizeMake(thread_count, 1, 1)];
    }

    static id<MTLBuffer> require_buffer(
        const TensorStorage& storage
    ) {
        if (storage.backend() != ExecutionBackend::Metal) {
            throw std::invalid_argument(
                "Metal neural tensors must use Metal storage"
            );
        }
        const void* handle = storage.native_handle();
        if (handle == nullptr) {
            throw std::logic_error(
                "Metal tensor is missing its persistent buffer"
            );
        }
        return (__bridge id<MTLBuffer>)
            const_cast<void*>(handle);
    }

    static id<MTLBuffer> require_buffer(
        TensorStorage& storage
    ) {
        if (storage.backend() != ExecutionBackend::Metal) {
            throw std::invalid_argument(
                "Metal neural tensors must use Metal storage"
            );
        }
        void* handle = storage.native_handle();
        if (handle == nullptr) {
            throw std::logic_error(
                "Metal tensor is missing its persistent buffer"
            );
        }
        return (__bridge id<MTLBuffer>)handle;
    }

    [[nodiscard]] id<MTLBuffer> make_status_buffer() const {
        id<MTLBuffer> buffer =
            [device_
                newBufferWithLength:sizeof(std::uint32_t)
                            options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw std::bad_alloc();
        }
        *static_cast<std::uint32_t*>(buffer.contents) = 0;
        return buffer;
    }

    [[nodiscard]] id<MTLBuffer> make_float_buffer(
        std::size_t count
    ) const {
        id<MTLBuffer> buffer =
            [device_
                newBufferWithLength:checked_bytes<float>(count)
                            options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw std::bad_alloc();
        }
        return buffer;
    }

    [[nodiscard]] id<MTLBuffer> reusable_paged_block_table(
        std::span<const std::uint32_t> values
    ) {
        const std::size_t required_bytes =
            checked_bytes<std::uint32_t>(values.size());
        if (paged_block_table_ == nil ||
            paged_block_table_.length < required_bytes) {
            paged_block_table_ =
                [device_
                    newBufferWithLength:required_bytes
                                options:MTLResourceStorageModeShared];
            if (paged_block_table_ == nil) {
                throw std::bad_alloc();
            }
        }
        std::memcpy(
            paged_block_table_.contents,
            values.data(),
            required_bytes
        );
        return paged_block_table_;
    }

    [[nodiscard]] id<MTLBuffer> reusable_paged_probabilities(
        std::size_t count
    ) {
        const std::size_t required_bytes =
            checked_bytes<float>(count);
        if (paged_probabilities_ == nil ||
            paged_probabilities_.length < required_bytes) {
            paged_probabilities_ =
                [device_
                    newBufferWithLength:required_bytes
                                options:MTLResourceStorageModeShared];
            if (paged_probabilities_ == nil) {
                throw std::bad_alloc();
            }
        }
        return paged_probabilities_;
    }

    [[nodiscard]] id<MTLBuffer> reusable_paged_status() {
        if (paged_status_ == nil) {
            paged_status_ = make_status_buffer();
        } else {
            *static_cast<std::uint32_t*>(
                 paged_status_.contents
             ) = 0;
        }
        return paged_status_;
    }

    template <typename Value>
    [[nodiscard]] id<MTLBuffer> make_span_buffer(
        std::span<const Value> values
    ) const {
        id<MTLBuffer> buffer =
            [device_
                newBufferWithBytes:values.data()
                             length:checked_bytes<Value>(
                                        values.size()
                                    )
                            options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw std::bad_alloc();
        }
        return buffer;
    }

    [[nodiscard]] id<MTLBuffer> make_vector_buffer(
        const std::vector<std::uint64_t>& values
    ) const {
        if (values.empty()) {
            const std::uint64_t dummy = 0;
            return make_span_buffer<std::uint64_t>(
                {&dummy, 1}
            );
        }
        return make_span_buffer<std::uint64_t>(values);
    }

    static void require_status_clear(
        id<MTLBuffer> status,
        const char* message
    ) {
        if (
            *static_cast<const std::uint32_t*>(
                status.contents
            ) != 0U
        ) {
            throw std::domain_error(message);
        }
    }

    static void require_cross_entropy_status(
        id<MTLBuffer> status
    ) {
        const auto value =
            *static_cast<const std::uint32_t*>(
                status.contents
            );
        if ((value & 2U) != 0U) {
            throw std::domain_error(
                "cross-entropy received invalid logits"
            );
        }
        if ((value & 4U) != 0U) {
            throw std::overflow_error(
                "cross entropy loss exceeds finite float range"
            );
        }
        if (value != 0U) {
            throw std::domain_error(
                "cross-entropy received invalid logits"
            );
        }
    }

    static void set_two_buffers(
        id<MTLComputeCommandEncoder> encoder,
        const TensorStorage& input,
        TensorStorage& output
    ) {
        [encoder
            setBuffer:require_buffer(input)
               offset:0
              atIndex:0];
        [encoder
            setBuffer:require_buffer(output)
               offset:0
              atIndex:1];
    }

    static void set_three_buffers(
        id<MTLComputeCommandEncoder> encoder,
        const TensorStorage& first,
        const TensorStorage& second,
        TensorStorage& output
    ) {
        [encoder
            setBuffer:require_buffer(first)
               offset:0
              atIndex:0];
        [encoder
            setBuffer:require_buffer(second)
               offset:0
              atIndex:1];
        [encoder
            setBuffer:require_buffer(output)
               offset:0
              atIndex:2];
    }

    static void set_axis_dimensions(
        id<MTLComputeCommandEncoder> encoder,
        const AxisDimensions& dimensions,
        NSUInteger start_index
    ) {
        const std::uint64_t outer = dimensions.outer;
        const std::uint64_t width = dimensions.width;
        const std::uint64_t inner = dimensions.inner;
        [encoder
            setBytes:&outer
               length:sizeof(outer)
              atIndex:start_index];
        [encoder
            setBytes:&width
               length:sizeof(width)
              atIndex:start_index + 1];
        [encoder
            setBytes:&inner
               length:sizeof(inner)
              atIndex:start_index + 2];
    }

    void unary_count_operation(
        NSString* function_name,
        const char* description,
        const TensorStorage& input,
        TensorStorage& output,
        std::size_t count
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(function_name, description);
            auto command = command_encoder(description);
            const std::uint64_t element_count = count;
            [command.encoder setComputePipelineState:pipeline];
            set_two_buffers(command.encoder, input, output);
            [command.encoder
                setBytes:&element_count
                   length:sizeof(element_count)
                  atIndex:2];
            dispatch(command.encoder, pipeline, count);
            complete(command, description);
        }
    }

    void softmax_forward_impl(
        NSString* function_name,
        const char* description,
        const TensorStorage& input,
        TensorStorage& output,
        const AxisDimensions& dimensions
    ) {
        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto pipeline =
                require_pipeline(function_name, description);
            const auto status = make_status_buffer();
            auto command = command_encoder(description);
            [command.encoder setComputePipelineState:pipeline];
            set_two_buffers(command.encoder, input, output);
            [command.encoder setBuffer:status offset:0 atIndex:2];
            set_axis_dimensions(command.encoder, dimensions, 3);
            dispatch(
                command.encoder,
                pipeline,
                dimensions.outer * dimensions.inner
            );
            complete(command, description);
            require_status_clear(
                status,
                "softmax received a non-finite row"
            );
        }
    }

    [[nodiscard]] static AttentionConstants attention_constants(
        const MaterializedCausalAttentionDimensions& dimensions
    ) {
        const auto row_count =
            dimensions.batch *
            dimensions.heads *
            dimensions.time;
        const auto vector_count =
            row_count * dimensions.head_width;
        return {
            static_cast<std::uint64_t>(row_count),
            static_cast<std::uint64_t>(vector_count),
            static_cast<std::uint64_t>(dimensions.time),
            static_cast<std::uint64_t>(dimensions.head_width),
            1.0F /
                std::sqrt(
                    static_cast<float>(
                        dimensions.head_width
                    )
                ),
        };
    }

    [[nodiscard]] static FlashAttentionConstants
    flash_attention_constants(
        const FlashCausalAttentionDimensions& dimensions
    ) {
        if (
            dimensions.batch >
            std::numeric_limits<std::size_t>::max() /
                dimensions.heads
        ) {
            throw std::overflow_error(
                "Metal Flash attention batch-head count overflows"
            );
        }
        const std::size_t batch_heads =
            dimensions.batch * dimensions.heads;
        const std::size_t query_tile_count =
            (dimensions.time - 1) / 8 + 1;
        return {
            static_cast<std::uint64_t>(batch_heads),
            static_cast<std::uint64_t>(dimensions.time),
            static_cast<std::uint64_t>(dimensions.head_width),
            1.0F /
                std::sqrt(
                    static_cast<float>(
                        dimensions.head_width
                    )
                ),
            query_tile_count,
        };
    }

    [[nodiscard]] static NSUInteger flash_scratch_bytes(
        std::size_t head_width,
        std::size_t width_factor,
        std::size_t fixed_float_count
    ) {
        if (
            head_width >
            (
                std::numeric_limits<std::size_t>::max() -
                fixed_float_count
            ) /
                width_factor
        ) {
            throw std::overflow_error(
                "Metal Flash attention threadgroup storage overflows"
            );
        }
        return checked_bytes<float>(
            width_factor * head_width + fixed_float_count
        );
    }

    void require_flash_resources(
        id<MTLComputePipelineState> pipeline,
        NSUInteger scratch_bytes,
        const char* description
    ) const {
        const NSUInteger static_bytes =
            pipeline.staticThreadgroupMemoryLength;
        const NSUInteger maximum_bytes =
            device_.maxThreadgroupMemoryLength;
        if (
            static_bytes > maximum_bytes ||
            scratch_bytes > maximum_bytes - static_bytes
        ) {
            throw std::runtime_error(
                std::string(description) +
                " requires more threadgroup memory than this "
                "Metal device provides; reduce the per-head width "
                "or select materialized attention"
            );
        }
        if (pipeline.maxTotalThreadsPerThreadgroup < 64) {
            throw std::runtime_error(
                std::string(description) +
                " requires a 64-thread Metal threadgroup"
            );
        }
    }

    static void set_flash_attention_constants(
        id<MTLComputeCommandEncoder> encoder,
        const FlashAttentionConstants& constants,
        NSUInteger start_index
    ) {
        [encoder
            setBytes:&constants.batch_heads
               length:sizeof(constants.batch_heads)
              atIndex:start_index];
        [encoder
            setBytes:&constants.time
               length:sizeof(constants.time)
              atIndex:start_index + 1];
        [encoder
            setBytes:&constants.head_width
               length:sizeof(constants.head_width)
              atIndex:start_index + 2];
        [encoder
            setBytes:&constants.score_scale
               length:sizeof(constants.score_scale)
              atIndex:start_index + 3];
    }

    static void set_flash_backward_inputs(
        id<MTLComputeCommandEncoder> encoder,
        const FlashCausalAttentionBackwardRequest& request,
        id<MTLBuffer> delta
    ) {
        [encoder
            setBuffer:require_buffer(request.queries)
               offset:0
              atIndex:0];
        [encoder
            setBuffer:require_buffer(request.keys)
               offset:0
              atIndex:1];
        [encoder
            setBuffer:require_buffer(request.values)
               offset:0
              atIndex:2];
        [encoder
            setBuffer:require_buffer(request.row_maxima)
               offset:0
              atIndex:3];
        [encoder
            setBuffer:require_buffer(request.row_exp_sums)
               offset:0
              atIndex:4];
        [encoder
            setBuffer:require_buffer(request.upstream_context)
               offset:0
              atIndex:5];
        [encoder setBuffer:delta offset:0 atIndex:6];
    }

    static void set_attention_probability_constants(
        id<MTLComputeCommandEncoder> encoder,
        const AttentionConstants& dimensions,
        NSUInteger start_index
    ) {
        [encoder
            setBytes:&dimensions.row_count
               length:sizeof(dimensions.row_count)
              atIndex:start_index];
        [encoder
            setBytes:&dimensions.time
               length:sizeof(dimensions.time)
              atIndex:start_index + 1];
        [encoder
            setBytes:&dimensions.head_width
               length:sizeof(dimensions.head_width)
              atIndex:start_index + 2];
        [encoder
            setBytes:&dimensions.score_scale
               length:sizeof(dimensions.score_scale)
              atIndex:start_index + 3];
    }

    static void set_attention_vector_constants(
        id<MTLComputeCommandEncoder> encoder,
        const AttentionConstants& dimensions,
        NSUInteger start_index
    ) {
        [encoder
            setBytes:&dimensions.vector_count
               length:sizeof(dimensions.vector_count)
              atIndex:start_index];
        [encoder
            setBytes:&dimensions.time
               length:sizeof(dimensions.time)
              atIndex:start_index + 1];
        [encoder
            setBytes:&dimensions.head_width
               length:sizeof(dimensions.head_width)
              atIndex:start_index + 2];
    }

    static void encode_query_key_gradients(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> query_pipeline,
        id<MTLComputePipelineState> key_pipeline,
        const TensorStorage& queries,
        const TensorStorage& keys,
        id<MTLBuffer> score_gradient,
        TensorStorage& query_gradient,
        TensorStorage& key_gradient,
        const AttentionConstants& dimensions
    ) {
        [encoder setComputePipelineState:query_pipeline];
        [encoder
            setBuffer:require_buffer(keys)
               offset:0
              atIndex:0];
        [encoder
            setBuffer:score_gradient offset:0 atIndex:1];
        [encoder
            setBuffer:require_buffer(query_gradient)
               offset:0
              atIndex:2];
        set_attention_vector_constants(
            encoder,
            dimensions,
            3
        );
        dispatch(
            encoder,
            query_pipeline,
            dimensions.vector_count
        );

        [encoder setComputePipelineState:key_pipeline];
        [encoder
            setBuffer:require_buffer(queries)
               offset:0
              atIndex:0];
        [encoder
            setBuffer:score_gradient offset:0 atIndex:1];
        [encoder
            setBuffer:require_buffer(key_gradient)
               offset:0
              atIndex:2];
        set_attention_vector_constants(
            encoder,
            dimensions,
            3
        );
        dispatch(
            encoder,
            key_pipeline,
            dimensions.vector_count
        );
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    NSMutableDictionary<
        NSString*,
        id<MTLComputePipelineState>
    >* pipelines_ = nil;
    std::exception_ptr library_error_;
    std::unordered_map<std::string, std::exception_ptr>
        pipeline_errors_;
    id<MTLBuffer> paged_block_table_ = nil;
    id<MTLBuffer> paged_probabilities_ = nil;
    id<MTLBuffer> paged_status_ = nil;
    std::mutex mutex_;
};

MetalNeuralRuntime& runtime() {
    static MetalNeuralRuntime instance;
    return instance;
}

}  // namespace

void metal_nn_unary_elementwise(
    const UnaryElementwiseRequest& request
) {
    runtime().unary_elementwise(request);
}

void metal_nn_binary_elementwise(
    const BinaryElementwiseRequest& request
) {
    runtime().binary_elementwise(request);
}

void metal_nn_scale(const ScaleRequest& request) {
    runtime().scale(request);
}

void metal_nn_gelu_forward(const GeluForwardRequest& request) {
    runtime().gelu_forward(request);
}

void metal_nn_gelu_backward(const GeluBackwardRequest& request) {
    runtime().gelu_backward(request);
}

void metal_nn_reduce(const ReductionRequest& request) {
    runtime().reduce(request);
}

void metal_nn_copy(const CopyRequest& request) {
    runtime().copy(request);
}

void metal_nn_permute(const PermuteRequest& request) {
    runtime().permute(request);
}

void metal_nn_broadcast(const BroadcastRequest& request) {
    runtime().broadcast(request);
}

void metal_nn_sum_to_shape(const SumToShapeRequest& request) {
    runtime().sum_to_shape(request);
}

void metal_nn_softmax_forward(
    const SoftmaxForwardRequest& request
) {
    runtime().softmax_forward(request);
}

void metal_nn_softmax_backward(
    const SoftmaxBackwardRequest& request
) {
    runtime().softmax_backward(request);
}

void metal_nn_causal_softmax_forward(
    const CausalSoftmaxForwardRequest& request
) {
    runtime().causal_softmax_forward(request);
}

void metal_nn_causal_softmax_backward(
    const CausalSoftmaxBackwardRequest& request
) {
    runtime().causal_softmax_backward(request);
}

void metal_nn_gather_rows(const GatherRowsRequest& request) {
    runtime().gather_rows(request);
}

void metal_nn_scatter_add_rows(
    const ScatterAddRowsRequest& request
) {
    runtime().scatter_add_rows(request);
}

void metal_nn_layer_norm_forward(
    const LayerNormForwardRequest& request
) {
    runtime().layer_norm_forward(request);
}

void metal_nn_layer_norm_backward(
    const LayerNormBackwardRequest& request
) {
    runtime().layer_norm_backward(request);
}

void metal_nn_cross_entropy_forward(
    const CrossEntropyForwardRequest& request
) {
    runtime().cross_entropy_forward(request);
}

void metal_nn_materialized_causal_attention_forward(
    const MaterializedCausalAttentionForwardRequest& request
) {
    runtime().materialized_causal_attention_forward(request);
}

void metal_nn_materialized_causal_attention_context_backward(
    const MaterializedCausalAttentionContextBackwardRequest& request
) {
    runtime().materialized_causal_attention_context_backward(request);
}

void metal_nn_materialized_causal_attention_probabilities_backward(
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
) {
    runtime().materialized_causal_attention_probabilities_backward(request);
}

void metal_nn_flash_causal_attention_forward(
    const FlashCausalAttentionForwardRequest& request
) {
    runtime().flash_causal_attention_forward(request);
}

void metal_nn_flash_causal_attention_backward(
    const FlashCausalAttentionBackwardRequest& request
) {
    runtime().flash_causal_attention_backward(request);
}

void metal_nn_paged_decode_attention_forward(
    const PagedDecodeAttentionForwardRequest& request
) {
    runtime().paged_decode_attention_forward(request);
}

}  // namespace riftco_transformer::backend_detail
