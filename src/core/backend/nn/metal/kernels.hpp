#pragma once

namespace riftco_transformer::backend_detail::nn_metal_detail {

inline constexpr char kNeuralKernelSource[] = R"METAL(
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

}  // namespace riftco_transformer::backend_detail::nn_metal_detail
