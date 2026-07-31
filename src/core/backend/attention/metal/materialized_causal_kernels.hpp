#pragma once

namespace transformer_lab::backend_detail::attention_metal_detail {

// Baseline full-sequence attention with an explicit [B,H,T,T] probability
// tensor. FlashAttention is a separate source fragment and contract.
inline constexpr char kMaterializedCausalAttentionKernelSource[] = R"METAL(
kernel void tl_materialized_causal_attention_probabilities(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* probabilities [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ulong& row_count [[buffer(4)]],
    constant ulong& time [[buffer(5)]],
    constant ulong& head_width [[buffer(6)]],
    constant float& score_scale [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= row_count) {
        return;
    }
    const ulong query = row % time;
    const ulong head_base =
        (ulong(row) / time) * time * head_width;
    const ulong query_base =
        head_base + query * head_width;
    const ulong probability_base = ulong(row) * time;
    float maximum = -INFINITY;
    bool valid = true;
    for (ulong key = 0; key <= query; ++key) {
        const ulong key_base =
            head_base + key * head_width;
        float score = 0.0f;
        for (
            ulong column = 0;
            column < head_width;
            ++column
        ) {
            score +=
                queries[query_base + column] *
                keys[key_base + column];
        }
        score *= score_scale;
        probabilities[probability_base + key] = score;
        if (isnan(score) || score == INFINITY) {
            valid = false;
        }
        maximum = max(maximum, score);
    }
    for (ulong key = query + 1; key < time; ++key) {
        probabilities[probability_base + key] = 0.0f;
    }
    if (!valid || maximum == -INFINITY) {
        tl_flag(status, TL_STATUS_INVALID_ROW);
        for (ulong key = 0; key <= query; ++key) {
            probabilities[probability_base + key] = 0.0f;
        }
        return;
    }
    float denominator_sum = 0.0f;
    float denominator_correction = 0.0f;
    for (ulong key = 0; key <= query; ++key) {
        tl_compensated_add(
            exp(
                probabilities[probability_base + key] -
                maximum
            ),
            denominator_sum,
            denominator_correction
        );
    }
    const float denominator =
        denominator_sum + denominator_correction;
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        tl_flag(status, TL_STATUS_INVALID_ROW);
        return;
    }
    for (ulong key = 0; key <= query; ++key) {
        probabilities[probability_base + key] =
            exp(
                probabilities[probability_base + key] -
                maximum
            ) /
            denominator;
    }
}

kernel void tl_materialized_causal_attention_context(
    device const float* probabilities [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* context [[buffer(2)]],
    constant ulong& context_count [[buffer(3)]],
    constant ulong& time [[buffer(4)]],
    constant ulong& head_width [[buffer(5)]],
    uint context_index [[thread_position_in_grid]]
) {
    if (context_index >= context_count) {
        return;
    }
    const ulong row = ulong(context_index) / head_width;
    const ulong column = ulong(context_index) % head_width;
    const ulong query = row % time;
    const ulong head_base =
        (row / time) * time * head_width;
    float total = 0.0f;
    for (ulong key = 0; key <= query; ++key) {
        total +=
            probabilities[row * time + key] *
            values[head_base + key * head_width + column];
    }
    context[context_index] = total;
}

kernel void tl_materialized_causal_attention_context_score_backward(
    device const float* probabilities [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const float* upstream_context [[buffer(2)]],
    device float* score_gradient [[buffer(3)]],
    constant ulong& row_count [[buffer(4)]],
    constant ulong& time [[buffer(5)]],
    constant ulong& head_width [[buffer(6)]],
    constant float& score_scale [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= row_count) {
        return;
    }
    const ulong query = row % time;
    const ulong head_base =
        (ulong(row) / time) * time * head_width;
    const ulong upstream_base = ulong(row) * head_width;
    const ulong probability_base = ulong(row) * time;
    float probability_dot_sum = 0.0f;
    float probability_dot_correction = 0.0f;
    for (ulong key = 0; key <= query; ++key) {
        float probability_gradient = 0.0f;
        const ulong value_base =
            head_base + key * head_width;
        for (
            ulong column = 0;
            column < head_width;
            ++column
        ) {
            probability_gradient +=
                upstream_context[upstream_base + column] *
                values[value_base + column];
        }
        score_gradient[probability_base + key] =
            probability_gradient;
        tl_compensated_add(
            probabilities[probability_base + key] *
                probability_gradient,
            probability_dot_sum,
            probability_dot_correction
        );
    }
    const float probability_dot =
        probability_dot_sum + probability_dot_correction;
    for (ulong key = 0; key <= query; ++key) {
        score_gradient[probability_base + key] =
            score_scale *
            probabilities[probability_base + key] *
            (
                score_gradient[probability_base + key] -
                probability_dot
            );
    }
    for (ulong key = query + 1; key < time; ++key) {
        score_gradient[probability_base + key] = 0.0f;
    }
}

kernel void tl_materialized_causal_attention_probability_score_backward(
    device const float* probabilities [[buffer(0)]],
    device const float* upstream_probabilities [[buffer(1)]],
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
    const ulong base = ulong(row) * time;
    float dot_sum = 0.0f;
    float dot_correction = 0.0f;
    for (ulong key = 0; key <= query; ++key) {
        tl_compensated_add(
            probabilities[base + key] *
                upstream_probabilities[base + key],
            dot_sum,
            dot_correction
        );
    }
    const float dot = dot_sum + dot_correction;
    for (ulong key = 0; key <= query; ++key) {
        score_gradient[base + key] =
            score_scale * probabilities[base + key] *
            (upstream_probabilities[base + key] - dot);
    }
    for (ulong key = query + 1; key < time; ++key) {
        score_gradient[base + key] = 0.0f;
    }
}

kernel void tl_materialized_causal_attention_query_backward(
    device const float* keys [[buffer(0)]],
    device const float* score_gradient [[buffer(1)]],
    device float* query_gradient [[buffer(2)]],
    constant ulong& vector_count [[buffer(3)]],
    constant ulong& time [[buffer(4)]],
    constant ulong& head_width [[buffer(5)]],
    uint output_index [[thread_position_in_grid]]
) {
    if (output_index >= vector_count) {
        return;
    }
    const ulong row = ulong(output_index) / head_width;
    const ulong column = ulong(output_index) % head_width;
    const ulong query = row % time;
    const ulong head_base =
        (row / time) * time * head_width;
    float total = 0.0f;
    for (ulong key = 0; key <= query; ++key) {
        total +=
            score_gradient[row * time + key] *
            keys[head_base + key * head_width + column];
    }
    query_gradient[output_index] = total;
}

kernel void tl_materialized_causal_attention_key_backward(
    device const float* queries [[buffer(0)]],
    device const float* score_gradient [[buffer(1)]],
    device float* key_gradient [[buffer(2)]],
    constant ulong& vector_count [[buffer(3)]],
    constant ulong& time [[buffer(4)]],
    constant ulong& head_width [[buffer(5)]],
    uint output_index [[thread_position_in_grid]]
) {
    if (output_index >= vector_count) {
        return;
    }
    const ulong vector = ulong(output_index) / head_width;
    const ulong column = ulong(output_index) % head_width;
    const ulong key = vector % time;
    const ulong head = vector / time;
    const ulong head_base = head * time * head_width;
    float total = 0.0f;
    for (ulong query = key; query < time; ++query) {
        const ulong row = head * time + query;
        total +=
            score_gradient[row * time + key] *
            queries[head_base + query * head_width + column];
    }
    key_gradient[output_index] = total;
}

kernel void tl_materialized_causal_attention_value_backward(
    device const float* probabilities [[buffer(0)]],
    device const float* upstream_context [[buffer(1)]],
    device float* value_gradient [[buffer(2)]],
    constant ulong& vector_count [[buffer(3)]],
    constant ulong& time [[buffer(4)]],
    constant ulong& head_width [[buffer(5)]],
    uint output_index [[thread_position_in_grid]]
) {
    if (output_index >= vector_count) {
        return;
    }
    const ulong vector = ulong(output_index) / head_width;
    const ulong column = ulong(output_index) % head_width;
    const ulong key = vector % time;
    const ulong head = vector / time;
    float total = 0.0f;
    for (ulong query = key; query < time; ++query) {
        const ulong row = head * time + query;
        total +=
            probabilities[row * time + key] *
            upstream_context[row * head_width + column];
    }
    value_gradient[output_index] = total;
}
)METAL";

}  // namespace transformer_lab::backend_detail::attention_metal_detail
