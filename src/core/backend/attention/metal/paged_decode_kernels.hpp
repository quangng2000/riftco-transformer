#pragma once

namespace riftco_transformer::backend_detail::attention_metal_detail {

// Compiled into the shared Metal neural-kernel library. Paged decode remains
// an independent sibling of materialized and Flash full-sequence attention
// while sharing Metal runtime ownership and command-queue synchronization.
inline constexpr char kPagedDecodeAttentionKernelSource[] = R"METAL(
kernel void rt_paged_decode_attention_probabilities(
    device const float* queries [[buffer(0)]],
    device const float* key_pages [[buffer(1)]],
    device const uint* block_table [[buffer(2)]],
    device float* probabilities [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant ulong& heads [[buffer(5)]],
    constant ulong& head_width [[buffer(6)]],
    constant ulong& block_size [[buffer(7)]],
    constant ulong& sequence_length [[buffer(8)]],
    constant float& score_scale [[buffer(9)]],
    uint head [[thread_position_in_grid]]
) {
    if (head >= heads) {
        return;
    }
    const ulong query_base = ulong(head) * head_width;
    const ulong probability_base =
        ulong(head) * sequence_length;
    float maximum = -INFINITY;
    bool valid = true;
    for (ulong position = 0; position < sequence_length; ++position) {
        const ulong physical_block =
            ulong(block_table[position / block_size]);
        const ulong block_offset = position % block_size;
        const ulong key_base =
            (
                (
                    physical_block * heads + ulong(head)
                ) * block_size + block_offset
            ) * head_width;
        float score = 0.0f;
        for (ulong channel = 0; channel < head_width; ++channel) {
            score +=
                queries[query_base + channel] *
                key_pages[key_base + channel];
        }
        score *= score_scale;
        probabilities[probability_base + position] = score;
        if (isnan(score) || score == INFINITY) {
            valid = false;
        }
        maximum = max(maximum, score);
    }
    if (!valid || maximum == -INFINITY) {
        rt_flag(status, RT_STATUS_INVALID_ROW);
        for (ulong position = 0;
             position < sequence_length;
             ++position) {
            probabilities[probability_base + position] = 0.0f;
        }
        return;
    }

    float denominator_sum = 0.0f;
    float denominator_correction = 0.0f;
    for (ulong position = 0; position < sequence_length; ++position) {
        rt_compensated_add(
            exp(
                probabilities[probability_base + position] -
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
    for (ulong position = 0; position < sequence_length; ++position) {
        probabilities[probability_base + position] =
            exp(
                probabilities[probability_base + position] -
                maximum
            ) /
            denominator;
    }
}

kernel void rt_paged_decode_attention_context(
    device const float* probabilities [[buffer(0)]],
    device const float* value_pages [[buffer(1)]],
    device const uint* block_table [[buffer(2)]],
    device float* context [[buffer(3)]],
    constant ulong& heads [[buffer(4)]],
    constant ulong& head_width [[buffer(5)]],
    constant ulong& block_size [[buffer(6)]],
    constant ulong& sequence_length [[buffer(7)]],
    uint context_index [[thread_position_in_grid]]
) {
    const ulong context_count = heads * head_width;
    if (context_index >= context_count) {
        return;
    }
    const ulong head = ulong(context_index) / head_width;
    const ulong channel = ulong(context_index) % head_width;
    const ulong probability_base = head * sequence_length;
    float total = 0.0f;
    for (ulong position = 0; position < sequence_length; ++position) {
        const ulong physical_block =
            ulong(block_table[position / block_size]);
        const ulong block_offset = position % block_size;
        const ulong value_index =
            (
                (
                    physical_block * heads + head
                ) * block_size + block_offset
            ) * head_width + channel;
        total +=
            probabilities[probability_base + position] *
            value_pages[value_index];
    }
    context[context_index] = total;
}
)METAL";

}  // namespace riftco_transformer::backend_detail::attention_metal_detail
