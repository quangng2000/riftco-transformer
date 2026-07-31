#pragma once

namespace transformer_lab::backend_detail::attention_metal_detail {

// Exact, memory-linear causal attention. Every 64-thread group owns eight
// query (or key) rows and reuses 8xD Q/K/V tiles from threadgroup memory.
// Probabilities are reconstructed from saved row maxima and exponential sums;
// no [B,H,T,T] buffer is allocated in either direction.
inline constexpr char kFlashCausalAttentionKernelSource[] = R"METAL(
constant uint TL_FLASH_TILE = 8u;
constant uint TL_FLASH_THREADS = 64u;

inline ulong tl_flash_offset(
    ulong batch_head,
    ulong position,
    ulong channel,
    ulong time,
    ulong head_width
) {
    return (
        batch_head * time + position
    ) * head_width + channel;
}

inline float tl_flash_probability(
    float score,
    float row_maximum,
    float row_exp_sum
) {
    return score == -INFINITY
        ? 0.0f
        : exp(score - row_maximum) / row_exp_sum;
}

kernel void tl_flash_causal_attention_forward(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* row_maxima [[buffer(3)]],
    device float* row_exp_sums [[buffer(4)]],
    device float* context [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    constant ulong& batch_heads [[buffer(7)]],
    constant ulong& time [[buffer(8)]],
    constant ulong& head_width [[buffer(9)]],
    constant float& score_scale [[buffer(10)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const ulong batch_head = ulong(group.y);
    const ulong query_start = ulong(group.x) * TL_FLASH_TILE;
    const ulong tile_elements = TL_FLASH_TILE * head_width;

    threadgroup float* query_tile = scratch;
    threadgroup float* key_tile = query_tile + tile_elements;
    threadgroup float* value_tile = key_tile + tile_elements;
    threadgroup float* output_tile = value_tile + tile_elements;
    threadgroup float* scores = output_tile + tile_elements;
    threadgroup float* maxima = scores + TL_FLASH_TILE * TL_FLASH_TILE;
    threadgroup float* sums = maxima + TL_FLASH_TILE;
    threadgroup float* old_scales = sums + TL_FLASH_TILE;
    threadgroup float* valid_rows = old_scales + TL_FLASH_TILE;

    for (ulong element = thread_index;
         element < tile_elements;
         element += TL_FLASH_THREADS) {
        const ulong local_query = element / head_width;
        const ulong channel = element % head_width;
        const ulong query = query_start + local_query;
        query_tile[element] =
            batch_head < batch_heads && query < time
                ? queries[tl_flash_offset(
                      batch_head,
                      query,
                      channel,
                      time,
                      head_width
                  )]
                : 0.0f;
        output_tile[element] = 0.0f;
    }
    if (thread_index < TL_FLASH_TILE) {
        maxima[thread_index] = -INFINITY;
        sums[thread_index] = 0.0f;
        old_scales[thread_index] = 0.0f;
        valid_rows[thread_index] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong query_limit = min(time, query_start + TL_FLASH_TILE);
    for (ulong key_start = 0;
         key_start < query_limit;
         key_start += TL_FLASH_TILE) {
        for (ulong element = thread_index;
             element < tile_elements;
             element += TL_FLASH_THREADS) {
            const ulong local_key = element / head_width;
            const ulong channel = element % head_width;
            const ulong key = key_start + local_key;
            const bool in_bounds =
                batch_head < batch_heads && key < time;
            key_tile[element] = in_bounds
                ? keys[tl_flash_offset(
                      batch_head,
                      key,
                      channel,
                      time,
                      head_width
                  )]
                : 0.0f;
            value_tile[element] = in_bounds
                ? values[tl_flash_offset(
                      batch_head,
                      key,
                      channel,
                      time,
                      head_width
                  )]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint local_query = thread_index / TL_FLASH_TILE;
        const uint local_key = thread_index % TL_FLASH_TILE;
        const ulong query = query_start + ulong(local_query);
        const ulong key = key_start + ulong(local_key);
        float score = -INFINITY;
        if (
            batch_head < batch_heads &&
            query < time &&
            key < time &&
            key <= query
        ) {
            score = 0.0f;
            for (ulong channel = 0;
                 channel < head_width;
                 ++channel) {
                score +=
                    query_tile[
                        ulong(local_query) * head_width + channel
                    ] *
                    key_tile[
                        ulong(local_key) * head_width + channel
                    ];
            }
            score *= score_scale;
        }
        scores[thread_index] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index < TL_FLASH_TILE) {
            const uint row = thread_index;
            const ulong query_position = query_start + ulong(row);
            if (
                batch_head < batch_heads &&
                query_position < time
            ) {
                float tile_maximum = -INFINITY;
                bool row_valid = valid_rows[row] != 0.0f;
                for (uint column = 0;
                     column < TL_FLASH_TILE;
                     ++column) {
                    const ulong key_position =
                        key_start + ulong(column);
                    if (
                        key_position >= time ||
                        key_position > query_position
                    ) {
                        continue;
                    }
                    const float candidate =
                        scores[row * TL_FLASH_TILE + column];
                    if (isnan(candidate) || candidate == INFINITY) {
                        row_valid = false;
                    } else {
                        tile_maximum = max(
                            tile_maximum,
                            candidate
                        );
                    }
                }

                const float old_maximum = maxima[row];
                const float old_sum = sums[row];
                float new_maximum = old_maximum;
                float old_scale = 1.0f;
                if (tile_maximum != -INFINITY) {
                    new_maximum =
                        old_sum == 0.0f
                            ? tile_maximum
                            : max(old_maximum, tile_maximum);
                    old_scale =
                        old_sum == 0.0f
                            ? 0.0f
                            : exp(old_maximum - new_maximum);
                }
                float tile_sum = 0.0f;
                if (new_maximum != -INFINITY) {
                    for (uint column = 0;
                         column < TL_FLASH_TILE;
                         ++column) {
                        const ulong key_position =
                            key_start + ulong(column);
                        const float candidate =
                            scores[row * TL_FLASH_TILE + column];
                        if (
                            key_position < time &&
                            key_position <= query_position &&
                            candidate != -INFINITY &&
                            !isnan(candidate) &&
                            candidate != INFINITY
                        ) {
                            tile_sum +=
                                exp(candidate - new_maximum);
                        }
                    }
                }
                maxima[row] = new_maximum;
                sums[row] = old_scale * old_sum + tile_sum;
                old_scales[row] = old_scale;
                valid_rows[row] = row_valid ? 1.0f : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ulong element = thread_index;
             element < tile_elements;
             element += TL_FLASH_THREADS) {
            const ulong local_query_index =
                element / head_width;
            const ulong channel = element % head_width;
            const ulong query =
                query_start + local_query_index;
            if (batch_head >= batch_heads || query >= time) {
                continue;
            }
            float total =
                old_scales[local_query_index] *
                output_tile[element];
            const float maximum = maxima[local_query_index];
            for (uint local_key_index = 0;
                 local_key_index < TL_FLASH_TILE;
                 ++local_key_index) {
                const ulong key =
                    key_start + ulong(local_key_index);
                const float candidate = scores[
                    local_query_index * TL_FLASH_TILE +
                    ulong(local_key_index)
                ];
                if (
                    key < time &&
                    key <= query &&
                    candidate != -INFINITY &&
                    !isnan(candidate) &&
                    candidate != INFINITY
                ) {
                    total +=
                        exp(candidate - maximum) *
                        value_tile[
                            ulong(local_key_index) *
                                head_width +
                            channel
                        ];
                }
            }
            output_tile[element] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index < TL_FLASH_TILE) {
        const ulong query = query_start + ulong(thread_index);
        if (batch_head < batch_heads && query < time) {
            const bool valid =
                valid_rows[thread_index] != 0.0f &&
                maxima[thread_index] != -INFINITY &&
                sums[thread_index] > 0.0f &&
                isfinite(sums[thread_index]);
            valid_rows[thread_index] = valid ? 1.0f : 0.0f;
            const ulong row = batch_head * time + query;
            row_maxima[row] =
                valid ? maxima[thread_index] : 0.0f;
            row_exp_sums[row] =
                valid ? sums[thread_index] : 0.0f;
            if (!valid) {
                tl_flag(status, TL_STATUS_INVALID_ROW);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ulong element = thread_index;
         element < tile_elements;
         element += TL_FLASH_THREADS) {
        const ulong local_query = element / head_width;
        const ulong channel = element % head_width;
        const ulong query = query_start + local_query;
        if (batch_head < batch_heads && query < time) {
            context[tl_flash_offset(
                batch_head,
                query,
                channel,
                time,
                head_width
            )] =
                valid_rows[local_query] != 0.0f
                    ? output_tile[element] / sums[local_query]
                    : 0.0f;
        }
    }
}

kernel void tl_flash_causal_attention_delta(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* row_maxima [[buffer(3)]],
    device const float* row_exp_sums [[buffer(4)]],
    device const float* upstream [[buffer(5)]],
    device float* delta [[buffer(6)]],
    constant ulong& batch_heads [[buffer(7)]],
    constant ulong& time [[buffer(8)]],
    constant ulong& head_width [[buffer(9)]],
    constant float& score_scale [[buffer(10)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const ulong batch_head = ulong(group.y);
    const ulong query_start = ulong(group.x) * TL_FLASH_TILE;
    const ulong tile_elements = TL_FLASH_TILE * head_width;
    threadgroup float* query_tile = scratch;
    threadgroup float* key_tile = query_tile + tile_elements;
    threadgroup float* value_tile = key_tile + tile_elements;
    threadgroup float* upstream_tile = value_tile + tile_elements;
    threadgroup float* score_tile = upstream_tile + tile_elements;
    threadgroup float* dot_tile =
        score_tile + TL_FLASH_TILE * TL_FLASH_TILE;
    threadgroup float* delta_tile =
        dot_tile + TL_FLASH_TILE * TL_FLASH_TILE;

    for (ulong element = thread_index;
         element < tile_elements;
         element += TL_FLASH_THREADS) {
        const ulong local_query = element / head_width;
        const ulong channel = element % head_width;
        const ulong query = query_start + local_query;
        const bool valid =
            batch_head < batch_heads && query < time;
        const ulong index = valid
            ? tl_flash_offset(
                  batch_head, query, channel, time, head_width
              )
            : 0;
        query_tile[element] = valid ? queries[index] : 0.0f;
        upstream_tile[element] = valid ? upstream[index] : 0.0f;
    }
    if (thread_index < TL_FLASH_TILE) {
        delta_tile[thread_index] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong query_limit = min(time, query_start + TL_FLASH_TILE);
    for (ulong key_start = 0;
         key_start < query_limit;
         key_start += TL_FLASH_TILE) {
        for (ulong element = thread_index;
             element < tile_elements;
             element += TL_FLASH_THREADS) {
            const ulong local_key = element / head_width;
            const ulong channel = element % head_width;
            const ulong key = key_start + local_key;
            const bool valid =
                batch_head < batch_heads && key < time;
            const ulong index = valid
                ? tl_flash_offset(
                      batch_head, key, channel, time, head_width
                  )
                : 0;
            key_tile[element] = valid ? keys[index] : 0.0f;
            value_tile[element] = valid ? values[index] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint local_query = thread_index / TL_FLASH_TILE;
        const uint local_key = thread_index % TL_FLASH_TILE;
        const ulong query = query_start + ulong(local_query);
        const ulong key = key_start + ulong(local_key);
        float score = -INFINITY;
        float upstream_dot_value = 0.0f;
        if (
            batch_head < batch_heads &&
            query < time &&
            key < time &&
            key <= query
        ) {
            score = 0.0f;
            for (ulong channel = 0;
                 channel < head_width;
                 ++channel) {
                score +=
                    query_tile[
                        ulong(local_query) * head_width + channel
                    ] *
                    key_tile[
                        ulong(local_key) * head_width + channel
                    ];
                upstream_dot_value +=
                    upstream_tile[
                        ulong(local_query) * head_width + channel
                    ] *
                    value_tile[
                        ulong(local_key) * head_width + channel
                    ];
            }
            score *= score_scale;
        }
        score_tile[thread_index] = score;
        dot_tile[thread_index] = upstream_dot_value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index < TL_FLASH_TILE) {
            const ulong query_position =
                query_start + ulong(thread_index);
            if (
                batch_head < batch_heads &&
                query_position < time
            ) {
                const ulong row =
                    batch_head * time + query_position;
                float total = delta_tile[thread_index];
                for (uint local_key_index = 0;
                     local_key_index < TL_FLASH_TILE;
                     ++local_key_index) {
                    const ulong key_position =
                        key_start + ulong(local_key_index);
                    if (
                        key_position < time &&
                        key_position <= query_position
                    ) {
                        const uint tile_index =
                            thread_index * TL_FLASH_TILE +
                            local_key_index;
                        total +=
                            tl_flash_probability(
                                score_tile[tile_index],
                                row_maxima[row],
                                row_exp_sums[row]
                            ) *
                            dot_tile[tile_index];
                    }
                }
                delta_tile[thread_index] = total;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index < TL_FLASH_TILE) {
        const ulong query = query_start + ulong(thread_index);
        if (batch_head < batch_heads && query < time) {
            delta[batch_head * time + query] =
                delta_tile[thread_index];
        }
    }
}

kernel void tl_flash_causal_attention_query_backward(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* row_maxima [[buffer(3)]],
    device const float* row_exp_sums [[buffer(4)]],
    device const float* upstream [[buffer(5)]],
    device const float* delta [[buffer(6)]],
    device float* query_gradient [[buffer(7)]],
    constant ulong& batch_heads [[buffer(8)]],
    constant ulong& time [[buffer(9)]],
    constant ulong& head_width [[buffer(10)]],
    constant float& score_scale [[buffer(11)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const ulong batch_head = ulong(group.y);
    const ulong query_start = ulong(group.x) * TL_FLASH_TILE;
    const ulong tile_elements = TL_FLASH_TILE * head_width;
    threadgroup float* query_tile = scratch;
    threadgroup float* key_tile = query_tile + tile_elements;
    threadgroup float* value_tile = key_tile + tile_elements;
    threadgroup float* upstream_tile = value_tile + tile_elements;
    threadgroup float* gradient_tile = upstream_tile + tile_elements;
    threadgroup float* score_gradient =
        gradient_tile + tile_elements;

    for (ulong element = thread_index;
         element < tile_elements;
         element += TL_FLASH_THREADS) {
        const ulong local_query = element / head_width;
        const ulong channel = element % head_width;
        const ulong query = query_start + local_query;
        const bool valid =
            batch_head < batch_heads && query < time;
        const ulong index = valid
            ? tl_flash_offset(
                  batch_head, query, channel, time, head_width
              )
            : 0;
        query_tile[element] = valid ? queries[index] : 0.0f;
        upstream_tile[element] = valid ? upstream[index] : 0.0f;
        gradient_tile[element] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong query_limit = min(time, query_start + TL_FLASH_TILE);
    for (ulong key_start = 0;
         key_start < query_limit;
         key_start += TL_FLASH_TILE) {
        for (ulong element = thread_index;
             element < tile_elements;
             element += TL_FLASH_THREADS) {
            const ulong local_key = element / head_width;
            const ulong channel = element % head_width;
            const ulong key = key_start + local_key;
            const bool valid =
                batch_head < batch_heads && key < time;
            const ulong index = valid
                ? tl_flash_offset(
                      batch_head, key, channel, time, head_width
                  )
                : 0;
            key_tile[element] = valid ? keys[index] : 0.0f;
            value_tile[element] = valid ? values[index] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint local_query = thread_index / TL_FLASH_TILE;
        const uint local_key = thread_index % TL_FLASH_TILE;
        const ulong query = query_start + ulong(local_query);
        const ulong key = key_start + ulong(local_key);
        float derivative = 0.0f;
        if (
            batch_head < batch_heads &&
            query < time &&
            key < time &&
            key <= query
        ) {
            float score = 0.0f;
            float upstream_dot_value = 0.0f;
            for (ulong channel = 0;
                 channel < head_width;
                 ++channel) {
                score +=
                    query_tile[
                        ulong(local_query) * head_width + channel
                    ] *
                    key_tile[
                        ulong(local_key) * head_width + channel
                    ];
                upstream_dot_value +=
                    upstream_tile[
                        ulong(local_query) * head_width + channel
                    ] *
                    value_tile[
                        ulong(local_key) * head_width + channel
                    ];
            }
            score *= score_scale;
            const ulong row = batch_head * time + query;
            const float probability = tl_flash_probability(
                score,
                row_maxima[row],
                row_exp_sums[row]
            );
            derivative =
                score_scale * probability *
                (upstream_dot_value - delta[row]);
        }
        score_gradient[thread_index] = derivative;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ulong element = thread_index;
             element < tile_elements;
             element += TL_FLASH_THREADS) {
            const ulong local_query_index =
                element / head_width;
            const ulong channel = element % head_width;
            const ulong query =
                query_start + local_query_index;
            if (batch_head >= batch_heads || query >= time) {
                continue;
            }
            float total = gradient_tile[element];
            for (uint local_key_index = 0;
                 local_key_index < TL_FLASH_TILE;
                 ++local_key_index) {
                const ulong key =
                    key_start + ulong(local_key_index);
                if (key < time && key <= query) {
                    total +=
                        score_gradient[
                            local_query_index *
                                TL_FLASH_TILE +
                            ulong(local_key_index)
                        ] *
                        key_tile[
                            ulong(local_key_index) *
                                head_width +
                            channel
                        ];
                }
            }
            gradient_tile[element] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (ulong element = thread_index;
         element < tile_elements;
         element += TL_FLASH_THREADS) {
        const ulong local_query = element / head_width;
        const ulong channel = element % head_width;
        const ulong query = query_start + local_query;
        if (batch_head < batch_heads && query < time) {
            query_gradient[tl_flash_offset(
                batch_head,
                query,
                channel,
                time,
                head_width
            )] = gradient_tile[element];
        }
    }
}

kernel void tl_flash_causal_attention_key_value_backward(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* row_maxima [[buffer(3)]],
    device const float* row_exp_sums [[buffer(4)]],
    device const float* upstream [[buffer(5)]],
    device const float* delta [[buffer(6)]],
    device float* key_gradient [[buffer(7)]],
    device float* value_gradient [[buffer(8)]],
    constant ulong& batch_heads [[buffer(9)]],
    constant ulong& time [[buffer(10)]],
    constant ulong& head_width [[buffer(11)]],
    constant float& score_scale [[buffer(12)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    const ulong batch_head = ulong(group.y);
    const ulong key_start = ulong(group.x) * TL_FLASH_TILE;
    const ulong tile_elements = TL_FLASH_TILE * head_width;
    threadgroup float* key_tile = scratch;
    threadgroup float* value_tile = key_tile + tile_elements;
    threadgroup float* key_gradient_tile =
        value_tile + tile_elements;
    threadgroup float* value_gradient_tile =
        key_gradient_tile + tile_elements;
    threadgroup float* query_tile =
        value_gradient_tile + tile_elements;
    threadgroup float* upstream_tile =
        query_tile + tile_elements;
    threadgroup float* probability_tile =
        upstream_tile + tile_elements;
    threadgroup float* score_gradient =
        probability_tile + TL_FLASH_TILE * TL_FLASH_TILE;

    for (ulong element = thread_index;
         element < tile_elements;
         element += TL_FLASH_THREADS) {
        const ulong local_key = element / head_width;
        const ulong channel = element % head_width;
        const ulong key = key_start + local_key;
        const bool valid =
            batch_head < batch_heads && key < time;
        const ulong index = valid
            ? tl_flash_offset(
                  batch_head, key, channel, time, head_width
              )
            : 0;
        key_tile[element] = valid ? keys[index] : 0.0f;
        value_tile[element] = valid ? values[index] : 0.0f;
        key_gradient_tile[element] = 0.0f;
        value_gradient_tile[element] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (
        ulong query_start =
            (key_start / TL_FLASH_TILE) * TL_FLASH_TILE;
        query_start < time;
        query_start += TL_FLASH_TILE
    ) {
        for (ulong element = thread_index;
             element < tile_elements;
             element += TL_FLASH_THREADS) {
            const ulong local_query = element / head_width;
            const ulong channel = element % head_width;
            const ulong query = query_start + local_query;
            const bool valid =
                batch_head < batch_heads && query < time;
            const ulong index = valid
                ? tl_flash_offset(
                      batch_head, query, channel, time, head_width
                  )
                : 0;
            query_tile[element] = valid ? queries[index] : 0.0f;
            upstream_tile[element] = valid ? upstream[index] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint local_key = thread_index / TL_FLASH_TILE;
        const uint local_query = thread_index % TL_FLASH_TILE;
        const ulong key = key_start + ulong(local_key);
        const ulong query = query_start + ulong(local_query);
        float probability = 0.0f;
        float derivative = 0.0f;
        if (
            batch_head < batch_heads &&
            key < time &&
            query < time &&
            key <= query
        ) {
            float score = 0.0f;
            float upstream_dot_value = 0.0f;
            for (ulong channel = 0;
                 channel < head_width;
                 ++channel) {
                score +=
                    query_tile[
                        ulong(local_query) * head_width + channel
                    ] *
                    key_tile[
                        ulong(local_key) * head_width + channel
                    ];
                upstream_dot_value +=
                    upstream_tile[
                        ulong(local_query) * head_width + channel
                    ] *
                    value_tile[
                        ulong(local_key) * head_width + channel
                    ];
            }
            score *= score_scale;
            const ulong row = batch_head * time + query;
            probability = tl_flash_probability(
                score,
                row_maxima[row],
                row_exp_sums[row]
            );
            derivative =
                score_scale * probability *
                (upstream_dot_value - delta[row]);
        }
        probability_tile[thread_index] = probability;
        score_gradient[thread_index] = derivative;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ulong element = thread_index;
             element < tile_elements;
             element += TL_FLASH_THREADS) {
            const ulong local_key_index =
                element / head_width;
            const ulong channel = element % head_width;
            const ulong key =
                key_start + local_key_index;
            if (batch_head >= batch_heads || key >= time) {
                continue;
            }
            float key_total = key_gradient_tile[element];
            float value_total = value_gradient_tile[element];
            for (uint local_query_index = 0;
                 local_query_index < TL_FLASH_TILE;
                 ++local_query_index) {
                const ulong query =
                    query_start + ulong(local_query_index);
                if (query < time && key <= query) {
                    const ulong tile_index =
                        local_key_index * TL_FLASH_TILE +
                        ulong(local_query_index);
                    key_total +=
                        score_gradient[tile_index] *
                        query_tile[
                            ulong(local_query_index) *
                                head_width +
                            channel
                        ];
                    value_total +=
                        probability_tile[tile_index] *
                        upstream_tile[
                            ulong(local_query_index) *
                                head_width +
                            channel
                        ];
                }
            }
            key_gradient_tile[element] = key_total;
            value_gradient_tile[element] = value_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (ulong element = thread_index;
         element < tile_elements;
         element += TL_FLASH_THREADS) {
        const ulong local_key = element / head_width;
        const ulong channel = element % head_width;
        const ulong key = key_start + local_key;
        if (batch_head < batch_heads && key < time) {
            const ulong index = tl_flash_offset(
                batch_head,
                key,
                channel,
                time,
                head_width
            );
            key_gradient[index] = key_gradient_tile[element];
            value_gradient[index] = value_gradient_tile[element];
        }
    }
}
)METAL";

}  // namespace transformer_lab::backend_detail::attention_metal_detail
