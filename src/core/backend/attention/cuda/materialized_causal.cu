#include "core/backend/attention/cuda/launch.hpp"

#include "core/backend/attention/cuda/common.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace riftco_transformer::backend_detail {
namespace {

using attention_cuda_detail::DeviceBuffer;
using attention_cuda_detail::kThreadsPerBlock;

__global__ void materialized_probabilities_kernel(const float* queries,
                                                  const float* keys,
                                                  float* probabilities,
                                                  unsigned int* status,
                                                  std::size_t row_count,
                                                  std::size_t time,
                                                  std::size_t head_width,
                                                  float score_scale) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (row < row_count) {
        const std::size_t query_time = row % time;
        const std::size_t head_base = (row / time) * time * head_width;
        const std::size_t query_base = head_base + query_time * head_width;
        const std::size_t probability_base = row * time;
        float maximum = -CUDART_INF_F;
        bool valid = true;

        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            const std::size_t key_base = head_base + key_time * head_width;
            float score = 0.0F;
            for (std::size_t channel = 0; channel < head_width; ++channel) {
                score +=
                    queries[query_base + channel] * keys[key_base + channel];
            }
            score *= score_scale;
            probabilities[probability_base + key_time] = score;
            if (isnan(score) || score == CUDART_INF_F) {
                valid = false;
            }
            maximum = fmaxf(maximum, score);
        }
        for (std::size_t key_time = query_time + 1; key_time < time;
             ++key_time) {
            probabilities[probability_base + key_time] = 0.0F;
        }

        if (!valid || maximum == -CUDART_INF_F) {
            atomicExch(status, 1U);
            for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
                probabilities[probability_base + key_time] = 0.0F;
            }
            row += stride;
            continue;
        }

        double denominator = 0.0;
        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            const double exponential = exp(static_cast<double>(
                probabilities[probability_base + key_time] - maximum));
            probabilities[probability_base + key_time] =
                static_cast<float>(exponential);
            denominator += exponential;
        }
        if (!(denominator > 0.0) || !isfinite(denominator)) {
            atomicExch(status, 1U);
            for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
                probabilities[probability_base + key_time] = 0.0F;
            }
            row += stride;
            continue;
        }
        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            probabilities[probability_base + key_time] = static_cast<float>(
                static_cast<double>(
                    probabilities[probability_base + key_time]) /
                denominator);
        }
        row += stride;
    }
}

__global__ void materialized_context_kernel(const float* probabilities,
                                            const float* values,
                                            float* context,
                                            std::size_t context_count,
                                            std::size_t time,
                                            std::size_t head_width) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (output_index < context_count) {
        const std::size_t row = output_index / head_width;
        const std::size_t channel = output_index % head_width;
        const std::size_t query_time = row % time;
        const std::size_t head_base = (row / time) * time * head_width;
        float total = 0.0F;
        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            total += probabilities[row * time + key_time] *
                     values[head_base + key_time * head_width + channel];
        }
        context[output_index] = total;
        output_index += stride;
    }
}

__global__ void materialized_context_score_backward_kernel(
    const float* probabilities,
    const float* values,
    const float* upstream_context,
    float* score_gradient,
    std::size_t row_count,
    std::size_t time,
    std::size_t head_width,
    float score_scale) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (row < row_count) {
        const std::size_t query_time = row % time;
        const std::size_t head_base = (row / time) * time * head_width;
        const std::size_t upstream_base = row * head_width;
        const std::size_t probability_base = row * time;
        double weighted_gradient = 0.0;

        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            float probability_gradient = 0.0F;
            const std::size_t value_base = head_base + key_time * head_width;
            for (std::size_t channel = 0; channel < head_width; ++channel) {
                probability_gradient +=
                    upstream_context[upstream_base + channel] *
                    values[value_base + channel];
            }
            score_gradient[probability_base + key_time] = probability_gradient;
            weighted_gradient +=
                static_cast<double>(
                    probabilities[probability_base + key_time]) *
                static_cast<double>(probability_gradient);
        }
        const float weighted = static_cast<float>(weighted_gradient);
        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            score_gradient[probability_base + key_time] =
                score_scale * probabilities[probability_base + key_time] *
                (score_gradient[probability_base + key_time] - weighted);
        }
        for (std::size_t key_time = query_time + 1; key_time < time;
             ++key_time) {
            score_gradient[probability_base + key_time] = 0.0F;
        }
        row += stride;
    }
}

__global__ void materialized_probability_score_backward_kernel(
    const float* probabilities,
    const float* upstream_probabilities,
    float* score_gradient,
    std::size_t row_count,
    std::size_t time,
    float score_scale) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (row < row_count) {
        const std::size_t query_time = row % time;
        const std::size_t base = row * time;
        double weighted_gradient = 0.0;
        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            weighted_gradient +=
                static_cast<double>(probabilities[base + key_time]) *
                static_cast<double>(upstream_probabilities[base + key_time]);
        }
        const float weighted = static_cast<float>(weighted_gradient);
        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            score_gradient[base + key_time] =
                score_scale * probabilities[base + key_time] *
                (upstream_probabilities[base + key_time] - weighted);
        }
        for (std::size_t key_time = query_time + 1; key_time < time;
             ++key_time) {
            score_gradient[base + key_time] = 0.0F;
        }
        row += stride;
    }
}

__global__ void materialized_query_backward_kernel(const float* keys,
                                                   const float* score_gradient,
                                                   float* query_gradient,
                                                   std::size_t vector_count,
                                                   std::size_t time,
                                                   std::size_t head_width) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (output_index < vector_count) {
        const std::size_t row = output_index / head_width;
        const std::size_t channel = output_index % head_width;
        const std::size_t query_time = row % time;
        const std::size_t head_base = (row / time) * time * head_width;
        float total = 0.0F;
        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            total += score_gradient[row * time + key_time] *
                     keys[head_base + key_time * head_width + channel];
        }
        query_gradient[output_index] = total;
        output_index += stride;
    }
}

__global__ void materialized_key_backward_kernel(const float* queries,
                                                 const float* score_gradient,
                                                 float* key_gradient,
                                                 std::size_t vector_count,
                                                 std::size_t time,
                                                 std::size_t head_width) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (output_index < vector_count) {
        const std::size_t vector = output_index / head_width;
        const std::size_t channel = output_index % head_width;
        const std::size_t key_time = vector % time;
        const std::size_t head = vector / time;
        const std::size_t head_base = head * time * head_width;
        float total = 0.0F;
        for (std::size_t query_time = key_time; query_time < time;
             ++query_time) {
            const std::size_t row = head * time + query_time;
            total += score_gradient[row * time + key_time] *
                     queries[head_base + query_time * head_width + channel];
        }
        key_gradient[output_index] = total;
        output_index += stride;
    }
}

__global__ void materialized_value_backward_kernel(
    const float* probabilities,
    const float* upstream_context,
    float* value_gradient,
    std::size_t vector_count,
    std::size_t time,
    std::size_t head_width) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (output_index < vector_count) {
        const std::size_t vector = output_index / head_width;
        const std::size_t channel = output_index % head_width;
        const std::size_t key_time = vector % time;
        const std::size_t head = vector / time;
        float total = 0.0F;
        for (std::size_t query_time = key_time; query_time < time;
             ++query_time) {
            const std::size_t row = head * time + query_time;
            total += probabilities[row * time + key_time] *
                     upstream_context[row * head_width + channel];
        }
        value_gradient[output_index] = total;
        output_index += stride;
    }
}

void launch_query_key_gradients(const float* queries,
                                const float* keys,
                                const float* score_gradient,
                                float* query_gradient,
                                float* key_gradient,
                                std::size_t vector_count,
                                std::size_t time,
                                std::size_t head_width) {
    const unsigned int blocks =
        attention_cuda_detail::block_count_for(vector_count);
    materialized_query_backward_kernel<<<blocks, kThreadsPerBlock>>>(
        keys,
        score_gradient,
        query_gradient,
        vector_count,
        time,
        head_width);
    attention_cuda_detail::require_kernel_launch(
        "materialized attention query-backward kernel launch");
    materialized_key_backward_kernel<<<blocks, kThreadsPerBlock>>>(
        queries,
        score_gradient,
        key_gradient,
        vector_count,
        time,
        head_width);
    attention_cuda_detail::require_kernel_launch(
        "materialized attention key-backward kernel launch");
}

}  // namespace

void cuda_materialized_causal_attention_forward(
    const MaterializedCausalAttentionForwardRequest& request) {
    const auto& dimensions = request.dimensions;
    const std::size_t vector_count = request.queries.size();
    const std::size_t row_count = vector_count / dimensions.head_width;
    const float score_scale =
        1.0F / std::sqrt(static_cast<float>(dimensions.head_width));
    DeviceBuffer<unsigned int> status(1);
    status.zero("materialized attention status initialization");

    const float* queries = attention_cuda_detail::require_native_input(
        request.queries,
        "materialized attention forward");
    const float* keys = attention_cuda_detail::require_native_input(
        request.keys,
        "materialized attention forward");
    const float* values = attention_cuda_detail::require_native_input(
        request.values,
        "materialized attention forward");
    float* probabilities = attention_cuda_detail::require_native_output(
        request.probabilities,
        "materialized attention forward");
    float* context = attention_cuda_detail::require_native_output(
        request.context,
        "materialized attention forward");

    materialized_probabilities_kernel<<<
        attention_cuda_detail::block_count_for(row_count),
        kThreadsPerBlock>>>(queries,
                            keys,
                            probabilities,
                            status.data(),
                            row_count,
                            dimensions.time,
                            dimensions.head_width,
                            score_scale);
    attention_cuda_detail::require_kernel_launch(
        "materialized attention probability kernel launch");
    materialized_context_kernel<<<attention_cuda_detail::block_count_for(
                                      vector_count),
                                  kThreadsPerBlock>>>(probabilities,
                                                      values,
                                                      context,
                                                      vector_count,
                                                      dimensions.time,
                                                      dimensions.head_width);
    attention_cuda_detail::require_kernel_launch(
        "materialized attention context kernel launch");
    attention_cuda_detail::synchronize(
        "materialized attention forward synchronization");
    if (attention_cuda_detail::read_status(
            status,
            "materialized attention status read")) {
        throw std::domain_error(
            "materialized causal attention received non-finite scores");
    }
}

void cuda_materialized_causal_attention_context_backward(
    const MaterializedCausalAttentionContextBackwardRequest& request) {
    const auto& dimensions = request.dimensions;
    const std::size_t vector_count = request.queries.size();
    const std::size_t row_count = vector_count / dimensions.head_width;
    const std::size_t probability_count = request.probabilities.size();
    const float score_scale =
        1.0F / std::sqrt(static_cast<float>(dimensions.head_width));
    DeviceBuffer<float> score_gradient(probability_count);

    const float* queries = attention_cuda_detail::require_native_input(
        request.queries,
        "materialized attention context backward");
    const float* keys = attention_cuda_detail::require_native_input(
        request.keys,
        "materialized attention context backward");
    const float* values = attention_cuda_detail::require_native_input(
        request.values,
        "materialized attention context backward");
    const float* probabilities = attention_cuda_detail::require_native_input(
        request.probabilities,
        "materialized attention context backward");
    const float* upstream = attention_cuda_detail::require_native_input(
        request.upstream_context,
        "materialized attention context backward");
    float* query_gradient = attention_cuda_detail::require_native_output(
        request.query_gradient,
        "materialized attention context backward");
    float* key_gradient = attention_cuda_detail::require_native_output(
        request.key_gradient,
        "materialized attention context backward");
    float* value_gradient = attention_cuda_detail::require_native_output(
        request.value_gradient,
        "materialized attention context backward");

    materialized_context_score_backward_kernel<<<
        attention_cuda_detail::block_count_for(row_count),
        kThreadsPerBlock>>>(probabilities,
                            values,
                            upstream,
                            score_gradient.data(),
                            row_count,
                            dimensions.time,
                            dimensions.head_width,
                            score_scale);
    attention_cuda_detail::require_kernel_launch(
        "materialized attention score-backward kernel launch");
    launch_query_key_gradients(queries,
                               keys,
                               score_gradient.data(),
                               query_gradient,
                               key_gradient,
                               vector_count,
                               dimensions.time,
                               dimensions.head_width);
    materialized_value_backward_kernel<<<
        attention_cuda_detail::block_count_for(vector_count),
        kThreadsPerBlock>>>(probabilities,
                            upstream,
                            value_gradient,
                            vector_count,
                            dimensions.time,
                            dimensions.head_width);
    attention_cuda_detail::require_kernel_launch(
        "materialized attention value-backward kernel launch");
    attention_cuda_detail::synchronize(
        "materialized attention context-backward synchronization");
}

void cuda_materialized_causal_attention_probabilities_backward(
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request) {
    const auto& dimensions = request.dimensions;
    const std::size_t vector_count = request.queries.size();
    const std::size_t row_count = vector_count / dimensions.head_width;
    const std::size_t probability_count = request.probabilities.size();
    const float score_scale =
        1.0F / std::sqrt(static_cast<float>(dimensions.head_width));
    DeviceBuffer<float> score_gradient(probability_count);

    const float* queries = attention_cuda_detail::require_native_input(
        request.queries,
        "materialized attention probability backward");
    const float* keys = attention_cuda_detail::require_native_input(
        request.keys,
        "materialized attention probability backward");
    const float* probabilities = attention_cuda_detail::require_native_input(
        request.probabilities,
        "materialized attention probability backward");
    const float* upstream = attention_cuda_detail::require_native_input(
        request.upstream_probabilities,
        "materialized attention probability backward");
    float* query_gradient = attention_cuda_detail::require_native_output(
        request.query_gradient,
        "materialized attention probability backward");
    float* key_gradient = attention_cuda_detail::require_native_output(
        request.key_gradient,
        "materialized attention probability backward");

    materialized_probability_score_backward_kernel<<<
        attention_cuda_detail::block_count_for(row_count),
        kThreadsPerBlock>>>(probabilities,
                            upstream,
                            score_gradient.data(),
                            row_count,
                            dimensions.time,
                            score_scale);
    attention_cuda_detail::require_kernel_launch(
        "materialized probability score-backward kernel launch");
    launch_query_key_gradients(queries,
                               keys,
                               score_gradient.data(),
                               query_gradient,
                               key_gradient,
                               vector_count,
                               dimensions.time,
                               dimensions.head_width);
    attention_cuda_detail::synchronize(
        "materialized probability-backward synchronization");
}

}  // namespace riftco_transformer::backend_detail
