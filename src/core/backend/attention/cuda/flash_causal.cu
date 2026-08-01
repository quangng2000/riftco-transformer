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

__device__ float block_sum(float value, float* scratch) {
    const unsigned int thread = threadIdx.x;
    scratch[thread] = value;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset != 0; offset /= 2) {
        if (thread < offset) {
            scratch[thread] += scratch[thread + offset];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ std::size_t flash_tensor_offset(std::size_t row,
                                           std::size_t time_index,
                                           std::size_t channel,
                                           std::size_t time,
                                           std::size_t head_width) {
    const std::size_t head = row / time;
    return (head * time + time_index) * head_width + channel;
}

__global__ void flash_forward_kernel(const float* queries,
                                     const float* keys,
                                     const float* values,
                                     float* row_maxima,
                                     float* row_exp_sums,
                                     double* context_numerator,
                                     float* context,
                                     unsigned int* status,
                                     std::size_t row_count,
                                     std::size_t time,
                                     std::size_t head_width,
                                     float score_scale) {
    __shared__ float reduction[kThreadsPerBlock];
    __shared__ double previous_scale;
    __shared__ double key_weight;
    __shared__ float saved_exp_sum;
    __shared__ unsigned int row_is_valid;

    for (std::size_t row = blockIdx.x; row < row_count; row += gridDim.x) {
        const std::size_t query_time = row % time;
        const std::size_t query_base = row * head_width;
        for (std::size_t channel = threadIdx.x; channel < head_width;
             channel += blockDim.x) {
            context_numerator[query_base + channel] = 0.0;
            context[query_base + channel] = 0.0F;
        }
        __syncthreads();

        float running_maximum = -CUDART_INF_F;
        double running_exp_sum = 0.0;
        bool valid = true;
        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            float score_partial = 0.0F;
            const std::size_t key_base =
                flash_tensor_offset(row, key_time, 0, time, head_width);
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                score_partial +=
                    queries[query_base + channel] * keys[key_base + channel];
            }
            const float score =
                block_sum(score_partial, reduction) * score_scale;

            if (threadIdx.x == 0) {
                if (isnan(score) || score == CUDART_INF_F) {
                    valid = false;
                    previous_scale = 1.0;
                    key_weight = 0.0;
                } else if (score == -CUDART_INF_F &&
                           running_maximum == -CUDART_INF_F) {
                    // A leading negative-infinite score has zero mass. Avoid
                    // the undefined subtraction -infinity - -infinity.
                    previous_scale = 1.0;
                    key_weight = 0.0;
                } else {
                    const float updated_maximum = fmaxf(running_maximum, score);
                    previous_scale =
                        running_maximum == -CUDART_INF_F
                            ? 0.0
                            : exp(static_cast<double>(running_maximum -
                                                      updated_maximum));
                    key_weight =
                        exp(static_cast<double>(score - updated_maximum));
                    running_exp_sum =
                        running_exp_sum * previous_scale + key_weight;
                    running_maximum = updated_maximum;
                }
            }
            __syncthreads();
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                context_numerator[query_base + channel] =
                    context_numerator[query_base + channel] * previous_scale +
                    key_weight *
                        static_cast<double>(values[key_base + channel]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            row_is_valid = 1U;
            if (!valid || running_maximum == -CUDART_INF_F ||
                !(running_exp_sum > 0.0) || !isfinite(running_exp_sum)) {
                row_is_valid = 0U;
            } else {
                saved_exp_sum = static_cast<float>(running_exp_sum);
                if (!(saved_exp_sum > 0.0F) || !isfinite(saved_exp_sum)) {
                    row_is_valid = 0U;
                }
            }
            if (row_is_valid == 0U) {
                atomicExch(status, 1U);
                row_maxima[row] = -CUDART_INF_F;
                row_exp_sums[row] = 0.0F;
                saved_exp_sum = 1.0F;
            } else {
                row_maxima[row] = running_maximum;
                row_exp_sums[row] = saved_exp_sum;
            }
        }
        __syncthreads();
        for (std::size_t channel = threadIdx.x; channel < head_width;
             channel += blockDim.x) {
            context[query_base + channel] =
                row_is_valid == 0U
                    ? 0.0F
                    : static_cast<float>(
                          context_numerator[query_base + channel] /
                          static_cast<double>(saved_exp_sum));
        }
        __syncthreads();
    }
}

__global__ void flash_delta_kernel(const float* queries,
                                   const float* keys,
                                   const float* values,
                                   const float* row_maxima,
                                   const float* row_exp_sums,
                                   const float* upstream,
                                   float* delta,
                                   std::size_t row_count,
                                   std::size_t time,
                                   std::size_t head_width,
                                   float score_scale) {
    __shared__ float reduction[kThreadsPerBlock];
    for (std::size_t row = blockIdx.x; row < row_count; row += gridDim.x) {
        const std::size_t query_time = row % time;
        const std::size_t query_base = row * head_width;
        double weighted_gradient = 0.0;

        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            const std::size_t key_base =
                flash_tensor_offset(row, key_time, 0, time, head_width);
            float score_partial = 0.0F;
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                score_partial +=
                    queries[query_base + channel] * keys[key_base + channel];
            }
            const float score =
                block_sum(score_partial, reduction) * score_scale;

            float upstream_dot_value_partial = 0.0F;
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                upstream_dot_value_partial +=
                    upstream[query_base + channel] * values[key_base + channel];
            }
            const float upstream_dot_value =
                block_sum(upstream_dot_value_partial, reduction);
            if (threadIdx.x == 0) {
                const float probability = static_cast<float>(
                    exp(static_cast<double>(score - row_maxima[row])) /
                    static_cast<double>(row_exp_sums[row]));
                weighted_gradient += static_cast<double>(probability) *
                                     static_cast<double>(upstream_dot_value);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            delta[row] = static_cast<float>(weighted_gradient);
        }
        __syncthreads();
    }
}

__global__ void flash_query_backward_kernel(const float* queries,
                                            const float* keys,
                                            const float* values,
                                            const float* row_maxima,
                                            const float* row_exp_sums,
                                            const float* upstream,
                                            const float* delta,
                                            float* query_gradient,
                                            std::size_t row_count,
                                            std::size_t time,
                                            std::size_t head_width,
                                            float score_scale) {
    __shared__ float reduction[kThreadsPerBlock];
    __shared__ float score_derivative;

    for (std::size_t row = blockIdx.x; row < row_count; row += gridDim.x) {
        const std::size_t query_time = row % time;
        const std::size_t query_base = row * head_width;
        for (std::size_t channel = threadIdx.x; channel < head_width;
             channel += blockDim.x) {
            query_gradient[query_base + channel] = 0.0F;
        }
        __syncthreads();

        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            const std::size_t key_base =
                flash_tensor_offset(row, key_time, 0, time, head_width);
            float score_partial = 0.0F;
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                score_partial +=
                    queries[query_base + channel] * keys[key_base + channel];
            }
            const float score =
                block_sum(score_partial, reduction) * score_scale;

            float upstream_dot_value_partial = 0.0F;
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                upstream_dot_value_partial +=
                    upstream[query_base + channel] * values[key_base + channel];
            }
            const float upstream_dot_value =
                block_sum(upstream_dot_value_partial, reduction);
            if (threadIdx.x == 0) {
                const float probability = static_cast<float>(
                    exp(static_cast<double>(score - row_maxima[row])) /
                    static_cast<double>(row_exp_sums[row]));
                score_derivative = score_scale * probability *
                                   (upstream_dot_value - delta[row]);
            }
            __syncthreads();
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                query_gradient[query_base + channel] +=
                    score_derivative * keys[key_base + channel];
            }
            __syncthreads();
        }
    }
}

__global__ void flash_key_value_backward_kernel(const float* queries,
                                                const float* keys,
                                                const float* values,
                                                const float* row_maxima,
                                                const float* row_exp_sums,
                                                const float* upstream,
                                                const float* delta,
                                                float* key_gradient,
                                                float* value_gradient,
                                                std::size_t row_count,
                                                std::size_t time,
                                                std::size_t head_width,
                                                float score_scale) {
    __shared__ float reduction[kThreadsPerBlock];
    __shared__ float probability;
    __shared__ float score_derivative;

    for (std::size_t key_row = blockIdx.x; key_row < row_count;
         key_row += gridDim.x) {
        const std::size_t key_time = key_row % time;
        const std::size_t key_base = key_row * head_width;
        const std::size_t head = key_row / time;
        for (std::size_t channel = threadIdx.x; channel < head_width;
             channel += blockDim.x) {
            key_gradient[key_base + channel] = 0.0F;
            value_gradient[key_base + channel] = 0.0F;
        }
        __syncthreads();

        for (std::size_t query_time = key_time; query_time < time;
             ++query_time) {
            const std::size_t row = head * time + query_time;
            const std::size_t query_base = row * head_width;
            float score_partial = 0.0F;
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                score_partial +=
                    queries[query_base + channel] * keys[key_base + channel];
            }
            const float score =
                block_sum(score_partial, reduction) * score_scale;

            float upstream_dot_value_partial = 0.0F;
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                upstream_dot_value_partial +=
                    upstream[query_base + channel] * values[key_base + channel];
            }
            const float upstream_dot_value =
                block_sum(upstream_dot_value_partial, reduction);
            if (threadIdx.x == 0) {
                probability = static_cast<float>(
                    exp(static_cast<double>(score - row_maxima[row])) /
                    static_cast<double>(row_exp_sums[row]));
                score_derivative = score_scale * probability *
                                   (upstream_dot_value - delta[row]);
            }
            __syncthreads();
            for (std::size_t channel = threadIdx.x; channel < head_width;
                 channel += blockDim.x) {
                key_gradient[key_base + channel] +=
                    score_derivative * queries[query_base + channel];
                value_gradient[key_base + channel] +=
                    probability * upstream[query_base + channel];
            }
            __syncthreads();
        }
    }
}

}  // namespace

void cuda_flash_causal_attention_forward(
    const FlashCausalAttentionForwardRequest& request) {
    const auto& dimensions = request.dimensions;
    const std::size_t vector_count = request.queries.size();
    const std::size_t row_count = vector_count / dimensions.head_width;
    const float score_scale =
        1.0F / std::sqrt(static_cast<float>(dimensions.head_width));
    DeviceBuffer<double> context_numerator(vector_count);
    DeviceBuffer<unsigned int> status(1);
    status.zero("Flash attention status initialization");

    const float* queries =
        attention_cuda_detail::require_native_input(request.queries,
                                                    "Flash attention forward");
    const float* keys =
        attention_cuda_detail::require_native_input(request.keys,
                                                    "Flash attention forward");
    const float* values =
        attention_cuda_detail::require_native_input(request.values,
                                                    "Flash attention forward");
    float* row_maxima =
        attention_cuda_detail::require_native_output(request.row_maxima,
                                                     "Flash attention forward");
    float* row_exp_sums =
        attention_cuda_detail::require_native_output(request.row_exp_sums,
                                                     "Flash attention forward");
    float* context =
        attention_cuda_detail::require_native_output(request.context,
                                                     "Flash attention forward");

    flash_forward_kernel<<<attention_cuda_detail::row_block_count_for(
                               row_count),
                           kThreadsPerBlock>>>(queries,
                                               keys,
                                               values,
                                               row_maxima,
                                               row_exp_sums,
                                               context_numerator.data(),
                                               context,
                                               status.data(),
                                               row_count,
                                               dimensions.time,
                                               dimensions.head_width,
                                               score_scale);
    attention_cuda_detail::require_kernel_launch(
        "Flash attention forward kernel launch");
    attention_cuda_detail::synchronize(
        "Flash attention forward synchronization");
    if (attention_cuda_detail::read_status(status,
                                           "Flash attention status read")) {
        throw std::domain_error(
            "Flash causal attention received non-finite scores");
    }
}

void cuda_flash_causal_attention_backward(
    const FlashCausalAttentionBackwardRequest& request) {
    const auto& dimensions = request.dimensions;
    const std::size_t vector_count = request.queries.size();
    const std::size_t row_count = vector_count / dimensions.head_width;
    const float score_scale =
        1.0F / std::sqrt(static_cast<float>(dimensions.head_width));
    DeviceBuffer<float> delta(row_count);

    const float* queries =
        attention_cuda_detail::require_native_input(request.queries,
                                                    "Flash attention backward");
    const float* keys =
        attention_cuda_detail::require_native_input(request.keys,
                                                    "Flash attention backward");
    const float* values =
        attention_cuda_detail::require_native_input(request.values,
                                                    "Flash attention backward");
    const float* row_maxima =
        attention_cuda_detail::require_native_input(request.row_maxima,
                                                    "Flash attention backward");
    const float* row_exp_sums =
        attention_cuda_detail::require_native_input(request.row_exp_sums,
                                                    "Flash attention backward");
    const float* upstream =
        attention_cuda_detail::require_native_input(request.upstream_context,
                                                    "Flash attention backward");
    float* query_gradient = attention_cuda_detail::require_native_output(
        request.query_gradient,
        "Flash attention backward");
    float* key_gradient = attention_cuda_detail::require_native_output(
        request.key_gradient,
        "Flash attention backward");
    float* value_gradient = attention_cuda_detail::require_native_output(
        request.value_gradient,
        "Flash attention backward");
    const unsigned int blocks =
        attention_cuda_detail::row_block_count_for(row_count);

    flash_delta_kernel<<<blocks, kThreadsPerBlock>>>(queries,
                                                     keys,
                                                     values,
                                                     row_maxima,
                                                     row_exp_sums,
                                                     upstream,
                                                     delta.data(),
                                                     row_count,
                                                     dimensions.time,
                                                     dimensions.head_width,
                                                     score_scale);
    attention_cuda_detail::require_kernel_launch(
        "Flash attention delta kernel launch");
    flash_query_backward_kernel<<<blocks, kThreadsPerBlock>>>(
        queries,
        keys,
        values,
        row_maxima,
        row_exp_sums,
        upstream,
        delta.data(),
        query_gradient,
        row_count,
        dimensions.time,
        dimensions.head_width,
        score_scale);
    attention_cuda_detail::require_kernel_launch(
        "Flash attention query-backward kernel launch");
    flash_key_value_backward_kernel<<<blocks, kThreadsPerBlock>>>(
        queries,
        keys,
        values,
        row_maxima,
        row_exp_sums,
        upstream,
        delta.data(),
        key_gradient,
        value_gradient,
        row_count,
        dimensions.time,
        dimensions.head_width,
        score_scale);
    attention_cuda_detail::require_kernel_launch(
        "Flash attention key/value-backward kernel launch");
    attention_cuda_detail::synchronize(
        "Flash attention backward synchronization");
}

}  // namespace riftco_transformer::backend_detail
