#include "core/backend/attention/cuda/launch.hpp"

#include "core/backend/attention/cuda/common.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace riftco_transformer::backend_detail {
namespace {

using attention_cuda_detail::DeviceBuffer;
using attention_cuda_detail::kThreadsPerBlock;

__device__ std::size_t paged_offset(const std::uint32_t* block_table,
                                    std::size_t head,
                                    std::size_t position,
                                    std::size_t heads,
                                    std::size_t head_width,
                                    std::size_t block_size) {
    const std::size_t logical_block = position / block_size;
    const std::size_t block_offset = position % block_size;
    const std::size_t physical_block =
        static_cast<std::size_t>(block_table[logical_block]);
    return ((physical_block * heads + head) * block_size + block_offset) *
           head_width;
}

__global__ void paged_decode_probabilities_kernel(
    const float* queries,
    const float* key_pages,
    const std::uint32_t* block_table,
    float* probabilities,
    unsigned int* status,
    std::size_t heads,
    std::size_t head_width,
    std::size_t block_size,
    std::size_t sequence_length,
    float score_scale) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t head =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (head < heads) {
        const std::size_t query_base = head * head_width;
        const std::size_t probability_base = head * sequence_length;
        float maximum = -CUDART_INF_F;
        bool valid = true;

        for (std::size_t position = 0; position < sequence_length; ++position) {
            const std::size_t key_base = paged_offset(block_table,
                                                      head,
                                                      position,
                                                      heads,
                                                      head_width,
                                                      block_size);
            float score = 0.0F;
            for (std::size_t channel = 0; channel < head_width; ++channel) {
                score += queries[query_base + channel] *
                         key_pages[key_base + channel];
            }
            score *= score_scale;
            probabilities[probability_base + position] = score;
            if (isnan(score) || score == CUDART_INF_F) {
                valid = false;
            }
            maximum = fmaxf(maximum, score);
        }
        if (!valid || maximum == -CUDART_INF_F) {
            atomicExch(status, 1U);
            for (std::size_t position = 0; position < sequence_length;
                 ++position) {
                probabilities[probability_base + position] = 0.0F;
            }
            head += stride;
            continue;
        }

        double denominator = 0.0;
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const double exponential = exp(static_cast<double>(
                probabilities[probability_base + position] - maximum));
            probabilities[probability_base + position] =
                static_cast<float>(exponential);
            denominator += exponential;
        }
        if (!(denominator > 0.0) || !isfinite(denominator)) {
            atomicExch(status, 1U);
            for (std::size_t position = 0; position < sequence_length;
                 ++position) {
                probabilities[probability_base + position] = 0.0F;
            }
            head += stride;
            continue;
        }
        for (std::size_t position = 0; position < sequence_length; ++position) {
            probabilities[probability_base + position] = static_cast<float>(
                static_cast<double>(
                    probabilities[probability_base + position]) /
                denominator);
        }
        head += stride;
    }
}

__global__ void paged_decode_context_kernel(const float* probabilities,
                                            const float* value_pages,
                                            const std::uint32_t* block_table,
                                            float* context,
                                            std::size_t context_count,
                                            std::size_t heads,
                                            std::size_t head_width,
                                            std::size_t block_size,
                                            std::size_t sequence_length) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t context_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (context_index < context_count) {
        const std::size_t head = context_index / head_width;
        const std::size_t channel = context_index % head_width;
        const std::size_t probability_base = head * sequence_length;
        float total = 0.0F;
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const std::size_t value_base = paged_offset(block_table,
                                                        head,
                                                        position,
                                                        heads,
                                                        head_width,
                                                        block_size);
            total += probabilities[probability_base + position] *
                     value_pages[value_base + channel];
        }
        context[context_index] = total;
        context_index += stride;
    }
}

}  // namespace

void cuda_paged_decode_attention_forward(
    const PagedDecodeAttentionForwardRequest& request) {
    const auto& dimensions = request.dimensions;
    const std::size_t context_count = request.context.size();
    if (dimensions.heads >
        std::numeric_limits<std::size_t>::max() / dimensions.sequence_length) {
        throw std::overflow_error(
            "CUDA paged decode probability storage size overflow");
    }
    DeviceBuffer<std::uint32_t> block_table(request.block_table.size());
    DeviceBuffer<float> probabilities(dimensions.heads *
                                      dimensions.sequence_length);
    DeviceBuffer<unsigned int> status(1);
    status.zero("paged decode attention status initialization");
    attention_cuda_detail::require_cuda_success(
        cudaMemcpy(block_table.data(),
                   request.block_table.data(),
                   request.block_table.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
        "paged decode block-table upload");

    const float* queries =
        attention_cuda_detail::require_native_input(request.queries,
                                                    "paged decode attention");
    const float* key_pages =
        attention_cuda_detail::require_native_input(request.key_pages,
                                                    "paged decode attention");
    const float* value_pages =
        attention_cuda_detail::require_native_input(request.value_pages,
                                                    "paged decode attention");
    float* context =
        attention_cuda_detail::require_native_output(request.context,
                                                     "paged decode attention");
    const float score_scale =
        1.0F / std::sqrt(static_cast<float>(dimensions.head_width));

    paged_decode_probabilities_kernel<<<
        attention_cuda_detail::block_count_for(dimensions.heads),
        kThreadsPerBlock>>>(queries,
                            key_pages,
                            block_table.data(),
                            probabilities.data(),
                            status.data(),
                            dimensions.heads,
                            dimensions.head_width,
                            dimensions.block_size,
                            dimensions.sequence_length,
                            score_scale);
    attention_cuda_detail::require_kernel_launch(
        "paged decode probability kernel launch");
    paged_decode_context_kernel<<<
        attention_cuda_detail::block_count_for(context_count),
        kThreadsPerBlock>>>(probabilities.data(),
                            value_pages,
                            block_table.data(),
                            context,
                            context_count,
                            dimensions.heads,
                            dimensions.head_width,
                            dimensions.block_size,
                            dimensions.sequence_length);
    attention_cuda_detail::require_kernel_launch(
        "paged decode context kernel launch");
    attention_cuda_detail::synchronize(
        "paged decode attention synchronization");
    if (attention_cuda_detail::read_status(
            status,
            "paged decode attention status read")) {
        throw std::domain_error(
            "paged decode attention received non-finite scores");
    }
}

}  // namespace riftco_transformer::backend_detail
