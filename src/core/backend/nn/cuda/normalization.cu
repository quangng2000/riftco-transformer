#include "core/backend/nn/cuda/launch.hpp"

#include "core/backend/nn/cuda/common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace riftco_transformer::backend_detail {
namespace {

using nn_cuda_detail::kThreadsPerBlock;

__global__ void layer_norm_forward_kernel(const float* input,
                                          const float* scale, const float* bias,
                                          float* output, float* mean,
                                          float* inverse_standard_deviation,
                                          std::size_t rows, std::size_t width,
                                          float epsilon) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (row < rows) {
        const std::size_t base = row * width;
        float row_mean = 0.0F;
        for (std::size_t column = 0; column < width; ++column) {
            row_mean += input[base + column];
        }
        row_mean /= static_cast<float>(width);

        float variance = 0.0F;
        for (std::size_t column = 0; column < width; ++column) {
            const float centered = input[base + column] - row_mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(width);
        const float inverse_std = 1.0F / sqrtf(variance + epsilon);

        mean[row] = row_mean;
        inverse_standard_deviation[row] = inverse_std;
        for (std::size_t column = 0; column < width; ++column) {
            const float normalized =
                (input[base + column] - row_mean) * inverse_std;
            output[base + column] = normalized * scale[column] + bias[column];
        }
        row += stride;
    }
}

__global__ void layer_norm_input_backward_kernel(
    const float* input, const float* scale, const float* mean,
    const float* inverse_standard_deviation, const float* upstream,
    float* input_gradient, std::size_t rows, std::size_t width) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (row < rows) {
        const std::size_t base = row * width;
        float sum_scaled_upstream = 0.0F;
        float sum_scaled_upstream_times_normalized = 0.0F;
        for (std::size_t column = 0; column < width; ++column) {
            const float normalized = (input[base + column] - mean[row]) *
                                     inverse_standard_deviation[row];
            const float scaled_upstream =
                upstream[base + column] * scale[column];
            sum_scaled_upstream += scaled_upstream;
            sum_scaled_upstream_times_normalized +=
                scaled_upstream * normalized;
        }

        const float inverse_width = 1.0F / static_cast<float>(width);
        for (std::size_t column = 0; column < width; ++column) {
            const float normalized = (input[base + column] - mean[row]) *
                                     inverse_standard_deviation[row];
            const float scaled_upstream =
                upstream[base + column] * scale[column];
            input_gradient[base + column] =
                inverse_standard_deviation[row] *
                (scaled_upstream - sum_scaled_upstream * inverse_width -
                 normalized * sum_scaled_upstream_times_normalized *
                     inverse_width);
        }
        row += stride;
    }
}

__global__ void layer_norm_parameter_backward_kernel(
    const float* input, const float* mean,
    const float* inverse_standard_deviation, const float* upstream,
    float* scale_gradient, float* bias_gradient, std::size_t rows,
    std::size_t width) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (column < width) {
        float scale_total = 0.0F;
        float bias_total = 0.0F;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t offset = row * width + column;
            const float normalized =
                (input[offset] - mean[row]) * inverse_standard_deviation[row];
            scale_total += upstream[offset] * normalized;
            bias_total += upstream[offset];
        }
        scale_gradient[column] = scale_total;
        bias_gradient[column] = bias_total;
        column += stride;
    }
}

} // namespace

void cuda_nn_layer_norm_forward(const LayerNormForwardRequest& request) {
    const char* operation_name = "LayerNorm forward";
    layer_norm_forward_kernel<<<nn_cuda_detail::block_count_for(request.rows),
                                kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_input(request.scale, operation_name),
        nn_cuda_detail::require_native_input(request.bias, operation_name),
        nn_cuda_detail::require_native_output(request.output, operation_name),
        nn_cuda_detail::require_native_output(request.mean, operation_name),
        nn_cuda_detail::require_native_output(
            request.inverse_standard_deviation, operation_name),
        request.rows, request.width, request.epsilon);
    nn_cuda_detail::require_kernel_launch("LayerNorm forward kernel launch");
    nn_cuda_detail::synchronize("LayerNorm forward synchronization");
}

void cuda_nn_layer_norm_backward(const LayerNormBackwardRequest& request) {
    const char* operation_name = "LayerNorm backward";
    const float* input =
        nn_cuda_detail::require_native_input(request.input, operation_name);
    const float* scale =
        nn_cuda_detail::require_native_input(request.scale, operation_name);
    const float* mean =
        nn_cuda_detail::require_native_input(request.mean, operation_name);
    const float* inverse_standard_deviation =
        nn_cuda_detail::require_native_input(request.inverse_standard_deviation,
                                             operation_name);
    const float* upstream =
        nn_cuda_detail::require_native_input(request.upstream, operation_name);
    float* input_gradient = nn_cuda_detail::require_native_output(
        request.input_gradient, operation_name);
    float* scale_gradient = nn_cuda_detail::require_native_output(
        request.scale_gradient, operation_name);
    float* bias_gradient = nn_cuda_detail::require_native_output(
        request.bias_gradient, operation_name);

    layer_norm_input_backward_kernel<<<
        nn_cuda_detail::block_count_for(request.rows), kThreadsPerBlock>>>(
        input, scale, mean, inverse_standard_deviation, upstream,
        input_gradient, request.rows, request.width);
    nn_cuda_detail::require_kernel_launch(
        "LayerNorm input-backward kernel launch");
    layer_norm_parameter_backward_kernel<<<
        nn_cuda_detail::block_count_for(request.width), kThreadsPerBlock>>>(
        input, mean, inverse_standard_deviation, upstream, scale_gradient,
        bias_gradient, request.rows, request.width);
    nn_cuda_detail::require_kernel_launch(
        "LayerNorm parameter-backward kernel launch");
    nn_cuda_detail::synchronize("LayerNorm backward synchronization");
}

} // namespace riftco_transformer::backend_detail
