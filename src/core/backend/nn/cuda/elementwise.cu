#include "core/backend/nn/cuda/launch.hpp"

#include "core/backend/nn/cuda/common.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace riftco_transformer::backend_detail {
namespace {

using nn_cuda_detail::DeviceBuffer;
using nn_cuda_detail::kThreadsPerBlock;

constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
constexpr float kInverseSqrtTwoPi = 0.39894228040143267794F;

__global__ void unary_elementwise_kernel(UnaryOperation operation,
                                         const float* input, float* output,
                                         unsigned int* status,
                                         std::size_t element_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (index < element_count) {
        const float value = input[index];
        switch (operation) {
        case UnaryOperation::Negate:
            output[index] = -value;
            break;
        case UnaryOperation::Exp:
            output[index] = expf(value);
            break;
        case UnaryOperation::Log:
            if (value <= 0.0F) {
                atomicOr(status, 1U);
                output[index] = 0.0F;
            } else {
                output[index] = logf(value);
            }
            break;
        case UnaryOperation::Sqrt:
            if (value < 0.0F) {
                atomicOr(status, 1U);
                output[index] = 0.0F;
            } else {
                output[index] = sqrtf(value);
            }
            break;
        case UnaryOperation::Erf:
            output[index] = erff(value);
            break;
        }
        index += stride;
    }
}

__global__ void binary_elementwise_kernel(BinaryOperation operation,
                                          const float* left, const float* right,
                                          float* output, unsigned int* status,
                                          std::size_t element_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (index < element_count) {
        switch (operation) {
        case BinaryOperation::Add:
            output[index] = left[index] + right[index];
            break;
        case BinaryOperation::Subtract:
            output[index] = left[index] - right[index];
            break;
        case BinaryOperation::Multiply:
            output[index] = left[index] * right[index];
            break;
        case BinaryOperation::Divide:
            if (right[index] == 0.0F) {
                atomicOr(status, 1U);
                output[index] = 0.0F;
            } else {
                output[index] = left[index] / right[index];
            }
            break;
        }
        index += stride;
    }
}

__global__ void scale_kernel(const float* input, float* output,
                             std::size_t element_count, float scale) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (index < element_count) {
        output[index] = input[index] * scale;
        index += stride;
    }
}

__global__ void gelu_forward_kernel(const float* input, float* output,
                                    std::size_t element_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (index < element_count) {
        const float value = input[index];
        output[index] = 0.5F * value * (1.0F + erff(value * kInverseSqrtTwo));
        index += stride;
    }
}

__global__ void gelu_backward_kernel(const float* input, const float* upstream,
                                     float* input_gradient,
                                     std::size_t element_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (index < element_count) {
        const float value = input[index];
        const float derivative =
            0.5F * (1.0F + erff(value * kInverseSqrtTwo)) +
            value * kInverseSqrtTwoPi * expf(-0.5F * value * value);
        input_gradient[index] = upstream[index] * derivative;
        index += stride;
    }
}

} // namespace

void cuda_nn_unary_elementwise(const UnaryElementwiseRequest& request) {
    const char* operation_name = "unary elementwise";
    const float* input =
        nn_cuda_detail::require_native_input(request.input, operation_name);
    float* output =
        nn_cuda_detail::require_native_output(request.output, operation_name);
    DeviceBuffer<unsigned int> status(1);
    status.zero("unary elementwise status initialization");
    unary_elementwise_kernel<<<nn_cuda_detail::block_count_for(
                                   request.element_count),
                               kThreadsPerBlock>>>(
        request.operation, input, output, status.data(), request.element_count);
    nn_cuda_detail::require_kernel_launch("unary elementwise kernel launch");
    nn_cuda_detail::synchronize("unary elementwise synchronization");
    if (nn_cuda_detail::read_status(status, "unary elementwise status read") !=
        0U) {
        if (request.operation == UnaryOperation::Log) {
            throw std::domain_error("log requires strictly positive values");
        }
        throw std::domain_error("sqrt requires non-negative values");
    }
}

void cuda_nn_binary_elementwise(const BinaryElementwiseRequest& request) {
    const char* operation_name = "binary elementwise";
    const float* left =
        nn_cuda_detail::require_native_input(request.left, operation_name);
    const float* right =
        nn_cuda_detail::require_native_input(request.right, operation_name);
    float* output =
        nn_cuda_detail::require_native_output(request.output, operation_name);
    DeviceBuffer<unsigned int> status(1);
    status.zero("binary elementwise status initialization");
    binary_elementwise_kernel<<<
        nn_cuda_detail::block_count_for(request.element_count),
        kThreadsPerBlock>>>(request.operation, left, right, output,
                            status.data(), request.element_count);
    nn_cuda_detail::require_kernel_launch("binary elementwise kernel launch");
    nn_cuda_detail::synchronize("binary elementwise synchronization");
    if (nn_cuda_detail::read_status(status, "binary elementwise status read") !=
        0U) {
        throw std::domain_error("division by zero in tensor operation");
    }
}

void cuda_nn_scale(const ScaleRequest& request) {
    const char* operation_name = "scale";
    scale_kernel<<<nn_cuda_detail::block_count_for(request.element_count),
                   kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_output(request.output, operation_name),
        request.element_count, request.scale);
    nn_cuda_detail::require_kernel_launch("scale kernel launch");
    nn_cuda_detail::synchronize("scale synchronization");
}

void cuda_nn_gelu_forward(const GeluForwardRequest& request) {
    const char* operation_name = "GELU forward";
    gelu_forward_kernel<<<nn_cuda_detail::block_count_for(
                              request.element_count),
                          kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_output(request.output, operation_name),
        request.element_count);
    nn_cuda_detail::require_kernel_launch("GELU forward kernel launch");
    nn_cuda_detail::synchronize("GELU forward synchronization");
}

void cuda_nn_gelu_backward(const GeluBackwardRequest& request) {
    const char* operation_name = "GELU backward";
    gelu_backward_kernel<<<nn_cuda_detail::block_count_for(
                               request.element_count),
                           kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_input(request.upstream, operation_name),
        nn_cuda_detail::require_native_output(request.input_gradient,
                                              operation_name),
        request.element_count);
    nn_cuda_detail::require_kernel_launch("GELU backward kernel launch");
    nn_cuda_detail::synchronize("GELU backward synchronization");
}

} // namespace riftco_transformer::backend_detail
