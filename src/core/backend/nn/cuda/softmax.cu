#include "core/backend/nn/cuda/launch.hpp"

#include "core/backend/nn/cuda/common.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace riftco_transformer::backend_detail {
namespace {

using nn_cuda_detail::DeviceBuffer;
using nn_cuda_detail::kThreadsPerBlock;

constexpr unsigned int kInvalidValue = 1U;
constexpr unsigned int kNoFiniteValue = 2U;

__device__ std::size_t axis_offset(std::size_t slice, std::size_t coordinate,
                                   std::size_t width, std::size_t inner) {
    const std::size_t outer_coordinate = slice / inner;
    const std::size_t inner_coordinate = slice % inner;
    return (outer_coordinate * width + coordinate) * inner + inner_coordinate;
}

__global__ void softmax_forward_kernel(const float* input, float* probabilities,
                                       unsigned int* status, std::size_t outer,
                                       std::size_t width, std::size_t inner) {
    const std::size_t slice_count = outer * inner;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t slice =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (slice < slice_count) {
        float maximum = -CUDART_INF_F;
        bool invalid_value = false;
        for (std::size_t item = 0; item < width; ++item) {
            const float value = input[axis_offset(slice, item, width, inner)];
            if (isnan(value) || value == CUDART_INF_F) {
                invalid_value = true;
            }
            if (maximum < value) {
                maximum = value;
            }
        }
        if (invalid_value || maximum == -CUDART_INF_F) {
            atomicOr(status, invalid_value ? kInvalidValue : kNoFiniteValue);
            for (std::size_t item = 0; item < width; ++item) {
                probabilities[axis_offset(slice, item, width, inner)] = 0.0F;
            }
            slice += stride;
            continue;
        }

        double denominator = 0.0;
        for (std::size_t item = 0; item < width; ++item) {
            const std::size_t index = axis_offset(slice, item, width, inner);
            const double exponential =
                exp(static_cast<double>(input[index] - maximum));
            probabilities[index] = static_cast<float>(exponential);
            denominator += exponential;
        }
        if (!(denominator > 0.0) || !isfinite(denominator)) {
            atomicOr(status, kNoFiniteValue);
            for (std::size_t item = 0; item < width; ++item) {
                probabilities[axis_offset(slice, item, width, inner)] = 0.0F;
            }
            slice += stride;
            continue;
        }
        for (std::size_t item = 0; item < width; ++item) {
            const std::size_t index = axis_offset(slice, item, width, inner);
            probabilities[index] = static_cast<float>(
                static_cast<double>(probabilities[index]) / denominator);
        }
        slice += stride;
    }
}

__global__ void softmax_backward_kernel(const float* probabilities,
                                        const float* upstream,
                                        float* input_gradient,
                                        std::size_t outer, std::size_t width,
                                        std::size_t inner) {
    const std::size_t slice_count = outer * inner;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t slice =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (slice < slice_count) {
        double weighted_upstream = 0.0;
        for (std::size_t item = 0; item < width; ++item) {
            const std::size_t index = axis_offset(slice, item, width, inner);
            weighted_upstream += static_cast<double>(probabilities[index]) *
                                 static_cast<double>(upstream[index]);
        }
        const float weighted = static_cast<float>(weighted_upstream);
        for (std::size_t item = 0; item < width; ++item) {
            const std::size_t index = axis_offset(slice, item, width, inner);
            input_gradient[index] =
                probabilities[index] * (upstream[index] - weighted);
        }
        slice += stride;
    }
}

__global__ void
causal_softmax_forward_kernel(const float* scores, float* probabilities,
                              unsigned int* status, std::size_t row_count,
                              std::size_t time, float score_scale) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (row < row_count) {
        const std::size_t query = row % time;
        const std::size_t base = row * time;
        float maximum = -CUDART_INF_F;
        bool invalid_value = false;
        for (std::size_t key = 0; key <= query; ++key) {
            const float value = scores[base + key] * score_scale;
            if (isnan(value) || value == CUDART_INF_F) {
                invalid_value = true;
            }
            if (maximum < value) {
                maximum = value;
            }
        }
        for (std::size_t key = query + 1; key < time; ++key) {
            probabilities[base + key] = 0.0F;
        }
        if (invalid_value || maximum == -CUDART_INF_F) {
            atomicOr(status, invalid_value ? kInvalidValue : kNoFiniteValue);
            for (std::size_t key = 0; key <= query; ++key) {
                probabilities[base + key] = 0.0F;
            }
            row += stride;
            continue;
        }

        double denominator = 0.0;
        for (std::size_t key = 0; key <= query; ++key) {
            const double exponential = exp(static_cast<double>(
                scores[base + key] * score_scale - maximum));
            probabilities[base + key] = static_cast<float>(exponential);
            denominator += exponential;
        }
        if (!(denominator > 0.0) || !isfinite(denominator)) {
            atomicOr(status, kNoFiniteValue);
            for (std::size_t key = 0; key <= query; ++key) {
                probabilities[base + key] = 0.0F;
            }
            row += stride;
            continue;
        }
        for (std::size_t key = 0; key <= query; ++key) {
            probabilities[base + key] = static_cast<float>(
                static_cast<double>(probabilities[base + key]) / denominator);
        }
        row += stride;
    }
}

__global__ void causal_softmax_backward_kernel(
    const float* probabilities, const float* upstream, float* score_gradient,
    std::size_t row_count, std::size_t time, float score_scale) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (row < row_count) {
        const std::size_t query = row % time;
        const std::size_t base = row * time;
        double weighted_upstream = 0.0;
        for (std::size_t key = 0; key <= query; ++key) {
            weighted_upstream +=
                static_cast<double>(probabilities[base + key]) *
                static_cast<double>(upstream[base + key]);
        }
        const float weighted = static_cast<float>(weighted_upstream);
        for (std::size_t key = 0; key <= query; ++key) {
            score_gradient[base + key] = probabilities[base + key] *
                                         (upstream[base + key] - weighted) *
                                         score_scale;
        }
        for (std::size_t key = query + 1; key < time; ++key) {
            score_gradient[base + key] = 0.0F;
        }
        row += stride;
    }
}

void require_valid_softmax_status(unsigned int status) {
    if ((status & kInvalidValue) != 0U) {
        throw std::domain_error("softmax rejects NaN and positive infinity");
    }
    if (status != 0U) {
        throw std::domain_error("softmax requires one finite value per slice");
    }
}

} // namespace

void cuda_nn_softmax_forward(const SoftmaxForwardRequest& request) {
    const char* operation_name = "softmax forward";
    DeviceBuffer<unsigned int> status(1);
    status.zero("softmax status initialization");
    const std::size_t slice_count =
        request.dimensions.outer * request.dimensions.inner;
    softmax_forward_kernel<<<nn_cuda_detail::block_count_for(slice_count),
                             kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_output(request.probabilities,
                                              operation_name),
        status.data(), request.dimensions.outer, request.dimensions.width,
        request.dimensions.inner);
    nn_cuda_detail::require_kernel_launch("softmax forward kernel launch");
    nn_cuda_detail::synchronize("softmax forward synchronization");
    require_valid_softmax_status(
        nn_cuda_detail::read_status(status, "softmax status read"));
}

void cuda_nn_softmax_backward(const SoftmaxBackwardRequest& request) {
    const char* operation_name = "softmax backward";
    const std::size_t slice_count =
        request.dimensions.outer * request.dimensions.inner;
    softmax_backward_kernel<<<nn_cuda_detail::block_count_for(slice_count),
                              kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.probabilities,
                                             operation_name),
        nn_cuda_detail::require_native_input(request.upstream, operation_name),
        nn_cuda_detail::require_native_output(request.input_gradient,
                                              operation_name),
        request.dimensions.outer, request.dimensions.width,
        request.dimensions.inner);
    nn_cuda_detail::require_kernel_launch("softmax backward kernel launch");
    nn_cuda_detail::synchronize("softmax backward synchronization");
}

void cuda_nn_causal_softmax_forward(
    const CausalSoftmaxForwardRequest& request) {
    const char* operation_name = "causal softmax forward";
    const std::size_t row_count = request.batch * request.heads * request.time;
    DeviceBuffer<unsigned int> status(1);
    status.zero("causal softmax status initialization");
    causal_softmax_forward_kernel<<<nn_cuda_detail::block_count_for(row_count),
                                    kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.scores, operation_name),
        nn_cuda_detail::require_native_output(request.probabilities,
                                              operation_name),
        status.data(), row_count, request.time, request.score_scale);
    nn_cuda_detail::require_kernel_launch(
        "causal softmax forward kernel launch");
    nn_cuda_detail::synchronize("causal softmax forward synchronization");
    require_valid_softmax_status(
        nn_cuda_detail::read_status(status, "causal softmax status read"));
}

void cuda_nn_causal_softmax_backward(
    const CausalSoftmaxBackwardRequest& request) {
    const char* operation_name = "causal softmax backward";
    const std::size_t row_count = request.batch * request.heads * request.time;
    causal_softmax_backward_kernel<<<nn_cuda_detail::block_count_for(row_count),
                                     kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.probabilities,
                                             operation_name),
        nn_cuda_detail::require_native_input(request.upstream, operation_name),
        nn_cuda_detail::require_native_output(request.score_gradient,
                                              operation_name),
        row_count, request.time, request.score_scale);
    nn_cuda_detail::require_kernel_launch(
        "causal softmax backward kernel launch");
    nn_cuda_detail::synchronize("causal softmax backward synchronization");
}

} // namespace riftco_transformer::backend_detail
