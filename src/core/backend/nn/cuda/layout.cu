#include "core/backend/nn/cuda/launch.hpp"

#include "core/backend/nn/cuda/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

using nn_cuda_detail::DeviceBuffer;
using nn_cuda_detail::kThreadsPerBlock;

std::vector<std::size_t>
contiguous_strides(std::span<const std::size_t> shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index > 1; --index) {
        const std::size_t current = index - 2;
        strides[current] = strides[current + 1] * shape[current + 1];
    }
    return strides;
}

void upload_metadata(DeviceBuffer<std::size_t>& buffer,
                     const std::vector<std::size_t>& values,
                     const char* operation) {
    buffer.copy_from_host({values.data(), values.size()}, operation);
}

struct SumToMetadata {
    std::vector<std::size_t> reduction_axes;
    std::vector<std::size_t> reduction_strides;
    std::size_t reduction_count;
};

SumToMetadata make_sum_to_metadata(const SumToShapeRequest& request) {
    const std::size_t rank_offset =
        request.input_shape.size() - request.output_shape.size();
    std::vector<std::size_t> reduction_axes;
    std::vector<std::size_t> reduction_shape;
    reduction_axes.reserve(request.input_shape.size());
    reduction_shape.reserve(request.input_shape.size());
    for (std::size_t input_dimension = 0;
         input_dimension < request.input_shape.size(); ++input_dimension) {
        const bool leading_dimension = input_dimension < rank_offset;
        const bool collapsed_dimension =
            !leading_dimension &&
            request.output_shape[input_dimension - rank_offset] == 1;
        if (leading_dimension || collapsed_dimension) {
            reduction_axes.push_back(input_dimension);
            reduction_shape.push_back(request.input_shape[input_dimension]);
        }
    }
    if (request.output.size() == 0 ||
        request.input.size() % request.output.size() != 0) {
        throw std::logic_error(
            "CUDA sum-to-shape storage sizes are not reduction-compatible");
    }
    return {
        std::move(reduction_axes),
        contiguous_strides(reduction_shape),
        request.input.size() / request.output.size(),
    };
}

__global__ void copy_kernel(const float* input, float* output,
                            std::size_t element_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (index < element_count) {
        output[index] = input[index];
        index += stride;
    }
}

__global__ void permute_kernel(const float* input, float* output,
                               const std::size_t* input_strides,
                               const std::size_t* output_strides,
                               const std::size_t* axes, std::size_t rank,
                               std::size_t output_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (output_index < output_count) {
        std::size_t remainder = output_index;
        std::size_t input_index = 0;
        for (std::size_t dimension = 0; dimension < rank; ++dimension) {
            const std::size_t coordinate =
                remainder / output_strides[dimension];
            remainder %= output_strides[dimension];
            input_index += coordinate * input_strides[axes[dimension]];
        }
        output[output_index] = input[input_index];
        output_index += stride;
    }
}

__global__ void broadcast_kernel(
    const float* input, float* output, const std::size_t* input_shape,
    const std::size_t* input_strides, const std::size_t* output_strides,
    std::size_t input_rank, std::size_t output_rank, std::size_t output_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t rank_offset = output_rank - input_rank;
    while (output_index < output_count) {
        std::size_t remainder = output_index;
        std::size_t input_index = 0;
        for (std::size_t output_dimension = 0; output_dimension < output_rank;
             ++output_dimension) {
            const std::size_t coordinate =
                remainder / output_strides[output_dimension];
            remainder %= output_strides[output_dimension];
            if (output_dimension >= rank_offset) {
                const std::size_t input_dimension =
                    output_dimension - rank_offset;
                if (input_shape[input_dimension] != 1) {
                    input_index += coordinate * input_strides[input_dimension];
                }
            }
        }
        output[output_index] = input[input_index];
        output_index += stride;
    }
}

__global__ void sum_to_shape_kernel(
    const float* input, float* output, const std::size_t* input_strides,
    const std::size_t* output_shape, const std::size_t* output_strides,
    const std::size_t* reduction_axes, const std::size_t* reduction_strides,
    std::size_t input_rank, std::size_t output_rank, std::size_t reduction_rank,
    std::size_t reduction_count, std::size_t output_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t rank_offset = input_rank - output_rank;
    while (output_index < output_count) {
        std::size_t output_remainder = output_index;
        std::size_t input_base = 0;
        for (std::size_t output_dimension = 0; output_dimension < output_rank;
             ++output_dimension) {
            const std::size_t coordinate =
                output_remainder / output_strides[output_dimension];
            output_remainder %= output_strides[output_dimension];
            if (output_shape[output_dimension] != 1) {
                input_base +=
                    coordinate * input_strides[rank_offset + output_dimension];
            }
        }

        // The reduced axes are sorted by input dimension and their compact
        // strides are row-major. This visits matching input elements in the
        // same increasing flat-index order as the reference implementation.
        float total = 0.0F;
        for (std::size_t reduction_index = 0; reduction_index < reduction_count;
             ++reduction_index) {
            std::size_t reduction_remainder = reduction_index;
            std::size_t input_index = input_base;
            for (std::size_t reduction_dimension = 0;
                 reduction_dimension < reduction_rank; ++reduction_dimension) {
                const std::size_t coordinate =
                    reduction_remainder /
                    reduction_strides[reduction_dimension];
                reduction_remainder %= reduction_strides[reduction_dimension];
                input_index +=
                    coordinate *
                    input_strides[reduction_axes[reduction_dimension]];
            }
            total += input[input_index];
        }
        output[output_index] = total;
        output_index += stride;
    }
}

} // namespace

void cuda_nn_copy(const CopyRequest& request) {
    const char* operation_name = "copy";
    copy_kernel<<<nn_cuda_detail::block_count_for(request.element_count),
                  kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_output(request.output, operation_name),
        request.element_count);
    nn_cuda_detail::require_kernel_launch("copy kernel launch");
    nn_cuda_detail::synchronize("copy synchronization");
}

void cuda_nn_permute(const PermuteRequest& request) {
    const std::vector<std::size_t> input_strides =
        contiguous_strides(request.input_shape);
    std::vector<std::size_t> output_shape;
    output_shape.reserve(request.axes.size());
    for (const std::size_t axis : request.axes) {
        output_shape.push_back(request.input_shape[axis]);
    }
    const std::vector<std::size_t> output_strides =
        contiguous_strides(output_shape);
    const std::vector<std::size_t> axes(request.axes.begin(),
                                        request.axes.end());
    DeviceBuffer<std::size_t> input_stride_buffer(input_strides.size());
    DeviceBuffer<std::size_t> output_stride_buffer(output_strides.size());
    DeviceBuffer<std::size_t> axes_buffer(axes.size());
    upload_metadata(input_stride_buffer, input_strides,
                    "permute input-stride upload");
    upload_metadata(output_stride_buffer, output_strides,
                    "permute output-stride upload");
    upload_metadata(axes_buffer, axes, "permute axes upload");

    const char* operation_name = "permute";
    permute_kernel<<<nn_cuda_detail::block_count_for(request.output.size()),
                     kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_output(request.output, operation_name),
        input_stride_buffer.data(), output_stride_buffer.data(),
        axes_buffer.data(), request.input_shape.size(), request.output.size());
    nn_cuda_detail::require_kernel_launch("permute kernel launch");
    nn_cuda_detail::synchronize("permute synchronization");
}

void cuda_nn_broadcast(const BroadcastRequest& request) {
    const std::vector<std::size_t> input_shape(request.input_shape.begin(),
                                               request.input_shape.end());
    const std::vector<std::size_t> input_strides =
        contiguous_strides(request.input_shape);
    const std::vector<std::size_t> output_strides =
        contiguous_strides(request.output_shape);
    DeviceBuffer<std::size_t> input_shape_buffer(input_shape.size());
    DeviceBuffer<std::size_t> input_stride_buffer(input_strides.size());
    DeviceBuffer<std::size_t> output_stride_buffer(output_strides.size());
    upload_metadata(input_shape_buffer, input_shape,
                    "broadcast input-shape upload");
    upload_metadata(input_stride_buffer, input_strides,
                    "broadcast input-stride upload");
    upload_metadata(output_stride_buffer, output_strides,
                    "broadcast output-stride upload");

    const char* operation_name = "broadcast";
    broadcast_kernel<<<nn_cuda_detail::block_count_for(request.output.size()),
                       kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_output(request.output, operation_name),
        input_shape_buffer.data(), input_stride_buffer.data(),
        output_stride_buffer.data(), request.input_shape.size(),
        request.output_shape.size(), request.output.size());
    nn_cuda_detail::require_kernel_launch("broadcast kernel launch");
    nn_cuda_detail::synchronize("broadcast synchronization");
}

void cuda_nn_sum_to_shape(const SumToShapeRequest& request) {
    const std::vector<std::size_t> input_strides =
        contiguous_strides(request.input_shape);
    const std::vector<std::size_t> output_shape(request.output_shape.begin(),
                                                request.output_shape.end());
    const std::vector<std::size_t> output_strides =
        contiguous_strides(request.output_shape);
    const SumToMetadata reduction = make_sum_to_metadata(request);

    DeviceBuffer<std::size_t> input_stride_buffer(input_strides.size());
    DeviceBuffer<std::size_t> output_shape_buffer(output_shape.size());
    DeviceBuffer<std::size_t> output_stride_buffer(output_strides.size());
    DeviceBuffer<std::size_t> reduction_axis_buffer(
        reduction.reduction_axes.size());
    DeviceBuffer<std::size_t> reduction_stride_buffer(
        reduction.reduction_strides.size());
    upload_metadata(input_stride_buffer, input_strides,
                    "sum-to-shape input-stride upload");
    upload_metadata(output_shape_buffer, output_shape,
                    "sum-to-shape output-shape upload");
    upload_metadata(output_stride_buffer, output_strides,
                    "sum-to-shape output-stride upload");
    upload_metadata(reduction_axis_buffer, reduction.reduction_axes,
                    "sum-to-shape reduction-axis upload");
    upload_metadata(reduction_stride_buffer, reduction.reduction_strides,
                    "sum-to-shape reduction-stride upload");

    const char* operation_name = "sum-to-shape";
    sum_to_shape_kernel<<<nn_cuda_detail::block_count_for(
                              request.output.size()),
                          kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_output(request.output, operation_name),
        input_stride_buffer.data(), output_shape_buffer.data(),
        output_stride_buffer.data(), reduction_axis_buffer.data(),
        reduction_stride_buffer.data(), request.input_shape.size(),
        request.output_shape.size(), reduction.reduction_axes.size(),
        reduction.reduction_count, request.output.size());
    nn_cuda_detail::require_kernel_launch("sum-to-shape kernel launch");
    nn_cuda_detail::synchronize("sum-to-shape synchronization");
}

} // namespace riftco_transformer::backend_detail
