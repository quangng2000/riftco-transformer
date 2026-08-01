#include "core/backend/nn/cuda/launch.hpp"

#include "core/backend/nn/cuda/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace riftco_transformer::backend_detail {
namespace {

using nn_cuda_detail::kThreadsPerBlock;

__global__ void reduce_axis_kernel(const float* input, float* output,
                                   std::size_t outer, std::size_t width,
                                   std::size_t inner,
                                   ReductionOperation operation) {
    const std::size_t slice_count = outer * inner;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t slice =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (slice < slice_count) {
        const std::size_t outer_coordinate = slice / inner;
        const std::size_t inner_coordinate = slice % inner;
        const std::size_t base =
            outer_coordinate * width * inner + inner_coordinate;
        float total = 0.0F;
        for (std::size_t item = 0; item < width; ++item) {
            total += input[base + item * inner];
        }
        if (operation == ReductionOperation::Mean) {
            total /= static_cast<float>(width);
        }
        output[slice] = total;
        slice += stride;
    }
}

} // namespace

void cuda_nn_reduce(const ReductionRequest& request) {
    const char* operation_name = "axis reduction";
    const std::size_t slice_count =
        request.dimensions.outer * request.dimensions.inner;
    reduce_axis_kernel<<<nn_cuda_detail::block_count_for(slice_count),
                         kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.input, operation_name),
        nn_cuda_detail::require_native_output(request.output, operation_name),
        request.dimensions.outer, request.dimensions.width,
        request.dimensions.inner, request.operation);
    nn_cuda_detail::require_kernel_launch("axis reduction kernel launch");
    nn_cuda_detail::synchronize("axis reduction synchronization");
}

} // namespace riftco_transformer::backend_detail
