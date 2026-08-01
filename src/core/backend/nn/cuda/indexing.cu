#include "core/backend/nn/cuda/launch.hpp"

#include "core/backend/nn/cuda/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

using nn_cuda_detail::DeviceBuffer;
using nn_cuda_detail::kThreadsPerBlock;

struct ScatterGrouping {
    std::vector<std::size_t> row_offsets;
    std::vector<std::size_t> grouped_positions;
};

ScatterGrouping make_scatter_grouping(const ScatterAddRowsRequest& request) {
    if (request.row_count == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(
            "CUDA scatter row-offset table exceeds addressable storage");
    }
    ScatterGrouping grouping{
        std::vector<std::size_t>(request.row_count + 1, 0),
        std::vector<std::size_t>(request.row_indices.size(), 0),
    };
    for (const std::uint32_t row : request.row_indices) {
        ++grouping.row_offsets[static_cast<std::size_t>(row) + 1];
    }
    for (std::size_t row = 0; row < request.row_count; ++row) {
        grouping.row_offsets[row + 1] += grouping.row_offsets[row];
    }

    std::vector<std::size_t> next = grouping.row_offsets;
    for (std::size_t position = 0; position < request.row_indices.size();
         ++position) {
        const std::size_t row =
            static_cast<std::size_t>(request.row_indices[position]);
        grouping.grouped_positions[next[row]] = position;
        ++next[row];
    }
    return grouping;
}

__global__ void gather_rows_kernel(const float* table,
                                   const std::uint32_t* row_indices,
                                   float* output, std::size_t output_count,
                                   std::size_t width) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (output_index < output_count) {
        const std::size_t position = output_index / width;
        const std::size_t column = output_index % width;
        const std::size_t row = static_cast<std::size_t>(row_indices[position]);
        output[output_index] = table[row * width + column];
        output_index += stride;
    }
}

__global__ void scatter_add_rows_kernel(const float* upstream,
                                        const std::size_t* row_offsets,
                                        const std::size_t* grouped_positions,
                                        float* table_gradient,
                                        std::size_t output_count,
                                        std::size_t width) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (output_index < output_count) {
        const std::size_t row = output_index / width;
        const std::size_t column = output_index % width;
        float total = 0.0F;
        for (std::size_t grouped_index = row_offsets[row];
             grouped_index < row_offsets[row + 1]; ++grouped_index) {
            const std::size_t position = grouped_positions[grouped_index];
            total += upstream[position * width + column];
        }
        table_gradient[output_index] = total;
        output_index += stride;
    }
}

} // namespace

void cuda_nn_gather_rows(const GatherRowsRequest& request) {
    DeviceBuffer<std::uint32_t> indices(request.row_indices.size());
    indices.copy_from_host(request.row_indices,
                           "embedding gather index upload");
    const char* operation_name = "embedding gather";
    gather_rows_kernel<<<nn_cuda_detail::block_count_for(request.output.size()),
                         kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.table, operation_name),
        indices.data(),
        nn_cuda_detail::require_native_output(request.output, operation_name),
        request.output.size(), request.width);
    nn_cuda_detail::require_kernel_launch("embedding gather kernel launch");
    nn_cuda_detail::synchronize("embedding gather synchronization");
}

void cuda_nn_scatter_add_rows(const ScatterAddRowsRequest& request) {
    const ScatterGrouping grouping = make_scatter_grouping(request);
    DeviceBuffer<std::size_t> row_offsets(grouping.row_offsets.size());
    DeviceBuffer<std::size_t> grouped_positions(
        grouping.grouped_positions.size());
    row_offsets.copy_from_host(
        {
            grouping.row_offsets.data(),
            grouping.row_offsets.size(),
        },
        "embedding scatter row-offset upload");
    grouped_positions.copy_from_host(
        {
            grouping.grouped_positions.data(),
            grouping.grouped_positions.size(),
        },
        "embedding scatter position upload");

    const char* operation_name = "embedding scatter-add";
    scatter_add_rows_kernel<<<nn_cuda_detail::block_count_for(
                                  request.table_gradient.size()),
                              kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.upstream, operation_name),
        row_offsets.data(), grouped_positions.data(),
        nn_cuda_detail::require_native_output(request.table_gradient,
                                              operation_name),
        request.table_gradient.size(), request.width);
    nn_cuda_detail::require_kernel_launch(
        "embedding scatter-add kernel launch");
    nn_cuda_detail::synchronize("embedding scatter-add synchronization");
}

} // namespace riftco_transformer::backend_detail
