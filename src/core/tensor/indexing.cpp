#include "riftco_transformer/core/tensor_ops.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/dispatch.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer::tensor_ops {
namespace {

std::size_t checked_shape_numel(const Tensor::Shape& shape) {
    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "tensor dimensions must be greater than zero"
            );
        }
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor shape exceeds addressable size");
        }
        count *= dimension;
    }
    return count;
}

std::vector<std::uint32_t> checked_backend_indices(
    std::span<const std::size_t> row_indices,
    std::size_t row_count
) {
    std::vector<std::uint32_t> result;
    result.reserve(row_indices.size());
    for (const auto row : row_indices) {
        if (row >= row_count) {
            throw std::out_of_range("gather row index is outside the table");
        }
        if (row > static_cast<std::size_t>(
                      std::numeric_limits<std::uint32_t>::max()
                  )) {
            throw std::overflow_error(
                "row index exceeds the backend index range"
            );
        }
        result.push_back(static_cast<std::uint32_t>(row));
    }
    return result;
}

}  // namespace

Tensor gather_rows(
    const Tensor& table,
    std::span<const std::size_t> row_indices,
    Tensor::Shape index_shape
) {
    if (table.rank() != 2) {
        throw std::invalid_argument("gather_rows requires a rank-2 table");
    }
    if (checked_shape_numel(index_shape) != row_indices.size()) {
        throw std::invalid_argument(
            "gather row count does not match the index shape"
        );
    }

    const auto row_count = table.shape()[0];
    const auto width = table.shape()[1];
    Tensor::Shape output_shape = std::move(index_shape);
    output_shape.push_back(width);
    Tensor result(std::move(output_shape), table.backend());
    const auto backend_indices =
        checked_backend_indices(row_indices, row_count);
    backend_detail::dispatch_gather_rows(
        table.backend(),
        {
            backend_detail::tensor_storage(table),
            backend_indices,
            backend_detail::tensor_storage(result),
            row_count,
            width,
        }
    );
    return result;
}

Tensor scatter_add_rows(
    const Tensor& upstream,
    std::span<const std::size_t> row_indices,
    std::size_t row_count
) {
    if (upstream.rank() == 0) {
        throw std::invalid_argument(
            "scatter_add_rows requires an input with a final width"
        );
    }
    if (row_count == 0) {
        throw std::invalid_argument(
            "scatter_add_rows row count must be greater than zero"
        );
    }
    const auto width = upstream.shape().back();
    if (upstream.numel() / width != row_indices.size()) {
        throw std::invalid_argument(
            "scatter row count does not match the upstream shape"
        );
    }
    const auto backend_indices =
        checked_backend_indices(row_indices, row_count);
    Tensor result({row_count, width}, upstream.backend());
    backend_detail::dispatch_scatter_add_rows(
        upstream.backend(),
        {
            backend_detail::tensor_storage(upstream),
            backend_indices,
            backend_detail::tensor_storage(result),
            row_count,
            width,
        }
    );
    return result;
}

}  // namespace riftco_transformer::tensor_ops
