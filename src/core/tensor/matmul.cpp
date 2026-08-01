#include "riftco_transformer/core/tensor_ops.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/dispatch.hpp"

#include <stdexcept>
#include <utility>

namespace riftco_transformer::tensor_ops {

Tensor matmul(const Tensor& left, const Tensor& right) {
    return matmul(left, right, left.backend());
}

Tensor
matmul(const Tensor& left, const Tensor& right, ExecutionBackend backend) {
    if (left.backend() != right.backend()) {
        throw std::invalid_argument(
            "matmul inputs must use the same storage backend"
        );
    }
    if (left.rank() < 2 || right.rank() < 2) {
        throw std::invalid_argument(
            "matmul requires tensors with rank at least two"
        );
    }
    if (left.rank() != right.rank()) {
        throw std::invalid_argument("matmul inputs must have identical ranks");
    }

    const auto matrix_dimension = left.rank() - 2;
    for (std::size_t dimension = 0; dimension < matrix_dimension; ++dimension) {
        if (left.shape()[dimension] != right.shape()[dimension]) {
            throw std::invalid_argument(
                "matmul leading batch dimensions must be identical"
            );
        }
    }

    const auto rows = left.shape()[matrix_dimension];
    const auto shared = left.shape()[matrix_dimension + 1];
    const auto right_shared = right.shape()[matrix_dimension];
    const auto columns = right.shape()[matrix_dimension + 1];
    if (shared != right_shared) {
        throw std::invalid_argument(
            "matmul inner dimensions must be identical"
        );
    }

    Tensor::Shape output_shape = left.shape();
    output_shape[matrix_dimension + 1] = columns;
    Tensor result(std::move(output_shape), left.backend());
    const auto batch_count = left.numel() / (rows * shared);
    backend_detail::dispatch_matmul(
        backend,
        backend_detail::tensor_storage(left),
        backend_detail::tensor_storage(right),
        backend_detail::tensor_storage(result),
        {
            batch_count,
            rows,
            shared,
            columns,
        }
    );
    return result;
}

}  // namespace riftco_transformer::tensor_ops
