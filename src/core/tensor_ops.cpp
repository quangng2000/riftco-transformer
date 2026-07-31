#include "transformer_lab/core/tensor_ops.hpp"

#include "backend/adapter.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace transformer_lab::tensor_ops {
namespace {

void require_same_shape(const Tensor& left, const Tensor& right) {
    if (left.backend() != right.backend()) {
        throw std::invalid_argument(
            "tensor operations require inputs on the same backend"
        );
    }
    if (left.shape() != right.shape()) {
        throw std::invalid_argument(
            "elementwise operations require identical tensor shapes"
        );
    }
}

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

Tensor::Shape reduction_shape(
    const Tensor::Shape& input_shape,
    std::size_t axis,
    bool keep_dimensions
) {
    if (axis >= input_shape.size()) {
        throw std::out_of_range("reduction axis is outside tensor rank");
    }

    Tensor::Shape output_shape = input_shape;
    if (keep_dimensions) {
        output_shape[axis] = 1;
    } else {
        output_shape.erase(
            output_shape.begin() + static_cast<std::ptrdiff_t>(axis)
        );
    }
    return output_shape;
}

backend_detail::AxisDimensions
axis_layout(const Tensor& value, std::size_t axis) {
    if (axis >= value.rank()) {
        throw std::out_of_range("operation axis is outside tensor rank");
    }

    const auto width = value.shape()[axis];
    const auto inner = value.strides()[axis];
    return {
        value.numel() / (width * inner),
        width,
        inner,
    };
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

struct CausalSoftmaxDimensions {
    std::size_t batch;
    std::size_t heads;
    std::size_t time;
};

CausalSoftmaxDimensions
causal_softmax_dimensions(const Tensor& value) {
    if (value.rank() != 4 || value.shape()[2] != value.shape()[3]) {
        throw std::invalid_argument(
            "causal softmax requires square rank-four attention scores"
        );
    }
    return {
        value.shape()[0],
        value.shape()[1],
        value.shape()[2],
    };
}

}  // namespace

Tensor negate(const Tensor& value) {
    Tensor result(value.shape(), value.backend());
    backend_detail::dispatch_unary_elementwise(
        value.backend(),
        {
            backend_detail::UnaryOperation::Negate,
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.numel(),
        }
    );
    return result;
}

Tensor add(const Tensor& left, const Tensor& right) {
    require_same_shape(left, right);
    Tensor result(left.shape(), left.backend());
    backend_detail::dispatch_binary_elementwise(
        left.backend(),
        {
            backend_detail::BinaryOperation::Add,
            backend_detail::tensor_storage(left),
            backend_detail::tensor_storage(right),
            backend_detail::tensor_storage(result),
            left.numel(),
        }
    );
    return result;
}

Tensor subtract(const Tensor& left, const Tensor& right) {
    require_same_shape(left, right);
    Tensor result(left.shape(), left.backend());
    backend_detail::dispatch_binary_elementwise(
        left.backend(),
        {
            backend_detail::BinaryOperation::Subtract,
            backend_detail::tensor_storage(left),
            backend_detail::tensor_storage(right),
            backend_detail::tensor_storage(result),
            left.numel(),
        }
    );
    return result;
}

Tensor multiply(const Tensor& left, const Tensor& right) {
    require_same_shape(left, right);
    Tensor result(left.shape(), left.backend());
    backend_detail::dispatch_binary_elementwise(
        left.backend(),
        {
            backend_detail::BinaryOperation::Multiply,
            backend_detail::tensor_storage(left),
            backend_detail::tensor_storage(right),
            backend_detail::tensor_storage(result),
            left.numel(),
        }
    );
    return result;
}

Tensor divide(const Tensor& numerator, const Tensor& denominator) {
    require_same_shape(numerator, denominator);
    Tensor result(numerator.shape(), numerator.backend());
    backend_detail::dispatch_binary_elementwise(
        numerator.backend(),
        {
            backend_detail::BinaryOperation::Divide,
            backend_detail::tensor_storage(numerator),
            backend_detail::tensor_storage(denominator),
            backend_detail::tensor_storage(result),
            numerator.numel(),
        }
    );
    return result;
}

Tensor scale(const Tensor& value, float scalar) {
    Tensor result(value.shape(), value.backend());
    backend_detail::dispatch_scale(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.numel(),
            scalar,
        }
    );
    return result;
}

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

Tensor permute(const Tensor& value, Tensor::Shape axes) {
    if (axes.size() != value.rank()) {
        throw std::invalid_argument(
            "permutation must provide one axis per tensor dimension"
        );
    }

    std::vector<bool> seen(value.rank(), false);
    Tensor::Shape output_shape;
    output_shape.reserve(value.rank());
    for (const std::size_t axis : axes) {
        if (axis >= value.rank()) {
            throw std::out_of_range("permutation axis is outside tensor rank");
        }
        if (seen[axis]) {
            throw std::invalid_argument("permutation axes must be unique");
        }
        seen[axis] = true;
        output_shape.push_back(value.shape()[axis]);
    }

    Tensor result(std::move(output_shape), value.backend());
    backend_detail::dispatch_permute(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.shape(),
            axes,
        }
    );
    return result;
}

Tensor transpose_2d(const Tensor& value) {
    if (value.rank() != 2) {
        throw std::invalid_argument("transpose_2d requires a rank-2 tensor");
    }
    return permute(value, {1, 0});
}

Tensor broadcast_to(const Tensor& value, Tensor::Shape output_shape) {
    if (value.rank() > output_shape.size()) {
        throw std::invalid_argument(
            "broadcast output rank cannot be smaller than input rank"
        );
    }

    const auto rank_offset = output_shape.size() - value.rank();
    for (std::size_t input_dimension = 0; input_dimension < value.rank();
         ++input_dimension) {
        const auto output_dimension = rank_offset + input_dimension;
        if (value.shape()[input_dimension] != 1 &&
            value.shape()[input_dimension] != output_shape[output_dimension]) {
            throw std::invalid_argument(
                "tensor shape is not compatible with broadcast output"
            );
        }
    }

    Tensor result(std::move(output_shape), value.backend());
    backend_detail::dispatch_broadcast(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.shape(),
            result.shape(),
        }
    );
    return result;
}

Tensor sum_to_shape(const Tensor& value, Tensor::Shape output_shape) {
    if (output_shape.size() > value.rank()) {
        throw std::invalid_argument(
            "sum-to-shape output rank cannot exceed input rank"
        );
    }

    const auto rank_offset = value.rank() - output_shape.size();
    for (std::size_t output_dimension = 0;
         output_dimension < output_shape.size();
         ++output_dimension) {
        const auto input_dimension = rank_offset + output_dimension;
        if (output_shape[output_dimension] != 1 &&
            output_shape[output_dimension] != value.shape()[input_dimension]) {
            throw std::invalid_argument(
                "sum-to-shape output is not broadcast-compatible"
            );
        }
    }

    Tensor result(std::move(output_shape), value.backend());
    backend_detail::dispatch_sum_to_shape(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.shape(),
            result.shape(),
        }
    );
    return result;
}

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

Tensor sum(const Tensor& value) {
    Tensor result(Tensor::Shape{}, value.backend());
    backend_detail::dispatch_reduction(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            {1, value.numel(), 1},
            backend_detail::ReductionOperation::Sum,
        }
    );
    return result;
}

Tensor sum(const Tensor& value, std::size_t axis, bool keep_dimensions) {
    Tensor result(
        reduction_shape(value.shape(), axis, keep_dimensions),
        value.backend()
    );
    backend_detail::dispatch_reduction(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            axis_layout(value, axis),
            backend_detail::ReductionOperation::Sum,
        }
    );
    return result;
}

Tensor mean(const Tensor& value) {
    Tensor result(Tensor::Shape{}, value.backend());
    backend_detail::dispatch_reduction(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            {1, value.numel(), 1},
            backend_detail::ReductionOperation::Mean,
        }
    );
    return result;
}

Tensor mean(const Tensor& value, std::size_t axis, bool keep_dimensions) {
    Tensor result(
        reduction_shape(value.shape(), axis, keep_dimensions),
        value.backend()
    );
    backend_detail::dispatch_reduction(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            axis_layout(value, axis),
            backend_detail::ReductionOperation::Mean,
        }
    );
    return result;
}

Tensor exp(const Tensor& value) {
    Tensor result(value.shape(), value.backend());
    backend_detail::dispatch_unary_elementwise(
        value.backend(),
        {
            backend_detail::UnaryOperation::Exp,
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.numel(),
        }
    );
    return result;
}

Tensor log(const Tensor& value) {
    Tensor result(value.shape(), value.backend());
    backend_detail::dispatch_unary_elementwise(
        value.backend(),
        {
            backend_detail::UnaryOperation::Log,
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.numel(),
        }
    );
    return result;
}

Tensor sqrt(const Tensor& value) {
    Tensor result(value.shape(), value.backend());
    backend_detail::dispatch_unary_elementwise(
        value.backend(),
        {
            backend_detail::UnaryOperation::Sqrt,
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.numel(),
        }
    );
    return result;
}

Tensor erf(const Tensor& value) {
    Tensor result(value.shape(), value.backend());
    backend_detail::dispatch_unary_elementwise(
        value.backend(),
        {
            backend_detail::UnaryOperation::Erf,
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.numel(),
        }
    );
    return result;
}

Tensor gelu(const Tensor& value) {
    Tensor result(value.shape(), value.backend());
    backend_detail::dispatch_gelu_forward(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            value.numel(),
        }
    );
    return result;
}

Tensor gelu_backward(const Tensor& input, const Tensor& upstream) {
    require_same_shape(input, upstream);
    Tensor result(input.shape(), input.backend());
    backend_detail::dispatch_gelu_backward(
        input.backend(),
        {
            backend_detail::tensor_storage(input),
            backend_detail::tensor_storage(upstream),
            backend_detail::tensor_storage(result),
            input.numel(),
        }
    );
    return result;
}

Tensor softmax(const Tensor& value, std::size_t axis) {
    const auto layout = axis_layout(value, axis);
    Tensor result(value.shape(), value.backend());
    backend_detail::dispatch_softmax_forward(
        value.backend(),
        {
            backend_detail::tensor_storage(value),
            backend_detail::tensor_storage(result),
            layout,
        }
    );
    return result;
}

Tensor softmax_backward(
    const Tensor& probabilities,
    const Tensor& upstream,
    std::size_t axis
) {
    require_same_shape(probabilities, upstream);
    const auto layout = axis_layout(probabilities, axis);
    Tensor result(probabilities.shape(), probabilities.backend());
    backend_detail::dispatch_softmax_backward(
        probabilities.backend(),
        {
            backend_detail::tensor_storage(probabilities),
            backend_detail::tensor_storage(upstream),
            backend_detail::tensor_storage(result),
            layout,
        }
    );
    return result;
}

Tensor causal_softmax(const Tensor& scores, float score_scale) {
    if (!std::isfinite(score_scale) || score_scale <= 0.0F) {
        throw std::invalid_argument(
            "causal softmax scale must be finite and positive"
        );
    }
    const auto dimensions = causal_softmax_dimensions(scores);
    Tensor result(scores.shape(), scores.backend());
    backend_detail::dispatch_causal_softmax_forward(
        scores.backend(),
        {
            backend_detail::tensor_storage(scores),
            backend_detail::tensor_storage(result),
            dimensions.batch,
            dimensions.heads,
            dimensions.time,
            score_scale,
        }
    );
    return result;
}

Tensor causal_softmax_backward(
    const Tensor& probabilities,
    const Tensor& upstream,
    float score_scale
) {
    require_same_shape(probabilities, upstream);
    if (!std::isfinite(score_scale) || score_scale <= 0.0F) {
        throw std::invalid_argument(
            "causal softmax scale must be finite and positive"
        );
    }
    const auto dimensions = causal_softmax_dimensions(probabilities);
    Tensor result(probabilities.shape(), probabilities.backend());
    backend_detail::dispatch_causal_softmax_backward(
        probabilities.backend(),
        {
            backend_detail::tensor_storage(probabilities),
            backend_detail::tensor_storage(upstream),
            backend_detail::tensor_storage(result),
            dimensions.batch,
            dimensions.heads,
            dimensions.time,
            score_scale,
        }
    );
    return result;
}

}  // namespace transformer_lab::tensor_ops
