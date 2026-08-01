#include "core/autograd/detail/node.hpp"

#include "riftco_transformer/core/tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace riftco_transformer {
namespace {

Variable constant_like(const Variable& value, float scalar) {
    return Variable(
        Tensor::full(value.value().shape(), scalar, value.value().backend()),
        false
    );
}

void require_differentiable_sqrt_domain(const Tensor& value) {
    for (const float element : value.data()) {
        if (element <= 0.0F) {
            throw std::domain_error(
                "sqrt autograd requires strictly positive values"
            );
        }
    }
}

Tensor::Shape swap_last_two_axes(std::size_t rank) {
    Tensor::Shape axes;
    axes.reserve(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
        axes.push_back(axis);
    }
    std::swap(axes[rank - 2], axes[rank - 1]);
    return axes;
}

}  // namespace


Variable operator+(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    return Variable::from_operation(
        tensor_ops::add(left.value(), right.value()),
        {left_node, right_node},
        [left_node, right_node](const Tensor& upstream) {
            Variable::accumulate_gradient(left_node, upstream);
            Variable::accumulate_gradient(right_node, upstream);
        }
    );
}

Variable operator-(const Variable& value) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::negate(value.value()),
        {input_node},
        [input_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::negate(upstream)
            );
        }
    );
}

Variable operator-(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    return Variable::from_operation(
        tensor_ops::subtract(left.value(), right.value()),
        {left_node, right_node},
        [left_node, right_node](const Tensor& upstream) {
            Variable::accumulate_gradient(left_node, upstream);
            Variable::accumulate_gradient(
                right_node,
                tensor_ops::negate(upstream)
            );
        }
    );
}

Variable operator*(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    return Variable::from_operation(
        tensor_ops::multiply(left.value(), right.value()),
        {left_node, right_node},
        [left_node, right_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                left_node,
                tensor_ops::multiply(upstream, right_node->value)
            );
            Variable::accumulate_gradient(
                right_node,
                tensor_ops::multiply(upstream, left_node->value)
            );
        }
    );
}

Variable operator/(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    return Variable::from_operation(
        tensor_ops::divide(left.value(), right.value()),
        {left_node, right_node},
        [left_node, right_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                left_node,
                tensor_ops::divide(upstream, right_node->value)
            );
            const Tensor denominator_squared =
                tensor_ops::multiply(right_node->value, right_node->value);
            const Tensor numerator =
                tensor_ops::multiply(upstream, left_node->value);
            Variable::accumulate_gradient(
                right_node,
                tensor_ops::negate(
                    tensor_ops::divide(numerator, denominator_squared)
                )
            );
        }
    );
}

Variable operator+(const Variable& value, float scalar) {
    return value + constant_like(value, scalar);
}

Variable operator+(float scalar, const Variable& value) {
    return value + scalar;
}

Variable operator-(const Variable& value, float scalar) {
    return value - constant_like(value, scalar);
}

Variable operator-(float scalar, const Variable& value) {
    return constant_like(value, scalar) - value;
}

Variable operator*(const Variable& value, float scalar) {
    return value * constant_like(value, scalar);
}

Variable operator*(float scalar, const Variable& value) {
    return value * scalar;
}

Variable operator/(const Variable& value, float scalar) {
    return value / constant_like(value, scalar);
}

Variable operator/(float scalar, const Variable& value) {
    return constant_like(value, scalar) / value;
}

Variable matmul(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    const auto backend = left.value().backend();
    return Variable::from_operation(
        tensor_ops::matmul(left.value(), right.value(), backend),
        {left_node, right_node},
        [left_node, right_node, backend](const Tensor& upstream) {
            const auto right_transpose_axes =
                swap_last_two_axes(right_node->value.rank());
            Variable::accumulate_gradient(
                left_node,
                tensor_ops::matmul(
                    upstream,
                    tensor_ops::permute(
                        right_node->value,
                        right_transpose_axes
                    ),
                    backend
                )
            );
            const auto left_transpose_axes =
                swap_last_two_axes(left_node->value.rank());
            Variable::accumulate_gradient(
                right_node,
                tensor_ops::matmul(
                    tensor_ops::permute(left_node->value, left_transpose_axes),
                    upstream,
                    backend
                )
            );
        }
    );
}

Variable permute(const Variable& value, Tensor::Shape axes) {
    Tensor::Shape inverse_axes(axes.size(), 0);
    for (std::size_t output_axis = 0; output_axis < axes.size();
         ++output_axis) {
        const auto input_axis = axes[output_axis];
        if (input_axis < inverse_axes.size()) {
            inverse_axes[input_axis] = output_axis;
        }
    }

    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::permute(value.value(), std::move(axes)),
        {input_node},
        [input_node, inverse_axes](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::permute(upstream, inverse_axes)
            );
        }
    );
}

Variable sum(const Variable& value) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::sum(value.value()),
        {input_node},
        [input_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::broadcast_to(upstream, input_node->value.shape())
            );
        }
    );
}

Variable sum(const Variable& value, std::size_t axis, bool keep_dimensions) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::sum(value.value(), axis, keep_dimensions),
        {input_node},
        [input_node, axis, keep_dimensions](const Tensor& upstream) {
            Tensor broadcast_source = upstream;
            if (!keep_dimensions) {
                auto kept_shape = input_node->value.shape();
                kept_shape[axis] = 1;
                broadcast_source = upstream.reshape(std::move(kept_shape));
            }
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::broadcast_to(
                    broadcast_source,
                    input_node->value.shape()
                )
            );
        }
    );
}

Variable mean(const Variable& value) {
    const auto element_count = value.value().numel();
    const auto input_shape = value.value().shape();
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::mean(value.value()),
        {input_node},
        [input_node, input_shape, element_count](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::broadcast_to(
                    tensor_ops::scale(
                        upstream,
                        1.0F / static_cast<float>(element_count)
                    ),
                    input_shape
                )
            );
        }
    );
}

Variable mean(const Variable& value, std::size_t axis, bool keep_dimensions) {
    if (axis >= value.value().rank()) {
        throw std::out_of_range("mean axis is outside tensor rank");
    }
    const auto input_shape = value.value().shape();
    const auto width = value.value().shape()[axis];
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::mean(value.value(), axis, keep_dimensions),
        {input_node},
        [input_node, input_shape, axis, keep_dimensions, width](
            const Tensor& upstream
        ) {
            Tensor broadcast_source =
                tensor_ops::scale(upstream, 1.0F / static_cast<float>(width));
            if (!keep_dimensions) {
                auto kept_shape = input_shape;
                kept_shape[axis] = 1;
                broadcast_source =
                    broadcast_source.reshape(std::move(kept_shape));
            }
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::broadcast_to(broadcast_source, input_shape)
            );
        }
    );
}

Variable exp(const Variable& value) {
    const Tensor output = tensor_ops::exp(value.value());
    const Tensor derivative = output;
    const auto input_node = value.node_;
    return Variable::from_operation(
        output,
        {input_node},
        [input_node, derivative](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::multiply(upstream, derivative)
            );
        }
    );
}

Variable log(const Variable& value) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::log(value.value()),
        {input_node},
        [input_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::divide(upstream, input_node->value)
            );
        }
    );
}

Variable sqrt(const Variable& value) {
    require_differentiable_sqrt_domain(value.value());
    const Tensor output = tensor_ops::sqrt(value.value());
    const Tensor derivative_denominator = tensor_ops::scale(output, 2.0F);
    const auto input_node = value.node_;
    return Variable::from_operation(
        output,
        {input_node},
        [input_node, derivative_denominator](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::divide(upstream, derivative_denominator)
            );
        }
    );
}

Variable erf(const Variable& value) {
    const Tensor input_squared =
        tensor_ops::multiply(value.value(), value.value());
    const Tensor derivative = tensor_ops::scale(
        tensor_ops::exp(tensor_ops::negate(input_squared)),
        2.0F / std::sqrt(std::numbers::pi_v<float>)
    );
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::erf(value.value()),
        {input_node},
        [input_node, derivative](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::multiply(upstream, derivative)
            );
        }
    );
}

Variable reshape(const Variable& value, Tensor::Shape new_shape) {
    const auto original_shape = value.value().shape();
    const auto input_node = value.node_;
    return Variable::from_operation(
        value.value().reshape(std::move(new_shape)),
        {input_node},
        [input_node, original_shape](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                upstream.reshape(original_shape)
            );
        }
    );
}

Variable transpose_2d(const Variable& value) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::transpose_2d(value.value()),
        {input_node},
        [input_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::transpose_2d(upstream)
            );
        }
    );
}

Variable broadcast_to(const Variable& value, Tensor::Shape output_shape) {
    const auto input_shape = value.value().shape();
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::broadcast_to(value.value(), std::move(output_shape)),
        {input_node},
        [input_node, input_shape](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::sum_to_shape(upstream, input_shape)
            );
        }
    );
}

Variable gather_rows(
    const Variable& table,
    std::span<const std::size_t> row_indices,
    Tensor::Shape index_shape
) {
    const Tensor output = tensor_ops::gather_rows(
        table.value(),
        row_indices,
        std::move(index_shape)
    );
    const auto table_shape = table.value().shape();
    const std::vector<std::size_t> owned_indices(
        row_indices.begin(),
        row_indices.end()
    );
    const auto table_node = table.node_;

    return Variable::from_operation(
        output,
        {table_node},
        [table_node, table_shape, owned_indices](const Tensor& upstream) {
            Variable::accumulate_gradient(
                table_node,
                tensor_ops::scatter_add_rows(
                    upstream,
                    owned_indices,
                    table_shape[0]
                )
            );
        }
    );
}

}  // namespace riftco_transformer
