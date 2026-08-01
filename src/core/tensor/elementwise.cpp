#include "riftco_transformer/core/tensor_ops.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/dispatch.hpp"
#include "core/tensor/detail/validation.hpp"

namespace riftco_transformer::tensor_ops {

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
    detail::require_same_shape(left, right);
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
    detail::require_same_shape(left, right);
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
    detail::require_same_shape(left, right);
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
    detail::require_same_shape(numerator, denominator);
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
    detail::require_same_shape(input, upstream);
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

}  // namespace riftco_transformer::tensor_ops
