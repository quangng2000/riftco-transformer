#include "riftco_transformer/core/tensor_ops.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/dispatch.hpp"
#include "core/tensor/detail/validation.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::tensor_ops {
namespace {

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

}  // namespace

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
            detail::axis_layout(value, axis),
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
            detail::axis_layout(value, axis),
            backend_detail::ReductionOperation::Mean,
        }
    );
    return result;
}

}  // namespace riftco_transformer::tensor_ops
