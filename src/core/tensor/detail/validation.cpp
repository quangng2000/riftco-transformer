#include "core/tensor/detail/validation.hpp"

#include <stdexcept>

namespace riftco_transformer::tensor_ops::detail {

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

}  // namespace riftco_transformer::tensor_ops::detail
