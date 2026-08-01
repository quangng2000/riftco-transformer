#pragma once

#include "riftco_transformer/core/tensor.hpp"

#include "core/backend/nn/contracts.hpp"

#include <cstddef>

namespace riftco_transformer::tensor_ops::detail {

void require_same_shape(const Tensor& left, const Tensor& right);

[[nodiscard]] backend_detail::AxisDimensions
axis_layout(const Tensor& value, std::size_t axis);

}  // namespace riftco_transformer::tensor_ops::detail
