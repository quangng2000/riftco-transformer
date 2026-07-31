#pragma once

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/tensor.hpp"

#include <cstddef>
#include <span>

namespace riftco_transformer::tensor_ops {

[[nodiscard]] Tensor negate(const Tensor& value);
[[nodiscard]] Tensor add(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor subtract(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor multiply(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor divide(const Tensor& numerator, const Tensor& denominator);
[[nodiscard]] Tensor scale(const Tensor& value, float scalar);

// Matmul accepts [..., rows, shared] and [..., shared, columns]. Inputs must
// share a storage backend, leading batch dimensions must be identical, and
// batch broadcasting is intentionally unsupported. The default overload runs
// on the input backend; the explicit overload may stage through another
// implementation while returning an output on the input storage backend.
[[nodiscard]] Tensor matmul(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor
matmul(const Tensor& left, const Tensor& right, ExecutionBackend backend);
[[nodiscard]] Tensor permute(const Tensor& value, Tensor::Shape axes);
[[nodiscard]] Tensor transpose_2d(const Tensor& value);
[[nodiscard]] Tensor
broadcast_to(const Tensor& value, Tensor::Shape output_shape);
[[nodiscard]] Tensor
sum_to_shape(const Tensor& value, Tensor::Shape output_shape);
[[nodiscard]] Tensor gather_rows(
    const Tensor& table,
    std::span<const std::size_t> row_indices,
    Tensor::Shape index_shape
);
// Adjoint of gather_rows. Repeated indices are accumulated, rather than
// overwritten, into a new [row_count, width] table.
[[nodiscard]] Tensor scatter_add_rows(
    const Tensor& upstream,
    std::span<const std::size_t> row_indices,
    std::size_t row_count
);

[[nodiscard]] Tensor sum(const Tensor& value);
[[nodiscard]] Tensor
sum(const Tensor& value, std::size_t axis, bool keep_dimensions = false);
[[nodiscard]] Tensor mean(const Tensor& value);
[[nodiscard]] Tensor
mean(const Tensor& value, std::size_t axis, bool keep_dimensions = false);

[[nodiscard]] Tensor exp(const Tensor& value);
[[nodiscard]] Tensor log(const Tensor& value);
[[nodiscard]] Tensor sqrt(const Tensor& value);
[[nodiscard]] Tensor erf(const Tensor& value);
[[nodiscard]] Tensor gelu(const Tensor& value);
[[nodiscard]] Tensor gelu_backward(const Tensor& input, const Tensor& upstream);

// Softmax supports any axis and permits negative infinity for future masks.
[[nodiscard]] Tensor softmax(const Tensor& value, std::size_t axis);
[[nodiscard]] Tensor softmax_backward(
    const Tensor& probabilities,
    const Tensor& upstream,
    std::size_t axis
);
// Fused causal softmax for square rank-four attention scores
// [batch, head, query_time, key_time]. Future probabilities are written as
// exact zero. score_scale is applied before the softmax and by its VJP.
[[nodiscard]] Tensor
causal_softmax(const Tensor& scores, float score_scale = 1.0F);
[[nodiscard]] Tensor causal_softmax_backward(
    const Tensor& probabilities,
    const Tensor& upstream,
    float score_scale = 1.0F
);

}  // namespace riftco_transformer::tensor_ops
