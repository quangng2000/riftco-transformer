#include "riftco_transformer/core/tensor_ops.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/dispatch.hpp"
#include "core/tensor/detail/validation.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace riftco_transformer::tensor_ops {
namespace {

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

Tensor softmax(const Tensor& value, std::size_t axis) {
    const auto layout = detail::axis_layout(value, axis);
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
    detail::require_same_shape(probabilities, upstream);
    const auto layout = detail::axis_layout(probabilities, axis);
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
    detail::require_same_shape(probabilities, upstream);
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

}  // namespace riftco_transformer::tensor_ops
