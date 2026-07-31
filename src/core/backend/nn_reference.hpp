#pragma once

#include "adapter.hpp"

namespace riftco_transformer::backend_detail {

void nn_reference_unary_elementwise(
    const UnaryElementwiseRequest& request
);
void nn_reference_binary_elementwise(
    const BinaryElementwiseRequest& request
);
void nn_reference_scale(const ScaleRequest& request);
void nn_reference_gelu_forward(const GeluForwardRequest& request);
void nn_reference_gelu_backward(const GeluBackwardRequest& request);
void nn_reference_reduce(const ReductionRequest& request);
void nn_reference_copy(const CopyRequest& request);
void nn_reference_permute(const PermuteRequest& request);
void nn_reference_broadcast(const BroadcastRequest& request);
void nn_reference_sum_to_shape(const SumToShapeRequest& request);
void nn_reference_softmax_forward(
    const SoftmaxForwardRequest& request
);
void nn_reference_softmax_backward(
    const SoftmaxBackwardRequest& request
);
void nn_reference_causal_softmax_forward(
    const CausalSoftmaxForwardRequest& request
);
void nn_reference_causal_softmax_backward(
    const CausalSoftmaxBackwardRequest& request
);
void nn_reference_gather_rows(const GatherRowsRequest& request);
void nn_reference_scatter_add_rows(
    const ScatterAddRowsRequest& request
);
void nn_reference_layer_norm_forward(
    const LayerNormForwardRequest& request
);
void nn_reference_layer_norm_backward(
    const LayerNormBackwardRequest& request
);
void nn_reference_cross_entropy_forward(
    const CrossEntropyForwardRequest& request
);

}  // namespace riftco_transformer::backend_detail
