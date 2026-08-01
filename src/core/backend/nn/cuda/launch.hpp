#pragma once

#include "core/backend/nn/contracts.hpp"

namespace riftco_transformer::backend_detail {

void cuda_nn_unary_elementwise(const UnaryElementwiseRequest& request);
void cuda_nn_binary_elementwise(const BinaryElementwiseRequest& request);
void cuda_nn_scale(const ScaleRequest& request);
void cuda_nn_gelu_forward(const GeluForwardRequest& request);
void cuda_nn_gelu_backward(const GeluBackwardRequest& request);
void cuda_nn_reduce(const ReductionRequest& request);
void cuda_nn_copy(const CopyRequest& request);
void cuda_nn_permute(const PermuteRequest& request);
void cuda_nn_broadcast(const BroadcastRequest& request);
void cuda_nn_sum_to_shape(const SumToShapeRequest& request);
void cuda_nn_softmax_forward(const SoftmaxForwardRequest& request);
void cuda_nn_softmax_backward(const SoftmaxBackwardRequest& request);
void cuda_nn_causal_softmax_forward(const CausalSoftmaxForwardRequest& request);
void cuda_nn_causal_softmax_backward(
    const CausalSoftmaxBackwardRequest& request);
void cuda_nn_gather_rows(const GatherRowsRequest& request);
void cuda_nn_scatter_add_rows(const ScatterAddRowsRequest& request);
void cuda_nn_layer_norm_forward(const LayerNormForwardRequest& request);
void cuda_nn_layer_norm_backward(const LayerNormBackwardRequest& request);
void cuda_nn_cross_entropy_forward(const CrossEntropyForwardRequest& request);

} // namespace riftco_transformer::backend_detail
