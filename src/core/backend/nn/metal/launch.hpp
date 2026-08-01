#pragma once

#include "core/backend/nn/contracts.hpp"

namespace riftco_transformer::backend_detail {

// Objective-C++ execution boundary for the native Metal NN kernels. The
// shared Metal runtime owns the device, command queue, and compiled library;
// this header keeps the NN capability surface independent of that runtime.
void metal_nn_unary_elementwise(const UnaryElementwiseRequest& request);
void metal_nn_binary_elementwise(const BinaryElementwiseRequest& request);
void metal_nn_scale(const ScaleRequest& request);
void metal_nn_gelu_forward(const GeluForwardRequest& request);
void metal_nn_gelu_backward(const GeluBackwardRequest& request);
void metal_nn_reduce(const ReductionRequest& request);
void metal_nn_copy(const CopyRequest& request);
void metal_nn_permute(const PermuteRequest& request);
void metal_nn_broadcast(const BroadcastRequest& request);
void metal_nn_sum_to_shape(const SumToShapeRequest& request);
void metal_nn_softmax_forward(const SoftmaxForwardRequest& request);
void metal_nn_softmax_backward(const SoftmaxBackwardRequest& request);
void metal_nn_causal_softmax_forward(
    const CausalSoftmaxForwardRequest& request);
void metal_nn_causal_softmax_backward(
    const CausalSoftmaxBackwardRequest& request);
void metal_nn_gather_rows(const GatherRowsRequest& request);
void metal_nn_scatter_add_rows(const ScatterAddRowsRequest& request);
void metal_nn_layer_norm_forward(const LayerNormForwardRequest& request);
void metal_nn_layer_norm_backward(const LayerNormBackwardRequest& request);
void metal_nn_cross_entropy_forward(const CrossEntropyForwardRequest& request);

} // namespace riftco_transformer::backend_detail
