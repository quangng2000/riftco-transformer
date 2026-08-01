#pragma once

#include "core/backend/nn/contracts.hpp"
#include "riftco_transformer/core/backend.hpp"

namespace riftco_transformer::backend_detail {

void dispatch_unary_elementwise(ExecutionBackend backend,
                                const UnaryElementwiseRequest& request);
void dispatch_binary_elementwise(ExecutionBackend backend,
                                 const BinaryElementwiseRequest& request);
void dispatch_scale(ExecutionBackend backend, const ScaleRequest& request);
void dispatch_gelu_forward(ExecutionBackend backend,
                           const GeluForwardRequest& request);
void dispatch_gelu_backward(ExecutionBackend backend,
                            const GeluBackwardRequest& request);
void dispatch_reduction(ExecutionBackend backend,
                        const ReductionRequest& request);
void dispatch_copy(ExecutionBackend backend, const CopyRequest& request);
void dispatch_permute(ExecutionBackend backend, const PermuteRequest& request);
void dispatch_broadcast(ExecutionBackend backend,
                        const BroadcastRequest& request);
void dispatch_sum_to_shape(ExecutionBackend backend,
                           const SumToShapeRequest& request);
void dispatch_softmax_forward(ExecutionBackend backend,
                              const SoftmaxForwardRequest& request);
void dispatch_softmax_backward(ExecutionBackend backend,
                               const SoftmaxBackwardRequest& request);
void dispatch_causal_softmax_forward(
    ExecutionBackend backend, const CausalSoftmaxForwardRequest& request);
void dispatch_causal_softmax_backward(
    ExecutionBackend backend, const CausalSoftmaxBackwardRequest& request);
void dispatch_gather_rows(ExecutionBackend backend,
                          const GatherRowsRequest& request);
void dispatch_scatter_add_rows(ExecutionBackend backend,
                               const ScatterAddRowsRequest& request);
void dispatch_layer_norm_forward(ExecutionBackend backend,
                                 const LayerNormForwardRequest& request);
void dispatch_layer_norm_backward(ExecutionBackend backend,
                                  const LayerNormBackwardRequest& request);
void dispatch_cross_entropy_forward(ExecutionBackend backend,
                                    const CrossEntropyForwardRequest& request);

} // namespace riftco_transformer::backend_detail
