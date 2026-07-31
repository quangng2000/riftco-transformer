#pragma once

#include "adapter.hpp"

namespace transformer_lab::backend_detail {

// Objective-C++ implementation boundary for neural-network Metal kernels.
// The public backend adapter remains the single facade; this split keeps the
// kernel/runtime details independent from storage, matmul, and Adam.
void metal_nn_unary_elementwise(
    const UnaryElementwiseRequest& request
);
void metal_nn_binary_elementwise(
    const BinaryElementwiseRequest& request
);
void metal_nn_scale(const ScaleRequest& request);
void metal_nn_gelu_forward(const GeluForwardRequest& request);
void metal_nn_gelu_backward(const GeluBackwardRequest& request);
void metal_nn_reduce(const ReductionRequest& request);
void metal_nn_copy(const CopyRequest& request);
void metal_nn_permute(const PermuteRequest& request);
void metal_nn_broadcast(const BroadcastRequest& request);
void metal_nn_sum_to_shape(const SumToShapeRequest& request);
void metal_nn_softmax_forward(
    const SoftmaxForwardRequest& request
);
void metal_nn_softmax_backward(
    const SoftmaxBackwardRequest& request
);
void metal_nn_causal_softmax_forward(
    const CausalSoftmaxForwardRequest& request
);
void metal_nn_causal_softmax_backward(
    const CausalSoftmaxBackwardRequest& request
);
void metal_nn_gather_rows(const GatherRowsRequest& request);
void metal_nn_scatter_add_rows(
    const ScatterAddRowsRequest& request
);
void metal_nn_layer_norm_forward(
    const LayerNormForwardRequest& request
);
void metal_nn_layer_norm_backward(
    const LayerNormBackwardRequest& request
);
void metal_nn_cross_entropy_forward(
    const CrossEntropyForwardRequest& request
);
void metal_nn_materialized_causal_attention_forward(
    const MaterializedCausalAttentionForwardRequest& request
);
void metal_nn_materialized_causal_attention_context_backward(
    const MaterializedCausalAttentionContextBackwardRequest& request
);
void metal_nn_materialized_causal_attention_probabilities_backward(
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
);
void metal_nn_flash_causal_attention_forward(
    const FlashCausalAttentionForwardRequest& request
);
void metal_nn_flash_causal_attention_backward(
    const FlashCausalAttentionBackwardRequest& request
);
void metal_nn_paged_decode_attention_forward(
    const PagedDecodeAttentionForwardRequest& request
);

}  // namespace transformer_lab::backend_detail
