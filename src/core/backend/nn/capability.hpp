#pragma once

#include "core/backend/nn/contracts.hpp"

namespace riftco_transformer::backend_detail {

class ElementwiseCapability {
  public:
    virtual ~ElementwiseCapability() = default;

    virtual void
    unary_elementwise(const UnaryElementwiseRequest& request) const = 0;
    virtual void
    binary_elementwise(const BinaryElementwiseRequest& request) const = 0;
    virtual void scale(const ScaleRequest& request) const = 0;
    virtual void gelu_forward(const GeluForwardRequest& request) const = 0;
    virtual void gelu_backward(const GeluBackwardRequest& request) const = 0;
};

class ReductionCapability {
  public:
    virtual ~ReductionCapability() = default;

    virtual void reduce(const ReductionRequest& request) const = 0;
};

class LayoutCapability {
  public:
    virtual ~LayoutCapability() = default;

    virtual void copy(const CopyRequest& request) const = 0;
    virtual void permute(const PermuteRequest& request) const = 0;
    virtual void broadcast(const BroadcastRequest& request) const = 0;
    virtual void sum_to_shape(const SumToShapeRequest& request) const = 0;
};

class SoftmaxCapability {
  public:
    virtual ~SoftmaxCapability() = default;

    virtual void
    softmax_forward(const SoftmaxForwardRequest& request) const = 0;
    virtual void
    softmax_backward(const SoftmaxBackwardRequest& request) const = 0;
    virtual void causal_softmax_forward(
        const CausalSoftmaxForwardRequest& request) const = 0;
    virtual void causal_softmax_backward(
        const CausalSoftmaxBackwardRequest& request) const = 0;
};

class IndexingCapability {
  public:
    virtual ~IndexingCapability() = default;

    virtual void gather_rows(const GatherRowsRequest& request) const = 0;
    virtual void
    scatter_add_rows(const ScatterAddRowsRequest& request) const = 0;
};

class NormalizationCapability {
  public:
    virtual ~NormalizationCapability() = default;

    virtual void
    layer_norm_forward(const LayerNormForwardRequest& request) const = 0;
    virtual void
    layer_norm_backward(const LayerNormBackwardRequest& request) const = 0;
};

class LossCapability {
  public:
    virtual ~LossCapability() = default;

    virtual void
    cross_entropy_forward(const CrossEntropyForwardRequest& request) const = 0;
};

} // namespace riftco_transformer::backend_detail
