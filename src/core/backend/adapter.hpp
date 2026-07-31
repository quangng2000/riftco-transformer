#pragma once

#include "core/backend/attention/capability.hpp"
#include "core/backend/attention/dispatch.hpp"
#include "storage.hpp"
#include "riftco_transformer/core/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace riftco_transformer::backend_detail {

struct MatmulDimensions {
    std::size_t batch_count;
    std::size_t rows;
    std::size_t shared;
    std::size_t columns;
};

struct MatmulRequest {
    const TensorStorage& left;
    const TensorStorage& right;
    TensorStorage& output;
    MatmulDimensions dimensions;
};

enum class UnaryOperation : std::uint8_t {
    Negate,
    Exp,
    Log,
    Sqrt,
    Erf,
};

enum class BinaryOperation : std::uint8_t {
    Add,
    Subtract,
    Multiply,
    Divide,
};

enum class ReductionOperation : std::uint8_t {
    Sum,
    Mean,
};

struct AxisDimensions {
    std::size_t outer;
    std::size_t width;
    std::size_t inner;
};

struct UnaryElementwiseRequest {
    UnaryOperation operation;
    const TensorStorage& input;
    TensorStorage& output;
    std::size_t element_count;
};

struct BinaryElementwiseRequest {
    BinaryOperation operation;
    const TensorStorage& left;
    const TensorStorage& right;
    TensorStorage& output;
    std::size_t element_count;
};

struct ScaleRequest {
    const TensorStorage& input;
    TensorStorage& output;
    std::size_t element_count;
    float scale;
};

struct GeluForwardRequest {
    const TensorStorage& input;
    TensorStorage& output;
    std::size_t element_count;
};

struct GeluBackwardRequest {
    const TensorStorage& input;
    const TensorStorage& upstream;
    TensorStorage& input_gradient;
    std::size_t element_count;
};

struct ReductionRequest {
    const TensorStorage& input;
    TensorStorage& output;
    AxisDimensions dimensions;
    ReductionOperation operation;
};

struct CopyRequest {
    const TensorStorage& input;
    TensorStorage& output;
    std::size_t element_count;
};

struct PermuteRequest {
    const TensorStorage& input;
    TensorStorage& output;
    std::span<const std::size_t> input_shape;
    std::span<const std::size_t> axes;
};

struct BroadcastRequest {
    const TensorStorage& input;
    TensorStorage& output;
    std::span<const std::size_t> input_shape;
    std::span<const std::size_t> output_shape;
};

struct SumToShapeRequest {
    const TensorStorage& input;
    TensorStorage& output;
    std::span<const std::size_t> input_shape;
    std::span<const std::size_t> output_shape;
};

struct SoftmaxForwardRequest {
    const TensorStorage& input;
    TensorStorage& probabilities;
    AxisDimensions dimensions;
};

struct SoftmaxBackwardRequest {
    const TensorStorage& probabilities;
    const TensorStorage& upstream;
    TensorStorage& input_gradient;
    AxisDimensions dimensions;
};

struct CausalSoftmaxForwardRequest {
    const TensorStorage& scores;
    TensorStorage& probabilities;
    std::size_t batch;
    std::size_t heads;
    std::size_t time;
    float score_scale;
};

struct CausalSoftmaxBackwardRequest {
    const TensorStorage& probabilities;
    const TensorStorage& upstream;
    TensorStorage& score_gradient;
    std::size_t batch;
    std::size_t heads;
    std::size_t time;
    float score_scale;
};

struct GatherRowsRequest {
    const TensorStorage& table;
    std::span<const std::uint32_t> row_indices;
    TensorStorage& output;
    std::size_t row_count;
    std::size_t width;
};

struct ScatterAddRowsRequest {
    const TensorStorage& upstream;
    std::span<const std::uint32_t> row_indices;
    TensorStorage& table_gradient;
    std::size_t row_count;
    std::size_t width;
};

struct LayerNormForwardRequest {
    const TensorStorage& input;
    const TensorStorage& scale;
    const TensorStorage& bias;
    TensorStorage& output;
    TensorStorage& mean;
    TensorStorage& inverse_standard_deviation;
    std::size_t rows;
    std::size_t width;
    float epsilon;
};

struct LayerNormBackwardRequest {
    const TensorStorage& input;
    const TensorStorage& scale;
    const TensorStorage& mean;
    const TensorStorage& inverse_standard_deviation;
    const TensorStorage& upstream;
    TensorStorage& input_gradient;
    TensorStorage& scale_gradient;
    TensorStorage& bias_gradient;
    std::size_t rows;
    std::size_t width;
};

struct CrossEntropyForwardRequest {
    const TensorStorage& logits;
    std::span<const std::uint32_t> targets;
    TensorStorage& loss;
    TensorStorage& base_gradient;
    std::size_t positions;
    std::size_t classes;
};

struct AdamTensorUpdate {
    const TensorStorage& value;
    const TensorStorage& gradient;
    const TensorStorage& first_moment;
    const TensorStorage& second_moment;
    TensorStorage& next_value;
    TensorStorage& next_first_moment;
    TensorStorage& next_second_moment;
};

struct AdamUpdateRequest {
    std::span<const AdamTensorUpdate> tensors;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    double clip_scale;
    double first_correction;
    double second_correction;
};

class StorageCapability {
public:
    virtual ~StorageCapability() = default;

    [[nodiscard]] virtual std::unique_ptr<TensorStorage> make_storage(
        std::size_t element_count,
        float fill_value
    ) const = 0;
    [[nodiscard]] virtual std::unique_ptr<TensorStorage> make_storage(
        std::vector<float> values
    ) const = 0;
};

class MatmulCapability {
public:
    virtual ~MatmulCapability() = default;

    // Shape validation belongs to tensor_ops. Adapters receive contiguous,
    // correctly sized storage and must finish writing output before returning.
    virtual void matmul(const MatmulRequest& request) const = 0;
};

class ElementwiseCapability {
public:
    virtual ~ElementwiseCapability() = default;

    virtual void unary_elementwise(
        const UnaryElementwiseRequest& request
    ) const = 0;
    virtual void binary_elementwise(
        const BinaryElementwiseRequest& request
    ) const = 0;
    virtual void scale(const ScaleRequest& request) const = 0;
    virtual void gelu_forward(
        const GeluForwardRequest& request
    ) const = 0;
    virtual void gelu_backward(
        const GeluBackwardRequest& request
    ) const = 0;
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
    virtual void sum_to_shape(
        const SumToShapeRequest& request
    ) const = 0;
};

class SoftmaxCapability {
public:
    virtual ~SoftmaxCapability() = default;

    virtual void softmax_forward(
        const SoftmaxForwardRequest& request
    ) const = 0;
    virtual void softmax_backward(
        const SoftmaxBackwardRequest& request
    ) const = 0;
    virtual void causal_softmax_forward(
        const CausalSoftmaxForwardRequest& request
    ) const = 0;
    virtual void causal_softmax_backward(
        const CausalSoftmaxBackwardRequest& request
    ) const = 0;
};

class IndexingCapability {
public:
    virtual ~IndexingCapability() = default;

    virtual void gather_rows(
        const GatherRowsRequest& request
    ) const = 0;
    virtual void scatter_add_rows(
        const ScatterAddRowsRequest& request
    ) const = 0;
};

class NormalizationCapability {
public:
    virtual ~NormalizationCapability() = default;

    virtual void layer_norm_forward(
        const LayerNormForwardRequest& request
    ) const = 0;
    virtual void layer_norm_backward(
        const LayerNormBackwardRequest& request
    ) const = 0;
};

class LossCapability {
public:
    virtual ~LossCapability() = default;

    virtual void cross_entropy_forward(
        const CrossEntropyForwardRequest& request
    ) const = 0;
};

class AdamCapability {
public:
    virtual ~AdamCapability() = default;

    // Writes only next-state buffers. A backend must complete the entire batch
    // or throw before returning; callers commit only after success.
    virtual void adam_update(const AdamUpdateRequest& request) const = 0;
};

// Internal Adapter facade. Capability-specific base interfaces keep storage,
// linear algebra, and optimizer growth independent.
class BackendAdapter
    : public StorageCapability,
      public MatmulCapability,
      public ElementwiseCapability,
      public ReductionCapability,
      public LayoutCapability,
      public SoftmaxCapability,
      public IndexingCapability,
      public NormalizationCapability,
      public LossCapability,
      public AttentionCapability,
      public AdamCapability {
public:
    ~BackendAdapter() override = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool is_available() const noexcept = 0;
};

// Implemented by the platform-neutral CPU adapter.
[[nodiscard]] const BackendAdapter& cpu_backend_adapter() noexcept;

// Implemented by metal_adapter.mm on Apple and by the unavailable stub on
// other platforms.
[[nodiscard]] const BackendAdapter& metal_backend_adapter() noexcept;

// The registry is deliberately closed and internal for now. Public callers
// select a stable ExecutionBackend value rather than owning adapter objects.
[[nodiscard]] const BackendAdapter* find_backend_adapter(
    ExecutionBackend backend
) noexcept;

[[nodiscard]] std::unique_ptr<TensorStorage> make_tensor_storage(
    ExecutionBackend backend,
    std::size_t element_count,
    float fill_value
);
[[nodiscard]] std::unique_ptr<TensorStorage> make_tensor_storage(
    ExecutionBackend backend,
    std::vector<float> values
);

void dispatch_matmul(
    ExecutionBackend backend,
    const TensorStorage& left,
    const TensorStorage& right,
    TensorStorage& output,
    MatmulDimensions dimensions
);

void dispatch_unary_elementwise(
    ExecutionBackend backend,
    const UnaryElementwiseRequest& request
);
void dispatch_binary_elementwise(
    ExecutionBackend backend,
    const BinaryElementwiseRequest& request
);
void dispatch_scale(
    ExecutionBackend backend,
    const ScaleRequest& request
);
void dispatch_gelu_forward(
    ExecutionBackend backend,
    const GeluForwardRequest& request
);
void dispatch_gelu_backward(
    ExecutionBackend backend,
    const GeluBackwardRequest& request
);
void dispatch_reduction(
    ExecutionBackend backend,
    const ReductionRequest& request
);
void dispatch_copy(
    ExecutionBackend backend,
    const CopyRequest& request
);
void dispatch_permute(
    ExecutionBackend backend,
    const PermuteRequest& request
);
void dispatch_broadcast(
    ExecutionBackend backend,
    const BroadcastRequest& request
);
void dispatch_sum_to_shape(
    ExecutionBackend backend,
    const SumToShapeRequest& request
);
void dispatch_softmax_forward(
    ExecutionBackend backend,
    const SoftmaxForwardRequest& request
);
void dispatch_softmax_backward(
    ExecutionBackend backend,
    const SoftmaxBackwardRequest& request
);
void dispatch_causal_softmax_forward(
    ExecutionBackend backend,
    const CausalSoftmaxForwardRequest& request
);
void dispatch_causal_softmax_backward(
    ExecutionBackend backend,
    const CausalSoftmaxBackwardRequest& request
);
void dispatch_gather_rows(
    ExecutionBackend backend,
    const GatherRowsRequest& request
);
void dispatch_scatter_add_rows(
    ExecutionBackend backend,
    const ScatterAddRowsRequest& request
);
void dispatch_layer_norm_forward(
    ExecutionBackend backend,
    const LayerNormForwardRequest& request
);
void dispatch_layer_norm_backward(
    ExecutionBackend backend,
    const LayerNormBackwardRequest& request
);
void dispatch_cross_entropy_forward(
    ExecutionBackend backend,
    const CrossEntropyForwardRequest& request
);
void dispatch_adam_update(
    ExecutionBackend backend,
    const AdamUpdateRequest& request
);

}  // namespace riftco_transformer::backend_detail
