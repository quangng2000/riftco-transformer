#pragma once

#include "core/backend/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace riftco_transformer::backend_detail {

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

} // namespace riftco_transformer::backend_detail
