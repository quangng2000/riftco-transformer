#pragma once

#include "core/backend/nn/quantized_linear/storage.hpp"
#include "core/backend/storage.hpp"

#include <cstddef>

namespace riftco_transformer::backend_detail {

struct QuantizedLinearDimensions {
    std::size_t rows;
    std::size_t input_width;
    std::size_t output_width;
    std::size_t block_size;
};

// Weight storage is row-major [output_width, input_width]. The input and output
// are flattened views of [rows, input_width] and [rows, output_width]. Bias and
// floating-point LoRA branches deliberately remain outside this primitive.
struct QuantizedLinearForwardRequest {
    const TensorStorage& input;
    const QuantizedWeightStorage& weight;
    TensorStorage& output;
    QuantizedLinearDimensions dimensions;
};

// Computes only dL/dInput. A frozen QuantizedWeight is not an autograd input,
// so the contract has no weight-gradient output by construction.
struct QuantizedLinearInputBackwardRequest {
    const TensorStorage& upstream;
    const QuantizedWeightStorage& weight;
    TensorStorage& input_gradient;
    QuantizedLinearDimensions dimensions;
};

}  // namespace riftco_transformer::backend_detail
