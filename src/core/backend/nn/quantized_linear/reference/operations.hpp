#pragma once

#include "core/backend/nn/quantized_linear/contracts.hpp"

namespace riftco_transformer::backend_detail {

void reference_quantized_linear_forward(
    const QuantizedLinearForwardRequest& request
);

void reference_quantized_linear_input_backward(
    const QuantizedLinearInputBackwardRequest& request
);

}  // namespace riftco_transformer::backend_detail
