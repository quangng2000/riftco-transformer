#pragma once

#include "core/backend/nn/quantized_linear/contracts.hpp"
#include "riftco_transformer/core/backend.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace riftco_transformer::backend_detail {

[[nodiscard]] std::unique_ptr<QuantizedWeightStorage>
make_nf4_weight_storage(
    ExecutionBackend backend,
    std::vector<std::uint8_t> packed_codes,
    Nf4ScaleStorageData scales
);

void dispatch_quantized_linear_forward(
    ExecutionBackend backend,
    const QuantizedLinearForwardRequest& request
);

void dispatch_quantized_linear_input_backward(
    ExecutionBackend backend,
    const QuantizedLinearInputBackwardRequest& request
);

}  // namespace riftco_transformer::backend_detail
