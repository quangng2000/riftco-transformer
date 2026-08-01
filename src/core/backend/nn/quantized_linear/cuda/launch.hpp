#pragma once

#include "core/backend/nn/quantized_linear/contracts.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace riftco_transformer::backend_detail {

[[nodiscard]] std::unique_ptr<QuantizedWeightStorage>
cuda_make_nf4_weight_storage(std::vector<std::uint8_t> packed_codes,
                             Nf4ScaleStorageData scales);

void cuda_quantized_linear_forward(
    const QuantizedLinearForwardRequest &request);

void cuda_quantized_linear_input_backward(
    const QuantizedLinearInputBackwardRequest &request);

} // namespace riftco_transformer::backend_detail
