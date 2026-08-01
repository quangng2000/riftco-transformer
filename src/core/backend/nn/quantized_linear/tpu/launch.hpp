#pragma once

#include "core/backend/nn/quantized_linear/contracts.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace riftco_transformer::backend_detail {

// TPU quantized weights keep the canonical NF4 payload packed in persistent
// host storage. PJRT execution uploads that payload and performs unpacking and
// dequantization inside the StableHLO program; no persistent FP32 weight is
// created.
[[nodiscard]] std::unique_ptr<QuantizedWeightStorage>
tpu_make_nf4_weight_storage(
    std::vector<std::uint8_t> packed_codes,
    Nf4ScaleStorageData scales
);

void tpu_quantized_linear_forward(
    const QuantizedLinearForwardRequest& request
);

void tpu_quantized_linear_input_backward(
    const QuantizedLinearInputBackwardRequest& request
);

}  // namespace riftco_transformer::backend_detail
