#pragma once

#include "core/backend/nn/quantized_linear/contracts.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer::backend_detail {

// Defaults make quantized execution an additive backend capability. Backends
// that have not implemented it remain source-compatible and fail explicitly
// instead of silently materializing a float weight.
class QuantizedStorageCapability {
public:
    virtual ~QuantizedStorageCapability() = default;

    [[nodiscard]] virtual std::unique_ptr<QuantizedWeightStorage>
    make_nf4_weight_storage(
        std::vector<std::uint8_t>,
        Nf4ScaleStorageData
    ) const {
        throw std::runtime_error(
            "NF4 weight storage is unsupported by this execution backend"
        );
    }
};

class QuantizedLinearCapability {
public:
    virtual ~QuantizedLinearCapability() = default;

    virtual void quantized_linear_forward(
        const QuantizedLinearForwardRequest&
    ) const {
        throw std::runtime_error(
            "quantized linear is unsupported by this execution backend"
        );
    }

    virtual void quantized_linear_input_backward(
        const QuantizedLinearInputBackwardRequest&
    ) const {
        throw std::runtime_error(
            "quantized linear backward is unsupported by this execution backend"
        );
    }
};

}  // namespace riftco_transformer::backend_detail
