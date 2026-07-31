#pragma once

#include <cstdint>

namespace riftco_transformer {

// Selects how a full-sequence model forward retains autograd activations.
// Serving decode is stateful and intentionally never checkpointed.
enum class ActivationCheckpointingKind : std::uint8_t {
    Disabled = 0,
    TransformerBlock = 1,
};

}  // namespace riftco_transformer
