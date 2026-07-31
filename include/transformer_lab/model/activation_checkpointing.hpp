#pragma once

#include <cstdint>

namespace transformer_lab {

// Selects how a full-sequence model forward retains autograd activations.
// Serving decode is stateful and intentionally never checkpointed.
enum class ActivationCheckpointingKind : std::uint8_t {
    Disabled = 0,
    TransformerBlock = 1,
};

}  // namespace transformer_lab
