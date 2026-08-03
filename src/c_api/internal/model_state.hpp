#pragma once

#include "bridge.hpp"

namespace riftco_transformer::c_api::detail {

std::vector<std::uint8_t> packed_model_state_bytes(
    const riftco_transformer::DecoderOnlyTransformer& model
);
std::vector<riftco_transformer::PackedLinearWeightState>
parse_packed_model_state(std::span<const std::uint8_t> bytes);

}  // namespace riftco_transformer::c_api::detail
