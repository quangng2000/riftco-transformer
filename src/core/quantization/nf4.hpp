#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace riftco_transformer::quantization::detail {

[[nodiscard]] bool nf4_block_size_supported(std::size_t block_size) noexcept;
[[nodiscard]] std::uint8_t nearest_nf4_code(float normalized_value) noexcept;
[[nodiscard]] std::uint8_t
unpack_nf4_code(std::span<const std::uint8_t> packed_codes,
                std::size_t flat_index) noexcept;
void pack_nf4_code(std::span<std::uint8_t> packed_codes, std::size_t flat_index,
                   std::uint8_t code) noexcept;

} // namespace riftco_transformer::quantization::detail
