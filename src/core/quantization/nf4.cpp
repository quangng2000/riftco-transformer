#include "core/quantization/nf4.hpp"

#include "riftco_transformer/core/quantized_weight.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace riftco_transformer::quantization {
namespace {

// NF4 is an index into this table, not a sign/exponent/mantissa encoding. Keep
// these binary32 constants stable because their indices are serialized.
constexpr std::array<float, 16> kNf4Codebook = {
    -1.0F,
    -0.6961928009986877F,
    -0.5250730514526367F,
    -0.39491748809814453F,
    -0.28444138169288635F,
    -0.18477343022823334F,
    -0.09105003625154495F,
    0.0F,
    0.07958029955625534F,
    0.16093020141124725F,
    0.24611230194568634F,
    0.33791524171829224F,
    0.44070982933044434F,
    0.5626170039176941F,
    0.7229568362236023F,
    1.0F,
};

} // namespace

std::span<const float, 16> nf4_codebook() noexcept { return kNf4Codebook; }

namespace detail {

bool nf4_block_size_supported(std::size_t block_size) noexcept {
  switch (block_size) {
  case 32:
  case 64:
  case 128:
  case 256:
  case 512:
  case 1024:
  case 2048:
  case 4096:
    return true;
  default:
    return false;
  }
}

std::uint8_t nearest_nf4_code(float normalized_value) noexcept {
  const auto codebook = nf4_codebook();
  std::uint8_t best_code = 0;
  float best_distance = std::fabs(normalized_value - codebook[0]);
  for (std::uint8_t code = 1; code < codebook.size(); ++code) {
    const float distance = std::fabs(normalized_value - codebook[code]);
    // A strict comparison makes exact midpoint ties choose the lower index.
    if (distance < best_distance) {
      best_distance = distance;
      best_code = code;
    }
  }
  return best_code;
}

std::uint8_t unpack_nf4_code(std::span<const std::uint8_t> packed_codes,
                             std::size_t flat_index) noexcept {
  const std::uint8_t packed = packed_codes[flat_index / 2];
  const unsigned shift = flat_index % 2 == 0 ? 0U : 4U;
  return static_cast<std::uint8_t>((packed >> shift) & 0x0FU);
}

void pack_nf4_code(std::span<std::uint8_t> packed_codes, std::size_t flat_index,
                   std::uint8_t code) noexcept {
  const std::size_t byte_index = flat_index / 2;
  const unsigned shift = flat_index % 2 == 0 ? 0U : 4U;
  const std::uint8_t mask = static_cast<std::uint8_t>(0x0FU << shift);
  packed_codes[byte_index] = static_cast<std::uint8_t>(
      (packed_codes[byte_index] & static_cast<std::uint8_t>(~mask)) |
      static_cast<std::uint8_t>((code & 0x0FU) << shift));
}

} // namespace detail
} // namespace riftco_transformer::quantization
