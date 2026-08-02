#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/quantized_weight.hpp"
#include "riftco_transformer/core/tensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using riftco_transformer::ExecutionBackend;
using riftco_transformer::Nf4DoubleQuantizedScales;
using riftco_transformer::Nf4Payload;
using riftco_transformer::QuantizationFormat;
using riftco_transformer::QuantizedWeight;
using riftco_transformer::Tensor;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(float actual, float expected, const std::string &message,
                   float tolerance = 1.0e-6F) {
  const float scale = std::max({1.0F, std::fabs(actual), std::fabs(expected)});
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > tolerance * scale) {
    throw std::runtime_error(message + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

template <typename Function>
void require_throws(Function &&function, const std::string &message) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception &) {
    threw = true;
  }
  require(threw, message);
}

std::uint8_t unpack(const std::vector<std::uint8_t> &packed,
                    std::size_t index) {
  const unsigned shift = index % 2 == 0 ? 0U : 4U;
  return static_cast<std::uint8_t>((packed[index / 2] >> shift) & 0x0FU);
}

void test_codebook_and_canonical_packing() {
  constexpr std::array<float, 16> expected_codebook = {
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
  const auto codebook = riftco_transformer::quantization::nf4_codebook();
  for (std::size_t index = 0; index < codebook.size(); ++index) {
    require(codebook[index] == expected_codebook[index],
            "NF4 codebook literal mismatch");
    if (index != 0) {
      require(codebook[index - 1] < codebook[index],
              "NF4 codebook must be strictly increasing");
    }
  }
  require(codebook[7] == 0.0F, "NF4 zero must use code 7");

  const Tensor source(
      {expected_codebook.size()},
      std::vector<float>(expected_codebook.begin(), expected_codebook.end()),
      ExecutionBackend::Cpu);
  const QuantizedWeight weight = QuantizedWeight::quantize_nf4(source, 32);
  const Nf4Payload payload = weight.copy_payload_to_host();
  const std::vector<std::uint8_t> expected_bytes = {
      0x10U, 0x32U, 0x54U, 0x76U, 0x98U, 0xBAU, 0xDCU, 0xFEU,
  };
  require(payload.packed_codes == expected_bytes,
          "NF4 packing must put each even code in the low nibble");
  require(payload.block_scales == std::vector<float>({1.0F}),
          "the codebook endpoints should establish an exact unit scale");

  const Tensor restored = weight.dequantize();
  require(restored.shape() == source.shape() &&
              restored.backend() == ExecutionBackend::Cpu,
          "dequantization must preserve shape and backend");
  for (std::size_t index = 0; index < restored.numel(); ++index) {
    require(restored.flat(index) == expected_codebook[index],
            "codebook values should round-trip exactly");
  }
}

void test_tie_breaking_zero_blocks_and_partial_blocks() {
  const auto codebook = riftco_transformer::quantization::nf4_codebook();
  std::vector<float> midpoint_values(32, 0.0F);
  midpoint_values[0] = codebook[8] * 0.5F;
  midpoint_values.back() = 1.0F;
  const auto midpoint_payload =
      QuantizedWeight::quantize_nf4(Tensor({32}, std::move(midpoint_values)),
                                    32)
          .copy_payload_to_host();
  require(unpack(midpoint_payload.packed_codes, 0) == 7,
          "an exact NF4 midpoint tie must choose the lower code");

  const QuantizedWeight zeros = QuantizedWeight::quantize_nf4(
      Tensor::zeros({33}, ExecutionBackend::Cpu), 32);
  const Nf4Payload zero_payload = zeros.copy_payload_to_host();
  require(zero_payload.block_scales == std::vector<float>({0.0F, 0.0F}),
          "zero blocks must store zero scales");
  for (std::size_t index = 0; index < 33; ++index) {
    require(unpack(zero_payload.packed_codes, index) == 7,
            "zero blocks must use canonical code 7");
  }
  require(zero_payload.packed_codes.back() == 0x07U,
          "an odd final code must leave the unused high nibble zero");

  std::vector<float> values(33, 0.25F);
  values[0] = -2.0F;
  values.back() = 10.0F;
  const QuantizedWeight partial =
      QuantizedWeight::quantize_nf4(Tensor({33}, std::move(values)), 32);
  const Nf4Payload partial_payload = partial.copy_payload_to_host();
  require(partial_payload.block_scales == std::vector<float>({2.0F, 10.0F}),
          "each complete or partial block needs its own absmax scale");
  require(unpack(partial_payload.packed_codes, 0) == 0 &&
              unpack(partial_payload.packed_codes, 32) == 15,
          "block endpoints should map to NF4 endpoint codes");
  const Tensor restored = partial.dequantize();
  require_close(restored.flat(0), -2.0F, "first block endpoint");
  require_close(restored.flat(32), 10.0F, "partial block endpoint");
}

void test_double_quantized_scale_packing_and_memory() {
  constexpr std::size_t first_level_block_size = 32;
  constexpr std::size_t scale_block_size = 32;
  constexpr std::size_t scale_count = 33;
  std::vector<float> values(scale_count * first_level_block_size, 0.0F);
  for (std::size_t block = 0; block < scale_count; ++block) {
    const auto begin = values.begin() + static_cast<std::ptrdiff_t>(
                                           block * first_level_block_size);
    std::fill_n(begin,
                first_level_block_size, static_cast<float>(block));
  }
  const std::size_t value_count = values.size();

  const QuantizedWeight weight =
      QuantizedWeight::quantize_nf4_double_quantized(
          Tensor({value_count}, std::move(values), ExecutionBackend::Cpu),
          first_level_block_size, scale_block_size);
  require(weight.uses_double_quantized_scales(),
          "double quantization should be observable without materialization");
  require(weight.scale_quantization_block_size() == scale_block_size,
          "double-quantization scale block size should be retained");

  const Nf4Payload payload = weight.copy_payload_to_host();
  require(payload.block_scales.empty() &&
              payload.double_quantized_scales.has_value(),
          "double-quantized payload must not retain FP32 first-level scales");
  const auto &nested = *payload.double_quantized_scales;
  require(nested.scale_codes.size() == scale_count &&
              nested.second_level_scales ==
                  std::vector<float>({16.0F, 16.0F}) &&
              nested.offset == 16.0F,
          "nested scale payload should use centered uint8 codes and FP32 absmaxes");
  require(nested.scale_codes[0] == 1 && nested.scale_codes[16] == 128 &&
              nested.scale_codes[32] == 255,
          "nested scale endpoints and center should have canonical uint8 codes");

  const Tensor restored = weight.dequantize();
  require(restored.flat(0) == 0.0F && restored.flat(16 * 32) == 16.0F &&
              restored.flat(32 * 32) == 32.0F,
          "nested scale decoding should reconstruct exact endpoints and center");

  const auto usage = weight.memory_usage();
  require(usage.packed_code_bytes == 528 && usage.scale_bytes == 45 &&
              usage.logical_payload_bytes == 573 &&
              usage.resident_payload_bytes == 573 &&
              usage.fp32_equivalent_bytes == 4224 &&
              usage.fp32_scale_bytes == 0 &&
              usage.scale_code_bytes == 33 &&
              usage.second_level_scale_bytes == 8 &&
              usage.scale_offset_bytes == 4,
          "double-quantized memory accounting must include only encoded scales");

  const QuantizedWeight imported = QuantizedWeight::from_packed_nf4(
      weight.shape(), first_level_block_size, payload, ExecutionBackend::Cpu);
  require(imported.copy_payload_to_host().block_scales.empty() &&
              imported.copy_payload_to_host()
                      .double_quantized_scales->scale_codes ==
                  nested.scale_codes,
          "double-quantized payload import must preserve encoded scales");
}

void test_payload_validation() {
  const auto scalar_zero = [] {
    return Nf4Payload{{0x07U}, {0.0F}, std::nullopt};
  };
  const auto double_scalar_zero = [] {
    return Nf4Payload{
        {0x07U},
        {},
        Nf4DoubleQuantizedScales{{128U}, {0.0F}, 32, 0.0F},
    };
  };
  const QuantizedWeight scalar = QuantizedWeight::from_packed_nf4(
      {}, 32, scalar_zero(), ExecutionBackend::Cpu);
  require(scalar.rank() == 0 && scalar.numel() == 1 &&
              scalar.dequantize().flat(0) == 0.0F,
          "empty shape should retain Tensor scalar semantics");

  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 16, scalar_zero(), ExecutionBackend::Cpu));
      },
      "unsupported NF4 block size should throw");
  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {0}, 32, scalar_zero(), ExecutionBackend::Cpu));
      },
      "zero shape dimension should throw");
  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {std::numeric_limits<std::size_t>::max(), 2}, 32, {},
            ExecutionBackend::Cpu));
      },
      "shape product overflow should throw");
  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, {{}, {0.0F}, std::nullopt}, ExecutionBackend::Cpu));
      },
      "wrong packed byte count should throw");
  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, {{0x07U}, {}, std::nullopt}, ExecutionBackend::Cpu));
      },
      "wrong scale count should throw");
  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, {{0xF7U}, {0.0F}, std::nullopt},
            ExecutionBackend::Cpu));
      },
      "nonzero unused high nibble should throw");
  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, {{0x08U}, {0.0F}, std::nullopt},
            ExecutionBackend::Cpu));
      },
      "zero-scale block with a nonzero code should throw");
  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, {{0x07U}, {-1.0F}, std::nullopt},
            ExecutionBackend::Cpu));
      },
      "negative scale should throw");
  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32,
            {{0x07U}, {std::numeric_limits<float>::infinity()}, std::nullopt},
            ExecutionBackend::Cpu));
      },
      "nonfinite scale should throw");

  require_throws(
      [&] {
        auto payload = double_scalar_zero();
        payload.block_scales = {0.0F};
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, std::move(payload), ExecutionBackend::Cpu));
      },
      "double-quantized payload must reject a hidden FP32 scale vector");
  require_throws(
      [&] {
        auto payload = double_scalar_zero();
        payload.double_quantized_scales->scale_codes.clear();
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, std::move(payload), ExecutionBackend::Cpu));
      },
      "double-quantized payload must validate its scale-code count");
  require_throws(
      [&] {
        auto payload = double_scalar_zero();
        payload.double_quantized_scales->scale_block_size = 16;
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, std::move(payload), ExecutionBackend::Cpu));
      },
      "double-quantized payload must validate its scale block size");
  require_throws(
      [&] {
        auto payload = double_scalar_zero();
        payload.double_quantized_scales->second_level_scales.clear();
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, std::move(payload), ExecutionBackend::Cpu));
      },
      "double-quantized payload must validate second-level scale count");
  require_throws(
      [&] {
        auto payload = double_scalar_zero();
        payload.double_quantized_scales->second_level_scales[0] = -1.0F;
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, std::move(payload), ExecutionBackend::Cpu));
      },
      "double-quantized payload must reject negative second-level scales");
  require_throws(
      [&] {
        auto payload = double_scalar_zero();
        payload.double_quantized_scales->offset =
            std::numeric_limits<float>::infinity();
        static_cast<void>(QuantizedWeight::from_packed_nf4(
            {1}, 32, std::move(payload), ExecutionBackend::Cpu));
      },
      "double-quantized payload must reject nonfinite offsets");

  Tensor nonfinite = Tensor::zeros({32}, ExecutionBackend::Cpu);
  nonfinite.flat(0) = std::numeric_limits<float>::quiet_NaN();
  require_throws(
      [&] { static_cast<void>(QuantizedWeight::quantize_nf4(nonfinite)); },
      "nonfinite source value should throw");
  require_throws(
      [&] {
        static_cast<void>(QuantizedWeight::quantize_nf4_double_quantized(
            Tensor::zeros({32}, ExecutionBackend::Cpu), 32, 16));
      },
      "double quantization should reject an unsupported scale block size");
}

void test_memory_ownership_and_transfer() {
  const QuantizedWeight original =
      QuantizedWeight::quantize_nf4(Tensor::zeros({64}, ExecutionBackend::Cpu));
  require(original.format() == QuantizationFormat::Nf4, "format identity");
  require(original.shape() == Tensor::Shape({64}), "shape identity");
  require(original.rank() == 1 && original.numel() == 64, "shape metrics");
  require(original.block_size() == 64 && original.block_count() == 1,
          "block metrics");
  require(original.packed_byte_count() == 32, "packed byte count");
  require(original.backend() == ExecutionBackend::Cpu, "backend identity");

  const auto usage = original.memory_usage();
  require(usage.packed_code_bytes == 32 && usage.scale_bytes == 4 &&
              usage.logical_payload_bytes == 36 &&
              usage.resident_payload_bytes == 36 &&
              usage.fp32_equivalent_bytes == 256 &&
              usage.fp32_scale_bytes == 4 && usage.scale_code_bytes == 0 &&
              usage.second_level_scale_bytes == 0 &&
              usage.scale_offset_bytes == 0,
          "64-value NF4 payload should use 36 resident bytes instead of 256");

  QuantizedWeight shared_copy = original;
  require(shared_copy.copy_payload_to_host().packed_codes ==
              original.copy_payload_to_host().packed_codes,
          "immutable copies should retain the same packed value");
  const QuantizedWeight same_backend = original.to(ExecutionBackend::Cpu);
  require(same_backend.copy_payload_to_host().block_scales ==
              original.copy_payload_to_host().block_scales,
          "same-backend transfer should preserve packed scales");

  QuantizedWeight moved = std::move(shared_copy);
  require(shared_copy.numel() == 0 && shared_copy.block_count() == 0 &&
              shared_copy.packed_byte_count() == 0,
          "moved-from quantized weight should be singular");
  require_throws([&] { static_cast<void>(shared_copy.copy_payload_to_host()); },
                 "moved-from payload read should throw");
  require_throws([&] { static_cast<void>(shared_copy.dequantize()); },
                 "moved-from dequantize should throw");
  require_throws(
      [&] { static_cast<void>(shared_copy.to(ExecutionBackend::Cpu)); },
      "moved-from transfer should throw");
  require(moved.numel() == 64, "move should retain destination state");

  if (riftco_transformer::execution_backend_available(
          ExecutionBackend::Metal)) {
    const QuantizedWeight metal = original.to(ExecutionBackend::Metal);
    require(metal.backend() == ExecutionBackend::Metal,
            "packed transfer should update backend identity");
    const Nf4Payload metal_payload = metal.copy_payload_to_host();
    const Nf4Payload cpu_payload = original.copy_payload_to_host();
    require(metal_payload.packed_codes == cpu_payload.packed_codes &&
                metal_payload.block_scales == cpu_payload.block_scales,
            "packed backend transfer must be byte-preserving");
    require(metal.dequantize().backend() == ExecutionBackend::Metal,
            "dequantized Tensor should remain on the weight backend");
  }

  const QuantizedWeight double_quantized =
      QuantizedWeight::quantize_nf4_double_quantized(
          Tensor::zeros({64}, ExecutionBackend::Cpu), 64, 32);
  const QuantizedWeight same_double_backend =
      double_quantized.to(ExecutionBackend::Cpu);
  require(same_double_backend.copy_payload_to_host().block_scales.empty() &&
              same_double_backend.uses_double_quantized_scales(),
          "same-backend transfer must retain double-quantized scale encoding");
  if (riftco_transformer::execution_backend_available(
          ExecutionBackend::Metal)) {
    const QuantizedWeight metal =
        double_quantized.to(ExecutionBackend::Metal);
    const auto metal_payload = metal.copy_payload_to_host();
    const auto cpu_payload = double_quantized.copy_payload_to_host();
    require(metal.uses_double_quantized_scales() &&
                metal_payload.block_scales.empty() &&
                metal_payload.double_quantized_scales->scale_codes ==
                    cpu_payload.double_quantized_scales->scale_codes &&
                metal_payload.double_quantized_scales->second_level_scales ==
                    cpu_payload.double_quantized_scales->second_level_scales,
            "Metal transfer must preserve double-quantized scale storage");
  }
}

} // namespace

int main() {
  try {
    test_codebook_and_canonical_packing();
    test_tie_breaking_zero_blocks_and_partial_blocks();
    test_double_quantized_scale_packing_and_memory();
    test_payload_validation();
    test_memory_ownership_and_transfer();
    std::cout << "NF4 quantization tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "NF4 quantization test failure: " << error.what() << '\n';
    return 1;
  }
}
