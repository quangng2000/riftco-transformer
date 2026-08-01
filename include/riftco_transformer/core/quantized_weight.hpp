#pragma once

#include "riftco_transformer/core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace riftco_transformer {

class QuantizedWeight;

namespace backend_detail {
class QuantizedWeightStorage;
[[nodiscard]] const QuantizedWeightStorage &
quantized_weight_storage(const QuantizedWeight &weight) noexcept;
} // namespace backend_detail

enum class QuantizationFormat : std::uint8_t {
  Nf4 = 0,
};

// Nested 8-bit representation of the first-level NF4 block scales. Each scale
// code is centered at 128. The corresponding FP32 second-level scale applies
// to scale_block_size consecutive codes; offset restores the shared mean.
// Keeping this representation explicit prevents a hidden persistent FP32
// first-level scale vector in double-quantized weights.
struct Nf4DoubleQuantizedScales {
  std::vector<std::uint8_t> scale_codes;
  std::vector<float> second_level_scales;
  std::size_t scale_block_size = 256;
  float offset = 0.0F;
};

// Canonical host representation of one blockwise NF4 weight. Flat even-indexed
// codes occupy the low nibble and odd-indexed codes occupy the high nibble.
// Legacy payloads populate block_scales. Double-quantized payloads leave
// block_scales empty and populate double_quantized_scales instead.
struct Nf4Payload {
  std::vector<std::uint8_t> packed_codes;
  std::vector<float> block_scales;
  std::optional<Nf4DoubleQuantizedScales> double_quantized_scales;
};

// Counts persistent quantized payload, not C++ object or allocator metadata.
// resident_payload_bytes includes every backend allocation and persistent
// mirror used for codes and scales, making hidden full-precision mirrors
// observable.
struct QuantizedMemoryUsage {
  std::size_t packed_code_bytes = 0;
  std::size_t scale_bytes = 0;
  std::size_t logical_payload_bytes = 0;
  std::size_t resident_payload_bytes = 0;
  std::size_t fp32_equivalent_bytes = 0;
  std::size_t fp32_scale_bytes = 0;
  std::size_t scale_code_bytes = 0;
  // Counts FP32 second-level scales only. scale_offset_bytes separately counts
  // the persistent FP32 centering offset.
  std::size_t second_level_scale_bytes = 0;
  std::size_t scale_offset_bytes = 0;
};

// Immutable, blockwise-quantized weight storage. Copies share immutable packed
// state; to() is the explicit operation that creates storage on another
// backend. A moved-from value is a safe singular object with numel()==0.
class QuantizedWeight {
public:
  using Shape = Tensor::Shape;
  static constexpr std::size_t kDefaultNf4BlockSize = 64;

  [[nodiscard]] static QuantizedWeight
  quantize_nf4(const Tensor &values,
               std::size_t block_size = kDefaultNf4BlockSize);
  [[nodiscard]] static QuantizedWeight quantize_nf4_double_quantized(
      const Tensor &values, std::size_t block_size = kDefaultNf4BlockSize,
      std::size_t scale_block_size = 256);
  [[nodiscard]] static QuantizedWeight
  from_packed_nf4(Shape shape, std::size_t block_size, Nf4Payload payload);
  [[nodiscard]] static QuantizedWeight
  from_packed_nf4(Shape shape, std::size_t block_size, Nf4Payload payload,
                  ExecutionBackend backend);

  QuantizedWeight(const QuantizedWeight &) noexcept = default;
  QuantizedWeight &operator=(const QuantizedWeight &) noexcept = default;
  QuantizedWeight(QuantizedWeight &&) noexcept = default;
  QuantizedWeight &operator=(QuantizedWeight &&) noexcept = default;
  ~QuantizedWeight();

  [[nodiscard]] const Shape &shape() const noexcept;
  [[nodiscard]] std::size_t rank() const noexcept;
  [[nodiscard]] std::size_t numel() const noexcept;
  [[nodiscard]] QuantizationFormat format() const noexcept;
  [[nodiscard]] std::size_t block_size() const noexcept;
  [[nodiscard]] std::size_t block_count() const noexcept;
  [[nodiscard]] std::size_t packed_byte_count() const noexcept;
  [[nodiscard]] bool uses_double_quantized_scales() const noexcept;
  [[nodiscard]] std::size_t scale_quantization_block_size() const noexcept;
  [[nodiscard]] ExecutionBackend backend() const noexcept;
  [[nodiscard]] QuantizedMemoryUsage memory_usage() const noexcept;

  // Explicit readback intended for serialization and diagnostics. It copies
  // the packed representation but never creates a full-precision weight.
  [[nodiscard]] Nf4Payload copy_payload_to_host() const;

  // Explicitly materializes a full-precision Tensor on this weight's backend.
  // Quantized training kernels must consume the packed storage directly.
  [[nodiscard]] Tensor dequantize() const;
  [[nodiscard]] QuantizedWeight to(ExecutionBackend backend) const;

private:
  struct State;
  std::shared_ptr<const State> state_;

  QuantizedWeight(
      Shape shape, std::size_t element_count, std::size_t block_size,
      std::shared_ptr<const backend_detail::QuantizedWeightStorage> storage);

  friend const backend_detail::QuantizedWeightStorage &
  backend_detail::quantized_weight_storage(
      const QuantizedWeight &weight) noexcept;
};

namespace quantization {

// Stable NF4 index-to-value mapping used by serialized packed weights.
[[nodiscard]] std::span<const float, 16> nf4_codebook() noexcept;

} // namespace quantization
} // namespace riftco_transformer
