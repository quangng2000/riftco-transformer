#include "riftco_transformer/core/quantized_weight.hpp"

#include "core/backend/nn/quantized_linear/dispatch.hpp"
#include "core/quantization/nf4.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer {
namespace {

[[nodiscard]] std::size_t checked_numel(const QuantizedWeight::Shape &shape) {
  std::size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension == 0) {
      throw std::invalid_argument(
          "quantized weight dimensions must be greater than zero");
    }
    if (count > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error(
          "quantized weight shape exceeds addressable size");
    }
    count *= dimension;
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error(
        "quantized weight FP32 equivalent exceeds addressable size");
  }
  return count;
}

[[nodiscard]] std::size_t nf4_block_count(std::size_t element_count,
                                          std::size_t block_size) noexcept {
  return element_count / block_size +
         static_cast<std::size_t>(element_count % block_size != 0);
}

[[nodiscard]] std::size_t
nf4_packed_byte_count(std::size_t element_count) noexcept {
  return element_count / 2 + static_cast<std::size_t>(element_count % 2 != 0);
}

void require_supported_block_size(std::size_t block_size) {
  if (!quantization::detail::nf4_block_size_supported(block_size)) {
    throw std::invalid_argument("NF4 block size must be one of "
                                "32, 64, 128, 256, 512, 1024, 2048, or 4096");
  }
}

[[nodiscard]] std::size_t checked_scale_payload_bytes(
    const Nf4Payload &payload) {
  if (!payload.double_quantized_scales.has_value()) {
    if (payload.block_scales.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::overflow_error("NF4 scale storage exceeds addressable size");
    }
    return payload.block_scales.size() * sizeof(float);
  }

  const auto &nested = *payload.double_quantized_scales;
  if (nested.second_level_scales.size() >
      (std::numeric_limits<std::size_t>::max() - sizeof(float)) /
          sizeof(float)) {
    throw std::overflow_error(
        "double-quantized NF4 scale storage exceeds addressable size");
  }
  const std::size_t second_level_bytes =
      nested.second_level_scales.size() * sizeof(float) + sizeof(float);
  if (nested.scale_codes.size() >
      std::numeric_limits<std::size_t>::max() - second_level_bytes) {
    throw std::overflow_error(
        "double-quantized NF4 scale storage exceeds addressable size");
  }
  return nested.scale_codes.size() + second_level_bytes;
}

[[nodiscard]] float decoded_payload_scale(const Nf4Payload &payload,
                                          std::size_t scale_index) noexcept {
  if (!payload.double_quantized_scales.has_value()) {
    return payload.block_scales[scale_index];
  }
  const auto &nested = *payload.double_quantized_scales;
  const int centered_code =
      static_cast<int>(nested.scale_codes[scale_index]) - 128;
  const float factor = static_cast<float>(centered_code) / 127.0F;
  const float delta =
      nested.second_level_scales[scale_index / nested.scale_block_size] *
      factor;
  if (delta > 0.0F &&
      nested.offset > std::numeric_limits<float>::max() - delta) {
    return std::numeric_limits<float>::max();
  }
  const float reconstructed = nested.offset + delta;
  return reconstructed > 0.0F ? reconstructed : 0.0F;
}

void require_canonical_payload(std::size_t element_count,
                               std::size_t block_size,
                               const Nf4Payload &payload) {
  const auto expected_packed_bytes = nf4_packed_byte_count(element_count);
  const auto expected_blocks = nf4_block_count(element_count, block_size);
  if (payload.packed_codes.size() != expected_packed_bytes) {
    throw std::invalid_argument(
        "NF4 packed byte count does not match the weight shape");
  }
  if (element_count % 2 != 0 && (payload.packed_codes.back() & 0xF0U) != 0) {
    throw std::invalid_argument("NF4 unused high nibble must be zero");
  }

  const bool double_quantized = payload.double_quantized_scales.has_value();
  if (!double_quantized) {
    if (payload.block_scales.size() != expected_blocks) {
      throw std::invalid_argument(
          "NF4 scale count does not match the weight shape and block size");
    }
  } else {
    const auto &nested = *payload.double_quantized_scales;
    if (!payload.block_scales.empty()) {
      throw std::invalid_argument(
          "double-quantized NF4 payload must not retain FP32 block scales");
    }
    require_supported_block_size(nested.scale_block_size);
    if (nested.scale_codes.size() != expected_blocks) {
      throw std::invalid_argument(
          "double-quantized NF4 scale-code count does not match the weight");
    }
    const std::size_t expected_second_level_scales =
        nf4_block_count(expected_blocks, nested.scale_block_size);
    if (nested.second_level_scales.size() != expected_second_level_scales) {
      throw std::invalid_argument(
          "double-quantized NF4 second-level scale count is invalid");
    }
    if (!std::isfinite(nested.offset) || nested.offset < 0.0F) {
      throw std::invalid_argument(
          "double-quantized NF4 scale offset must be finite and nonnegative");
    }
    for (const float scale : nested.second_level_scales) {
      if (!std::isfinite(scale) || scale < 0.0F) {
        throw std::invalid_argument(
            "NF4 second-level scales must be finite and nonnegative");
      }
    }
  }

  static_cast<void>(checked_scale_payload_bytes(payload));

  for (std::size_t block = 0; block < expected_blocks; ++block) {
    const float scale = decoded_payload_scale(payload, block);
    if (!std::isfinite(scale) || scale < 0.0F) {
      throw std::invalid_argument(
          "NF4 block scales must be finite and nonnegative");
    }
    // Nested quantization can round a small positive scale down to zero. Its
    // original NF4 codes remain valid and deterministically decode to zero.
    if (scale != 0.0F || double_quantized) {
      continue;
    }

    const std::size_t begin = block * block_size;
    const std::size_t end = std::min(begin + block_size, element_count);
    for (std::size_t index = begin; index < end; ++index) {
      if (quantization::detail::unpack_nf4_code(payload.packed_codes, index) !=
          7) {
        throw std::invalid_argument(
            "zero-scale NF4 blocks must use canonical zero code 7");
      }
    }
  }
}

[[nodiscard]] Nf4Payload pack_nf4(const Tensor &values,
                                  std::size_t block_size) {
  const std::size_t element_count = values.numel();
  Nf4Payload payload{
      std::vector<std::uint8_t>(nf4_packed_byte_count(element_count), 0),
      std::vector<float>(nf4_block_count(element_count, block_size), 0.0F),
      std::nullopt,
  };
  const auto source = values.data();
  for (std::size_t block = 0; block < payload.block_scales.size(); ++block) {
    const std::size_t begin = block * block_size;
    const std::size_t end = std::min(begin + block_size, element_count);
    float scale = 0.0F;
    for (std::size_t index = begin; index < end; ++index) {
      if (!std::isfinite(source[index])) {
        throw std::invalid_argument("NF4 source values must be finite");
      }
      scale = std::max(scale, std::fabs(source[index]));
    }
    payload.block_scales[block] = scale;

    for (std::size_t index = begin; index < end; ++index) {
      const std::uint8_t code =
          scale == 0.0F
              ? std::uint8_t{7}
              : quantization::detail::nearest_nf4_code(source[index] / scale);
      quantization::detail::pack_nf4_code(payload.packed_codes, index, code);
    }
  }
  return payload;
}

[[nodiscard]] Nf4DoubleQuantizedScales double_quantize_scales(
    std::vector<float> block_scales, std::size_t scale_block_size) {
  Nf4DoubleQuantizedScales nested;
  nested.scale_block_size = scale_block_size;
  nested.scale_codes.resize(block_scales.size(), std::uint8_t{128});
  nested.second_level_scales.resize(
      nf4_block_count(block_scales.size(), scale_block_size), 0.0F);

  double sum = 0.0;
  for (const float scale : block_scales) {
    sum += static_cast<double>(scale);
  }
  nested.offset =
      static_cast<float>(sum / static_cast<double>(block_scales.size()));
  if (!std::isfinite(nested.offset)) {
    throw std::overflow_error(
        "NF4 scale mean exceeds the FP32 double-quantization range");
  }

  for (std::size_t group = 0; group < nested.second_level_scales.size();
       ++group) {
    const std::size_t begin = group * scale_block_size;
    const std::size_t end =
        std::min(begin + scale_block_size, block_scales.size());
    float absmax = 0.0F;
    for (std::size_t index = begin; index < end; ++index) {
      absmax = std::max(
          absmax, std::fabs(block_scales[index] - nested.offset));
    }
    nested.second_level_scales[group] = absmax;
    if (absmax == 0.0F) {
      continue;
    }

    for (std::size_t index = begin; index < end; ++index) {
      const float normalized =
          (block_scales[index] - nested.offset) / absmax;
      const float scaled = normalized * 127.0F;
      const float rounded = scaled >= 0.0F ? std::floor(scaled + 0.5F)
                                            : std::ceil(scaled - 0.5F);
      const int centered = std::clamp(static_cast<int>(rounded), -127, 127);
      nested.scale_codes[index] =
          static_cast<std::uint8_t>(centered + 128);
    }
  }
  return nested;
}

[[nodiscard]] backend_detail::Nf4ScaleStorageData
make_scale_storage_data(Nf4Payload &payload) {
  backend_detail::Nf4ScaleStorageData scales;
  if (!payload.double_quantized_scales.has_value()) {
    scales.fp32_scales = std::move(payload.block_scales);
    return scales;
  }
  auto &nested = *payload.double_quantized_scales;
  scales.encoding = backend_detail::Nf4ScaleEncoding::DoubleQuantizedUInt8;
  scales.quantized_scales = std::move(nested.scale_codes);
  scales.second_level_scales = std::move(nested.second_level_scales);
  scales.second_level_block_size = nested.scale_block_size;
  scales.offset = nested.offset;
  return scales;
}

[[nodiscard]] std::shared_ptr<const backend_detail::QuantizedWeightStorage>
make_storage(ExecutionBackend backend, Nf4Payload payload) {
  auto scales = make_scale_storage_data(payload);
  auto storage = backend_detail::make_nf4_weight_storage(
      backend, std::move(payload.packed_codes), std::move(scales));
  return std::shared_ptr<const backend_detail::QuantizedWeightStorage>(
      std::move(storage));
}

} // namespace

struct QuantizedWeight::State {
  Shape shape;
  std::size_t element_count;
  std::size_t block_size;
  std::shared_ptr<const backend_detail::QuantizedWeightStorage> storage;
};

QuantizedWeight::QuantizedWeight(
    Shape shape, std::size_t element_count, std::size_t block_size,
    std::shared_ptr<const backend_detail::QuantizedWeightStorage> storage)
    : state_(std::make_shared<const State>(State{
          std::move(shape),
          element_count,
          block_size,
          std::move(storage),
      })) {}

QuantizedWeight::~QuantizedWeight() = default;

QuantizedWeight QuantizedWeight::quantize_nf4(const Tensor &values,
                                              std::size_t block_size) {
  if (values.numel() == 0) {
    throw std::logic_error("cannot quantize a moved-from tensor");
  }
  require_supported_block_size(block_size);

  return from_packed_nf4(values.shape(), block_size,
                         pack_nf4(values, block_size),
                         values.backend());
}

QuantizedWeight QuantizedWeight::quantize_nf4_double_quantized(
    const Tensor &values, std::size_t block_size,
    std::size_t scale_block_size) {
  if (values.numel() == 0) {
    throw std::logic_error("cannot quantize a moved-from tensor");
  }
  require_supported_block_size(block_size);
  require_supported_block_size(scale_block_size);

  Nf4Payload payload = pack_nf4(values, block_size);
  payload.double_quantized_scales =
      double_quantize_scales(std::move(payload.block_scales),
                             scale_block_size);
  payload.block_scales.clear();
  return from_packed_nf4(values.shape(), block_size, std::move(payload),
                         values.backend());
}

QuantizedWeight QuantizedWeight::from_packed_nf4(Shape shape,
                                                 std::size_t block_size,
                                                 Nf4Payload payload) {
  return from_packed_nf4(std::move(shape), block_size, std::move(payload),
                         execution_backend());
}

QuantizedWeight QuantizedWeight::from_packed_nf4(Shape shape,
                                                 std::size_t block_size,
                                                 Nf4Payload payload,
                                                 ExecutionBackend backend) {
  require_supported_block_size(block_size);
  const std::size_t element_count = checked_numel(shape);
  require_canonical_payload(element_count, block_size, payload);
  auto storage = make_storage(backend, std::move(payload));
  if (storage == nullptr) {
    throw std::logic_error("backend returned null NF4 weight storage");
  }
  return QuantizedWeight(std::move(shape), element_count, block_size,
                         std::move(storage));
}

const QuantizedWeight::Shape &QuantizedWeight::shape() const noexcept {
  static const Shape singular_shape;
  return state_ == nullptr ? singular_shape : state_->shape;
}

std::size_t QuantizedWeight::rank() const noexcept { return shape().size(); }

std::size_t QuantizedWeight::numel() const noexcept {
  return state_ == nullptr ? 0 : state_->element_count;
}

QuantizationFormat QuantizedWeight::format() const noexcept {
  return QuantizationFormat::Nf4;
}

std::size_t QuantizedWeight::block_size() const noexcept {
  return state_ == nullptr ? 0 : state_->block_size;
}

std::size_t QuantizedWeight::block_count() const noexcept {
  return state_ == nullptr
             ? 0
             : nf4_block_count(state_->element_count, state_->block_size);
}

std::size_t QuantizedWeight::packed_byte_count() const noexcept {
  return state_ == nullptr ? 0 : nf4_packed_byte_count(state_->element_count);
}

bool QuantizedWeight::uses_double_quantized_scales() const noexcept {
  return state_ != nullptr &&
         state_->storage->scale_storage().encoding ==
             backend_detail::Nf4ScaleEncoding::DoubleQuantizedUInt8;
}

std::size_t
QuantizedWeight::scale_quantization_block_size() const noexcept {
  return uses_double_quantized_scales()
             ? state_->storage->scale_storage().second_level_block_size
             : 0;
}

ExecutionBackend QuantizedWeight::backend() const noexcept {
  return state_ == nullptr ? ExecutionBackend::Cpu : state_->storage->backend();
}

QuantizedMemoryUsage QuantizedWeight::memory_usage() const noexcept {
  if (state_ == nullptr) {
    return {};
  }
  const std::size_t code_bytes = packed_byte_count();
  const std::size_t scales_bytes =
      backend_detail::nf4_scale_payload_bytes(state_->storage->scale_storage());
  const auto scales = state_->storage->scale_storage();
  const bool double_quantized =
      scales.encoding ==
      backend_detail::Nf4ScaleEncoding::DoubleQuantizedUInt8;
  return {
      code_bytes,
      scales_bytes,
      code_bytes + scales_bytes,
      state_->storage->resident_payload_bytes(),
      state_->element_count * sizeof(float),
      double_quantized ? 0 : scales.fp32_scales.size() * sizeof(float),
      double_quantized ? scales.quantized_scales.size() : 0,
      double_quantized
          ? scales.second_level_scales.size() * sizeof(float)
          : 0,
      double_quantized ? sizeof(float) : 0,
  };
}

Nf4Payload QuantizedWeight::copy_payload_to_host() const {
  if (state_ == nullptr) {
    throw std::logic_error("cannot read a moved-from quantized weight");
  }
  const auto codes = state_->storage->packed_codes();
  const auto scales = state_->storage->scale_storage();
  Nf4Payload payload{
      std::vector<std::uint8_t>(codes.begin(), codes.end()),
      {},
      std::nullopt,
  };
  if (scales.encoding == backend_detail::Nf4ScaleEncoding::Float32) {
    payload.block_scales.assign(scales.fp32_scales.begin(),
                                scales.fp32_scales.end());
  } else {
    payload.double_quantized_scales = Nf4DoubleQuantizedScales{
        std::vector<std::uint8_t>(scales.quantized_scales.begin(),
                                  scales.quantized_scales.end()),
        std::vector<float>(scales.second_level_scales.begin(),
                           scales.second_level_scales.end()),
        scales.second_level_block_size,
        scales.offset,
    };
  }
  return payload;
}

Tensor QuantizedWeight::dequantize() const {
  if (state_ == nullptr) {
    throw std::logic_error("cannot dequantize a moved-from quantized weight");
  }
  const auto payload = copy_payload_to_host();
  const auto codebook = quantization::nf4_codebook();
  std::vector<float> values(state_->element_count, 0.0F);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const std::size_t block = index / state_->block_size;
    const auto code =
        quantization::detail::unpack_nf4_code(payload.packed_codes, index);
    values[index] = decoded_payload_scale(payload, block) * codebook[code];
  }
  return Tensor(state_->shape, std::move(values), backend());
}

QuantizedWeight QuantizedWeight::to(ExecutionBackend target_backend) const {
  if (state_ == nullptr) {
    throw std::logic_error("cannot transfer a moved-from quantized weight");
  }
  if (target_backend == backend()) {
    return *this;
  }
  return from_packed_nf4(state_->shape, state_->block_size,
                         copy_payload_to_host(), target_backend);
}

namespace backend_detail {

const QuantizedWeightStorage &
quantized_weight_storage(const QuantizedWeight &weight) noexcept {
  return *weight.state_->storage;
}

} // namespace backend_detail
} // namespace riftco_transformer
