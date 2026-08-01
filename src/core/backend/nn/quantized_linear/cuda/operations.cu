#include "core/backend/nn/quantized_linear/cuda/launch.hpp"

#include "core/backend/nn/cuda/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace riftco_transformer::backend_detail {
namespace {

__device__ __constant__ float kNf4Codebook[16] = {
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

// Decode only the scale required by the current multiply. For double
// quantization, the uint8 first-level scale stays packed and is reconstructed
// from its FP32 second-level block without creating an FP32 scale vector.
__device__ __forceinline__ float
decode_block_scale(const void *primary_scales, const float *secondary_scales,
                   Nf4ScaleEncoding scale_encoding,
                   std::size_t second_level_block_size, float scale_offset,
                   std::size_t scale_index) {
  if (scale_encoding == Nf4ScaleEncoding::Float32) {
    return static_cast<const float *>(primary_scales)[scale_index];
  }

  const auto scale_code =
      static_cast<const std::uint8_t *>(primary_scales)[scale_index];
  const float normalized_code =
      static_cast<float>(static_cast<int>(scale_code) - 128) / 127.0F;
  const float delta =
      secondary_scales[scale_index / second_level_block_size] * normalized_code;
  constexpr float maximum_finite_float = 0x1.fffffep+127F;
  if (delta > 0.0F && scale_offset > maximum_finite_float - delta) {
    return maximum_finite_float;
  }
  const float decoded = scale_offset + delta;
  return decoded > 0.0F ? decoded : 0.0F;
}

__device__ __forceinline__ float
decode_nf4(const std::uint8_t *packed_codes, const void *primary_scales,
           const float *secondary_scales, Nf4ScaleEncoding scale_encoding,
           std::size_t second_level_block_size, float scale_offset,
           std::size_t block_size, std::size_t flat_index) {
  const std::uint8_t packed = packed_codes[flat_index >> 1U];
  const std::uint8_t code = (flat_index & 1U) == 0U
                                ? static_cast<std::uint8_t>(packed & 0x0FU)
                                : static_cast<std::uint8_t>(packed >> 4U);
  const std::size_t scale_index = flat_index / block_size;
  return kNf4Codebook[code] *
         decode_block_scale(primary_scales, secondary_scales, scale_encoding,
                            second_level_block_size, scale_offset, scale_index);
}

__global__ void quantized_linear_forward_kernel(
    const float *input, const std::uint8_t *packed_codes,
    const void *primary_scales, const float *secondary_scales,
    Nf4ScaleEncoding scale_encoding, std::size_t second_level_block_size,
    float scale_offset, float *output, std::size_t output_count,
    std::size_t input_width, std::size_t output_width, std::size_t block_size) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  std::size_t output_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  while (output_index < output_count) {
    const std::size_t row = output_index / output_width;
    const std::size_t output_column = output_index % output_width;
    const std::size_t input_offset = row * input_width;
    const std::size_t weight_offset = output_column * input_width;

    float total = 0.0F;
    for (std::size_t input_column = 0; input_column < input_width;
         ++input_column) {
      total += input[input_offset + input_column] *
               decode_nf4(packed_codes, primary_scales, secondary_scales,
                          scale_encoding, second_level_block_size, scale_offset,
                          block_size, weight_offset + input_column);
    }
    output[output_index] = total;
    output_index += stride;
  }
}

__global__ void quantized_linear_input_backward_kernel(
    const float *upstream, const std::uint8_t *packed_codes,
    const void *primary_scales, const float *secondary_scales,
    Nf4ScaleEncoding scale_encoding, std::size_t second_level_block_size,
    float scale_offset, float *input_gradient, std::size_t gradient_count,
    std::size_t input_width, std::size_t output_width, std::size_t block_size) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  std::size_t gradient_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  while (gradient_index < gradient_count) {
    const std::size_t row = gradient_index / input_width;
    const std::size_t input_column = gradient_index % input_width;
    const std::size_t upstream_offset = row * output_width;

    float total = 0.0F;
    for (std::size_t output_column = 0; output_column < output_width;
         ++output_column) {
      const std::size_t weight_index =
          output_column * input_width + input_column;
      total += upstream[upstream_offset + output_column] *
               decode_nf4(packed_codes, primary_scales, secondary_scales,
                          scale_encoding, second_level_block_size, scale_offset,
                          block_size, weight_index);
    }
    input_gradient[gradient_index] = total;
    gradient_index += stride;
  }
}

[[nodiscard]] const std::uint8_t *
require_packed_codes(const QuantizedWeightStorage &weight,
                     std::string_view operation) {
  if (weight.backend() != ExecutionBackend::Cuda) {
    throw std::invalid_argument("CUDA " + std::string(operation) +
                                " requires CUDA quantized-weight storage");
  }
  const void *handle = weight.packed_codes_native_handle();
  if (handle == nullptr) {
    throw std::logic_error(
        "CUDA " + std::string(operation) +
        " received packed codes without a native allocation");
  }
  return static_cast<const std::uint8_t *>(handle);
}

struct NativeNf4Scales {
  const void *primary;
  const float *secondary;
  Nf4ScaleEncoding encoding;
  std::size_t second_level_block_size;
  float offset;
};

[[nodiscard]] NativeNf4Scales
require_native_scales(const QuantizedWeightStorage &weight,
                      std::string_view operation) {
  const auto scales = weight.scale_storage();
  const void *primary = weight.primary_scales_native_handle();
  if (primary == nullptr) {
    throw std::logic_error(
        "CUDA " + std::string(operation) +
        " received primary scales without a native allocation");
  }

  if (scales.encoding == Nf4ScaleEncoding::Float32) {
    if (scales.fp32_scales.empty()) {
      throw std::logic_error("CUDA " + std::string(operation) +
                             " received empty FP32 block scales");
    }
    return {primary, nullptr, scales.encoding, 0, 0.0F};
  }

  if (scales.encoding != Nf4ScaleEncoding::DoubleQuantizedUInt8) {
    throw std::logic_error("CUDA " + std::string(operation) +
                           " received an unknown scale encoding");
  }

  const void *secondary = weight.secondary_scales_native_handle();
  if (scales.quantized_scales.empty() || scales.second_level_scales.empty() ||
      scales.second_level_block_size == 0 || secondary == nullptr) {
    throw std::logic_error(
        "CUDA " + std::string(operation) +
        " received incomplete double-quantized scale storage");
  }
  return {primary, static_cast<const float *>(secondary), scales.encoding,
          scales.second_level_block_size, scales.offset};
}

} // namespace

void cuda_quantized_linear_forward(
    const QuantizedLinearForwardRequest &request) {
  constexpr std::string_view operation = "quantized-linear forward";
  const std::size_t output_count = request.output.size();
  const NativeNf4Scales scales =
      require_native_scales(request.weight, operation);
  quantized_linear_forward_kernel<<<nn_cuda_detail::block_count_for(
                                        output_count),
                                    nn_cuda_detail::kThreadsPerBlock>>>(
      nn_cuda_detail::require_native_input(request.input, operation),
      require_packed_codes(request.weight, operation), scales.primary,
      scales.secondary, scales.encoding, scales.second_level_block_size,
      scales.offset,
      nn_cuda_detail::require_native_output(request.output, operation),
      output_count, request.dimensions.input_width,
      request.dimensions.output_width, request.dimensions.block_size);
  nn_cuda_detail::require_kernel_launch(
      "quantized-linear forward kernel launch");
  nn_cuda_detail::synchronize("quantized-linear forward synchronization");
}

void cuda_quantized_linear_input_backward(
    const QuantizedLinearInputBackwardRequest &request) {
  constexpr std::string_view operation = "quantized-linear input backward";
  const std::size_t gradient_count = request.input_gradient.size();
  const NativeNf4Scales scales =
      require_native_scales(request.weight, operation);
  quantized_linear_input_backward_kernel<<<nn_cuda_detail::block_count_for(
                                               gradient_count),
                                           nn_cuda_detail::kThreadsPerBlock>>>(
      nn_cuda_detail::require_native_input(request.upstream, operation),
      require_packed_codes(request.weight, operation), scales.primary,
      scales.secondary, scales.encoding, scales.second_level_block_size,
      scales.offset,
      nn_cuda_detail::require_native_output(request.input_gradient, operation),
      gradient_count, request.dimensions.input_width,
      request.dimensions.output_width, request.dimensions.block_size);
  nn_cuda_detail::require_kernel_launch(
      "quantized-linear input-backward kernel launch");
  nn_cuda_detail::synchronize(
      "quantized-linear input-backward synchronization");
}

} // namespace riftco_transformer::backend_detail
