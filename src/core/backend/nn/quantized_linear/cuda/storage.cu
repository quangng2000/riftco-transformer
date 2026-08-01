#include "core/backend/nn/quantized_linear/cuda/launch.hpp"

#include "core/backend/nn/cuda/common.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

// Unified allocations satisfy the host-visible serialization contract and are
// also passed directly to CUDA kernels. There is no duplicate host payload and
// no persistent dequantized scale or weight allocation.
template <typename Element> class CudaManagedQuantizedBuffer final {
public:
  explicit CudaManagedQuantizedBuffer(std::vector<Element> values)
      : element_count_(values.size()) {
    if (element_count_ == 0) {
      return;
    }
    if (element_count_ >
        std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error(
          "CUDA quantized-weight allocation size overflow");
    }

    void *allocation = nullptr;
    nn_cuda_detail::require_cuda_success(
        cudaMallocManaged(&allocation, element_count_ * sizeof(Element),
                          cudaMemAttachGlobal),
        "quantized-weight managed allocation");
    values_ = static_cast<Element *>(allocation);
    std::copy(values.begin(), values.end(), values_);
  }

  CudaManagedQuantizedBuffer(const CudaManagedQuantizedBuffer &) = delete;
  CudaManagedQuantizedBuffer &
  operator=(const CudaManagedQuantizedBuffer &) = delete;

  ~CudaManagedQuantizedBuffer() {
    if (values_ != nullptr) {
      static_cast<void>(cudaFree(values_));
    }
  }

  [[nodiscard]] std::span<const Element> values() const noexcept {
    return {values_, element_count_};
  }

  [[nodiscard]] const Element *native_handle() const noexcept {
    return values_;
  }

  [[nodiscard]] std::size_t byte_count() const noexcept {
    return element_count_ * sizeof(Element);
  }

private:
  std::size_t element_count_ = 0;
  Element *values_ = nullptr;
};

class CudaNf4WeightStorage final : public QuantizedWeightStorage {
public:
  CudaNf4WeightStorage(std::vector<std::uint8_t> packed_codes,
                       Nf4ScaleStorageData scales)
      : packed_codes_(std::move(packed_codes)), encoding_(scales.encoding),
        fp32_scales_(std::move(scales.fp32_scales)),
        quantized_scales_(std::move(scales.quantized_scales)),
        second_level_scales_(std::move(scales.second_level_scales)),
        second_level_block_size_(scales.second_level_block_size),
        offset_(scales.offset) {}

  [[nodiscard]] ExecutionBackend backend() const noexcept override {
    return ExecutionBackend::Cuda;
  }

  [[nodiscard]] std::span<const std::uint8_t>
  packed_codes() const noexcept override {
    return packed_codes_.values();
  }

  [[nodiscard]] Nf4ScaleStorageView scale_storage() const noexcept override {
    return {
        encoding_,
        fp32_scales_.values(),
        quantized_scales_.values(),
        second_level_scales_.values(),
        second_level_block_size_,
        offset_,
    };
  }

  [[nodiscard]] const void *
  packed_codes_native_handle() const noexcept override {
    return packed_codes_.native_handle();
  }

  [[nodiscard]] const void *
  primary_scales_native_handle() const noexcept override {
    return encoding_ == Nf4ScaleEncoding::Float32
               ? static_cast<const void *>(fp32_scales_.native_handle())
               : static_cast<const void *>(quantized_scales_.native_handle());
  }

  [[nodiscard]] const void *
  secondary_scales_native_handle() const noexcept override {
    return encoding_ == Nf4ScaleEncoding::DoubleQuantizedUInt8
               ? static_cast<const void *>(second_level_scales_.native_handle())
               : nullptr;
  }

  [[nodiscard]] std::size_t resident_payload_bytes() const noexcept override {
    return packed_codes_.byte_count() +
           nf4_scale_payload_bytes(scale_storage());
  }

private:
  CudaManagedQuantizedBuffer<std::uint8_t> packed_codes_;
  Nf4ScaleEncoding encoding_ = Nf4ScaleEncoding::Float32;
  CudaManagedQuantizedBuffer<float> fp32_scales_;
  CudaManagedQuantizedBuffer<std::uint8_t> quantized_scales_;
  CudaManagedQuantizedBuffer<float> second_level_scales_;
  std::size_t second_level_block_size_ = 0;
  float offset_ = 0.0F;
};

} // namespace

std::unique_ptr<QuantizedWeightStorage>
cuda_make_nf4_weight_storage(std::vector<std::uint8_t> packed_codes,
                             Nf4ScaleStorageData scales) {
  return std::make_unique<CudaNf4WeightStorage>(std::move(packed_codes),
                                                std::move(scales));
}

} // namespace riftco_transformer::backend_detail
