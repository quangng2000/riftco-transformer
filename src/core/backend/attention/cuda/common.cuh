#pragma once

#include "core/backend/storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

namespace riftco_transformer::backend_detail::attention_cuda_detail {

inline constexpr unsigned int kThreadsPerBlock = 256;
inline constexpr std::size_t kMaximumBlockCount = 65'535;

[[noreturn]] inline void throw_cuda_error(cudaError_t status,
                                          std::string_view operation) {
    if (status == cudaErrorMemoryAllocation) {
        throw std::bad_alloc();
    }
    throw std::runtime_error("CUDA " + std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
}

inline void require_cuda_success(cudaError_t status,
                                 std::string_view operation) {
    if (status != cudaSuccess) {
        throw_cuda_error(status, operation);
    }
}

inline unsigned int block_count_for(std::size_t element_count) {
    if (element_count == 0) {
        throw std::logic_error("CUDA attention launch requires non-empty work");
    }
    const std::size_t needed = 1 + (element_count - 1) / kThreadsPerBlock;
    return static_cast<unsigned int>(std::min(needed, kMaximumBlockCount));
}

inline unsigned int row_block_count_for(std::size_t row_count) {
    if (row_count == 0) {
        throw std::logic_error("CUDA attention launch requires non-empty rows");
    }
    return static_cast<unsigned int>(std::min(row_count, kMaximumBlockCount));
}

inline void require_kernel_launch(std::string_view operation) {
    require_cuda_success(cudaGetLastError(), operation);
}

inline void synchronize(std::string_view operation) {
    require_cuda_success(cudaDeviceSynchronize(), operation);
}

inline const float* require_native_input(const TensorStorage& storage,
                                         std::string_view operation) {
    const void* handle = storage.native_handle();
    if (handle == nullptr) {
        throw std::logic_error("CUDA " + std::string(operation) +
                               " received storage without a native allocation");
    }
    return static_cast<const float*>(handle);
}

inline float* require_native_output(TensorStorage& storage,
                                    std::string_view operation) {
    void* handle = storage.native_handle();
    if (handle == nullptr) {
        throw std::logic_error("CUDA " + std::string(operation) +
                               " received storage without a native allocation");
    }
    return static_cast<float*>(handle);
}

template <typename Element>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t element_count)
        : element_count_(element_count) {
        if (element_count_ == 0) {
            throw std::logic_error(
                "CUDA attention temporary storage must not be empty");
        }
        if (element_count_ >
            std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
            throw std::overflow_error(
                "CUDA attention temporary allocation size overflow");
        }
        void* allocation = nullptr;
        require_cuda_success(
            cudaMalloc(&allocation, element_count_ * sizeof(Element)),
            "attention temporary allocation");
        values_ = static_cast<Element*>(allocation);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (values_ != nullptr) {
            static_cast<void>(cudaFree(values_));
        }
    }

    [[nodiscard]] Element* data() noexcept { return values_; }
    [[nodiscard]] const Element* data() const noexcept { return values_; }
    [[nodiscard]] std::size_t size() const noexcept { return element_count_; }

    void zero(std::string_view operation) {
        require_cuda_success(
            cudaMemset(values_, 0, element_count_ * sizeof(Element)),
            operation);
    }

private:
    std::size_t element_count_;
    Element* values_ = nullptr;
};

inline bool read_status(const DeviceBuffer<unsigned int>& status,
                        std::string_view operation) {
    unsigned int host_status = 0;
    require_cuda_success(cudaMemcpy(&host_status,
                                    status.data(),
                                    sizeof(host_status),
                                    cudaMemcpyDeviceToHost),
                         operation);
    return host_status != 0;
}

}  // namespace riftco_transformer::backend_detail::attention_cuda_detail
