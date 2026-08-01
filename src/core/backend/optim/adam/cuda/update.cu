#include "core/backend/optim/adam/cuda/launch.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumBlockCount = 65'535;

[[noreturn]] void throw_cuda_error(cudaError_t status,
                                   std::string_view operation) {
    if (status == cudaErrorMemoryAllocation) {
        throw std::bad_alloc();
    }
    throw std::runtime_error("CUDA " + std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
}

void require_cuda_success(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw_cuda_error(status, operation);
    }
}

class DeviceStatus final {
  public:
    DeviceStatus() {
        void* allocation = nullptr;
        require_cuda_success(cudaMalloc(&allocation, sizeof(unsigned int)),
                             "Adam status allocation");
        value_ = static_cast<unsigned int*>(allocation);
        try {
            require_cuda_success(cudaMemset(value_, 0, sizeof(unsigned int)),
                                 "Adam status initialization");
        } catch (...) {
            static_cast<void>(cudaFree(value_));
            value_ = nullptr;
            throw;
        }
    }

    DeviceStatus(const DeviceStatus&) = delete;
    DeviceStatus& operator=(const DeviceStatus&) = delete;

    ~DeviceStatus() {
        if (value_ != nullptr) {
            static_cast<void>(cudaFree(value_));
        }
    }

    [[nodiscard]] unsigned int* data() noexcept {
        return value_;
    }

    [[nodiscard]] bool requested_overflow() const {
        unsigned int result = 0;
        require_cuda_success(
            cudaMemcpy(&result, value_, sizeof(result), cudaMemcpyDeviceToHost),
            "Adam status download");
        return result != 0;
    }

  private:
    unsigned int* value_ = nullptr;
};

struct NativeAdamTensorUpdate {
    const float* value;
    const float* gradient;
    const float* first_moment;
    const float* second_moment;
    float* next_value;
    float* next_first_moment;
    float* next_second_moment;
    std::size_t element_count;
};

const float* require_native_input(const TensorStorage& storage,
                                  std::string_view description) {
    const void* handle = storage.native_handle();
    if (handle == nullptr) {
        throw std::logic_error("CUDA Adam " + std::string(description) +
                               " is missing its native allocation");
    }
    return static_cast<const float*>(handle);
}

float* require_native_output(TensorStorage& storage,
                             std::string_view description) {
    void* handle = storage.native_handle();
    if (handle == nullptr) {
        throw std::logic_error("CUDA Adam " + std::string(description) +
                               " is missing its native allocation");
    }
    return static_cast<float*>(handle);
}

NativeAdamTensorUpdate require_native_tensor(const AdamTensorUpdate& tensor) {
    const std::size_t element_count = tensor.value.size();
    if (element_count == 0) {
        throw std::invalid_argument("CUDA Adam tensors must not be empty");
    }
    if (tensor.value.backend() != ExecutionBackend::Cuda ||
        tensor.gradient.backend() != ExecutionBackend::Cuda ||
        tensor.first_moment.backend() != ExecutionBackend::Cuda ||
        tensor.second_moment.backend() != ExecutionBackend::Cuda ||
        tensor.next_value.backend() != ExecutionBackend::Cuda ||
        tensor.next_first_moment.backend() != ExecutionBackend::Cuda ||
        tensor.next_second_moment.backend() != ExecutionBackend::Cuda) {
        throw std::invalid_argument("CUDA Adam tensors must use CUDA storage");
    }
    if (tensor.gradient.size() != element_count ||
        tensor.first_moment.size() != element_count ||
        tensor.second_moment.size() != element_count ||
        tensor.next_value.size() != element_count ||
        tensor.next_first_moment.size() != element_count ||
        tensor.next_second_moment.size() != element_count) {
        throw std::logic_error("CUDA Adam tensor storage sizes do not match");
    }

    return {
        require_native_input(tensor.value, "parameter value"),
        require_native_input(tensor.gradient, "gradient"),
        require_native_input(tensor.first_moment, "first moment"),
        require_native_input(tensor.second_moment, "second moment"),
        require_native_output(tensor.next_value, "next parameter value"),
        require_native_output(tensor.next_first_moment, "next first moment"),
        require_native_output(tensor.next_second_moment, "next second moment"),
        element_count,
    };
}

__global__ void
adam_update_kernel(const float* value, const float* gradient,
                   const float* first_moment, const float* second_moment,
                   float* next_value, float* next_first_moment,
                   float* next_second_moment, std::size_t element_count,
                   float learning_rate, float beta1, float beta2, float epsilon,
                   double clip_scale, double first_correction,
                   double second_correction, unsigned int* overflow_requested) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    while (index < element_count) {
        const double clipped_gradient =
            static_cast<double>(gradient[index]) * clip_scale;
        const double first_value =
            static_cast<double>(beta1) *
                static_cast<double>(first_moment[index]) +
            (1.0 - static_cast<double>(beta1)) * clipped_gradient;
        const double second_value =
            static_cast<double>(beta2) *
                static_cast<double>(second_moment[index]) +
            (1.0 - static_cast<double>(beta2)) * clipped_gradient *
                clipped_gradient;
        const double corrected_first = first_value / first_correction;
        const double corrected_second = second_value / second_correction;
        const double update =
            static_cast<double>(learning_rate) * corrected_first /
            (sqrt(corrected_second) + static_cast<double>(epsilon));
        const double updated_value = static_cast<double>(value[index]) - update;

        const float stored_first = static_cast<float>(first_value);
        const float stored_second = static_cast<float>(second_value);
        const float stored_value = static_cast<float>(updated_value);
        if (!isfinite(first_value) || !isfinite(second_value) ||
            !isfinite(updated_value) || !isfinite(stored_first) ||
            !isfinite(stored_second) || !isfinite(stored_value)) {
            atomicExch(overflow_requested, 1U);
        } else {
            next_first_moment[index] = stored_first;
            next_second_moment[index] = stored_second;
            next_value[index] = stored_value;
        }
        index += stride;
    }
}

unsigned int block_count_for(std::size_t element_count) {
    const std::size_t needed = 1 + (element_count - 1) / kThreadsPerBlock;
    return static_cast<unsigned int>(std::min(needed, kMaximumBlockCount));
}

void launch_tensor_update(const NativeAdamTensorUpdate& tensor,
                          const AdamUpdateRequest& request,
                          unsigned int* overflow_requested) {
    const unsigned int block_count = block_count_for(tensor.element_count);
    adam_update_kernel<<<block_count, kThreadsPerBlock>>>(
        tensor.value, tensor.gradient, tensor.first_moment,
        tensor.second_moment, tensor.next_value, tensor.next_first_moment,
        tensor.next_second_moment, tensor.element_count, request.learning_rate,
        request.beta1, request.beta2, request.epsilon, request.clip_scale,
        request.first_correction, request.second_correction,
        overflow_requested);
    require_cuda_success(cudaGetLastError(), "Adam update kernel launch");
}

} // namespace

void cuda_adam_update(const AdamUpdateRequest& request) {
    if (request.tensors.empty()) {
        throw std::invalid_argument("CUDA Adam requires at least one tensor");
    }

    // Resolve and validate every native allocation before any device work is
    // launched. A bad later tensor therefore cannot leave an earlier launch in
    // flight when this function reports a precondition failure.
    std::vector<NativeAdamTensorUpdate> native_tensors;
    native_tensors.reserve(request.tensors.size());
    for (const auto& tensor : request.tensors) {
        native_tensors.push_back(require_native_tensor(tensor));
    }

    DeviceStatus status;
    for (const auto& tensor : native_tensors) {
        launch_tensor_update(tensor, request, status.data());
    }
    require_cuda_success(cudaDeviceSynchronize(),
                         "Adam update synchronization");
    if (status.requested_overflow()) {
        throw std::overflow_error("Adam produced a non-finite CUDA candidate");
    }
}

} // namespace riftco_transformer::backend_detail
