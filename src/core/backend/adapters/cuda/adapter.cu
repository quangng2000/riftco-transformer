#include "core/backend/adapter.hpp"
#include "core/backend/attention/cuda/launch.hpp"
#include "core/backend/nn/cuda/launch.hpp"
#include "core/backend/nn/quantized_linear/cuda/launch.hpp"
#include "core/backend/optim/adam/cuda/launch.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

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

bool cuda_device_available() noexcept {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        // Availability probes are intentionally non-throwing. Clear the
        // thread-local runtime error so a later successful call is not blamed
        // for an absent driver or device.
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return device_count > 0;
}

class CudaManagedTensorStorage final : public TensorStorage {
public:
    CudaManagedTensorStorage(std::size_t element_count, float fill_value)
        : element_count_(element_count) {
        allocate();
        if (element_count_ != 0) {
            std::fill_n(values_, element_count_, fill_value);
        }
    }

    explicit CudaManagedTensorStorage(std::vector<float> values)
        : element_count_(values.size()) {
        allocate();
        if (element_count_ != 0) {
            std::copy(values.begin(), values.end(), values_);
        }
    }

    CudaManagedTensorStorage(const CudaManagedTensorStorage&) = delete;
    CudaManagedTensorStorage& operator=(const CudaManagedTensorStorage&) =
        delete;

    ~CudaManagedTensorStorage() override {
        if (values_ != nullptr) {
            static_cast<void>(cudaFree(values_));
        }
    }

    [[nodiscard]] ExecutionBackend backend() const noexcept override {
        return ExecutionBackend::Cuda;
    }

    [[nodiscard]] std::size_t size() const noexcept override {
        return element_count_;
    }

    [[nodiscard]] std::span<float> data() noexcept override {
        return {values_, element_count_};
    }

    [[nodiscard]] std::span<const float> data() const noexcept override {
        return {values_, element_count_};
    }

    [[nodiscard]] void* native_handle() noexcept override { return values_; }

    [[nodiscard]] const void* native_handle() const noexcept override {
        return values_;
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> clone() const override {
        if (element_count_ == 0) {
            return std::make_unique<CudaManagedTensorStorage>(0, 0.0F);
        }
        return std::make_unique<CudaManagedTensorStorage>(
            std::vector<float>(values_, values_ + element_count_));
    }

private:
    void allocate() {
        if (element_count_ == 0) {
            return;
        }
        if (element_count_ >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("CUDA tensor allocation size overflow");
        }
        void* allocation = nullptr;
        require_cuda_success(cudaMallocManaged(&allocation,
                                               element_count_ * sizeof(float),
                                               cudaMemAttachGlobal),
                             "managed allocation");
        values_ = static_cast<float*>(allocation);
    }

    std::size_t element_count_ = 0;
    float* values_ = nullptr;
};

__global__ void batched_matmul_kernel(const float* left,
                                      const float* right,
                                      float* output,
                                      std::size_t output_count,
                                      std::size_t rows,
                                      std::size_t shared,
                                      std::size_t columns) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t matrix_size = rows * columns;
    while (output_index < output_count) {
        const std::size_t batch = output_index / matrix_size;
        const std::size_t within_matrix = output_index % matrix_size;
        const std::size_t row = within_matrix / columns;
        const std::size_t column = within_matrix % columns;
        const std::size_t left_offset = batch * rows * shared;
        const std::size_t right_offset = batch * shared * columns;

        float total = 0.0F;
        for (std::size_t inner = 0; inner < shared; ++inner) {
            total += left[left_offset + row * shared + inner] *
                     right[right_offset + inner * columns + column];
        }
        output[output_index] = total;
        output_index += stride;
    }
}

void launch_native_batched_matmul(const MatmulRequest& request) {
    constexpr unsigned int threads_per_block = 256;
    constexpr std::size_t maximum_block_count = 65535;
    const std::size_t output_count = request.output.size();
    const std::size_t needed_blocks =
        1 + (output_count - 1) / threads_per_block;
    const unsigned int block_count =
        static_cast<unsigned int>(std::min(needed_blocks, maximum_block_count));
    const auto& dimensions = request.dimensions;

    batched_matmul_kernel<<<block_count, threads_per_block>>>(
        request.left.data().data(),
        request.right.data().data(),
        request.output.data().data(),
        output_count,
        dimensions.rows,
        dimensions.shared,
        dimensions.columns);
    require_cuda_success(cudaGetLastError(), "matmul kernel launch");
    require_cuda_success(cudaDeviceSynchronize(), "matmul synchronization");
}

void launch_batched_matmul(const MatmulRequest& request) {
    if (request.left.backend() == ExecutionBackend::Cuda) {
        if (request.right.backend() != ExecutionBackend::Cuda ||
            request.output.backend() != ExecutionBackend::Cuda ||
            request.left.native_handle() == nullptr ||
            request.right.native_handle() == nullptr ||
            request.output.native_handle() == nullptr) {
            throw std::logic_error(
                "CUDA matmul received invalid native storage");
        }
        launch_native_batched_matmul(request);
        return;
    }

    // tensor_ops::matmul permits an explicit execution backend that differs
    // from the tensors' storage backend. Stage host-visible CPU or Metal
    // storage through managed CUDA allocations before launching the kernel,
    // then return the result to the original output storage.
    const auto left_values = request.left.data();
    const auto right_values = request.right.data();
    CudaManagedTensorStorage staged_left(
        std::vector<float>(left_values.begin(), left_values.end()));
    CudaManagedTensorStorage staged_right(
        std::vector<float>(right_values.begin(), right_values.end()));
    CudaManagedTensorStorage staged_output(request.output.size(), 0.0F);
    launch_native_batched_matmul({
        staged_left,
        staged_right,
        staged_output,
        request.dimensions,
    });
    const auto staged_values = staged_output.data();
    std::copy(staged_values.begin(),
              staged_values.end(),
              request.output.data().begin());
}

class CudaBackendAdapter final : public BackendAdapter {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "cuda";
    }

    [[nodiscard]] bool is_available() const noexcept override {
        return cuda_device_available();
    }

    [[nodiscard]] std::string_view unavailability_reason()
        const noexcept override {
        if (cuda_device_available()) {
            return {};
        }
        return "no CUDA device is visible to the CUDA runtime";
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::size_t element_count,
        float fill_value) const override {
        return std::make_unique<CudaManagedTensorStorage>(element_count,
                                                          fill_value);
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::vector<float> values) const override {
        return std::make_unique<CudaManagedTensorStorage>(std::move(values));
    }

    [[nodiscard]] std::unique_ptr<QuantizedWeightStorage>
    make_nf4_weight_storage(
        std::vector<std::uint8_t> packed_codes,
        Nf4ScaleStorageData scales
    ) const override {
        return cuda_make_nf4_weight_storage(
            std::move(packed_codes),
            std::move(scales)
        );
    }

    void matmul(const MatmulRequest& request) const override {
        launch_batched_matmul(request);
    }

    void quantized_linear_forward(
        const QuantizedLinearForwardRequest& request
    ) const override {
        cuda_quantized_linear_forward(request);
    }

    void quantized_linear_input_backward(
        const QuantizedLinearInputBackwardRequest& request
    ) const override {
        cuda_quantized_linear_input_backward(request);
    }

    // Managed allocations preserve the public host-visible tensor contract;
    // every operation below still executes through a synchronous CUDA kernel.
    void unary_elementwise(
        const UnaryElementwiseRequest& request) const override {
        cuda_nn_unary_elementwise(request);
    }
    void binary_elementwise(
        const BinaryElementwiseRequest& request) const override {
        cuda_nn_binary_elementwise(request);
    }
    void scale(const ScaleRequest& request) const override {
        cuda_nn_scale(request);
    }
    void gelu_forward(const GeluForwardRequest& request) const override {
        cuda_nn_gelu_forward(request);
    }
    void gelu_backward(const GeluBackwardRequest& request) const override {
        cuda_nn_gelu_backward(request);
    }
    void reduce(const ReductionRequest& request) const override {
        cuda_nn_reduce(request);
    }
    void copy(const CopyRequest& request) const override {
        cuda_nn_copy(request);
    }
    void permute(const PermuteRequest& request) const override {
        cuda_nn_permute(request);
    }
    void broadcast(const BroadcastRequest& request) const override {
        cuda_nn_broadcast(request);
    }
    void sum_to_shape(const SumToShapeRequest& request) const override {
        cuda_nn_sum_to_shape(request);
    }
    void softmax_forward(const SoftmaxForwardRequest& request) const override {
        cuda_nn_softmax_forward(request);
    }
    void softmax_backward(
        const SoftmaxBackwardRequest& request) const override {
        cuda_nn_softmax_backward(request);
    }
    void causal_softmax_forward(
        const CausalSoftmaxForwardRequest& request) const override {
        cuda_nn_causal_softmax_forward(request);
    }
    void causal_softmax_backward(
        const CausalSoftmaxBackwardRequest& request) const override {
        cuda_nn_causal_softmax_backward(request);
    }
    void gather_rows(const GatherRowsRequest& request) const override {
        cuda_nn_gather_rows(request);
    }
    void scatter_add_rows(const ScatterAddRowsRequest& request) const override {
        cuda_nn_scatter_add_rows(request);
    }
    void layer_norm_forward(
        const LayerNormForwardRequest& request) const override {
        cuda_nn_layer_norm_forward(request);
    }
    void layer_norm_backward(
        const LayerNormBackwardRequest& request) const override {
        cuda_nn_layer_norm_backward(request);
    }
    void cross_entropy_forward(
        const CrossEntropyForwardRequest& request) const override {
        cuda_nn_cross_entropy_forward(request);
    }
    void materialized_causal_attention_forward(
        const MaterializedCausalAttentionForwardRequest& request)
        const override {
        cuda_materialized_causal_attention_forward(request);
    }
    void materialized_causal_attention_context_backward(
        const MaterializedCausalAttentionContextBackwardRequest& request)
        const override {
        cuda_materialized_causal_attention_context_backward(request);
    }
    void materialized_causal_attention_probabilities_backward(
        const MaterializedCausalAttentionProbabilitiesBackwardRequest& request)
        const override {
        cuda_materialized_causal_attention_probabilities_backward(request);
    }
    void flash_causal_attention_forward(
        const FlashCausalAttentionForwardRequest& request) const override {
        cuda_flash_causal_attention_forward(request);
    }
    void flash_causal_attention_backward(
        const FlashCausalAttentionBackwardRequest& request) const override {
        cuda_flash_causal_attention_backward(request);
    }
    void paged_decode_attention_forward(
        const PagedDecodeAttentionForwardRequest& request) const override {
        cuda_paged_decode_attention_forward(request);
    }
    void adam_update(const AdamUpdateRequest& request) const override {
        cuda_adam_update(request);
    }
};

}  // namespace

const BackendAdapter& cuda_backend_adapter() noexcept {
    static const CudaBackendAdapter adapter;
    return adapter;
}

}  // namespace riftco_transformer::backend_detail
