#include "core/backend/adapter.hpp"
#include "core/backend/attention/metal/launch.hpp"
#include "core/backend/nn/metal/launch.hpp"
#include "core/backend/nn/quantized_linear/metal/launch.hpp"
#include "core/backend/optim/adam/metal/diagnostics.hpp"
#include "core/backend/optim/adam/metal/kernels.hpp"
#include "core/backend/optim/adam/metal/launch.hpp"
#include "core/backend/optim/adam/reference/update.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

constexpr const char* kMatmulKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void riftco_transformer_batched_matmul(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong& rows [[buffer(3)]],
    constant ulong& shared [[buffer(4)]],
    constant ulong& columns [[buffer(5)]],
    uint3 position [[thread_position_in_grid]]
) {
    const ulong column = position.x;
    const ulong row = position.y;
    const ulong batch = position.z;
    if (row >= rows || column >= columns) {
        return;
    }

    const ulong left_offset = batch * rows * shared;
    const ulong right_offset = batch * shared * columns;
    const ulong output_offset = batch * rows * columns;
    float total = 0.0f;
    for (ulong inner = 0; inner < shared; ++inner) {
        total +=
            left[left_offset + row * shared + inner] *
            right[right_offset + inner * columns + column];
    }
    output[output_offset + row * columns + column] = total;
}
)METAL";

std::atomic<std::uint64_t> fused_adam_batches{0};
std::atomic<std::uint64_t> reference_adam_batches{0};

std::string error_description(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const char* description =
        error.localizedDescription.UTF8String;
    return description == nullptr
               ? "unknown Metal error"
               : std::string(description);
}

NSUInteger byte_count(std::size_t element_count) {
    if (element_count >
        std::numeric_limits<NSUInteger>::max() / sizeof(float)) {
        throw std::overflow_error(
            "Metal buffer size exceeds NSUInteger"
        );
    }
    return static_cast<NSUInteger>(
        element_count * sizeof(float)
    );
}

class MetalTensorStorage final : public TensorStorage {
public:
    MetalTensorStorage(
        id<MTLDevice> device,
        std::size_t element_count,
        float fill_value
    )
        : device_(device),
          element_count_(element_count),
          buffer_(
              [device
                  newBufferWithLength:byte_count(element_count)
                              options:MTLResourceStorageModeShared]
          ) {
        if (buffer_ == nil) {
            throw std::bad_alloc();
        }
        std::fill(data().begin(), data().end(), fill_value);
    }

    MetalTensorStorage(
        id<MTLDevice> device,
        std::vector<float> values
    )
        : device_(device),
          element_count_(values.size()),
          buffer_(
              [device
                  newBufferWithBytes:values.data()
                             length:byte_count(values.size())
                            options:MTLResourceStorageModeShared]
          ) {
        if (buffer_ == nil) {
            throw std::bad_alloc();
        }
    }

    [[nodiscard]] ExecutionBackend backend() const noexcept override {
        return ExecutionBackend::Metal;
    }

    [[nodiscard]] std::size_t size() const noexcept override {
        return element_count_;
    }

    [[nodiscard]] std::span<float> data() noexcept override {
        return {
            static_cast<float*>(buffer_.contents),
            element_count_,
        };
    }

    [[nodiscard]] std::span<const float> data() const noexcept override {
        return {
            static_cast<const float*>(buffer_.contents),
            element_count_,
        };
    }

    [[nodiscard]] void* native_handle() noexcept override {
        return (__bridge void*)buffer_;
    }

    [[nodiscard]] const void* native_handle() const noexcept override {
        return (__bridge const void*)buffer_;
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> clone() const override {
        return std::make_unique<MetalTensorStorage>(
            device_,
            std::vector<float>(data().begin(), data().end())
        );
    }

private:
    id<MTLDevice> device_ = nil;
    std::size_t element_count_ = 0;
    id<MTLBuffer> buffer_ = nil;
};

std::size_t checked_product(
    std::initializer_list<std::size_t> factors
) {
    std::size_t result = 1;
    for (const auto factor : factors) {
        if (factor != 0 &&
            result >
                std::numeric_limits<std::size_t>::max() / factor) {
            throw std::overflow_error(
                "Metal operation size exceeds addressable storage"
            );
        }
        result *= factor;
    }
    return result;
}

id<MTLBuffer> persistent_buffer(
    const TensorStorage& storage
) {
    if (storage.backend() != ExecutionBackend::Metal) {
        return nil;
    }
    const void* handle = storage.native_handle();
    if (handle == nullptr) {
        return nil;
    }
    return (__bridge id<MTLBuffer>)const_cast<void*>(handle);
}

id<MTLBuffer> persistent_buffer(TensorStorage& storage) {
    if (storage.backend() != ExecutionBackend::Metal) {
        return nil;
    }
    void* handle = storage.native_handle();
    if (handle == nullptr) {
        return nil;
    }
    return (__bridge id<MTLBuffer>)handle;
}

class MetalRuntime {
public:
    MetalRuntime()
        : device_(MTLCreateSystemDefaultDevice()) {
        if (device_ == nil) {
            throw std::runtime_error(
                "no Metal device is available"
            );
        }
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) {
            throw std::runtime_error(
                "could not create a Metal command queue"
            );
        }
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::size_t element_count,
        float fill_value
    ) const {
        @autoreleasepool {
            return std::make_unique<MetalTensorStorage>(
                device_,
                element_count,
                fill_value
            );
        }
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::vector<float> values
    ) const {
        @autoreleasepool {
            return std::make_unique<MetalTensorStorage>(
                device_,
                std::move(values)
            );
        }
    }

    void matmul(const MatmulRequest& request) {
        const auto& dimensions = request.dimensions;
        if (dimensions.batch_count >
                std::numeric_limits<std::uint32_t>::max() ||
            dimensions.rows >
                std::numeric_limits<std::uint32_t>::max() ||
            dimensions.columns >
                std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "Metal matmul grid dimensions exceed uint32"
            );
        }
        const auto expected_left = checked_product({
            dimensions.batch_count,
            dimensions.rows,
            dimensions.shared,
        });
        const auto expected_right = checked_product({
            dimensions.batch_count,
            dimensions.shared,
            dimensions.columns,
        });
        const auto expected_output = checked_product({
            dimensions.batch_count,
            dimensions.rows,
            dimensions.columns,
        });
        if (request.left.size() != expected_left ||
            request.right.size() != expected_right ||
            request.output.size() != expected_output) {
            throw std::logic_error(
                "Metal matmul storage size does not match dimensions"
            );
        }

        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const id<MTLComputePipelineState> pipeline =
                matmul_pipeline();
            id<MTLBuffer> left_buffer =
                persistent_buffer(request.left);
            id<MTLBuffer> right_buffer =
                persistent_buffer(request.right);
            id<MTLBuffer> output_buffer =
                persistent_buffer(request.output);
            const bool copy_output = output_buffer == nil;

            if (left_buffer == nil) {
                const auto left = request.left.data();
                left_buffer =
                    [device_
                        newBufferWithBytes:left.data()
                                    length:byte_count(left.size())
                                   options:MTLResourceStorageModeShared];
            }
            if (right_buffer == nil) {
                const auto right = request.right.data();
                right_buffer =
                    [device_
                        newBufferWithBytes:right.data()
                                    length:byte_count(right.size())
                                   options:MTLResourceStorageModeShared];
            }
            if (output_buffer == nil) {
                output_buffer =
                    [device_
                        newBufferWithLength:byte_count(
                                                request.output.size()
                                            )
                                    options:MTLResourceStorageModeShared];
            }
            if (left_buffer == nil ||
                right_buffer == nil ||
                output_buffer == nil) {
                throw std::bad_alloc();
            }

            id<MTLCommandBuffer> command_buffer =
                [queue_ commandBuffer];
            if (command_buffer == nil) {
                throw std::runtime_error(
                    "could not create Metal matmul command buffer"
                );
            }
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            if (encoder == nil) {
                throw std::runtime_error(
                    "could not create Metal matmul command encoder"
                );
            }

            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:left_buffer offset:0 atIndex:0];
            [encoder setBuffer:right_buffer offset:0 atIndex:1];
            [encoder setBuffer:output_buffer offset:0 atIndex:2];

            const std::uint64_t rows = dimensions.rows;
            const std::uint64_t shared = dimensions.shared;
            const std::uint64_t columns = dimensions.columns;
            [encoder setBytes:&rows
                       length:sizeof(rows)
                      atIndex:3];
            [encoder setBytes:&shared
                       length:sizeof(shared)
                      atIndex:4];
            [encoder setBytes:&columns
                       length:sizeof(columns)
                      atIndex:5];

            const MTLSize grid = MTLSizeMake(
                static_cast<NSUInteger>(dimensions.columns),
                static_cast<NSUInteger>(dimensions.rows),
                static_cast<NSUInteger>(dimensions.batch_count)
            );
            const NSUInteger thread_width =
                std::min<NSUInteger>(
                    8,
                    pipeline.threadExecutionWidth
                );
            const NSUInteger thread_height =
                std::max<NSUInteger>(
                    1,
                    std::min<NSUInteger>(
                        8,
                        pipeline.
                            maxTotalThreadsPerThreadgroup /
                            thread_width
                    )
                );
            [encoder
                dispatchThreads:grid
                threadsPerThreadgroup:
                    MTLSizeMake(
                        thread_width,
                        thread_height,
                        1
                    )];
            [encoder endEncoding];
            finish(command_buffer, "matmul");

            if (copy_output) {
                std::memcpy(
                    request.output.data().data(),
                    output_buffer.contents,
                    byte_count(request.output.size())
                );
            }
        }
    }

    void adam_update(const AdamUpdateRequest& request) {
        if (request.tensors.empty()) {
            throw std::invalid_argument(
                "Metal Adam requires at least one tensor"
            );
        }
        if (!std::isfinite(request.clip_scale) ||
            request.clip_scale <= 0.0 ||
            !std::isfinite(request.first_correction) ||
            request.first_correction <= 0.0 ||
            !std::isfinite(request.second_correction) ||
            request.second_correction <= 0.0) {
            throw std::invalid_argument(
                "Metal Adam received invalid scalar state"
            );
        }

        // Validate every native-buffer invariant before selecting or encoding
        // an execution path.
        for (const auto& tensor : request.tensors) {
            validate_adam_tensor(tensor);
            static_cast<void>(
                require_persistent_buffer(tensor.value)
            );
            static_cast<void>(
                require_persistent_buffer(tensor.gradient)
            );
            static_cast<void>(
                require_persistent_buffer(tensor.first_moment)
            );
            static_cast<void>(
                require_persistent_buffer(tensor.second_moment)
            );
            static_cast<void>(
                require_persistent_buffer(tensor.next_value)
            );
            static_cast<void>(
                require_persistent_buffer(
                    tensor.next_first_moment
                )
            );
            static_cast<void>(
                require_persistent_buffer(
                    tensor.next_second_moment
                )
            );
        }

        float clip_mantissa = 1.0F;
        std::int32_t clip_exponent = 0;
        if (request.clip_scale != 1.0) {
            int clip_exponent_value = 0;
            const double clip_mantissa_value =
                std::frexp(
                    request.clip_scale,
                    &clip_exponent_value
                );
            clip_mantissa =
                static_cast<float>(clip_mantissa_value);
            clip_exponent =
                static_cast<std::int32_t>(clip_exponent_value);
        }
        const float first_correction =
            static_cast<float>(request.first_correction);
        const float second_correction =
            static_cast<float>(request.second_correction);
        if (clip_mantissa == 0.0F ||
            !std::isfinite(clip_mantissa) ||
            first_correction == 0.0F ||
            !std::isfinite(first_correction) ||
            second_correction == 0.0F ||
            !std::isfinite(second_correction)) {
            adam_reference_update(request);
            reference_adam_batches.fetch_add(
                1,
                std::memory_order_relaxed
            );
            return;
        }

        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const id<MTLComputePipelineState> pipeline =
                adam_pipeline();
            id<MTLBuffer> reference_buffer =
                [device_
                    newBufferWithLength:sizeof(std::uint32_t)
                                options:MTLResourceStorageModeShared];
            if (reference_buffer == nil) {
                throw std::bad_alloc();
            }
            *static_cast<std::uint32_t*>(
                 reference_buffer.contents
             ) = 0;

            id<MTLCommandBuffer> command_buffer =
                [queue_ commandBuffer];
            if (command_buffer == nil) {
                throw std::runtime_error(
                    "could not create Metal Adam command buffer"
                );
            }
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            if (encoder == nil) {
                throw std::runtime_error(
                    "could not create Metal Adam command encoder"
                );
            }
            [encoder setComputePipelineState:pipeline];

            for (const auto& tensor : request.tensors) {
                const std::uint64_t element_count =
                    tensor.value.size();

                [encoder
                    setBuffer:require_persistent_buffer(
                                  tensor.value
                              )
                       offset:0
                      atIndex:0];
                [encoder
                    setBuffer:require_persistent_buffer(
                                  tensor.gradient
                              )
                       offset:0
                      atIndex:1];
                [encoder
                    setBuffer:require_persistent_buffer(
                                  tensor.first_moment
                              )
                       offset:0
                      atIndex:2];
                [encoder
                    setBuffer:require_persistent_buffer(
                                  tensor.second_moment
                              )
                       offset:0
                      atIndex:3];
                [encoder
                    setBuffer:require_persistent_buffer(
                                  tensor.next_value
                              )
                       offset:0
                      atIndex:4];
                [encoder
                    setBuffer:require_persistent_buffer(
                                  tensor.next_first_moment
                              )
                       offset:0
                      atIndex:5];
                [encoder
                    setBuffer:require_persistent_buffer(
                                  tensor.next_second_moment
                              )
                       offset:0
                      atIndex:6];
                [encoder setBuffer:reference_buffer
                            offset:0
                           atIndex:7];
                [encoder setBytes:&element_count
                           length:sizeof(element_count)
                          atIndex:8];
                [encoder setBytes:&request.learning_rate
                           length:sizeof(request.learning_rate)
                          atIndex:9];
                [encoder setBytes:&request.beta1
                           length:sizeof(request.beta1)
                          atIndex:10];
                [encoder setBytes:&request.beta2
                           length:sizeof(request.beta2)
                          atIndex:11];
                [encoder setBytes:&request.epsilon
                           length:sizeof(request.epsilon)
                          atIndex:12];
                [encoder setBytes:&clip_mantissa
                           length:sizeof(clip_mantissa)
                          atIndex:13];
                [encoder setBytes:&clip_exponent
                           length:sizeof(clip_exponent)
                          atIndex:14];
                [encoder setBytes:&first_correction
                           length:sizeof(first_correction)
                          atIndex:15];
                [encoder setBytes:&second_correction
                           length:sizeof(second_correction)
                          atIndex:16];

                const NSUInteger thread_count =
                    static_cast<NSUInteger>(tensor.value.size());
                const NSUInteger group_width =
                    std::max<NSUInteger>(
                        1,
                        std::min<NSUInteger>(
                            pipeline.
                                maxTotalThreadsPerThreadgroup,
                            pipeline.threadExecutionWidth
                        )
                    );
                [encoder
                    dispatchThreads:MTLSizeMake(
                                        thread_count,
                                        1,
                                        1
                                    )
                    threadsPerThreadgroup:
                        MTLSizeMake(group_width, 1, 1)];
            }

            [encoder endEncoding];
            finish(command_buffer, "Adam");
            if (*static_cast<const std::uint32_t*>(
                    reference_buffer.contents
                ) != 0) {
                // The kernel observed its actual precise-float operation
                // sequence becoming subnormal, non-finite, or ill
                // conditioned. Shared candidate buffers allow a wide
                // reference retry without migrating backend storage.
                adam_reference_update(request);
                reference_adam_batches.fetch_add(
                    1,
                    std::memory_order_relaxed
                );
                return;
            }
            fused_adam_batches.fetch_add(
                1,
                std::memory_order_relaxed
            );
        }
    }

private:
    [[nodiscard]] id<MTLComputePipelineState> make_pipeline(
        const char* source_text,
        NSString* function_name,
        const char* description,
        bool fast_math_enabled
    ) {
        NSString* source =
            [NSString stringWithUTF8String:source_text];
        MTLCompileOptions* options = [MTLCompileOptions new];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, *)) {
            options.mathMode =
                fast_math_enabled
                    ? MTLMathModeFast
                    : MTLMathModeSafe;
            options.mathFloatingPointFunctions =
                fast_math_enabled
                    ? MTLMathFloatingPointFunctionsFast
                    : MTLMathFloatingPointFunctionsPrecise;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled =
                fast_math_enabled ? YES : NO;
#pragma clang diagnostic pop
        }
#else
        options.fastMathEnabled =
            fast_math_enabled ? YES : NO;
#endif
        NSError* library_error = nil;
        id<MTLLibrary> library =
            [device_ newLibraryWithSource:source
                                  options:options
                                    error:&library_error];
        if (library == nil) {
            throw std::runtime_error(
                std::string("could not compile Metal ") +
                description + " kernel: " +
                error_description(library_error)
            );
        }
        id<MTLFunction> function =
            [library newFunctionWithName:function_name];
        if (function == nil) {
            throw std::runtime_error(
                std::string("compiled Metal library is missing ") +
                description + " kernel"
            );
        }
        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            [device_
                newComputePipelineStateWithFunction:function
                                               error:&pipeline_error];
        if (pipeline == nil) {
            throw std::runtime_error(
                std::string("could not create Metal ") +
                description + " pipeline: " +
                error_description(pipeline_error)
            );
        }
        return pipeline;
    }

    [[nodiscard]] id<MTLComputePipelineState> matmul_pipeline() {
        if (matmul_pipeline_error_ != nullptr) {
            std::rethrow_exception(matmul_pipeline_error_);
        }
        if (matmul_pipeline_ == nil) {
            try {
                matmul_pipeline_ = make_pipeline(
                    kMatmulKernelSource,
                    @"riftco_transformer_batched_matmul",
                    "matmul",
                    true
                );
            } catch (...) {
                matmul_pipeline_error_ = std::current_exception();
                throw;
            }
        }
        return matmul_pipeline_;
    }

    [[nodiscard]] id<MTLComputePipelineState> adam_pipeline() {
        if (adam_pipeline_error_ != nullptr) {
            std::rethrow_exception(adam_pipeline_error_);
        }
        if (adam_pipeline_ == nil) {
            try {
                adam_pipeline_ = make_pipeline(
                    adam_metal_detail::kAdamKernelSource,
                    @"riftco_transformer_adam_update",
                    "Adam",
                    false
                );
            } catch (...) {
                adam_pipeline_error_ = std::current_exception();
                throw;
            }
        }
        return adam_pipeline_;
    }

    static void finish(
        id<MTLCommandBuffer> command_buffer,
        const char* description
    ) {
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status !=
            MTLCommandBufferStatusCompleted) {
            throw std::runtime_error(
                std::string("Metal ") + description +
                " command failed: " +
                error_description(command_buffer.error)
            );
        }
    }

    static id<MTLBuffer> require_persistent_buffer(
        const TensorStorage& storage
    ) {
        if (storage.backend() != ExecutionBackend::Metal) {
            throw std::invalid_argument(
                "Metal Adam tensors must use Metal storage"
            );
        }
        id<MTLBuffer> buffer = persistent_buffer(storage);
        if (buffer == nil) {
            throw std::logic_error(
                "Metal tensor is missing its persistent buffer"
            );
        }
        return buffer;
    }

    static id<MTLBuffer> require_persistent_buffer(
        TensorStorage& storage
    ) {
        if (storage.backend() != ExecutionBackend::Metal) {
            throw std::invalid_argument(
                "Metal Adam tensors must use Metal storage"
            );
        }
        id<MTLBuffer> buffer = persistent_buffer(storage);
        if (buffer == nil) {
            throw std::logic_error(
                "Metal tensor is missing its persistent buffer"
            );
        }
        return buffer;
    }

    static void validate_adam_tensor(
        const AdamTensorUpdate& tensor
    ) {
        const auto element_count = tensor.value.size();
        if (element_count == 0 ||
            element_count >
                std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "Metal Adam tensor size exceeds uint32 grid range"
            );
        }
        if (tensor.gradient.size() != element_count ||
            tensor.first_moment.size() != element_count ||
            tensor.second_moment.size() != element_count ||
            tensor.next_value.size() != element_count ||
            tensor.next_first_moment.size() != element_count ||
            tensor.next_second_moment.size() != element_count) {
            throw std::logic_error(
                "Metal Adam tensor storage sizes do not match"
            );
        }
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> matmul_pipeline_ = nil;
    id<MTLComputePipelineState> adam_pipeline_ = nil;
    std::exception_ptr matmul_pipeline_error_;
    std::exception_ptr adam_pipeline_error_;
    std::mutex mutex_;
};

MetalRuntime& metal_runtime_instance() {
    static MetalRuntime instance;
    return instance;
}

class MetalBackendAdapter final : public BackendAdapter {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "metal";
    }

    [[nodiscard]] bool is_available() const noexcept override {
        @autoreleasepool {
            try {
                static_cast<void>(metal_runtime_instance());
                return true;
            } catch (...) {
                return false;
            }
        }
    }

    [[nodiscard]] std::string_view unavailability_reason()
        const noexcept override {
        if (is_available()) {
            return {};
        }
        return "no usable Metal device and command queue are available";
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::size_t element_count,
        float fill_value
    ) const override {
        return metal_runtime_instance().make_storage(
            element_count,
            fill_value
        );
    }

    [[nodiscard]] std::unique_ptr<QuantizedWeightStorage>
    make_nf4_weight_storage(
        std::vector<std::uint8_t> packed_codes,
        Nf4ScaleStorageData scales
    ) const override {
        @autoreleasepool {
            return metal_make_nf4_weight_storage(
                std::move(packed_codes),
                std::move(scales)
            );
        }
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::vector<float> values
    ) const override {
        return metal_runtime_instance().make_storage(
            std::move(values)
        );
    }

    void matmul(const MatmulRequest& request) const override {
        @autoreleasepool {
            metal_runtime_instance().matmul(request);
        }
    }

    void quantized_linear_forward(
        const QuantizedLinearForwardRequest& request
    ) const override {
        @autoreleasepool {
            metal_quantized_linear_forward(request);
        }
    }

    void quantized_linear_input_backward(
        const QuantizedLinearInputBackwardRequest& request
    ) const override {
        @autoreleasepool {
            metal_quantized_linear_input_backward(request);
        }
    }

    void unary_elementwise(
        const UnaryElementwiseRequest& request
    ) const override {
        metal_nn_unary_elementwise(request);
    }

    void binary_elementwise(
        const BinaryElementwiseRequest& request
    ) const override {
        metal_nn_binary_elementwise(request);
    }

    void scale(const ScaleRequest& request) const override {
        metal_nn_scale(request);
    }

    void gelu_forward(
        const GeluForwardRequest& request
    ) const override {
        metal_nn_gelu_forward(request);
    }

    void gelu_backward(
        const GeluBackwardRequest& request
    ) const override {
        metal_nn_gelu_backward(request);
    }

    void reduce(const ReductionRequest& request) const override {
        metal_nn_reduce(request);
    }

    void copy(const CopyRequest& request) const override {
        metal_nn_copy(request);
    }

    void permute(const PermuteRequest& request) const override {
        metal_nn_permute(request);
    }

    void broadcast(
        const BroadcastRequest& request
    ) const override {
        metal_nn_broadcast(request);
    }

    void sum_to_shape(
        const SumToShapeRequest& request
    ) const override {
        metal_nn_sum_to_shape(request);
    }

    void softmax_forward(
        const SoftmaxForwardRequest& request
    ) const override {
        metal_nn_softmax_forward(request);
    }

    void softmax_backward(
        const SoftmaxBackwardRequest& request
    ) const override {
        metal_nn_softmax_backward(request);
    }

    void causal_softmax_forward(
        const CausalSoftmaxForwardRequest& request
    ) const override {
        metal_nn_causal_softmax_forward(request);
    }

    void causal_softmax_backward(
        const CausalSoftmaxBackwardRequest& request
    ) const override {
        metal_nn_causal_softmax_backward(request);
    }

    void gather_rows(
        const GatherRowsRequest& request
    ) const override {
        metal_nn_gather_rows(request);
    }

    void scatter_add_rows(
        const ScatterAddRowsRequest& request
    ) const override {
        metal_nn_scatter_add_rows(request);
    }

    void layer_norm_forward(
        const LayerNormForwardRequest& request
    ) const override {
        metal_nn_layer_norm_forward(request);
    }

    void layer_norm_backward(
        const LayerNormBackwardRequest& request
    ) const override {
        metal_nn_layer_norm_backward(request);
    }

    void cross_entropy_forward(
        const CrossEntropyForwardRequest& request
    ) const override {
        metal_nn_cross_entropy_forward(request);
    }

    void materialized_causal_attention_forward(
        const MaterializedCausalAttentionForwardRequest& request
    ) const override {
        metal_attention_materialized_causal_forward(request);
    }

    void materialized_causal_attention_context_backward(
        const MaterializedCausalAttentionContextBackwardRequest& request
    ) const override {
        metal_attention_materialized_causal_context_backward(request);
    }

    void materialized_causal_attention_probabilities_backward(
        const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
    ) const override {
        metal_attention_materialized_causal_probabilities_backward(request);
    }

    void flash_causal_attention_forward(
        const FlashCausalAttentionForwardRequest& request
    ) const override {
        metal_attention_flash_causal_forward(request);
    }

    void flash_causal_attention_backward(
        const FlashCausalAttentionBackwardRequest& request
    ) const override {
        metal_attention_flash_causal_backward(request);
    }

    void paged_decode_attention_forward(
        const PagedDecodeAttentionForwardRequest& request
    ) const override {
        metal_attention_paged_decode_forward(request);
    }

    void adam_update(
        const AdamUpdateRequest& request
    ) const override {
        @autoreleasepool {
            metal_adam_update(request);
        }
    }
};

}  // namespace

const BackendAdapter& metal_backend_adapter() noexcept {
    static const MetalBackendAdapter adapter;
    return adapter;
}

void metal_adam_update(const AdamUpdateRequest& request) {
    metal_runtime_instance().adam_update(request);
}

void reset_metal_adam_path_counts() noexcept {
    fused_adam_batches.store(0, std::memory_order_relaxed);
    reference_adam_batches.store(0, std::memory_order_relaxed);
}

MetalAdamPathCounts metal_adam_path_counts() noexcept {
    return {
        fused_adam_batches.load(std::memory_order_relaxed),
        reference_adam_batches.load(std::memory_order_relaxed),
    };
}

}  // namespace riftco_transformer::backend_detail
