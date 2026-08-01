#include "core/backend/nn/quantized_linear/metal/launch.hpp"

#include "core/backend/nn/quantized_linear/metal/kernels.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
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

[[nodiscard]] std::string error_description(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const char* description = error.localizedDescription.UTF8String;
    return description == nullptr
               ? "unknown Metal error"
               : std::string(description);
}

template <typename Value>
[[nodiscard]] NSUInteger checked_bytes(std::size_t count) {
    if (count >
        std::numeric_limits<NSUInteger>::max() / sizeof(Value)) {
        throw std::overflow_error(
            "Metal NF4 buffer size exceeds NSUInteger"
        );
    }
    return static_cast<NSUInteger>(count * sizeof(Value));
}

class MetalNf4WeightStorage final : public QuantizedWeightStorage {
public:
    MetalNf4WeightStorage(
        id<MTLDevice> device,
        std::vector<std::uint8_t> packed_codes,
        Nf4ScaleStorageData scales
    )
        : packed_code_count_(packed_codes.size()),
          scale_encoding_(scales.encoding),
          primary_scale_count_(
              scales.encoding == Nf4ScaleEncoding::Float32
                  ? scales.fp32_scales.size()
                  : scales.quantized_scales.size()
          ),
          second_level_scale_count_(scales.second_level_scales.size()),
          second_level_block_size_(scales.second_level_block_size),
          offset_(scales.offset),
          packed_codes_(
              [device
                  newBufferWithBytes:packed_codes.data()
                             length:checked_bytes<std::uint8_t>(
                                        packed_codes.size()
                                    )
                            options:MTLResourceStorageModeShared]
          ) {
        if (scale_encoding_ == Nf4ScaleEncoding::Float32) {
            primary_scales_ = [device
                newBufferWithBytes:scales.fp32_scales.data()
                           length:checked_bytes<float>(
                                      scales.fp32_scales.size()
                                  )
                          options:MTLResourceStorageModeShared];
        } else {
            primary_scales_ = [device
                newBufferWithBytes:scales.quantized_scales.data()
                           length:checked_bytes<std::uint8_t>(
                                      scales.quantized_scales.size()
                                  )
                          options:MTLResourceStorageModeShared];
            second_level_scales_ = [device
                newBufferWithBytes:scales.second_level_scales.data()
                           length:checked_bytes<float>(
                                      scales.second_level_scales.size()
                                  )
                          options:MTLResourceStorageModeShared];
        }
        if (packed_codes_ == nil || primary_scales_ == nil ||
            (scale_encoding_ == Nf4ScaleEncoding::DoubleQuantizedUInt8 &&
             second_level_scales_ == nil)) {
            throw std::bad_alloc();
        }
    }

    [[nodiscard]] ExecutionBackend backend() const noexcept override {
        return ExecutionBackend::Metal;
    }

    [[nodiscard]] std::span<const std::uint8_t>
    packed_codes() const noexcept override {
        return {
            static_cast<const std::uint8_t*>(packed_codes_.contents),
            packed_code_count_,
        };
    }

    [[nodiscard]] Nf4ScaleStorageView
    scale_storage() const noexcept override {
        if (scale_encoding_ == Nf4ScaleEncoding::Float32) {
            return {
                scale_encoding_,
                {static_cast<const float*>(primary_scales_.contents),
                 primary_scale_count_},
                {},
                {},
                0,
                0.0F,
            };
        }
        return {
            scale_encoding_,
            {},
            {static_cast<const std::uint8_t*>(primary_scales_.contents),
             primary_scale_count_},
            {static_cast<const float*>(second_level_scales_.contents),
             second_level_scale_count_},
            second_level_block_size_,
            offset_,
        };
    }

    [[nodiscard]] const void*
    packed_codes_native_handle() const noexcept override {
        return (__bridge const void*)packed_codes_;
    }

    [[nodiscard]] const void*
    primary_scales_native_handle() const noexcept override {
        return (__bridge const void*)primary_scales_;
    }

    [[nodiscard]] const void*
    secondary_scales_native_handle() const noexcept override {
        return (__bridge const void*)second_level_scales_;
    }

    [[nodiscard]] std::size_t
    resident_payload_bytes() const noexcept override {
        return packed_code_count_ + nf4_scale_payload_bytes(scale_storage());
    }

private:
    std::size_t packed_code_count_ = 0;
    Nf4ScaleEncoding scale_encoding_ = Nf4ScaleEncoding::Float32;
    std::size_t primary_scale_count_ = 0;
    std::size_t second_level_scale_count_ = 0;
    std::size_t second_level_block_size_ = 0;
    float offset_ = 0.0F;
    id<MTLBuffer> packed_codes_ = nil;
    id<MTLBuffer> primary_scales_ = nil;
    id<MTLBuffer> second_level_scales_ = nil;
};

class MetalQuantizedLinearRuntime {
public:
    MetalQuantizedLinearRuntime()
        : device_(MTLCreateSystemDefaultDevice()) {
        if (device_ == nil) {
            throw std::runtime_error("no Metal device is available");
        }
        queue_ = [device_ newCommandQueue];
        if (queue_ == nil) {
            throw std::runtime_error(
                "could not create a Metal quantized-linear command queue"
            );
        }
    }

    [[nodiscard]] std::unique_ptr<QuantizedWeightStorage>
    make_nf4_weight_storage(
        std::vector<std::uint8_t> packed_codes,
        Nf4ScaleStorageData scales
    ) const {
        @autoreleasepool {
            return std::make_unique<MetalNf4WeightStorage>(
                device_,
                std::move(packed_codes),
                std::move(scales)
            );
        }
    }

    void forward(const QuantizedLinearForwardRequest& request) {
        encode(
            request.input,
            request.weight,
            request.output,
            request.dimensions,
            request.output.size(),
            false,
            "NF4 quantized-linear forward"
        );
    }

    void input_backward(
        const QuantizedLinearInputBackwardRequest& request
    ) {
        encode(
            request.upstream,
            request.weight,
            request.input_gradient,
            request.dimensions,
            request.input_gradient.size(),
            true,
            "NF4 quantized-linear input backward"
        );
    }

private:
    [[nodiscard]] id<MTLLibrary> library() {
        if (library_error_ != nullptr) {
            std::rethrow_exception(library_error_);
        }
        if (library_ != nil) {
            return library_;
        }
        try {
            NSString* source = [NSString
                stringWithUTF8String:
                    quantized_linear_metal_detail::
                        kQuantizedLinearKernelSource];
            MTLCompileOptions* options = [MTLCompileOptions new];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
                options.mathFloatingPointFunctions =
                    MTLMathFloatingPointFunctionsPrecise;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
#else
            options.fastMathEnabled = NO;
#endif
            NSError* error = nil;
            library_ = [device_ newLibraryWithSource:source
                                             options:options
                                               error:&error];
            if (library_ == nil) {
                throw std::runtime_error(
                    "could not compile Metal quantized-linear kernels: " +
                    error_description(error)
                );
            }
        } catch (...) {
            library_error_ = std::current_exception();
            throw;
        }
        return library_;
    }

    [[nodiscard]] id<MTLComputePipelineState> make_pipeline(
        NSString* function_name,
        const char* description
    ) {
        id<MTLFunction> function =
            [library() newFunctionWithName:function_name];
        if (function == nil) {
            throw std::runtime_error(
                std::string("compiled Metal library is missing ") +
                description + " kernel"
            );
        }
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device_ newComputePipelineStateWithFunction:function
                                                    error:&error];
        if (pipeline == nil) {
            throw std::runtime_error(
                std::string("could not create Metal ") + description +
                " pipeline: " + error_description(error)
            );
        }
        return pipeline;
    }

    [[nodiscard]] id<MTLComputePipelineState> forward_pipeline() {
        if (forward_pipeline_error_ != nullptr) {
            std::rethrow_exception(forward_pipeline_error_);
        }
        if (forward_pipeline_ == nil) {
            try {
                forward_pipeline_ = make_pipeline(
                    @"rt_nf4_linear_forward",
                    "NF4 quantized-linear forward"
                );
            } catch (...) {
                forward_pipeline_error_ = std::current_exception();
                throw;
            }
        }
        return forward_pipeline_;
    }

    [[nodiscard]] id<MTLComputePipelineState>
    input_backward_pipeline() {
        if (input_backward_pipeline_error_ != nullptr) {
            std::rethrow_exception(input_backward_pipeline_error_);
        }
        if (input_backward_pipeline_ == nil) {
            try {
                input_backward_pipeline_ = make_pipeline(
                    @"rt_nf4_linear_input_backward",
                    "NF4 quantized-linear input backward"
                );
            } catch (...) {
                input_backward_pipeline_error_ =
                    std::current_exception();
                throw;
            }
        }
        return input_backward_pipeline_;
    }

    [[nodiscard]] id<MTLComputePipelineState>
    double_quantized_forward_pipeline() {
        if (double_quantized_forward_pipeline_error_ != nullptr) {
            std::rethrow_exception(double_quantized_forward_pipeline_error_);
        }
        if (double_quantized_forward_pipeline_ == nil) {
            try {
                double_quantized_forward_pipeline_ = make_pipeline(
                    @"rt_nf4_double_quantized_linear_forward",
                    "double-quantized NF4 linear forward"
                );
            } catch (...) {
                double_quantized_forward_pipeline_error_ =
                    std::current_exception();
                throw;
            }
        }
        return double_quantized_forward_pipeline_;
    }

    [[nodiscard]] id<MTLComputePipelineState>
    double_quantized_input_backward_pipeline() {
        if (double_quantized_input_backward_pipeline_error_ != nullptr) {
            std::rethrow_exception(
                double_quantized_input_backward_pipeline_error_
            );
        }
        if (double_quantized_input_backward_pipeline_ == nil) {
            try {
                double_quantized_input_backward_pipeline_ = make_pipeline(
                    @"rt_nf4_double_quantized_linear_input_backward",
                    "double-quantized NF4 linear input backward"
                );
            } catch (...) {
                double_quantized_input_backward_pipeline_error_ =
                    std::current_exception();
                throw;
            }
        }
        return double_quantized_input_backward_pipeline_;
    }

    [[nodiscard]] static id<MTLBuffer> require_tensor_buffer(
        const TensorStorage& storage
    ) {
        if (storage.backend() != ExecutionBackend::Metal) {
            throw std::invalid_argument(
                "Metal quantized linear requires Metal tensor storage"
            );
        }
        const void* handle = storage.native_handle();
        if (handle == nullptr) {
            throw std::logic_error(
                "Metal tensor is missing its persistent buffer"
            );
        }
        return (__bridge id<MTLBuffer>)const_cast<void*>(handle);
    }

    [[nodiscard]] static id<MTLBuffer> require_weight_buffer(
        const void* handle,
        const char* description
    ) {
        if (handle == nullptr) {
            throw std::logic_error(
                std::string("Metal NF4 weight is missing its persistent ") +
                description + " buffer"
            );
        }
        return (__bridge id<MTLBuffer>)const_cast<void*>(handle);
    }

    void encode(
        const TensorStorage& input,
        const QuantizedWeightStorage& weight,
        TensorStorage& output,
        const QuantizedLinearDimensions& dimensions,
        std::size_t output_count,
        bool input_backward,
        const char* description
    ) {
        if (output_count == 0 ||
            output_count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "Metal quantized-linear grid size is outside uint32 range"
            );
        }
        if (weight.backend() != ExecutionBackend::Metal) {
            throw std::invalid_argument(
                "Metal quantized linear requires Metal weight storage"
            );
        }

        std::lock_guard lock(mutex_);
        @autoreleasepool {
            const auto scales = weight.scale_storage();
            const bool double_quantized =
                scales.encoding ==
                Nf4ScaleEncoding::DoubleQuantizedUInt8;
            const id<MTLComputePipelineState> pipeline =
                input_backward
                    ? (double_quantized
                           ? double_quantized_input_backward_pipeline()
                           : input_backward_pipeline())
                    : (double_quantized
                           ? double_quantized_forward_pipeline()
                           : forward_pipeline());
            id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
            if (command_buffer == nil) {
                throw std::runtime_error(
                    std::string("could not create Metal ") + description +
                    " command buffer"
                );
            }
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            if (encoder == nil) {
                throw std::runtime_error(
                    std::string("could not create Metal ") + description +
                    " command encoder"
                );
            }

            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:require_tensor_buffer(input)
                         offset:0
                        atIndex:0];
            [encoder
                setBuffer:require_weight_buffer(
                              weight.packed_codes_native_handle(),
                              "packed-code"
                          )
                   offset:0
                  atIndex:1];
            [encoder
                setBuffer:require_weight_buffer(
                              weight.primary_scales_native_handle(),
                              double_quantized ? "scale-code" : "block-scale"
                          )
                   offset:0
                  atIndex:2];
            const std::uint64_t rows = dimensions.rows;
            const std::uint64_t input_width = dimensions.input_width;
            const std::uint64_t output_width = dimensions.output_width;
            const std::uint64_t block_size = dimensions.block_size;
            if (double_quantized) {
                [encoder
                    setBuffer:require_weight_buffer(
                                  weight.secondary_scales_native_handle(),
                                  "second-level scale"
                              )
                       offset:0
                      atIndex:3];
                [encoder setBuffer:require_tensor_buffer(output)
                             offset:0
                            atIndex:4];
                [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
                [encoder setBytes:&input_width
                           length:sizeof(input_width)
                          atIndex:6];
                [encoder setBytes:&output_width
                           length:sizeof(output_width)
                          atIndex:7];
                [encoder setBytes:&block_size
                           length:sizeof(block_size)
                          atIndex:8];
                const std::uint64_t scale_block_size =
                    scales.second_level_block_size;
                [encoder setBytes:&scale_block_size
                           length:sizeof(scale_block_size)
                          atIndex:9];
                [encoder setBytes:&scales.offset
                           length:sizeof(scales.offset)
                          atIndex:10];
            } else {
                [encoder setBuffer:require_tensor_buffer(output)
                             offset:0
                            atIndex:3];
                [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
                [encoder setBytes:&input_width
                           length:sizeof(input_width)
                          atIndex:5];
                [encoder setBytes:&output_width
                           length:sizeof(output_width)
                          atIndex:6];
                [encoder setBytes:&block_size
                           length:sizeof(block_size)
                          atIndex:7];
            }

            const NSUInteger group_width =
                std::max<NSUInteger>(
                    1,
                    std::min<NSUInteger>(
                        pipeline.maxTotalThreadsPerThreadgroup,
                        pipeline.threadExecutionWidth
                    )
                );
            [encoder
                dispatchThreads:MTLSizeMake(
                                    static_cast<NSUInteger>(output_count),
                                    1,
                                    1
                                )
                threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)];
            [encoder endEncoding];
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
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    id<MTLComputePipelineState> forward_pipeline_ = nil;
    id<MTLComputePipelineState> input_backward_pipeline_ = nil;
    id<MTLComputePipelineState> double_quantized_forward_pipeline_ = nil;
    id<MTLComputePipelineState>
        double_quantized_input_backward_pipeline_ = nil;
    std::exception_ptr library_error_;
    std::exception_ptr forward_pipeline_error_;
    std::exception_ptr input_backward_pipeline_error_;
    std::exception_ptr double_quantized_forward_pipeline_error_;
    std::exception_ptr double_quantized_input_backward_pipeline_error_;
    std::mutex mutex_;
};

[[nodiscard]] MetalQuantizedLinearRuntime& runtime() {
    static MetalQuantizedLinearRuntime instance;
    return instance;
}

}  // namespace

std::unique_ptr<QuantizedWeightStorage> metal_make_nf4_weight_storage(
    std::vector<std::uint8_t> packed_codes,
    Nf4ScaleStorageData scales
) {
    return runtime().make_nf4_weight_storage(
        std::move(packed_codes),
        std::move(scales)
    );
}

void metal_quantized_linear_forward(
    const QuantizedLinearForwardRequest& request
) {
    runtime().forward(request);
}

void metal_quantized_linear_input_backward(
    const QuantizedLinearInputBackwardRequest& request
) {
    runtime().input_backward(request);
}

}  // namespace riftco_transformer::backend_detail
