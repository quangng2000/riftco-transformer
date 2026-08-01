#include "core/backend/nn/quantized_linear/dispatch.hpp"

#include "core/backend/adapter.hpp"
#include "core/quantization/nf4.hpp"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace riftco_transformer::backend_detail {
namespace {

[[nodiscard]] const BackendAdapter& require_backend_adapter(
    ExecutionBackend backend
) {
    const auto* adapter = find_backend_adapter(backend);
    if (adapter == nullptr) {
        throw std::invalid_argument("unknown execution backend");
    }
    if (!adapter->is_available()) {
        std::string message =
            std::string(adapter->name()) + " execution backend is unavailable";
        const std::string_view reason = adapter->unavailability_reason();
        if (!reason.empty()) {
            message += ": ";
            message += reason;
        }
        throw std::runtime_error(message);
    }
    return *adapter;
}

[[nodiscard]] std::size_t checked_product(
    std::initializer_list<std::size_t> factors
) {
    std::size_t result = 1;
    for (const std::size_t factor : factors) {
        if (factor == 0) {
            throw std::invalid_argument(
                "quantized linear dimensions must be positive"
            );
        }
        if (result > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::overflow_error(
                "quantized linear size exceeds addressable storage"
            );
        }
        result *= factor;
    }
    return result;
}

[[nodiscard]] std::size_t checked_payload_bytes(
    std::size_t packed_code_count,
    const Nf4ScaleStorageView& scales
) {
    const std::size_t float_scale_count =
        scales.encoding == Nf4ScaleEncoding::Float32
            ? scales.fp32_scales.size()
            : scales.second_level_scales.size();
    if (float_scale_count >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error(
            "NF4 scale storage exceeds addressable memory"
        );
    }
    std::size_t scale_bytes = float_scale_count * sizeof(float);
    if (scales.encoding == Nf4ScaleEncoding::DoubleQuantizedUInt8) {
        if (scale_bytes >
            std::numeric_limits<std::size_t>::max() - sizeof(float)) {
            throw std::overflow_error(
                "double-quantized NF4 scale storage exceeds addressable memory"
            );
        }
        scale_bytes += sizeof(float);
        if (scales.quantized_scales.size() >
            std::numeric_limits<std::size_t>::max() - scale_bytes) {
            throw std::overflow_error(
                "double-quantized NF4 scale storage exceeds addressable memory"
            );
        }
        scale_bytes += scales.quantized_scales.size();
    }
    if (packed_code_count >
        std::numeric_limits<std::size_t>::max() - scale_bytes) {
        throw std::overflow_error(
            "NF4 weight storage exceeds addressable memory"
        );
    }
    return packed_code_count + scale_bytes;
}

void require_finite_nonnegative_scales(std::span<const float> scales) {
    for (const float scale : scales) {
        if (!std::isfinite(scale) || scale < 0.0F) {
            throw std::invalid_argument(
                "NF4 block scales must be finite and non-negative"
            );
        }
    }
}

void require_scale_storage(
    const Nf4ScaleStorageView& scales,
    std::size_t expected_scale_count
) {
    switch (scales.encoding) {
    case Nf4ScaleEncoding::Float32:
        if (scales.fp32_scales.size() != expected_scale_count ||
            !scales.quantized_scales.empty() ||
            !scales.second_level_scales.empty() ||
            scales.second_level_block_size != 0 || scales.offset != 0.0F) {
            throw std::invalid_argument(
                "legacy NF4 scale storage has an invalid representation"
            );
        }
        require_finite_nonnegative_scales(scales.fp32_scales);
        return;
    case Nf4ScaleEncoding::DoubleQuantizedUInt8:
        if (!scales.fp32_scales.empty() ||
            scales.quantized_scales.size() != expected_scale_count ||
            scales.second_level_block_size == 0 ||
            !quantization::detail::nf4_block_size_supported(
                scales.second_level_block_size
            ) ||
            !std::isfinite(scales.offset) || scales.offset < 0.0F) {
            throw std::invalid_argument(
                "double-quantized NF4 scale storage has an invalid representation"
            );
        }
        if (scales.second_level_scales.size() !=
            1 + (expected_scale_count - 1) /
                    scales.second_level_block_size) {
            throw std::invalid_argument(
                "double-quantized NF4 second-level scale count is invalid"
            );
        }
        require_finite_nonnegative_scales(scales.second_level_scales);
        for (std::size_t index = 0; index < expected_scale_count; ++index) {
            if (!std::isfinite(decode_nf4_block_scale(scales, index))) {
                throw std::invalid_argument(
                    "double-quantized NF4 scale reconstruction is nonfinite"
                );
            }
        }
        return;
    }
    throw std::invalid_argument("unknown NF4 scale encoding");
}

[[nodiscard]] bool tensor_storage_aliases(
    const TensorStorage& left,
    const TensorStorage& right
) noexcept {
    if (&left == &right) {
        return true;
    }
    const void* left_handle = left.native_handle();
    const void* right_handle = right.native_handle();
    return left_handle != nullptr && left_handle == right_handle;
}

void require_tensor_storage(
    const TensorStorage& storage,
    ExecutionBackend backend,
    std::size_t expected_size
) {
    if (storage.backend() != backend) {
        throw std::invalid_argument(
            "quantized linear tensor storage does not match its backend"
        );
    }
    if (storage.size() != expected_size ||
        storage.data().size() != expected_size) {
        throw std::logic_error(
            "quantized linear tensor storage size does not match dimensions"
        );
    }
}

void require_weight_separate_from_output(
    const QuantizedWeightStorage& weight,
    const TensorStorage& output
) {
    const void* output_handle = output.native_handle();
    if (output_handle == nullptr) {
        return;
    }
    if (output_handle == weight.packed_codes_native_handle() ||
        output_handle == weight.primary_scales_native_handle() ||
        output_handle == weight.secondary_scales_native_handle()) {
        throw std::invalid_argument(
            "quantized linear weight and output storage must not alias"
        );
    }
}

void require_weight_storage(
    const QuantizedWeightStorage& weight,
    ExecutionBackend backend,
    const QuantizedLinearDimensions& dimensions
) {
    if (weight.backend() != backend) {
        throw std::invalid_argument(
            "quantized linear weight storage does not match its backend"
        );
    }
    if (dimensions.block_size == 0) {
        throw std::invalid_argument(
            "NF4 block size must be greater than zero"
        );
    }
    const std::size_t logical_size = checked_product({
        dimensions.output_width,
        dimensions.input_width,
    });
    const std::size_t expected_packed_codes =
        logical_size / 2 + logical_size % 2;
    const std::size_t expected_scales =
        1 + (logical_size - 1) / dimensions.block_size;
    if (weight.packed_codes().size() != expected_packed_codes) {
        throw std::logic_error(
            "NF4 packed-code count does not match weight dimensions"
        );
    }
    const auto scales = weight.scale_storage();
    require_scale_storage(scales, expected_scales);
    const std::size_t expected_payload = checked_payload_bytes(
        expected_packed_codes,
        scales
    );
    if (weight.resident_payload_bytes() < expected_payload) {
        throw std::logic_error(
            "NF4 resident payload is smaller than its packed storage"
        );
    }
}

void require_dimensions(const QuantizedLinearDimensions& dimensions) {
    static_cast<void>(checked_product({
        dimensions.rows,
        dimensions.input_width,
        dimensions.output_width,
    }));
    if (dimensions.block_size == 0) {
        throw std::invalid_argument(
            "NF4 block size must be greater than zero"
        );
    }
}

}  // namespace

std::unique_ptr<QuantizedWeightStorage> make_nf4_weight_storage(
    ExecutionBackend backend,
    std::vector<std::uint8_t> packed_codes,
    Nf4ScaleStorageData scales
) {
    const auto& adapter = require_backend_adapter(backend);
    if (packed_codes.empty()) {
        throw std::invalid_argument(
            "NF4 packed-code storage must not be empty"
        );
    }
    const std::size_t packed_count = packed_codes.size();
    const Nf4ScaleStorageView input_view{
        scales.encoding,
        scales.fp32_scales,
        scales.quantized_scales,
        scales.second_level_scales,
        scales.second_level_block_size,
        scales.offset,
    };
    const std::size_t scale_count = input_view.scale_count();
    if (scale_count == 0) {
        throw std::invalid_argument("NF4 scale storage must not be empty");
    }
    require_scale_storage(input_view, scale_count);
    const std::size_t expected_payload =
        checked_payload_bytes(packed_count, input_view);
    const auto encoding = input_view.encoding;
    const auto second_level_count = input_view.second_level_scales.size();
    const auto second_level_block_size = input_view.second_level_block_size;
    const float offset = input_view.offset;

    auto storage = adapter.make_nf4_weight_storage(
        std::move(packed_codes),
        std::move(scales)
    );
    if (storage == nullptr) {
        throw std::logic_error(
            "backend adapter returned null NF4 weight storage"
        );
    }
    const auto stored_scales = storage->scale_storage();
    if (storage->backend() != backend ||
        storage->packed_codes().size() != packed_count ||
        stored_scales.encoding != encoding ||
        stored_scales.scale_count() != scale_count ||
        stored_scales.second_level_scales.size() != second_level_count ||
        stored_scales.second_level_block_size != second_level_block_size ||
        stored_scales.offset != offset ||
        storage->resident_payload_bytes() < expected_payload) {
        throw std::logic_error(
            "backend adapter returned invalid NF4 weight storage"
        );
    }
    require_scale_storage(stored_scales, scale_count);
    return storage;
}

void dispatch_quantized_linear_forward(
    ExecutionBackend backend,
    const QuantizedLinearForwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_dimensions(request.dimensions);
    const std::size_t input_size = checked_product({
        request.dimensions.rows,
        request.dimensions.input_width,
    });
    const std::size_t output_size = checked_product({
        request.dimensions.rows,
        request.dimensions.output_width,
    });
    require_tensor_storage(request.input, backend, input_size);
    require_tensor_storage(request.output, backend, output_size);
    require_weight_storage(request.weight, backend, request.dimensions);
    if (tensor_storage_aliases(request.input, request.output)) {
        throw std::invalid_argument(
            "quantized linear input and output must not alias"
        );
    }
    require_weight_separate_from_output(request.weight, request.output);
    adapter.quantized_linear_forward(request);
}

void dispatch_quantized_linear_input_backward(
    ExecutionBackend backend,
    const QuantizedLinearInputBackwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_dimensions(request.dimensions);
    const std::size_t upstream_size = checked_product({
        request.dimensions.rows,
        request.dimensions.output_width,
    });
    const std::size_t gradient_size = checked_product({
        request.dimensions.rows,
        request.dimensions.input_width,
    });
    require_tensor_storage(request.upstream, backend, upstream_size);
    require_tensor_storage(request.input_gradient, backend, gradient_size);
    require_weight_storage(request.weight, backend, request.dimensions);
    if (tensor_storage_aliases(request.upstream, request.input_gradient)) {
        throw std::invalid_argument(
            "quantized linear upstream and input gradient must not alias"
        );
    }
    require_weight_separate_from_output(
        request.weight,
        request.input_gradient
    );
    adapter.quantized_linear_input_backward(request);
}

}  // namespace riftco_transformer::backend_detail
