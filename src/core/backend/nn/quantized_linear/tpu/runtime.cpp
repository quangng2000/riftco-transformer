#include "core/backend/nn/quantized_linear/tpu/launch.hpp"

#include "core/backend/adapters/tpu/runtime.hpp"
#include "riftco_transformer/core/quantized_weight.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

class TpuNf4WeightStorage final : public QuantizedWeightStorage {
public:
    TpuNf4WeightStorage(
        std::vector<std::uint8_t> packed_codes,
        Nf4ScaleStorageData scales
    )
        : packed_codes_(std::move(packed_codes)),
          scales_(std::move(scales)) {}

    [[nodiscard]] ExecutionBackend backend() const noexcept override {
        return ExecutionBackend::Tpu;
    }

    [[nodiscard]] std::span<const std::uint8_t>
    packed_codes() const noexcept override {
        return packed_codes_;
    }

    [[nodiscard]] Nf4ScaleStorageView
    scale_storage() const noexcept override {
        return {
            scales_.encoding,
            scales_.fp32_scales,
            scales_.quantized_scales,
            scales_.second_level_scales,
            scales_.second_level_block_size,
            scales_.offset,
        };
    }

    [[nodiscard]] const void*
    packed_codes_native_handle() const noexcept override {
        return packed_codes_.data();
    }

    [[nodiscard]] const void*
    primary_scales_native_handle() const noexcept override {
        if (scales_.encoding == Nf4ScaleEncoding::Float32) {
            return scales_.fp32_scales.data();
        }
        return scales_.quantized_scales.data();
    }

    [[nodiscard]] const void*
    secondary_scales_native_handle() const noexcept override {
        if (scales_.encoding == Nf4ScaleEncoding::Float32) {
            return nullptr;
        }
        return scales_.second_level_scales.data();
    }

    [[nodiscard]] std::size_t
    resident_payload_bytes() const noexcept override {
        return packed_codes_.size() +
            nf4_scale_payload_bytes(scale_storage());
    }

private:
    std::vector<std::uint8_t> packed_codes_;
    Nf4ScaleStorageData scales_;
};

struct QuantizedLinearShape {
    std::size_t rows;
    std::size_t input_width;
    std::size_t output_width;
    std::size_t block_size;
    std::size_t weight_elements;
    std::size_t packed_code_count;
    std::size_t block_count;
};

struct ScaleLayout {
    Nf4ScaleEncoding encoding;
    std::size_t second_level_block_size;
    std::size_t second_level_count;
};

[[nodiscard]] std::size_t checked_product(
    std::size_t left,
    std::size_t right
) {
    if (left == 0 || right == 0) {
        throw std::invalid_argument(
            "TPU quantized-linear dimensions must be positive"
        );
    }
    if (left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(
            "TPU quantized-linear weight size exceeds addressable storage"
        );
    }
    return left * right;
}

[[nodiscard]] QuantizedLinearShape shape_from(
    const QuantizedLinearDimensions& dimensions
) {
    if (dimensions.rows == 0 || dimensions.block_size == 0) {
        throw std::invalid_argument(
            "TPU quantized-linear dimensions must be positive"
        );
    }
    const std::size_t weight_elements = checked_product(
        dimensions.output_width,
        dimensions.input_width
    );
    const std::size_t packed_code_count =
        weight_elements / 2 + weight_elements % 2;
    const std::size_t block_count =
        1 + (weight_elements - 1) / dimensions.block_size;
    constexpr std::size_t largest_iota_size =
        static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max()
        ) + 1;
    if (weight_elements > largest_iota_size ||
        dimensions.block_size > static_cast<std::size_t>(
                                    std::numeric_limits<std::int32_t>::max()
                                )) {
        throw std::overflow_error(
            "TPU NF4 indexing exceeds the StableHLO s32 range"
        );
    }
    return {
        dimensions.rows,
        dimensions.input_width,
        dimensions.output_width,
        dimensions.block_size,
        weight_elements,
        packed_code_count,
        block_count,
    };
}

[[nodiscard]] ScaleLayout scale_layout_from(
    const Nf4ScaleStorageView& scales,
    const QuantizedLinearShape& shape
) {
    if (scales.scale_count() != shape.block_count) {
        throw std::logic_error(
            "TPU NF4 scale count does not match quantized-linear dimensions"
        );
    }
    if (scales.encoding == Nf4ScaleEncoding::Float32) {
        if (!scales.quantized_scales.empty() ||
            !scales.second_level_scales.empty()) {
            throw std::logic_error(
                "TPU FP32 scale storage contains double-quantized metadata"
            );
        }
        return {Nf4ScaleEncoding::Float32, 0, 0};
    }
    if (scales.second_level_block_size == 0 ||
        scales.second_level_block_size > static_cast<std::size_t>(
                                             std::numeric_limits<
                                                 std::int32_t>::max()
                                         )) {
        throw std::overflow_error(
            "TPU second-level scale block size exceeds StableHLO s32"
        );
    }
    const std::size_t expected_second_level_count =
        1 + (shape.block_count - 1) / scales.second_level_block_size;
    if (!scales.fp32_scales.empty() ||
        scales.second_level_scales.size() != expected_second_level_count) {
        throw std::logic_error(
            "TPU double-quantized scale metadata is inconsistent"
        );
    }
    return {
        Nf4ScaleEncoding::DoubleQuantizedUInt8,
        scales.second_level_block_size,
        expected_second_level_count,
    };
}

[[nodiscard]] std::int64_t checked_dimension(std::size_t value) {
    if (value == 0 || value > static_cast<std::size_t>(
                                  std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "TPU quantized-linear dimension exceeds the PJRT int64 range"
        );
    }
    return static_cast<std::int64_t>(value);
}

[[nodiscard]] std::string tensor_type(
    std::initializer_list<std::size_t> dimensions,
    std::string_view element_type
) {
    std::string result = "tensor<";
    bool first = true;
    for (const std::size_t dimension : dimensions) {
        if (!first) {
            result += "x";
        }
        result += std::to_string(dimension);
        first = false;
    }
    result += "x";
    result += element_type;
    result += ">";
    return result;
}

[[nodiscard]] std::string float_literal(float value) {
    std::ostringstream stream;
    stream << std::scientific
           << std::setprecision(std::numeric_limits<float>::max_digits10)
           << value;
    return stream.str();
}

void append_scalar_broadcast(
    std::ostringstream& program,
    std::string_view name,
    std::int32_t value,
    std::string_view vector_type
) {
    program << "    %" << name << "_scalar = \"stablehlo.constant\"() {"
            << "value = dense<" << value << "> : tensor<i32>} : () -> "
            << "tensor<i32>\n"
            << "    %" << name << " = \"stablehlo.broadcast_in_dim\"(%"
            << name << "_scalar) {broadcast_dimensions = array<i64>} : "
            << "(tensor<i32>) -> " << vector_type << "\n";
}

void append_float_scalar_broadcast(
    std::ostringstream& program,
    std::string_view name,
    float value,
    std::string_view vector_type
) {
    program << "    %" << name << "_scalar = \"stablehlo.constant\"() {"
            << "value = dense<" << float_literal(value)
            << "> : tensor<f32>} : () -> tensor<f32>\n"
            << "    %" << name << " = \"stablehlo.broadcast_in_dim\"(%"
            << name << "_scalar) {broadcast_dimensions = array<i64>} : "
            << "(tensor<f32>) -> " << vector_type << "\n";
}

void append_vector_gather(
    std::ostringstream& program,
    std::string_view result,
    std::string_view operand,
    std::string_view indices,
    std::string_view operand_type,
    std::string_view index_type,
    std::string_view result_type,
    bool indices_are_sorted
) {
    program << "    " << result << " = \"stablehlo.gather\"(" << operand
            << ", " << indices << ") {\n"
            << "      dimension_numbers = #stablehlo.gather<\n"
            << "        offset_dims = [],\n"
            << "        collapsed_slice_dims = [0],\n"
            << "        start_index_map = [0],\n"
            << "        index_vector_dim = 1>,\n"
            << "      slice_sizes = array<i64: 1>,\n"
            << "      indices_are_sorted = "
            << (indices_are_sorted ? "true" : "false") << "\n"
            << "    } : (" << operand_type << ", " << index_type << ") -> "
            << result_type << "\n";
}

void append_nf4_dequantization(
    std::ostringstream& program,
    const QuantizedLinearShape& shape,
    std::string_view packed_codes,
    std::string_view block_scales,
    std::string_view result
) {
    const std::string indices_type =
        tensor_type({shape.weight_elements}, "i32");
    const std::string values_type =
        tensor_type({shape.weight_elements}, "f32");
    const std::string packed_type =
        tensor_type({shape.packed_code_count}, "ui8");
    const std::string scales_type =
        tensor_type({shape.block_count}, "f32");
    const std::string codebook_type = tensor_type({16}, "f32");

    program << "    %flat_indices = \"stablehlo.iota\"() {"
            << "iota_dimension = 0 : i64} : () -> " << indices_type << "\n";
    append_scalar_broadcast(program, "two", 2, indices_type);
    program << "    %packed_indices = \"stablehlo.divide\"("
            << "%flat_indices, %two) : (" << indices_type << ", "
            << indices_type << ") -> " << indices_type << "\n";
    append_vector_gather(
        program,
        "%packed_for_elements_u8",
        packed_codes,
        "%packed_indices",
        packed_type,
        indices_type,
        tensor_type({shape.weight_elements}, "ui8"),
        true
    );
    program << "    %packed_for_elements = \"stablehlo.convert\"("
            << "%packed_for_elements_u8) : ("
            << tensor_type({shape.weight_elements}, "ui8") << ") -> "
            << indices_type << "\n";

    append_scalar_broadcast(program, "one", 1, indices_type);
    append_scalar_broadcast(program, "four", 4, indices_type);
    append_scalar_broadcast(program, "nibble_mask", 15, indices_type);
    append_scalar_broadcast(program, "zero", 0, indices_type);
    program << "    %parity = \"stablehlo.and\"(%flat_indices, %one) : ("
            << indices_type << ", " << indices_type << ") -> "
            << indices_type << "\n"
            << "    %low_nibbles = \"stablehlo.and\"("
            << "%packed_for_elements, %nibble_mask) : (" << indices_type
            << ", " << indices_type << ") -> " << indices_type << "\n"
            << "    %shifted = \"stablehlo.shift_right_logical\"("
            << "%packed_for_elements, %four) : (" << indices_type << ", "
            << indices_type << ") -> " << indices_type << "\n"
            << "    %high_nibbles = \"stablehlo.and\"("
            << "%shifted, %nibble_mask) : (" << indices_type << ", "
            << indices_type << ") -> " << indices_type << "\n"
            << "    %even_positions = \"stablehlo.compare\"("
            << "%parity, %zero) {comparison_direction = "
            << "#stablehlo<comparison_direction EQ>, compare_type = "
            << "#stablehlo<comparison_type SIGNED>} : (" << indices_type
            << ", " << indices_type << ") -> "
            << tensor_type({shape.weight_elements}, "i1") << "\n"
            << "    %nf4_codes = \"stablehlo.select\"("
            << "%even_positions, %low_nibbles, %high_nibbles) : ("
            << tensor_type({shape.weight_elements}, "i1") << ", "
            << indices_type << ", " << indices_type << ") -> "
            << indices_type << "\n";

    program << "    %nf4_codebook = \"stablehlo.constant\"() {value = "
            << "dense<[";
    const auto codebook = quantization::nf4_codebook();
    for (std::size_t index = 0; index < codebook.size(); ++index) {
        if (index != 0) {
            program << ", ";
        }
        program << float_literal(codebook[index]);
    }
    program << "]> : " << codebook_type << "} : () -> " << codebook_type
            << "\n";
    append_vector_gather(
        program,
        "%normalized_values",
        "%nf4_codebook",
        "%nf4_codes",
        codebook_type,
        indices_type,
        values_type,
        false
    );

    append_scalar_broadcast(
        program,
        "block_size",
        static_cast<std::int32_t>(shape.block_size),
        indices_type
    );
    program << "    %block_indices = \"stablehlo.divide\"("
            << "%flat_indices, %block_size) : (" << indices_type << ", "
            << indices_type << ") -> " << indices_type << "\n";
    append_vector_gather(
        program,
        "%scales_for_elements",
        block_scales,
        "%block_indices",
        scales_type,
        indices_type,
        values_type,
        true
    );
    program << "    " << result << " = \"stablehlo.multiply\"("
            << "%normalized_values, %scales_for_elements) : ("
            << values_type << ", " << values_type << ") -> "
            << values_type << "\n";
}

void append_double_quantized_scale_dequantization(
    std::ostringstream& program,
    const QuantizedLinearShape& shape,
    const ScaleLayout& scales
) {
    const std::string index_type = tensor_type({shape.block_count}, "i32");
    const std::string scale_type = tensor_type({shape.block_count}, "f32");
    const std::string code_type = tensor_type({shape.block_count}, "ui8");
    const std::string second_level_type =
        tensor_type({scales.second_level_count}, "f32");

    program << "    %scale_indices = \"stablehlo.iota\"() {"
            << "iota_dimension = 0 : i64} : () -> " << index_type << "\n";
    append_scalar_broadcast(
        program,
        "scale_block_size",
        static_cast<std::int32_t>(scales.second_level_block_size),
        index_type
    );
    program << "    %second_level_indices = \"stablehlo.divide\"("
            << "%scale_indices, %scale_block_size) : (" << index_type << ", "
            << index_type << ") -> " << index_type << "\n";
    append_vector_gather(
        program,
        "%second_level_for_scales",
        "%second_level_scales",
        "%second_level_indices",
        second_level_type,
        index_type,
        scale_type,
        true
    );
    program << "    %scale_codes_i32 = \"stablehlo.convert\"("
            << "%quantized_scales) : (" << code_type << ") -> "
            << index_type << "\n";
    append_scalar_broadcast(program, "scale_code_center", 128, index_type);
    program << "    %centered_scale_codes = \"stablehlo.subtract\"("
            << "%scale_codes_i32, %scale_code_center) : (" << index_type
            << ", " << index_type << ") -> " << index_type << "\n"
            << "    %centered_scale_codes_f32 = \"stablehlo.convert\"("
            << "%centered_scale_codes) : (" << index_type << ") -> "
            << scale_type << "\n";
    append_float_scalar_broadcast(
        program,
        "scale_code_divisor",
        127.0F,
        scale_type
    );
    program << "    %normalized_scale_codes = \"stablehlo.divide\"("
            << "%centered_scale_codes_f32, %scale_code_divisor) : ("
            << scale_type << ", " << scale_type << ") -> " << scale_type
            << "\n"
            << "    %scale_deltas = \"stablehlo.multiply\"("
            << "%normalized_scale_codes, %second_level_for_scales) : ("
            << scale_type << ", " << scale_type << ") -> " << scale_type
            << "\n"
            << "    %scale_offset_scalar = \"stablehlo.reshape\"("
            << "%scale_offset) : (tensor<1xf32>) -> tensor<f32>\n"
            << "    %scale_offsets = \"stablehlo.broadcast_in_dim\"("
            << "%scale_offset_scalar) {broadcast_dimensions = array<i64>} : "
            << "(tensor<f32>) -> " << scale_type << "\n"
            << "    %uncapped_block_scales = \"stablehlo.add\"("
            << "%scale_deltas, %scale_offsets) : (" << scale_type << ", "
            << scale_type << ") -> " << scale_type << "\n";
    append_float_scalar_broadcast(program, "zero_scale", 0.0F, scale_type);
    program << "    %nonnegative_block_scales = \"stablehlo.maximum\"("
            << "%uncapped_block_scales, %zero_scale) : (" << scale_type
            << ", " << scale_type << ") -> " << scale_type << "\n";
    append_float_scalar_broadcast(
        program,
        "maximum_scale",
        std::numeric_limits<float>::max(),
        scale_type
    );
    program << "    %block_scales = \"stablehlo.minimum\"("
            << "%nonnegative_block_scales, %maximum_scale) : ("
            << scale_type << ", " << scale_type << ") -> " << scale_type
            << "\n";
}

[[nodiscard]] std::string quantized_linear_program(
    const QuantizedLinearShape& shape,
    const ScaleLayout& scale_layout,
    bool input_backward
) {
    const std::string forward_input = tensor_type(
        {shape.rows, shape.input_width}, "f32"
    );
    const std::string upstream = tensor_type(
        {shape.rows, shape.output_width}, "f32"
    );
    const std::string packed = tensor_type(
        {shape.packed_code_count}, "ui8"
    );
    const std::string scales = tensor_type({shape.block_count}, "f32");
    const std::string weight_flat = tensor_type(
        {shape.weight_elements}, "f32"
    );
    const std::string weight = tensor_type(
        {shape.output_width, shape.input_width}, "f32"
    );
    const std::string output = input_backward ? forward_input : upstream;
    const std::string activation = input_backward ? upstream : forward_input;

    std::ostringstream program;
    program << "module @riftco_tpu_nf4_quantized_linear_"
            << (input_backward ? "input_backward" : "forward")
            << " attributes {mhlo.num_partitions = 1 : i32, "
            << "mhlo.num_replicas = 1 : i32} {\n"
            << "  func.func public @main(%activation: " << activation
            << ", %packed_codes: " << packed;
    if (scale_layout.encoding == Nf4ScaleEncoding::Float32) {
        program << ", %block_scales: " << scales;
    } else {
        program << ", %quantized_scales: "
                << tensor_type({shape.block_count}, "ui8")
                << ", %second_level_scales: "
                << tensor_type({scale_layout.second_level_count}, "f32")
                << ", %scale_offset: tensor<1xf32>";
    }
    program << ") -> " << output << " {\n";
    if (scale_layout.encoding == Nf4ScaleEncoding::DoubleQuantizedUInt8) {
        append_double_quantized_scale_dequantization(
            program,
            shape,
            scale_layout
        );
    }
    append_nf4_dequantization(
        program,
        shape,
        "%packed_codes",
        "%block_scales",
        "%weight_flat"
    );
    program << "    %weight = \"stablehlo.reshape\"(%weight_flat) : ("
            << weight_flat << ") -> " << weight << "\n"
            << "    %result = \"stablehlo.dot_general\"("
            << "%activation, %weight) {\n"
            << "      dot_dimension_numbers = #stablehlo.dot<\n"
            << "        lhs_batching_dimensions = [],\n"
            << "        rhs_batching_dimensions = [],\n"
            << "        lhs_contracting_dimensions = [1],\n"
            << "        rhs_contracting_dimensions = ["
            << (input_backward ? 0 : 1) << "]\n"
            << "      >,\n"
            << "      precision_config = [#stablehlo<precision HIGHEST>, "
            << "#stablehlo<precision HIGHEST>]\n"
            << "    } : (" << activation << ", " << weight << ") -> "
            << output << "\n"
            << "    func.return %result : " << output << "\n"
            << "  }\n"
            << "}\n";
    return program.str();
}

void execute_quantized_linear(
    const TensorStorage& activation,
    const QuantizedWeightStorage& weight,
    TensorStorage& output,
    const QuantizedLinearDimensions& dimensions,
    bool input_backward
) {
    if (activation.backend() != ExecutionBackend::Tpu ||
        output.backend() != ExecutionBackend::Tpu ||
        weight.backend() != ExecutionBackend::Tpu) {
        throw std::invalid_argument(
            "TPU quantized linear requires TPU-owned storage"
        );
    }
    const QuantizedLinearShape shape = shape_from(dimensions);
    const Nf4ScaleStorageView scales = weight.scale_storage();
    const ScaleLayout scale_layout = scale_layout_from(scales, shape);
    const std::array<std::int64_t, 2> activation_shape{
        checked_dimension(shape.rows),
        checked_dimension(
            input_backward ? shape.output_width : shape.input_width
        ),
    };
    const std::array<std::int64_t, 1> packed_shape{
        checked_dimension(shape.packed_code_count),
    };
    const std::array<std::int64_t, 1> scale_shape{
        checked_dimension(shape.block_count),
    };
    const std::array<std::int64_t, 1> second_level_shape{
        checked_dimension(
            scale_layout.second_level_count == 0
                ? 1
                : scale_layout.second_level_count
        ),
    };
    const std::array<std::int64_t, 1> offset_shape{1};
    const std::array<std::int64_t, 2> output_shape{
        checked_dimension(shape.rows),
        checked_dimension(
            input_backward ? shape.input_width : shape.output_width
        ),
    };

    const auto packed_codes = weight.packed_codes();
    if (packed_codes.size() != shape.packed_code_count) {
        throw std::logic_error(
            "TPU NF4 payload does not match quantized-linear dimensions"
        );
    }

    TpuProgram program{};
    program.key.kind = input_backward
        ? TpuProgramKind::QuantizedLinearInputBackward
        : TpuProgramKind::QuantizedLinearForward;
    program.key.dimensions = {
        shape.rows,
        shape.input_width,
        shape.output_width,
        shape.block_size,
        scale_layout.second_level_block_size,
        0,
    };
    program.key.dimension_count = 5;
    program.stablehlo = quantized_linear_program(
        shape,
        scale_layout,
        input_backward
    );
    program.operation_name = input_backward
        ? "StableHLO NF4 quantized-linear input backward"
        : "StableHLO NF4 quantized-linear forward";

    std::vector<TpuHostInput> inputs;
    inputs.reserve(
        scale_layout.encoding == Nf4ScaleEncoding::Float32 ? 3 : 5
    );
    inputs.push_back(
        TpuHostInput{
            activation.data().data(),
            activation.data().size_bytes(),
            TpuElementType::F32,
            activation_shape,
        }
    );
    inputs.push_back(
        TpuHostInput{
            packed_codes.data(),
            packed_codes.size_bytes(),
            TpuElementType::U8,
            packed_shape,
        }
    );
    const std::array<float, 1> scale_offset{scales.offset};
    if (scale_layout.encoding == Nf4ScaleEncoding::Float32) {
        inputs.push_back(
            TpuHostInput{
                scales.fp32_scales.data(),
                scales.fp32_scales.size_bytes(),
                TpuElementType::F32,
                scale_shape,
            }
        );
    } else {
        inputs.push_back(
            TpuHostInput{
                scales.quantized_scales.data(),
                scales.quantized_scales.size_bytes(),
                TpuElementType::U8,
                scale_shape,
            }
        );
        inputs.push_back(
            TpuHostInput{
                scales.second_level_scales.data(),
                scales.second_level_scales.size_bytes(),
                TpuElementType::F32,
                second_level_shape,
            }
        );
        inputs.push_back(
            TpuHostInput{
                scale_offset.data(),
                sizeof(float),
                TpuElementType::F32,
                offset_shape,
            }
        );
    }
    const std::array<TpuHostOutput, 1> outputs{
        TpuHostOutput{
            output.data().data(),
            output.data().size_bytes(),
            TpuElementType::F32,
            output_shape,
        },
    };
    tpu_runtime_execute(program, inputs, outputs);
}

}  // namespace

std::unique_ptr<QuantizedWeightStorage> tpu_make_nf4_weight_storage(
    std::vector<std::uint8_t> packed_codes,
    Nf4ScaleStorageData scales
) {
    return std::make_unique<TpuNf4WeightStorage>(
        std::move(packed_codes),
        std::move(scales)
    );
}

void tpu_quantized_linear_forward(
    const QuantizedLinearForwardRequest& request
) {
    execute_quantized_linear(
        request.input,
        request.weight,
        request.output,
        request.dimensions,
        false
    );
}

void tpu_quantized_linear_input_backward(
    const QuantizedLinearInputBackwardRequest& request
) {
    execute_quantized_linear(
        request.upstream,
        request.weight,
        request.input_gradient,
        request.dimensions,
        true
    );
}

}  // namespace riftco_transformer::backend_detail
