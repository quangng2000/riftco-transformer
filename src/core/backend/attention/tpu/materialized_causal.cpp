#include "core/backend/attention/tpu/materialized_causal.hpp"

#include "core/backend/adapters/tpu/runtime.hpp"
#include "core/backend/attention/tpu/common.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

using attention_tpu_detail::attention_scale;
using attention_tpu_detail::checked_dimension;
using attention_tpu_detail::float_literal;
using attention_tpu_detail::require_valid_softmax_status;
using attention_tpu_detail::tensor_type;

struct MaterializedShape {
    std::size_t batch;
    std::size_t heads;
    std::size_t time;
    std::size_t head_width;
};

void append_module_header(
    std::ostringstream& program,
    std::string_view module_name) {
    program << "module @" << module_name << " attributes {"
            << "mhlo.num_partitions = 1 : i32, "
            << "mhlo.num_replicas = 1 : i32} {\n";
}

void append_batched_attention_dot(
    std::ostringstream& program,
    std::string_view result,
    std::string_view left,
    std::string_view right,
    std::string_view left_type,
    std::string_view right_type,
    std::string_view output_type,
    std::size_t left_contract,
    std::size_t right_contract) {
    program << "    " << result << " = \"stablehlo.dot_general\"(" << left
            << ", " << right << ") {\n"
            << "      dot_dimension_numbers = #stablehlo.dot<\n"
            << "        lhs_batching_dimensions = [0, 1],\n"
            << "        rhs_batching_dimensions = [0, 1],\n"
            << "        lhs_contracting_dimensions = [" << left_contract
            << "],\n"
            << "        rhs_contracting_dimensions = [" << right_contract
            << "]\n"
            << "      >,\n"
            << "      precision_config = [#stablehlo<precision HIGHEST>, "
            << "#stablehlo<precision HIGHEST>]\n"
            << "    } : (" << left_type << ", " << right_type << ") -> "
            << output_type << "\n";
}

void append_causal_mask(
    std::ostringstream& program,
    const MaterializedShape& shape,
    std::string_view probability_type) {
    const std::string index_type =
        tensor_type({shape.batch, shape.heads, shape.time, shape.time}, "i32");
    const std::string predicate_type =
        tensor_type({shape.batch, shape.heads, shape.time, shape.time}, "i1");
    program << "    %query_positions = \"stablehlo.iota\"() {"
            << "iota_dimension = 2 : i64} : () -> " << index_type << "\n"
            << "    %key_positions = \"stablehlo.iota\"() {"
            << "iota_dimension = 3 : i64} : () -> " << index_type << "\n"
            << "    %causal = \"stablehlo.compare\"("
            << "%query_positions, %key_positions) {\n"
            << "      comparison_direction = "
            << "#stablehlo<comparison_direction GE>,\n"
            << "      compare_type = #stablehlo<comparison_type SIGNED>\n"
            << "    } : (" << index_type << ", " << index_type << ") -> "
            << predicate_type << "\n"
            << "    %zero_scalar = \"stablehlo.constant\"() {"
            << "value = dense<0.000000000e+00> : tensor<f32>} : () -> "
            << "tensor<f32>\n"
            << "    %zero_probabilities = \"stablehlo.broadcast_in_dim\"("
            << "%zero_scalar) {broadcast_dimensions = array<i64>} : "
            << "(tensor<f32>) -> " << probability_type << "\n";
}

std::string materialized_forward_program(const MaterializedShape& shape) {
    const std::string tensor = tensor_type({
        shape.batch,
        shape.heads,
        shape.time,
        shape.head_width,
    });
    const std::string probabilities = tensor_type({
        shape.batch,
        shape.heads,
        shape.time,
        shape.time,
    });
    const std::string rows = tensor_type({
        shape.batch,
        shape.heads,
        shape.time,
    });
    const std::string row_predicates = tensor_type(
        {
            shape.batch,
            shape.heads,
            shape.time,
        },
        "i1");

    std::ostringstream program;
    append_module_header(
        program, "riftco_tpu_materialized_causal_attention_forward");
    program << "  func.func public @main(%queries: " << tensor
            << ", %keys: " << tensor << ", %values: " << tensor << ") -> ("
            << probabilities << ", " << tensor << ", " << rows << ") {\n";
    append_batched_attention_dot(
        program,
        "%raw_scores",
        "%queries",
        "%keys",
        tensor,
        tensor,
        probabilities,
        3,
        3);
    program << "    %scale_scalar = \"stablehlo.constant\"() {value = dense<"
            << float_literal(attention_scale(shape.head_width))
            << "> : tensor<f32>} : () -> tensor<f32>\n"
            << "    %scale = \"stablehlo.broadcast_in_dim\"(%scale_scalar) "
            << "{broadcast_dimensions = array<i64>} : (tensor<f32>) -> "
            << probabilities << "\n"
            << "    %scaled_scores = \"stablehlo.multiply\"("
            << "%raw_scores, %scale) : (" << probabilities << ", "
            << probabilities << ") -> " << probabilities << "\n";
    append_causal_mask(program, shape, probabilities);
    program << "    %negative_infinity_scalar = \"stablehlo.constant\"() {"
            << "value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>\n"
            << "    %negative_infinity = \"stablehlo.broadcast_in_dim\"("
            << "%negative_infinity_scalar) {broadcast_dimensions = array<i64>} "
            << ": (tensor<f32>) -> " << probabilities << "\n"
            << "    %masked_scores = \"stablehlo.select\"("
            << "%causal, %scaled_scores, %negative_infinity) : ("
            << tensor_type(
                   {shape.batch, shape.heads, shape.time, shape.time}, "i1")
            << ", " << probabilities << ", " << probabilities << ") -> "
            << probabilities << "\n"
            << "    %row_maxima = \"stablehlo.reduce\"("
            << "%masked_scores, %negative_infinity_scalar) ({\n"
            << "      ^bb0(%left: tensor<f32>, %right: tensor<f32>):\n"
            << "        %maximum = \"stablehlo.maximum\"(%left, %right) : "
            << "(tensor<f32>, tensor<f32>) -> tensor<f32>\n"
            << "        \"stablehlo.return\"(%maximum) : (tensor<f32>) -> ()\n"
            << "    }) {dimensions = array<i64: 3>} : (" << probabilities
            << ", tensor<f32>) -> " << rows << "\n"
            << "    %row_maxima_expanded = \"stablehlo.broadcast_in_dim\"("
            << "%row_maxima) {broadcast_dimensions = array<i64: 0, 1, 2>} : "
            << "(" << rows << ") -> " << probabilities << "\n"
            << "    %centered = \"stablehlo.subtract\"("
            << "%masked_scores, %row_maxima_expanded) : (" << probabilities
            << ", " << probabilities << ") -> " << probabilities << "\n"
            << "    %exponentials = \"stablehlo.exponential\"(%centered) : ("
            << probabilities << ") -> " << probabilities << "\n"
            << "    %row_exp_sums = \"stablehlo.reduce\"("
            << "%exponentials, %zero_scalar) ({\n"
            << "      ^bb0(%left: tensor<f32>, %right: tensor<f32>):\n"
            << "        %sum = \"stablehlo.add\"(%left, %right) : "
            << "(tensor<f32>, tensor<f32>) -> tensor<f32>\n"
            << "        \"stablehlo.return\"(%sum) : (tensor<f32>) -> ()\n"
            << "    }) {dimensions = array<i64: 3>} : (" << probabilities
            << ", tensor<f32>) -> " << rows << "\n"
            << "    %row_exp_sums_expanded = \"stablehlo.broadcast_in_dim\"("
            << "%row_exp_sums) {broadcast_dimensions = array<i64: 0, 1, 2>} : "
            << "(" << rows << ") -> " << probabilities << "\n"
            << "    %probabilities = \"stablehlo.divide\"("
            << "%exponentials, %row_exp_sums_expanded) : (" << probabilities
            << ", " << probabilities << ") -> " << probabilities << "\n";
    append_batched_attention_dot(
        program,
        "%context",
        "%probabilities",
        "%values",
        probabilities,
        tensor,
        tensor,
        3,
        2);
    program << "    %maxima_finite = \"stablehlo.is_finite\"(%row_maxima) : ("
            << rows << ") -> " << row_predicates << "\n"
            << "    %sums_finite = \"stablehlo.is_finite\"(%row_exp_sums) : ("
            << rows << ") -> " << row_predicates << "\n"
            << "    %zero_rows = \"stablehlo.broadcast_in_dim\"(%zero_scalar) "
            << "{broadcast_dimensions = array<i64>} : (tensor<f32>) -> " << rows
            << "\n"
            << "    %sums_positive = \"stablehlo.compare\"("
            << "%row_exp_sums, %zero_rows) {comparison_direction = "
            << "#stablehlo<comparison_direction GT>, compare_type = "
            << "#stablehlo<comparison_type FLOAT>} : (" << rows << ", " << rows
            << ") -> " << row_predicates << "\n"
            << "    %finite = \"stablehlo.and\"("
            << "%maxima_finite, %sums_finite) : (" << row_predicates << ", "
            << row_predicates << ") -> " << row_predicates << "\n"
            << "    %valid = \"stablehlo.and\"(%finite, %sums_positive) : ("
            << row_predicates << ", " << row_predicates << ") -> "
            << row_predicates << "\n"
            << "    %one_scalar = \"stablehlo.constant\"() {"
            << "value = dense<1.000000000e+00> : tensor<f32>} : () -> "
            << "tensor<f32>\n"
            << "    %one_rows = \"stablehlo.broadcast_in_dim\"(%one_scalar) "
            << "{broadcast_dimensions = array<i64>} : (tensor<f32>) -> " << rows
            << "\n"
            << "    %status = \"stablehlo.select\"("
            << "%valid, %zero_rows, %one_rows) : (" << row_predicates << ", "
            << rows << ", " << rows << ") -> " << rows << "\n"
            << "    func.return %probabilities, %context, %status : "
            << probabilities << ", " << tensor << ", " << rows << "\n"
            << "  }\n"
            << "}\n";
    return program.str();
}

std::string
materialized_context_backward_program(const MaterializedShape& shape) {
    const std::string tensor = tensor_type({
        shape.batch,
        shape.heads,
        shape.time,
        shape.head_width,
    });
    const std::string probabilities = tensor_type({
        shape.batch,
        shape.heads,
        shape.time,
        shape.time,
    });
    const std::string rows = tensor_type({
        shape.batch,
        shape.heads,
        shape.time,
    });

    std::ostringstream program;
    append_module_header(
        program, "riftco_tpu_materialized_causal_attention_context_backward");
    program << "  func.func public @main(%queries: " << tensor
            << ", %keys: " << tensor << ", %values: " << tensor
            << ", %probabilities: " << probabilities
            << ", %upstream_context: " << tensor << ") -> (" << tensor << ", "
            << tensor << ", " << tensor << ") {\n";
    append_causal_mask(program, shape, probabilities);
    program << "    %causal_probabilities = \"stablehlo.select\"("
            << "%causal, %probabilities, %zero_probabilities) : ("
            << tensor_type(
                   {shape.batch, shape.heads, shape.time, shape.time}, "i1")
            << ", " << probabilities << ", " << probabilities << ") -> "
            << probabilities << "\n";
    append_batched_attention_dot(
        program,
        "%probability_gradient",
        "%upstream_context",
        "%values",
        tensor,
        tensor,
        probabilities,
        3,
        3);
    program << "    %weighted_terms = \"stablehlo.multiply\"("
            << "%causal_probabilities, %probability_gradient) : ("
            << probabilities << ", " << probabilities << ") -> "
            << probabilities << "\n"
            << "    %weighted_gradient = \"stablehlo.reduce\"("
            << "%weighted_terms, %zero_scalar) ({\n"
            << "      ^bb0(%left: tensor<f32>, %right: tensor<f32>):\n"
            << "        %sum = \"stablehlo.add\"(%left, %right) : "
            << "(tensor<f32>, tensor<f32>) -> tensor<f32>\n"
            << "        \"stablehlo.return\"(%sum) : (tensor<f32>) -> ()\n"
            << "    }) {dimensions = array<i64: 3>} : (" << probabilities
            << ", tensor<f32>) -> " << rows << "\n"
            << "    %weighted_expanded = \"stablehlo.broadcast_in_dim\"("
            << "%weighted_gradient) {broadcast_dimensions = "
            << "array<i64: 0, 1, 2>} : (" << rows << ") -> " << probabilities
            << "\n"
            << "    %centered_gradient = \"stablehlo.subtract\"("
            << "%probability_gradient, %weighted_expanded) : (" << probabilities
            << ", " << probabilities << ") -> " << probabilities << "\n"
            << "    %softmax_gradient = \"stablehlo.multiply\"("
            << "%causal_probabilities, %centered_gradient) : (" << probabilities
            << ", " << probabilities << ") -> " << probabilities << "\n"
            << "    %scale_scalar = \"stablehlo.constant\"() {value = dense<"
            << float_literal(attention_scale(shape.head_width))
            << "> : tensor<f32>} : () -> tensor<f32>\n"
            << "    %scale = \"stablehlo.broadcast_in_dim\"(%scale_scalar) "
            << "{broadcast_dimensions = array<i64>} : (tensor<f32>) -> "
            << probabilities << "\n"
            << "    %scaled_gradient = \"stablehlo.multiply\"("
            << "%softmax_gradient, %scale) : (" << probabilities << ", "
            << probabilities << ") -> " << probabilities << "\n"
            << "    %score_gradient = \"stablehlo.select\"("
            << "%causal, %scaled_gradient, %zero_probabilities) : ("
            << tensor_type(
                   {shape.batch, shape.heads, shape.time, shape.time}, "i1")
            << ", " << probabilities << ", " << probabilities << ") -> "
            << probabilities << "\n";
    append_batched_attention_dot(
        program,
        "%query_gradient",
        "%score_gradient",
        "%keys",
        probabilities,
        tensor,
        tensor,
        3,
        2);
    append_batched_attention_dot(
        program,
        "%key_gradient",
        "%score_gradient",
        "%queries",
        probabilities,
        tensor,
        tensor,
        2,
        2);
    append_batched_attention_dot(
        program,
        "%value_gradient",
        "%causal_probabilities",
        "%upstream_context",
        probabilities,
        tensor,
        tensor,
        2,
        2);
    program << "    func.return %query_gradient, %key_gradient, "
            << "%value_gradient : " << tensor << ", " << tensor << ", "
            << tensor << "\n"
            << "  }\n"
            << "}\n";
    return program.str();
}

std::string
materialized_probabilities_backward_program(const MaterializedShape& shape) {
    const std::string tensor = tensor_type({
        shape.batch,
        shape.heads,
        shape.time,
        shape.head_width,
    });
    const std::string probabilities = tensor_type({
        shape.batch,
        shape.heads,
        shape.time,
        shape.time,
    });
    const std::string rows = tensor_type({
        shape.batch,
        shape.heads,
        shape.time,
    });

    std::ostringstream program;
    append_module_header(
        program,
        "riftco_tpu_materialized_causal_attention_probabilities_backward");
    program << "  func.func public @main(%queries: " << tensor
            << ", %keys: " << tensor << ", %probabilities: " << probabilities
            << ", %upstream_probabilities: " << probabilities << ") -> ("
            << tensor << ", " << tensor << ") {\n";
    append_causal_mask(program, shape, probabilities);
    program
        << "    %causal_probabilities = \"stablehlo.select\"("
        << "%causal, %probabilities, %zero_probabilities) : ("
        << tensor_type({shape.batch, shape.heads, shape.time, shape.time}, "i1")
        << ", " << probabilities << ", " << probabilities << ") -> "
        << probabilities << "\n"
        << "    %weighted_terms = \"stablehlo.multiply\"("
        << "%causal_probabilities, %upstream_probabilities) : ("
        << probabilities << ", " << probabilities << ") -> " << probabilities
        << "\n"
        << "    %weighted_gradient = \"stablehlo.reduce\"("
        << "%weighted_terms, %zero_scalar) ({\n"
        << "      ^bb0(%left: tensor<f32>, %right: tensor<f32>):\n"
        << "        %sum = \"stablehlo.add\"(%left, %right) : "
        << "(tensor<f32>, tensor<f32>) -> tensor<f32>\n"
        << "        \"stablehlo.return\"(%sum) : (tensor<f32>) -> ()\n"
        << "    }) {dimensions = array<i64: 3>} : (" << probabilities
        << ", tensor<f32>) -> " << rows << "\n"
        << "    %weighted_expanded = \"stablehlo.broadcast_in_dim\"("
        << "%weighted_gradient) {broadcast_dimensions = "
        << "array<i64: 0, 1, 2>} : (" << rows << ") -> " << probabilities
        << "\n"
        << "    %centered_gradient = \"stablehlo.subtract\"("
        << "%upstream_probabilities, %weighted_expanded) : (" << probabilities
        << ", " << probabilities << ") -> " << probabilities << "\n"
        << "    %softmax_gradient = \"stablehlo.multiply\"("
        << "%causal_probabilities, %centered_gradient) : (" << probabilities
        << ", " << probabilities << ") -> " << probabilities << "\n"
        << "    %scale_scalar = \"stablehlo.constant\"() {value = dense<"
        << float_literal(attention_scale(shape.head_width))
        << "> : tensor<f32>} : () -> tensor<f32>\n"
        << "    %scale = \"stablehlo.broadcast_in_dim\"(%scale_scalar) "
        << "{broadcast_dimensions = array<i64>} : (tensor<f32>) -> "
        << probabilities << "\n"
        << "    %scaled_gradient = \"stablehlo.multiply\"("
        << "%softmax_gradient, %scale) : (" << probabilities << ", "
        << probabilities << ") -> " << probabilities << "\n"
        << "    %score_gradient = \"stablehlo.select\"("
        << "%causal, %scaled_gradient, %zero_probabilities) : ("
        << tensor_type({shape.batch, shape.heads, shape.time, shape.time}, "i1")
        << ", " << probabilities << ", " << probabilities << ") -> "
        << probabilities << "\n";
    append_batched_attention_dot(
        program,
        "%query_gradient",
        "%score_gradient",
        "%keys",
        probabilities,
        tensor,
        tensor,
        3,
        2);
    append_batched_attention_dot(
        program,
        "%key_gradient",
        "%score_gradient",
        "%queries",
        probabilities,
        tensor,
        tensor,
        2,
        2);
    program << "    func.return %query_gradient, %key_gradient : " << tensor
            << ", " << tensor << "\n"
            << "  }\n"
            << "}\n";
    return program.str();
}

MaterializedShape
shape_from(const MaterializedCausalAttentionDimensions& dimensions) {
    return {
        dimensions.batch,
        dimensions.heads,
        dimensions.time,
        dimensions.head_width,
    };
}

TpuProgram make_program(
    TpuProgramKind kind,
    const MaterializedShape& shape,
    std::string stablehlo,
    std::string operation_name) {
    TpuProgram result{};
    result.key.kind = kind;
    result.key.dimensions = {
        shape.batch,
        shape.heads,
        shape.time,
        shape.head_width,
        0,
        0,
    };
    result.key.dimension_count = 4;
    result.stablehlo = std::move(stablehlo);
    result.operation_name = std::move(operation_name);
    return result;
}

}  // namespace

void tpu_materialized_causal_attention_forward(
    const MaterializedCausalAttentionForwardRequest& request) {
    const MaterializedShape shape = shape_from(request.dimensions);
    const std::array<std::int64_t, 4> tensor_shape{
        checked_dimension(shape.batch),
        checked_dimension(shape.heads),
        checked_dimension(shape.time),
        checked_dimension(shape.head_width),
    };
    const std::array<std::int64_t, 4> probability_shape{
        checked_dimension(shape.batch),
        checked_dimension(shape.heads),
        checked_dimension(shape.time),
        checked_dimension(shape.time),
    };
    const std::array<std::int64_t, 3> row_shape{
        checked_dimension(shape.batch),
        checked_dimension(shape.heads),
        checked_dimension(shape.time),
    };

    std::vector<float> staged_probabilities(
        request.probabilities.data().size());
    std::vector<float> staged_context(request.context.data().size());
    std::vector<float> status(
        request.context.data().size() / shape.head_width, 0.0F);
    const TpuProgram program = make_program(
        TpuProgramKind::MaterializedCausalAttentionForward,
        shape,
        materialized_forward_program(shape),
        "StableHLO materialized causal attention forward");
    const std::array<TpuHostInput, 3> inputs{
        TpuHostInput{
            request.queries.data().data(),
            request.queries.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape,
        },
        TpuHostInput{
            request.keys.data().data(),
            request.keys.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape,
        },
        TpuHostInput{
            request.values.data().data(),
            request.values.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape,
        },
    };
    const std::array<TpuHostOutput, 3> outputs{
        TpuHostOutput{
            staged_probabilities.data(),
            staged_probabilities.size() * sizeof(float),
            TpuElementType::F32,
            probability_shape,
        },
        TpuHostOutput{
            staged_context.data(),
            staged_context.size() * sizeof(float),
            TpuElementType::F32,
            tensor_shape,
        },
        TpuHostOutput{
            status.data(),
            status.size() * sizeof(float),
            TpuElementType::F32,
            row_shape,
        },
    };
    tpu_runtime_execute(program, inputs, outputs);
    require_valid_softmax_status(status);
    std::copy(
        staged_probabilities.begin(),
        staged_probabilities.end(),
        request.probabilities.data().begin());
    std::copy(
        staged_context.begin(),
        staged_context.end(),
        request.context.data().begin());
}

void tpu_materialized_causal_attention_context_backward(
    const MaterializedCausalAttentionContextBackwardRequest& request) {
    const MaterializedShape shape = shape_from(request.dimensions);
    const std::array<std::int64_t, 4> tensor_shape{
        checked_dimension(shape.batch),
        checked_dimension(shape.heads),
        checked_dimension(shape.time),
        checked_dimension(shape.head_width),
    };
    const std::array<std::int64_t, 4> probability_shape{
        checked_dimension(shape.batch),
        checked_dimension(shape.heads),
        checked_dimension(shape.time),
        checked_dimension(shape.time),
    };
    const TpuProgram program = make_program(
        TpuProgramKind::MaterializedCausalAttentionContextBackward,
        shape,
        materialized_context_backward_program(shape),
        "StableHLO materialized causal attention context backward");
    const std::array<TpuHostInput, 5> inputs{
        TpuHostInput{
            request.queries.data().data(),
            request.queries.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
        TpuHostInput{
            request.keys.data().data(),
            request.keys.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
        TpuHostInput{
            request.values.data().data(),
            request.values.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
        TpuHostInput{
            request.probabilities.data().data(),
            request.probabilities.data().size_bytes(),
            TpuElementType::F32,
            probability_shape},
        TpuHostInput{
            request.upstream_context.data().data(),
            request.upstream_context.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
    };
    const std::array<TpuHostOutput, 3> outputs{
        TpuHostOutput{
            request.query_gradient.data().data(),
            request.query_gradient.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
        TpuHostOutput{
            request.key_gradient.data().data(),
            request.key_gradient.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
        TpuHostOutput{
            request.value_gradient.data().data(),
            request.value_gradient.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
    };
    tpu_runtime_execute(program, inputs, outputs);
}

void tpu_materialized_causal_attention_probabilities_backward(
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request) {
    const MaterializedShape shape = shape_from(request.dimensions);
    const std::array<std::int64_t, 4> tensor_shape{
        checked_dimension(shape.batch),
        checked_dimension(shape.heads),
        checked_dimension(shape.time),
        checked_dimension(shape.head_width),
    };
    const std::array<std::int64_t, 4> probability_shape{
        checked_dimension(shape.batch),
        checked_dimension(shape.heads),
        checked_dimension(shape.time),
        checked_dimension(shape.time),
    };
    const TpuProgram program = make_program(
        TpuProgramKind::MaterializedCausalAttentionProbabilitiesBackward,
        shape,
        materialized_probabilities_backward_program(shape),
        "StableHLO materialized causal attention probabilities backward");
    const std::array<TpuHostInput, 4> inputs{
        TpuHostInput{
            request.queries.data().data(),
            request.queries.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
        TpuHostInput{
            request.keys.data().data(),
            request.keys.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
        TpuHostInput{
            request.probabilities.data().data(),
            request.probabilities.data().size_bytes(),
            TpuElementType::F32,
            probability_shape},
        TpuHostInput{
            request.upstream_probabilities.data().data(),
            request.upstream_probabilities.data().size_bytes(),
            TpuElementType::F32,
            probability_shape},
    };
    const std::array<TpuHostOutput, 2> outputs{
        TpuHostOutput{
            request.query_gradient.data().data(),
            request.query_gradient.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
        TpuHostOutput{
            request.key_gradient.data().data(),
            request.key_gradient.data().size_bytes(),
            TpuElementType::F32,
            tensor_shape},
    };
    tpu_runtime_execute(program, inputs, outputs);
}

}  // namespace riftco_transformer::backend_detail
