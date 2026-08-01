#include "core/backend/attention/tpu/paged_decode.hpp"

#include "core/backend/adapters/tpu/runtime.hpp"
#include "core/backend/attention/tpu/common.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

using attention_tpu_detail::attention_scale;
using attention_tpu_detail::checked_dimension;
using attention_tpu_detail::float_literal;
using attention_tpu_detail::require_valid_softmax_status;
using attention_tpu_detail::tensor_type;

struct PagedShape {
    std::size_t heads;
    std::size_t head_width;
    std::size_t block_size;
    std::size_t physical_blocks;
    std::size_t sequence_bucket;
};

std::size_t checked_product(std::size_t left, std::size_t right) {
    if (right == 0) {
        throw std::invalid_argument(
            "TPU paged-attention dimensions must be positive");
    }
    if (left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error("TPU paged-attention shape overflow");
    }
    return left * right;
}

std::size_t sequence_bucket(std::size_t sequence_length) {
    std::size_t result = 1;
    while (result < sequence_length) {
        if (result > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::overflow_error(
                "TPU paged-attention sequence bucket overflow");
        }
        result *= 2;
    }
    return result;
}

void append_head_batched_dot(
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
            << "        lhs_batching_dimensions = [0],\n"
            << "        rhs_batching_dimensions = [0],\n"
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

std::string paged_decode_program(const PagedShape& shape) {
    const std::size_t flattened_positions =
        checked_product(shape.physical_blocks, shape.block_size);
    const std::string query = tensor_type({
        shape.heads,
        shape.head_width,
    });
    const std::string pages = tensor_type({
        shape.physical_blocks,
        shape.heads,
        shape.block_size,
        shape.head_width,
    });
    const std::string transposed_pages = tensor_type({
        shape.physical_blocks,
        shape.block_size,
        shape.heads,
        shape.head_width,
    });
    const std::string flat_pages = tensor_type({
        flattened_positions,
        shape.heads,
        shape.head_width,
    });
    const std::string indices = tensor_type({shape.sequence_bucket}, "i32");
    const std::string mask = tensor_type({shape.sequence_bucket});
    const std::string mask_predicate =
        tensor_type({shape.sequence_bucket}, "i1");
    const std::string gathered = tensor_type({
        shape.sequence_bucket,
        shape.heads,
        shape.head_width,
    });
    const std::string logical_pages = tensor_type({
        shape.heads,
        shape.sequence_bucket,
        shape.head_width,
    });
    const std::string scores = tensor_type({
        shape.heads,
        shape.sequence_bucket,
    });
    const std::string score_predicates = tensor_type(
        {
            shape.heads,
            shape.sequence_bucket,
        },
        "i1");
    const std::string rows = tensor_type({shape.heads});
    const std::string row_predicates = tensor_type({shape.heads}, "i1");

    std::ostringstream program;
    program << "module @riftco_tpu_paged_decode_attention_forward attributes {"
            << "mhlo.num_partitions = 1 : i32, "
            << "mhlo.num_replicas = 1 : i32} {\n"
            << "  func.func public @main(%query: " << query
            << ", %key_pages: " << pages << ", %value_pages: " << pages
            << ", %token_indices: " << indices << ", %valid_mask: " << mask
            << ") -> (" << query << ", " << rows << ") {\n"
            << "    %key_transposed = \"stablehlo.transpose\"(%key_pages) {"
            << "permutation = array<i64: 0, 2, 1, 3>} : (" << pages << ") -> "
            << transposed_pages << "\n"
            << "    %value_transposed = \"stablehlo.transpose\"(%value_pages) {"
            << "permutation = array<i64: 0, 2, 1, 3>} : (" << pages << ") -> "
            << transposed_pages << "\n"
            << "    %key_flat = \"stablehlo.reshape\"(%key_transposed) : ("
            << transposed_pages << ") -> " << flat_pages << "\n"
            << "    %value_flat = \"stablehlo.reshape\"(%value_transposed) : ("
            << transposed_pages << ") -> " << flat_pages << "\n"
            << "    %key_gathered = \"stablehlo.gather\"("
            << "%key_flat, %token_indices) {\n"
            << "      dimension_numbers = #stablehlo.gather<\n"
            << "        offset_dims = [1, 2],\n"
            << "        collapsed_slice_dims = [0],\n"
            << "        start_index_map = [0],\n"
            << "        index_vector_dim = 1>,\n"
            << "      slice_sizes = array<i64: 1, " << shape.heads << ", "
            << shape.head_width << ">,\n"
            << "      indices_are_sorted = false\n"
            << "    } : (" << flat_pages << ", " << indices << ") -> "
            << gathered << "\n"
            << "    %value_gathered = \"stablehlo.gather\"("
            << "%value_flat, %token_indices) {\n"
            << "      dimension_numbers = #stablehlo.gather<\n"
            << "        offset_dims = [1, 2],\n"
            << "        collapsed_slice_dims = [0],\n"
            << "        start_index_map = [0],\n"
            << "        index_vector_dim = 1>,\n"
            << "      slice_sizes = array<i64: 1, " << shape.heads << ", "
            << shape.head_width << ">,\n"
            << "      indices_are_sorted = false\n"
            << "    } : (" << flat_pages << ", " << indices << ") -> "
            << gathered << "\n"
            << "    %logical_keys = \"stablehlo.transpose\"(%key_gathered) {"
            << "permutation = array<i64: 1, 0, 2>} : (" << gathered << ") -> "
            << logical_pages << "\n"
            << "    %logical_values = \"stablehlo.transpose\"("
            << "%value_gathered) {permutation = array<i64: 1, 0, 2>} : ("
            << gathered << ") -> " << logical_pages << "\n";
    append_head_batched_dot(
        program,
        "%raw_scores",
        "%query",
        "%logical_keys",
        query,
        logical_pages,
        scores,
        1,
        2);
    program << "    %scale_scalar = \"stablehlo.constant\"() {value = dense<"
            << float_literal(attention_scale(shape.head_width))
            << "> : tensor<f32>} : () -> tensor<f32>\n"
            << "    %scale = \"stablehlo.broadcast_in_dim\"(%scale_scalar) "
            << "{broadcast_dimensions = array<i64>} : (tensor<f32>) -> "
            << scores << "\n"
            << "    %scaled_scores = \"stablehlo.multiply\"("
            << "%raw_scores, %scale) : (" << scores << ", " << scores << ") -> "
            << scores << "\n"
            << "    %zero_scalar = \"stablehlo.constant\"() {"
            << "value = dense<0.000000000e+00> : tensor<f32>} : () -> "
            << "tensor<f32>\n"
            << "    %zero_mask = \"stablehlo.broadcast_in_dim\"(%zero_scalar) "
            << "{broadcast_dimensions = array<i64>} : (tensor<f32>) -> " << mask
            << "\n"
            << "    %valid_positions = \"stablehlo.compare\"("
            << "%valid_mask, %zero_mask) {comparison_direction = "
            << "#stablehlo<comparison_direction GT>, compare_type = "
            << "#stablehlo<comparison_type FLOAT>} : (" << mask << ", " << mask
            << ") -> " << mask_predicate << "\n"
            << "    %valid_scores = \"stablehlo.broadcast_in_dim\"("
            << "%valid_positions) {broadcast_dimensions = array<i64: 1>} : ("
            << mask_predicate << ") -> " << score_predicates << "\n"
            << "    %negative_infinity_scalar = \"stablehlo.constant\"() {"
            << "value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>\n"
            << "    %negative_infinity = \"stablehlo.broadcast_in_dim\"("
            << "%negative_infinity_scalar) {broadcast_dimensions = array<i64>} "
            << ": (tensor<f32>) -> " << scores << "\n"
            << "    %masked_scores = \"stablehlo.select\"("
            << "%valid_scores, %scaled_scores, %negative_infinity) : ("
            << score_predicates << ", " << scores << ", " << scores << ") -> "
            << scores << "\n"
            << "    %row_maxima = \"stablehlo.reduce\"("
            << "%masked_scores, %negative_infinity_scalar) ({\n"
            << "      ^bb0(%left: tensor<f32>, %right: tensor<f32>):\n"
            << "        %maximum = \"stablehlo.maximum\"(%left, %right) : "
            << "(tensor<f32>, tensor<f32>) -> tensor<f32>\n"
            << "        \"stablehlo.return\"(%maximum) : (tensor<f32>) -> ()\n"
            << "    }) {dimensions = array<i64: 1>} : (" << scores
            << ", tensor<f32>) -> " << rows << "\n"
            << "    %row_maxima_expanded = \"stablehlo.broadcast_in_dim\"("
            << "%row_maxima) {broadcast_dimensions = array<i64: 0>} : (" << rows
            << ") -> " << scores << "\n"
            << "    %centered = \"stablehlo.subtract\"("
            << "%masked_scores, %row_maxima_expanded) : (" << scores << ", "
            << scores << ") -> " << scores << "\n"
            << "    %exponentials = \"stablehlo.exponential\"(%centered) : ("
            << scores << ") -> " << scores << "\n"
            << "    %row_exp_sums = \"stablehlo.reduce\"("
            << "%exponentials, %zero_scalar) ({\n"
            << "      ^bb0(%left: tensor<f32>, %right: tensor<f32>):\n"
            << "        %sum = \"stablehlo.add\"(%left, %right) : "
            << "(tensor<f32>, tensor<f32>) -> tensor<f32>\n"
            << "        \"stablehlo.return\"(%sum) : (tensor<f32>) -> ()\n"
            << "    }) {dimensions = array<i64: 1>} : (" << scores
            << ", tensor<f32>) -> " << rows << "\n"
            << "    %row_exp_sums_expanded = \"stablehlo.broadcast_in_dim\"("
            << "%row_exp_sums) {broadcast_dimensions = array<i64: 0>} : ("
            << rows << ") -> " << scores << "\n"
            << "    %probabilities = \"stablehlo.divide\"("
            << "%exponentials, %row_exp_sums_expanded) : (" << scores << ", "
            << scores << ") -> " << scores << "\n";
    append_head_batched_dot(
        program,
        "%context",
        "%probabilities",
        "%logical_values",
        scores,
        logical_pages,
        query,
        1,
        1);
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
            << "    func.return %context, %status : " << query << ", " << rows
            << "\n"
            << "  }\n"
            << "}\n";
    return program.str();
}

}  // namespace

void tpu_paged_decode_attention_forward(
    const PagedDecodeAttentionForwardRequest& request) {
    const auto& dimensions = request.dimensions;
    const std::size_t bucket = sequence_bucket(dimensions.sequence_length);
    const PagedShape shape{
        dimensions.heads,
        dimensions.head_width,
        dimensions.block_size,
        dimensions.physical_block_count,
        bucket,
    };
    const std::array<std::int64_t, 2> query_shape{
        checked_dimension(shape.heads),
        checked_dimension(shape.head_width),
    };
    const std::array<std::int64_t, 4> page_shape{
        checked_dimension(shape.physical_blocks),
        checked_dimension(shape.heads),
        checked_dimension(shape.block_size),
        checked_dimension(shape.head_width),
    };
    const std::array<std::int64_t, 1> sequence_shape{
        checked_dimension(shape.sequence_bucket),
    };
    const std::array<std::int64_t, 1> status_shape{
        checked_dimension(shape.heads),
    };

    std::vector<std::int32_t> token_indices(bucket, 0);
    std::vector<float> valid_mask(bucket, 0.0F);
    for (std::size_t position = 0; position < dimensions.sequence_length;
         ++position) {
        const std::size_t logical_block = position / dimensions.block_size;
        const std::size_t physical_block =
            static_cast<std::size_t>(request.block_table[logical_block]);
        const std::size_t page_offset = position % dimensions.block_size;
        const std::size_t flattened_base =
            checked_product(physical_block, dimensions.block_size);
        constexpr auto largest_index =
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
        if (flattened_base > largest_index ||
            page_offset > largest_index - flattened_base) {
            throw std::overflow_error(
                "TPU paged-attention indices exceed StableHLO s32");
        }
        const std::size_t flattened = flattened_base + page_offset;
        token_indices[position] = static_cast<std::int32_t>(flattened);
        valid_mask[position] = 1.0F;
    }

    std::vector<float> staged_context(request.context.data().size());
    std::vector<float> status(shape.heads, 0.0F);
    TpuProgram program{};
    program.key.kind = TpuProgramKind::PagedDecodeAttentionForward;
    program.key.dimensions = {
        shape.heads,
        shape.head_width,
        shape.block_size,
        shape.physical_blocks,
        shape.sequence_bucket,
        0,
    };
    program.key.dimension_count = 5;
    program.stablehlo = paged_decode_program(shape);
    program.operation_name = "StableHLO paged decode attention forward";

    const std::array<TpuHostInput, 5> inputs{
        TpuHostInput{
            request.queries.data().data(),
            request.queries.data().size_bytes(),
            TpuElementType::F32,
            query_shape},
        TpuHostInput{
            request.key_pages.data().data(),
            request.key_pages.data().size_bytes(),
            TpuElementType::F32,
            page_shape},
        TpuHostInput{
            request.value_pages.data().data(),
            request.value_pages.data().size_bytes(),
            TpuElementType::F32,
            page_shape},
        TpuHostInput{
            token_indices.data(),
            token_indices.size() * sizeof(std::int32_t),
            TpuElementType::S32,
            sequence_shape},
        TpuHostInput{
            valid_mask.data(),
            valid_mask.size() * sizeof(float),
            TpuElementType::F32,
            sequence_shape},
    };
    const std::array<TpuHostOutput, 2> outputs{
        TpuHostOutput{
            staged_context.data(),
            staged_context.size() * sizeof(float),
            TpuElementType::F32,
            query_shape},
        TpuHostOutput{
            status.data(),
            status.size() * sizeof(float),
            TpuElementType::F32,
            status_shape},
    };
    tpu_runtime_execute(program, inputs, outputs);
    require_valid_softmax_status(status);
    std::copy(
        staged_context.begin(),
        staged_context.end(),
        request.context.data().begin());
}

}  // namespace riftco_transformer::backend_detail
