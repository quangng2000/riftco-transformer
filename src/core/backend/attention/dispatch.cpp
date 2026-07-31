#include "core/backend/attention/dispatch.hpp"

#include "core/backend/adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace transformer_lab::backend_detail {
namespace {

const BackendAdapter& require_backend_adapter(
    ExecutionBackend backend
) {
    const auto* adapter = find_backend_adapter(backend);
    if (adapter == nullptr) {
        throw std::invalid_argument("unknown execution backend");
    }
    return *adapter;
}

void require_available(const BackendAdapter& adapter) {
    if (!adapter.is_available()) {
        throw std::runtime_error(
            std::string(adapter.name()) +
            " execution backend is unavailable"
        );
    }
}

std::size_t checked_product(
    std::initializer_list<std::size_t> factors
) {
    std::size_t result = 1;
    for (const std::size_t factor : factors) {
        if (
            factor != 0 &&
            result > std::numeric_limits<std::size_t>::max() / factor
        ) {
            throw std::overflow_error(
                "backend operation size exceeds addressable storage"
            );
        }
        result *= factor;
    }
    return result;
}

void require_operation_storage(
    const TensorStorage& storage,
    ExecutionBackend backend,
    std::size_t expected_size
) {
    if (storage.backend() != backend) {
        throw std::invalid_argument(
            "backend operation storage does not match its backend"
        );
    }
    if (
        expected_size == 0 ||
        storage.size() != expected_size ||
        storage.data().size() != expected_size
    ) {
        throw std::logic_error(
            "backend operation storage size does not match its dimensions"
        );
    }
}

bool storage_aliases(
    const TensorStorage& left,
    const TensorStorage& right
) noexcept {
    if (&left == &right) {
        return true;
    }
    const void* left_handle = left.native_handle();
    const void* right_handle = right.native_handle();
    return
        left_handle != nullptr &&
        left_handle == right_handle;
}

void require_distinct(
    const TensorStorage& left,
    const TensorStorage& right
) {
    if (storage_aliases(left, right)) {
        throw std::invalid_argument(
            "backend operation inputs and outputs must not alias"
        );
    }
}

void require_output_separation(
    std::initializer_list<const TensorStorage*> inputs,
    std::initializer_list<const TensorStorage*> outputs
) {
    for (const auto* output : outputs) {
        for (const auto* input : inputs) {
            require_distinct(*input, *output);
        }
    }
    for (auto left = outputs.begin(); left != outputs.end(); ++left) {
        for (auto right = left + 1; right != outputs.end(); ++right) {
            require_distinct(**left, **right);
        }
    }
}

std::size_t checked_materialized_tensor_size(
    const MaterializedCausalAttentionDimensions& dimensions
) {
    if (
        dimensions.batch == 0 ||
        dimensions.heads == 0 ||
        dimensions.time == 0 ||
        dimensions.head_width == 0
    ) {
        throw std::invalid_argument(
            "materialized causal attention dimensions must be positive"
        );
    }
    return checked_product({
        dimensions.batch,
        dimensions.heads,
        dimensions.time,
        dimensions.head_width,
    });
}

std::size_t checked_materialized_probability_size(
    const MaterializedCausalAttentionDimensions& dimensions
) {
    static_cast<void>(checked_materialized_tensor_size(dimensions));
    return checked_product({
        dimensions.batch,
        dimensions.heads,
        dimensions.time,
        dimensions.time,
    });
}

std::size_t checked_flash_tensor_size(
    const FlashCausalAttentionDimensions& dimensions
) {
    if (
        dimensions.batch == 0 ||
        dimensions.heads == 0 ||
        dimensions.time == 0 ||
        dimensions.head_width == 0
    ) {
        throw std::invalid_argument(
            "Flash causal attention dimensions must be positive"
        );
    }
    return checked_product({
        dimensions.batch,
        dimensions.heads,
        dimensions.time,
        dimensions.head_width,
    });
}

std::size_t checked_flash_row_count(
    const FlashCausalAttentionDimensions& dimensions
) {
    static_cast<void>(checked_flash_tensor_size(dimensions));
    return checked_product({
        dimensions.batch,
        dimensions.heads,
        dimensions.time,
    });
}

std::size_t checked_paged_decode_query_size(
    const PagedDecodeAttentionDimensions& dimensions
) {
    if (
        dimensions.heads == 0 ||
        dimensions.head_width == 0 ||
        dimensions.block_size == 0 ||
        dimensions.physical_block_count == 0 ||
        dimensions.sequence_length == 0
    ) {
        throw std::invalid_argument(
            "paged decode attention dimensions must be positive"
        );
    }
    if (
        dimensions.physical_block_count - 1 >
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )
    ) {
        throw std::overflow_error(
            "paged decode attention physical block IDs exceed uint32_t"
        );
    }
    return checked_product({
        dimensions.heads,
        dimensions.head_width,
    });
}

std::size_t checked_paged_decode_pool_size(
    const PagedDecodeAttentionDimensions& dimensions
) {
    static_cast<void>(checked_paged_decode_query_size(dimensions));
    return checked_product({
        dimensions.physical_block_count,
        dimensions.heads,
        dimensions.block_size,
        dimensions.head_width,
    });
}

}  // namespace

void dispatch_materialized_causal_attention_forward(
    ExecutionBackend backend,
    const MaterializedCausalAttentionForwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    const auto tensor_size =
        checked_materialized_tensor_size(request.dimensions);
    const auto probability_size =
        checked_materialized_probability_size(request.dimensions);
    require_operation_storage(request.queries, backend, tensor_size);
    require_operation_storage(request.keys, backend, tensor_size);
    require_operation_storage(request.values, backend, tensor_size);
    require_operation_storage(
        request.probabilities,
        backend,
        probability_size
    );
    require_operation_storage(request.context, backend, tensor_size);
    require_distinct(request.queries, request.probabilities);
    require_distinct(request.keys, request.probabilities);
    require_distinct(request.values, request.probabilities);
    require_distinct(request.queries, request.context);
    require_distinct(request.keys, request.context);
    require_distinct(request.values, request.context);
    require_distinct(request.probabilities, request.context);
    adapter.materialized_causal_attention_forward(request);
}

void dispatch_materialized_causal_attention_context_backward(
    ExecutionBackend backend,
    const MaterializedCausalAttentionContextBackwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    const auto tensor_size =
        checked_materialized_tensor_size(request.dimensions);
    const auto probability_size =
        checked_materialized_probability_size(request.dimensions);
    require_operation_storage(request.queries, backend, tensor_size);
    require_operation_storage(request.keys, backend, tensor_size);
    require_operation_storage(request.values, backend, tensor_size);
    require_operation_storage(
        request.probabilities,
        backend,
        probability_size
    );
    require_operation_storage(
        request.upstream_context,
        backend,
        tensor_size
    );
    require_operation_storage(
        request.query_gradient,
        backend,
        tensor_size
    );
    require_operation_storage(
        request.key_gradient,
        backend,
        tensor_size
    );
    require_operation_storage(
        request.value_gradient,
        backend,
        tensor_size
    );
    require_output_separation(
        {
            &request.queries,
            &request.keys,
            &request.values,
            &request.probabilities,
            &request.upstream_context,
        },
        {
            &request.query_gradient,
            &request.key_gradient,
            &request.value_gradient,
        }
    );
    adapter.materialized_causal_attention_context_backward(request);
}

void dispatch_materialized_causal_attention_probabilities_backward(
    ExecutionBackend backend,
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    const auto tensor_size =
        checked_materialized_tensor_size(request.dimensions);
    const auto probability_size =
        checked_materialized_probability_size(request.dimensions);
    require_operation_storage(request.queries, backend, tensor_size);
    require_operation_storage(request.keys, backend, tensor_size);
    require_operation_storage(
        request.probabilities,
        backend,
        probability_size
    );
    require_operation_storage(
        request.upstream_probabilities,
        backend,
        probability_size
    );
    require_operation_storage(
        request.query_gradient,
        backend,
        tensor_size
    );
    require_operation_storage(
        request.key_gradient,
        backend,
        tensor_size
    );
    require_output_separation(
        {
            &request.queries,
            &request.keys,
            &request.probabilities,
            &request.upstream_probabilities,
        },
        {
            &request.query_gradient,
            &request.key_gradient,
        }
    );
    adapter.materialized_causal_attention_probabilities_backward(
        request
    );
}

void dispatch_flash_causal_attention_forward(
    ExecutionBackend backend,
    const FlashCausalAttentionForwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    const auto tensor_size =
        checked_flash_tensor_size(request.dimensions);
    const auto row_count =
        checked_flash_row_count(request.dimensions);
    require_operation_storage(request.queries, backend, tensor_size);
    require_operation_storage(request.keys, backend, tensor_size);
    require_operation_storage(request.values, backend, tensor_size);
    require_operation_storage(request.row_maxima, backend, row_count);
    require_operation_storage(request.row_exp_sums, backend, row_count);
    require_operation_storage(request.context, backend, tensor_size);
    require_output_separation(
        {
            &request.queries,
            &request.keys,
            &request.values,
        },
        {
            &request.row_maxima,
            &request.row_exp_sums,
            &request.context,
        }
    );
    adapter.flash_causal_attention_forward(request);
}

void dispatch_flash_causal_attention_backward(
    ExecutionBackend backend,
    const FlashCausalAttentionBackwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    const auto tensor_size =
        checked_flash_tensor_size(request.dimensions);
    const auto row_count =
        checked_flash_row_count(request.dimensions);
    require_operation_storage(request.queries, backend, tensor_size);
    require_operation_storage(request.keys, backend, tensor_size);
    require_operation_storage(request.values, backend, tensor_size);
    require_operation_storage(request.row_maxima, backend, row_count);
    require_operation_storage(request.row_exp_sums, backend, row_count);
    require_operation_storage(
        request.upstream_context,
        backend,
        tensor_size
    );
    require_operation_storage(
        request.query_gradient,
        backend,
        tensor_size
    );
    require_operation_storage(
        request.key_gradient,
        backend,
        tensor_size
    );
    require_operation_storage(
        request.value_gradient,
        backend,
        tensor_size
    );
    require_output_separation(
        {
            &request.queries,
            &request.keys,
            &request.values,
            &request.row_maxima,
            &request.row_exp_sums,
            &request.upstream_context,
        },
        {
            &request.query_gradient,
            &request.key_gradient,
            &request.value_gradient,
        }
    );
    adapter.flash_causal_attention_backward(request);
}

void dispatch_paged_decode_attention_forward(
    ExecutionBackend backend,
    const PagedDecodeAttentionForwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);

    const auto query_size =
        checked_paged_decode_query_size(request.dimensions);
    const auto pool_size =
        checked_paged_decode_pool_size(request.dimensions);
    const std::size_t required_blocks =
        (request.dimensions.sequence_length - 1) /
            request.dimensions.block_size +
        1;
    if (request.block_table.size() != required_blocks) {
        throw std::invalid_argument(
            "paged decode attention block table size does not match "
            "the sequence"
        );
    }
    if (
        required_blocks >
        request.dimensions.physical_block_count
    ) {
        throw std::invalid_argument(
            "paged decode attention has fewer physical than logical blocks"
        );
    }

    // Repeated physical IDs are a valid low-level mapping (for example shared
    // immutable prefix pages). Cache allocators enforce copy-on-write before a
    // repeated page is mutated.
    for (const std::uint32_t block : request.block_table) {
        if (
            static_cast<std::size_t>(block) >=
            request.dimensions.physical_block_count
        ) {
            throw std::out_of_range(
                "paged decode attention block table contains an "
                "invalid block"
            );
        }
    }

    require_operation_storage(
        request.queries,
        backend,
        query_size
    );
    require_operation_storage(
        request.key_pages,
        backend,
        pool_size
    );
    require_operation_storage(
        request.value_pages,
        backend,
        pool_size
    );
    require_operation_storage(
        request.context,
        backend,
        query_size
    );
    require_distinct(request.queries, request.context);
    require_distinct(request.key_pages, request.context);
    require_distinct(request.value_pages, request.context);
    adapter.paged_decode_attention_forward(request);
}

}  // namespace transformer_lab::backend_detail
