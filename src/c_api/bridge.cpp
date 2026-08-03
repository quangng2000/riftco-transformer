#include "internal/bridge.hpp"

namespace riftco_transformer::c_api::detail {

riftco_transformer::ExecutionBackend checked_backend(
    rt_backend backend
) {
    riftco_transformer::ExecutionBackend result;
    switch (backend) {
        case RT_BACKEND_CPU:
            result = riftco_transformer::ExecutionBackend::Cpu;
            break;
        case RT_BACKEND_METAL:
            result = riftco_transformer::ExecutionBackend::Metal;
            break;
        case RT_BACKEND_CUDA:
            result = riftco_transformer::ExecutionBackend::Cuda;
            break;
        case RT_BACKEND_TPU:
            result = riftco_transformer::ExecutionBackend::Tpu;
            break;
        default:
            throw std::invalid_argument("unknown C API backend");
    }
    if (!riftco_transformer::execution_backend_available(result)) {
        std::string message = std::string(
            riftco_transformer::execution_backend_name(result)
        ) + " execution backend is unavailable";
        const std::string_view reason =
            riftco_transformer::execution_backend_unavailability_reason(
                result
            );
        if (!reason.empty()) {
            message += ": ";
            message += reason;
        }
        throw BackendUnavailable(message);
    }
    return result;
}

rt_backend c_backend(
    riftco_transformer::ExecutionBackend backend
) {
    switch (backend) {
        case riftco_transformer::ExecutionBackend::Cpu:
            return RT_BACKEND_CPU;
        case riftco_transformer::ExecutionBackend::Metal:
            return RT_BACKEND_METAL;
        case riftco_transformer::ExecutionBackend::Cuda:
            return RT_BACKEND_CUDA;
        case riftco_transformer::ExecutionBackend::Tpu:
            return RT_BACKEND_TPU;
    }
    throw std::invalid_argument("unknown native backend");
}

riftco_transformer::FullSequenceAttentionKind
checked_full_sequence_attention(
    rt_full_sequence_attention_kind kind
) {
    switch (kind) {
        case RT_FULL_SEQUENCE_ATTENTION_MATERIALIZED:
            return riftco_transformer::FullSequenceAttentionKind::Materialized;
        case RT_FULL_SEQUENCE_ATTENTION_FLASH:
            return riftco_transformer::FullSequenceAttentionKind::Flash;
        default:
            throw std::invalid_argument(
                "unknown C API full-sequence attention kind"
            );
    }
}

rt_full_sequence_attention_kind c_full_sequence_attention(
    riftco_transformer::FullSequenceAttentionKind kind
) {
    switch (kind) {
        case riftco_transformer::FullSequenceAttentionKind::Materialized:
            return RT_FULL_SEQUENCE_ATTENTION_MATERIALIZED;
        case riftco_transformer::FullSequenceAttentionKind::Flash:
            return RT_FULL_SEQUENCE_ATTENTION_FLASH;
    }
    throw std::invalid_argument(
        "unknown native full-sequence attention kind"
    );
}

std::size_t checked_size(
    std::uint64_t value,
    const char* description
) {
    if (value >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max()
        )) {
        throw std::overflow_error(
            std::string(description) + " exceeds size_t"
        );
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t checked_u64(
    std::size_t value,
    const char* description
) {
    if constexpr (
        std::numeric_limits<std::size_t>::max() >
        std::numeric_limits<std::uint64_t>::max()
    ) {
        if (value >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max()
            )) {
            throw std::overflow_error(
                std::string(description) + " exceeds uint64"
            );
        }
    }
    return static_cast<std::uint64_t>(value);
}

riftco_transformer::Tensor::Shape checked_shape(
    const std::uint64_t* shape,
    std::uint64_t rank
) {
    const std::size_t native_rank =
        checked_size(rank, "tensor rank");
    if (native_rank != 0 && shape == nullptr) {
        throw std::invalid_argument(
            "tensor shape pointer must not be null"
        );
    }

    riftco_transformer::Tensor::Shape result;
    result.reserve(native_rank);
    for (std::size_t index = 0;
         index < native_rank;
         ++index) {
        result.push_back(
            checked_size(shape[index], "tensor dimension")
        );
    }
    return result;
}

void checked_structure_size(
    std::uint64_t actual,
    std::size_t minimum,
    const char* description
) {
    if (actual < checked_u64(minimum, description)) {
        throw std::invalid_argument(
            std::string(description) + " is too small"
        );
    }
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* description
) {
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(
            std::string(description) + " overflows size_t"
        );
    }
    return left * right;
}

std::vector<riftco_transformer::TokenId> checked_token_ids(
    const std::uint32_t* values,
    std::uint64_t value_count,
    const char* description
) {
    const std::size_t count =
        checked_size(value_count, description);
    if (count != 0 && values == nullptr) {
        throw std::invalid_argument(
            std::string(description) +
            " pointer must not be null"
        );
    }
    if (count == 0) {
        return {};
    }
    return {values, values + count};
}

void require_model(const rt_model* model) {
    if (model == nullptr || model->state == nullptr) {
        throw std::invalid_argument(
            "model handle must not be null"
        );
    }
}

void require_no_active_decode_sessions(
    const TrainableOwnerState& state,
    const char* operation
) {
    if (state.active_decode_sessions.load(
            std::memory_order_relaxed
        ) != 0) {
        throw std::invalid_argument(
            std::string("cannot ") + operation +
            " while decode sessions are alive"
        );
    }
}

void require_parameter_list(
    const rt_parameter_list* parameters
) {
    if (parameters == nullptr ||
        parameters->owner == nullptr) {
        throw std::invalid_argument(
            "parameter-list handle must not be null"
        );
    }
}

const riftco_transformer::NamedParameter& checked_parameter(
    const rt_parameter_list* parameters,
    std::uint64_t index
) {
    require_parameter_list(parameters);
    const std::size_t native_index =
        checked_size(index, "parameter index");
    if (native_index >= parameters->value.size()) {
        throw std::out_of_range(
            "parameter index is outside the list"
        );
    }
    const auto& parameter = parameters->value[native_index];
    if (parameter.parameter == nullptr) {
        throw std::logic_error(
            "parameter list contains a null parameter"
        );
    }
    return parameter;
}

void require_epoch_increment_available(
    const TrainableOwnerState& state
) {
    if (state.parameter_epoch.load(
            std::memory_order_relaxed
        ) == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "model parameter epoch overflow"
        );
    }
}

void require_adam(const rt_adam* adam) {
    if (adam == nullptr || adam->owner == nullptr) {
        throw std::invalid_argument(
            "Adam handle must not be null"
        );
    }
}

void copy_tensor_shape(
    const riftco_transformer::Tensor& tensor,
    std::uint64_t* output_dimensions,
    std::uint64_t dimension_capacity
) {
    const std::size_t capacity =
        checked_size(
            dimension_capacity,
            "shape output capacity"
        );
    if (capacity < tensor.rank()) {
        throw std::invalid_argument(
            "shape output capacity is too small"
        );
    }
    if (tensor.rank() != 0 && output_dimensions == nullptr) {
        throw std::invalid_argument(
            "shape output must not be null"
        );
    }
    for (std::size_t index = 0;
         index < tensor.rank();
         ++index) {
        output_dimensions[index] = checked_u64(
            tensor.shape()[index],
            "tensor dimension"
        );
    }
}

void copy_tensor_values(
    const riftco_transformer::Tensor& tensor,
    float* output_values,
    std::uint64_t value_capacity
) {
    const std::size_t capacity =
        checked_size(
            value_capacity,
            "value output capacity"
        );
    if (capacity < tensor.numel()) {
        throw std::invalid_argument(
            "value output capacity is too small"
        );
    }
    if (tensor.numel() != 0 && output_values == nullptr) {
        throw std::invalid_argument(
            "value output must not be null"
        );
    }
    std::copy(
        tensor.data().begin(),
        tensor.data().end(),
        output_values
    );
}

std::vector<float> checked_f32_values(
    const float* values,
    std::uint64_t value_count,
    const char* description
) {
    const std::size_t count = checked_size(value_count, description);
    if (count != 0 && values == nullptr) {
        throw std::invalid_argument(
            std::string(description) + " must not be null"
        );
    }
    std::vector<float> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument(
                std::string(description) + " must contain finite values"
            );
        }
        result.push_back(values[index]);
    }
    return result;
}

}  // namespace riftco_transformer::c_api::detail
