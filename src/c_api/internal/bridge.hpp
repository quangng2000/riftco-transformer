#pragma once

#include "error.hpp"

namespace riftco_transformer::c_api::detail {

riftco_transformer::ExecutionBackend checked_backend(rt_backend backend);
rt_backend c_backend(riftco_transformer::ExecutionBackend backend);

riftco_transformer::FullSequenceAttentionKind
checked_full_sequence_attention(rt_full_sequence_attention_kind kind);
rt_full_sequence_attention_kind c_full_sequence_attention(
    riftco_transformer::FullSequenceAttentionKind kind
);

std::size_t checked_size(
    std::uint64_t value,
    const char* description
);
std::uint64_t checked_u64(
    std::size_t value,
    const char* description
);
riftco_transformer::Tensor::Shape checked_shape(
    const std::uint64_t* shape,
    std::uint64_t rank
);
void checked_structure_size(
    std::uint64_t actual,
    std::size_t minimum,
    const char* description
);
std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* description
);
std::vector<riftco_transformer::TokenId> checked_token_ids(
    const std::uint32_t* values,
    std::uint64_t value_count,
    const char* description
);

template <typename InputElement, typename OutputElement>
void copy_sized_output(
    std::span<const InputElement> values,
    OutputElement* output,
    std::uint64_t output_capacity,
    std::uint64_t* required_count,
    const char* description
) {
    if (required_count == nullptr) {
        throw std::invalid_argument(
            std::string(description) +
            " required-count output must not be null"
        );
    }
    *required_count = checked_u64(values.size(), description);
    const std::size_t capacity =
        checked_size(output_capacity, description);
    if (capacity == 0 && output == nullptr) {
        return;
    }
    if (output == nullptr) {
        throw std::invalid_argument(
            std::string(description) + " output must not be null"
        );
    }
    if (capacity < values.size()) {
        throw std::out_of_range(
            std::string(description) +
            " output capacity is too small"
        );
    }
    std::copy(values.begin(), values.end(), output);
}

void require_model(const rt_model* model);
void require_no_active_decode_sessions(
    const TrainableOwnerState& state,
    const char* operation
);
void require_parameter_list(const rt_parameter_list* parameters);
const riftco_transformer::NamedParameter& checked_parameter(
    const rt_parameter_list* parameters,
    std::uint64_t index
);
void require_epoch_increment_available(
    const TrainableOwnerState& state
);
void require_adam(const rt_adam* adam);

template <typename Handle>
void require_output(Handle** output) {
    if (output == nullptr) {
        throw std::invalid_argument(
            "output handle pointer must not be null"
        );
    }
    *output = nullptr;
}

void copy_tensor_shape(
    const riftco_transformer::Tensor& tensor,
    std::uint64_t* output_dimensions,
    std::uint64_t dimension_capacity
);
void copy_tensor_values(
    const riftco_transformer::Tensor& tensor,
    float* output_values,
    std::uint64_t value_capacity
);
std::vector<float> checked_f32_values(
    const float* values,
    std::uint64_t value_count,
    const char* description
);

}  // namespace riftco_transformer::c_api::detail
