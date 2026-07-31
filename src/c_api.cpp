#include "transformer_lab/c_api.h"

#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/core/autograd.hpp"
#include "transformer_lab/core/tensor.hpp"
#include "transformer_lab/core/tensor_ops.hpp"
#include "transformer_lab/data/tokenizer.hpp"
#include "transformer_lab/model/decoder_only_transformer.hpp"
#include "transformer_lab/model/lora.hpp"
#include "transformer_lab/nn/loss.hpp"
#include "transformer_lab/nn/parameter.hpp"
#include "transformer_lab/optim/adam.hpp"
#include "transformer_lab/stages/serving/kv_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

static_assert(sizeof(tl_transformer_config) == 64);
static_assert(sizeof(tl_lora_config) == 40);
static_assert(sizeof(tl_decode_session_options) == 24);
static_assert(sizeof(tl_adam_options) == 32);
static_assert(sizeof(tl_adam_step_stats) == 32);
static_assert(sizeof(tl_tokenizer_options) == 32);
static_assert(sizeof(tl_bpe_merge_rule) == 12);
static_assert(sizeof(tl_activation_checkpointing_kind) == 4);
static_assert(
    TL_LORA_TARGET_ATTENTION_QUERY ==
    transformer_lab::kLoraAttentionQuery
);
static_assert(
    TL_LORA_TARGET_ATTENTION_KEY ==
    transformer_lab::kLoraAttentionKey
);
static_assert(
    TL_LORA_TARGET_ATTENTION_VALUE ==
    transformer_lab::kLoraAttentionValue
);
static_assert(
    TL_LORA_TARGET_ATTENTION_OUTPUT ==
    transformer_lab::kLoraAttentionOutput
);
static_assert(
    TL_LORA_TARGET_FF_EXPAND ==
    transformer_lab::kLoraFeedForwardExpand
);
static_assert(
    TL_LORA_TARGET_FF_PROJECT ==
    transformer_lab::kLoraFeedForwardProject
);
static_assert(
    TL_LORA_TARGET_LM_HEAD ==
    transformer_lab::kLoraLanguageModelHead
);
static_assert(
    TL_LORA_TARGET_DEFAULT ==
    transformer_lab::kLoraDefaultTargets
);
static_assert(
    TL_LORA_TARGET_ALL_LINEAR ==
    transformer_lab::kLoraAllTargets
);

struct tl_context {
    transformer_lab::ExecutionBackend backend;
};

struct tl_tensor {
    transformer_lab::Tensor value;
};

struct tl_tokenizer {
    explicit tl_tokenizer(
        std::unique_ptr<transformer_lab::TokenizerStrategy> strategy
    )
        : value(std::move(strategy)) {
        if (value == nullptr) {
            throw std::invalid_argument(
                "tokenizer strategy must not be null"
            );
        }
    }

    std::unique_ptr<transformer_lab::TokenizerStrategy> value;
};

struct ModelState {
    ModelState(
        transformer_lab::TransformerDimensions dimensions,
        std::mt19937& random,
        float layer_norm_epsilon
    )
        : value(
              dimensions,
              random,
              layer_norm_epsilon
          ) {}

    transformer_lab::DecoderOnlyTransformer value;
    std::atomic_size_t active_variables{0};
    std::atomic_size_t active_optimizers{0};
    std::atomic_size_t active_lora_parameter_lists{0};
    std::atomic_size_t active_decode_sessions{0};
    std::atomic_uint64_t parameter_epoch{0};
};

struct tl_model {
    std::shared_ptr<ModelState> state;
};

struct tl_decode_session {
    tl_decode_session(
        std::shared_ptr<ModelState> model_owner,
        std::unique_ptr<transformer_lab::DecoderKeyValueCache>
            key_value_cache,
        tl_kv_cache_kind cache_kind,
        std::size_t cache_block_size,
        std::uint64_t epoch
    )
        : owner(std::move(model_owner)),
          cache(std::move(key_value_cache)),
          kind(cache_kind),
          block_size(cache_block_size),
          parameter_epoch(epoch) {
        if (owner == nullptr || cache == nullptr) {
            throw std::invalid_argument(
                "decode session requires a model and cache"
            );
        }
        owner->active_decode_sessions.fetch_add(
            1,
            std::memory_order_relaxed
        );
    }

    ~tl_decode_session() {
        owner->active_decode_sessions.fetch_sub(
            1,
            std::memory_order_relaxed
        );
    }

    tl_decode_session(const tl_decode_session&) = delete;
    tl_decode_session& operator=(const tl_decode_session&) = delete;
    tl_decode_session(tl_decode_session&&) = delete;
    tl_decode_session& operator=(tl_decode_session&&) = delete;

    std::shared_ptr<ModelState> owner;
    std::unique_ptr<transformer_lab::DecoderKeyValueCache> cache;
    tl_kv_cache_kind kind;
    std::size_t block_size;
    std::uint64_t parameter_epoch;
};

struct tl_parameter_list {
    tl_parameter_list(
        std::shared_ptr<ModelState> model_owner,
        transformer_lab::ParameterList parameters,
        bool tracks_lora_parameters
    )
        : owner(std::move(model_owner)),
          value(std::move(parameters)),
          tracks_lora(tracks_lora_parameters) {
        if (tracks_lora) {
            owner->active_lora_parameter_lists.fetch_add(
                1,
                std::memory_order_relaxed
            );
        }
    }

    ~tl_parameter_list() {
        if (tracks_lora) {
            owner->active_lora_parameter_lists.fetch_sub(
                1,
                std::memory_order_relaxed
            );
        }
    }

    tl_parameter_list(const tl_parameter_list&) = delete;
    tl_parameter_list& operator=(const tl_parameter_list&) = delete;
    tl_parameter_list(tl_parameter_list&&) = delete;
    tl_parameter_list& operator=(tl_parameter_list&&) = delete;

    std::shared_ptr<ModelState> owner;
    transformer_lab::ParameterList value;
    bool tracks_lora;
};

struct VariableGraphState {
    explicit VariableGraphState(std::uint64_t epoch)
        : parameter_epoch(epoch) {}

    std::uint64_t parameter_epoch;
    bool backward_consumed = false;
};

struct tl_variable {
    tl_variable(
        std::shared_ptr<ModelState> model_owner,
        std::shared_ptr<VariableGraphState> graph_state,
        transformer_lab::Variable variable
    )
        : owner(std::move(model_owner)),
          graph(std::move(graph_state)),
          value(std::move(variable)) {
        owner->active_variables.fetch_add(
            1,
            std::memory_order_relaxed
        );
    }

    ~tl_variable() {
        owner->active_variables.fetch_sub(
            1,
            std::memory_order_relaxed
        );
    }

    std::shared_ptr<ModelState> owner;
    std::shared_ptr<VariableGraphState> graph;
    transformer_lab::Variable value;
};

struct tl_adam {
    tl_adam(
        std::shared_ptr<ModelState> model_owner,
        transformer_lab::ParameterList parameters,
        transformer_lab::AdamOptions options
    )
        : owner(std::move(model_owner)),
          value(std::move(parameters), options) {
        owner->active_optimizers.fetch_add(
            1,
            std::memory_order_relaxed
        );
    }

    ~tl_adam() {
        owner->active_optimizers.fetch_sub(
            1,
            std::memory_order_relaxed
        );
    }

    std::shared_ptr<ModelState> owner;
    transformer_lab::Adam value;
};

namespace {

constexpr std::size_t kLastErrorCapacity = 1024;
thread_local std::array<char, kLastErrorCapacity> last_error{};

void clear_last_error() noexcept {
    last_error[0] = '\0';
}

void set_last_error(const char* message) noexcept {
    std::size_t index = 0;
    if (message != nullptr) {
        while (index + 1 < last_error.size() &&
               message[index] != '\0') {
            last_error[index] = message[index];
            ++index;
        }
    }
    last_error[index] = '\0';
}

class BackendUnavailable final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template <typename Function>
tl_status guard(Function&& function) noexcept {
    try {
        clear_last_error();
        function();
        return TL_STATUS_OK;
    } catch (const BackendUnavailable& error) {
        set_last_error(error.what());
        return TL_STATUS_BACKEND_UNAVAILABLE;
    } catch (const std::domain_error& error) {
        set_last_error(error.what());
        return TL_STATUS_INVALID_ARGUMENT;
    } catch (const std::invalid_argument& error) {
        set_last_error(error.what());
        return TL_STATUS_INVALID_ARGUMENT;
    } catch (const std::out_of_range& error) {
        set_last_error(error.what());
        return TL_STATUS_OUT_OF_RANGE;
    } catch (const std::overflow_error& error) {
        set_last_error(error.what());
        return TL_STATUS_OVERFLOW;
    } catch (const std::bad_alloc& error) {
        set_last_error(error.what());
        return TL_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return TL_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_last_error("unknown native error");
        return TL_STATUS_UNKNOWN_ERROR;
    }
}

transformer_lab::ExecutionBackend checked_backend(
    tl_backend backend
) {
    transformer_lab::ExecutionBackend result;
    switch (backend) {
        case TL_BACKEND_CPU:
            result = transformer_lab::ExecutionBackend::Cpu;
            break;
        case TL_BACKEND_METAL:
            result = transformer_lab::ExecutionBackend::Metal;
            break;
        default:
            throw std::invalid_argument("unknown C API backend");
    }
    if (!transformer_lab::execution_backend_available(result)) {
        throw BackendUnavailable(
            std::string(
                transformer_lab::execution_backend_name(result)
            ) +
            " execution backend is unavailable"
        );
    }
    return result;
}

tl_backend c_backend(
    transformer_lab::ExecutionBackend backend
) {
    switch (backend) {
        case transformer_lab::ExecutionBackend::Cpu:
            return TL_BACKEND_CPU;
        case transformer_lab::ExecutionBackend::Metal:
            return TL_BACKEND_METAL;
    }
    throw std::invalid_argument("unknown native backend");
}

transformer_lab::FullSequenceAttentionKind
checked_full_sequence_attention(
    tl_full_sequence_attention_kind kind
) {
    switch (kind) {
        case TL_FULL_SEQUENCE_ATTENTION_MATERIALIZED:
            return transformer_lab::FullSequenceAttentionKind::Materialized;
        case TL_FULL_SEQUENCE_ATTENTION_FLASH:
            return transformer_lab::FullSequenceAttentionKind::Flash;
        default:
            throw std::invalid_argument(
                "unknown C API full-sequence attention kind"
            );
    }
}

tl_full_sequence_attention_kind c_full_sequence_attention(
    transformer_lab::FullSequenceAttentionKind kind
) {
    switch (kind) {
        case transformer_lab::FullSequenceAttentionKind::Materialized:
            return TL_FULL_SEQUENCE_ATTENTION_MATERIALIZED;
        case transformer_lab::FullSequenceAttentionKind::Flash:
            return TL_FULL_SEQUENCE_ATTENTION_FLASH;
    }
    throw std::invalid_argument(
        "unknown native full-sequence attention kind"
    );
}

transformer_lab::ActivationCheckpointingKind
checked_activation_checkpointing(
    tl_activation_checkpointing_kind kind
) {
    switch (kind) {
        case TL_ACTIVATION_CHECKPOINTING_DISABLED:
            return transformer_lab::ActivationCheckpointingKind::Disabled;
        case TL_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK:
            return transformer_lab::ActivationCheckpointingKind::
                TransformerBlock;
        default:
            throw std::invalid_argument(
                "unknown C API activation checkpointing kind"
            );
    }
}

tl_activation_checkpointing_kind c_activation_checkpointing(
    transformer_lab::ActivationCheckpointingKind kind
) {
    switch (kind) {
        case transformer_lab::ActivationCheckpointingKind::Disabled:
            return TL_ACTIVATION_CHECKPOINTING_DISABLED;
        case transformer_lab::ActivationCheckpointingKind::
                TransformerBlock:
            return TL_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK;
    }
    throw std::invalid_argument(
        "unknown native activation checkpointing kind"
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

transformer_lab::Tensor::Shape checked_shape(
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

    transformer_lab::Tensor::Shape result;
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

std::vector<transformer_lab::TokenId> checked_token_ids(
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

std::string_view checked_bytes(
    const std::uint8_t* values,
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
    return {
        reinterpret_cast<const char*>(values),
        count,
    };
}

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
    *required_count = checked_u64(
        values.size(),
        description
    );
    const std::size_t capacity =
        checked_size(output_capacity, description);
    if (capacity == 0 && output == nullptr) {
        return;
    }
    if (output == nullptr) {
        throw std::invalid_argument(
            std::string(description) +
            " output must not be null"
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

void require_tokenizer(const tl_tokenizer* tokenizer) {
    if (tokenizer == nullptr || tokenizer->value == nullptr) {
        throw std::invalid_argument(
            "tokenizer handle must not be null"
        );
    }
}

transformer_lab::TokenizerMethod checked_tokenizer_method(
    tl_tokenizer_method method
) {
    switch (method) {
        case TL_TOKENIZER_METHOD_BYTE:
            return transformer_lab::TokenizerMethod::CorpusByte;
        case TL_TOKENIZER_METHOD_BPE:
            return transformer_lab::TokenizerMethod::BytePair;
        default:
            throw std::invalid_argument(
                "unknown C API tokenizer method"
            );
    }
}

tl_tokenizer_method c_tokenizer_method(
    transformer_lab::TokenizerMethod method
) {
    switch (method) {
        case transformer_lab::TokenizerMethod::CorpusByte:
            return TL_TOKENIZER_METHOD_BYTE;
        case transformer_lab::TokenizerMethod::BytePair:
            return TL_TOKENIZER_METHOD_BPE;
    }
    throw std::invalid_argument("unknown native tokenizer method");
}

transformer_lab::TokenizerOptions checked_tokenizer_options(
    const tl_tokenizer_options* options
) {
    if (options == nullptr) {
        return {};
    }
    constexpr std::size_t minimum_size =
        offsetof(tl_tokenizer_options, minimum_pair_frequency) +
        sizeof(std::uint64_t);
    checked_structure_size(
        options->struct_size,
        minimum_size,
        "tokenizer options structure"
    );
    if (options->reserved != 0) {
        throw std::invalid_argument(
            "tokenizer options reserved field must be zero"
        );
    }

    const auto method = checked_tokenizer_method(options->method);
    if (method == transformer_lab::TokenizerMethod::CorpusByte) {
        return {
            method,
            transformer_lab::TokenizerOptions{}.vocabulary_size,
            transformer_lab::TokenizerOptions{}.minimum_pair_frequency,
        };
    }

    if (options->vocabulary_size <
        static_cast<std::uint64_t>(256)) {
        throw std::invalid_argument(
            "BPE tokenizer vocabulary size must be at least 256"
        );
    }
    if (options->vocabulary_size >
        static_cast<std::uint64_t>(
            std::numeric_limits<transformer_lab::TokenId>::max()
        )) {
        throw std::overflow_error(
            "tokenizer vocabulary size exceeds token ID range"
        );
    }
    if (options->minimum_pair_frequency == 0) {
        throw std::invalid_argument(
            "tokenizer minimum pair frequency must be positive"
        );
    }

    return {
        method,
        checked_size(
            options->vocabulary_size,
            "tokenizer vocabulary size"
        ),
        checked_size(
            options->minimum_pair_frequency,
            "tokenizer minimum pair frequency"
        ),
    };
}

struct CheckedDecodeSessionOptions {
    tl_kv_cache_kind kind;
    std::size_t block_size;
};

CheckedDecodeSessionOptions checked_decode_session_options(
    const tl_decode_session_options* options
) {
    if (options == nullptr) {
        return {
            TL_KV_CACHE_PAGED,
            16,
        };
    }
    constexpr std::size_t minimum_size =
        offsetof(tl_decode_session_options, block_size) +
        sizeof(std::uint64_t);
    checked_structure_size(
        options->struct_size,
        minimum_size,
        "decode-session options structure"
    );
    if (options->reserved != 0) {
        throw std::invalid_argument(
            "decode-session options reserved field must be zero"
        );
    }
    switch (options->kind) {
        case TL_KV_CACHE_CONTIGUOUS:
        case TL_KV_CACHE_PAGED:
            break;
        default:
            throw std::invalid_argument(
                "unknown C API KV-cache kind"
            );
    }
    const std::size_t block_size = checked_size(
        options->block_size,
        "decode-session block size"
    );
    if (block_size == 0) {
        throw std::invalid_argument(
            "decode-session block size must be positive"
        );
    }
    return {
        options->kind,
        block_size,
    };
}

void require_context(const tl_context* context) {
    if (context == nullptr) {
        throw std::invalid_argument(
            "context handle must not be null"
        );
    }
}

void require_tensor(const tl_tensor* tensor) {
    if (tensor == nullptr) {
        throw std::invalid_argument(
            "tensor handle must not be null"
        );
    }
}

void require_model(const tl_model* model) {
    if (model == nullptr || model->state == nullptr) {
        throw std::invalid_argument(
            "model handle must not be null"
        );
    }
}

void require_decode_session(const tl_decode_session* session) {
    if (session == nullptr ||
        session->owner == nullptr ||
        session->cache == nullptr) {
        throw std::invalid_argument(
            "decode-session handle must not be null"
        );
    }
}

void require_current_decode_session(
    const tl_decode_session* session
) {
    require_decode_session(session);
    if (session->parameter_epoch !=
        session->owner->parameter_epoch.load(
            std::memory_order_relaxed
        )) {
        throw std::logic_error(
            "decode session cannot use stale model parameters"
        );
    }
}

void require_no_active_decode_sessions(
    const ModelState& state,
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
    const tl_parameter_list* parameters
) {
    if (parameters == nullptr ||
        parameters->owner == nullptr) {
        throw std::invalid_argument(
            "parameter-list handle must not be null"
        );
    }
}

const transformer_lab::NamedParameter& checked_parameter(
    const tl_parameter_list* parameters,
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

void require_variable(const tl_variable* variable) {
    if (variable == nullptr ||
        variable->owner == nullptr ||
        variable->graph == nullptr) {
        throw std::invalid_argument(
            "variable handle must not be null"
        );
    }
}

void require_current_graph(
    const tl_variable* variable,
    const char* operation
) {
    if (variable->graph->backward_consumed) {
        throw std::logic_error(
            std::string(operation) +
            " cannot use a graph after backward"
        );
    }
    if (variable->graph->parameter_epoch !=
        variable->owner->parameter_epoch.load(
            std::memory_order_relaxed
        )) {
        throw std::logic_error(
            std::string(operation) +
            " cannot use a graph after an optimizer step"
        );
    }
}

void require_epoch_increment_available(
    const ModelState& state
) {
    if (state.parameter_epoch.load(
            std::memory_order_relaxed
        ) == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error(
            "model parameter epoch overflow"
        );
    }
}

void require_adam(const tl_adam* adam) {
    if (adam == nullptr || adam->owner == nullptr) {
        throw std::invalid_argument(
            "Adam handle must not be null"
        );
    }
}

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
    const transformer_lab::Tensor& tensor,
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
    const transformer_lab::Tensor& tensor,
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

transformer_lab::ExecutionBackend model_backend(
    ModelState& state
) {
    const auto parameters = state.value.parameters();
    if (parameters.empty() ||
        parameters.front().parameter == nullptr) {
        throw std::logic_error(
            "transformer model has no registered parameters"
        );
    }
    return parameters.front().parameter->value().backend();
}

transformer_lab::AdamOptions checked_adam_options(
    const tl_adam_options* options
) {
    if (options == nullptr) {
        return {};
    }
    constexpr std::size_t minimum_size =
        offsetof(tl_adam_options, reserved) +
        sizeof(std::uint32_t);
    checked_structure_size(
        options->struct_size,
        minimum_size,
        "Adam options structure"
    );
    if (options->reserved != 0) {
        throw std::invalid_argument(
            "Adam options reserved field must be zero"
        );
    }
    return {
        options->learning_rate,
        options->beta1,
        options->beta2,
        options->epsilon,
        options->maximum_gradient_norm,
    };
}

transformer_lab::LoraConfig checked_lora_config(
    const tl_lora_config* config
) {
    if (config == nullptr) {
        return {};
    }
    constexpr std::size_t minimum_size =
        offsetof(tl_lora_config, reserved) +
        sizeof(std::uint64_t);
    checked_structure_size(
        config->struct_size,
        minimum_size,
        "LoRA config structure"
    );
    if (config->reserved != 0) {
        throw std::invalid_argument(
            "LoRA config reserved field must be zero"
        );
    }
    if (config->rank == 0) {
        throw std::invalid_argument(
            "LoRA rank must be greater than zero"
        );
    }
    if (!std::isfinite(config->alpha) ||
        config->alpha <= 0.0F) {
        throw std::invalid_argument(
            "LoRA alpha must be finite and positive"
        );
    }
    if (config->targets == 0) {
        throw std::invalid_argument(
            "LoRA targets must not be empty"
        );
    }
    if ((config->targets & ~TL_LORA_TARGET_ALL_LINEAR) != 0) {
        throw std::invalid_argument(
            "LoRA targets contain an unknown bit"
        );
    }
    return {
        checked_size(config->rank, "LoRA rank"),
        config->alpha,
        config->random_seed,
        static_cast<transformer_lab::LoraTargetMask>(
            config->targets
        ),
    };
}

void write_lora_config(
    const transformer_lab::LoraConfig& config,
    tl_lora_config* output
) {
    if (output == nullptr) {
        throw std::invalid_argument(
            "LoRA config output must not be null"
        );
    }
    constexpr std::size_t minimum_size =
        offsetof(tl_lora_config, reserved) +
        sizeof(std::uint64_t);
    checked_structure_size(
        output->struct_size,
        minimum_size,
        "LoRA config structure"
    );
    const std::uint64_t structure_size = output->struct_size;
    *output = {
        structure_size,
        checked_u64(config.rank, "LoRA rank"),
        config.alpha,
        config.random_seed,
        static_cast<tl_lora_target_mask>(config.targets),
        0,
    };
}

}  // namespace

extern "C" {

uint32_t TL_CALL tl_abi_version(void) {
    return TL_ABI_VERSION;
}

const char* TL_CALL tl_status_string(tl_status status) {
    switch (status) {
        case TL_STATUS_OK:
            return "ok";
        case TL_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case TL_STATUS_OUT_OF_RANGE:
            return "out of range";
        case TL_STATUS_OVERFLOW:
            return "overflow";
        case TL_STATUS_BACKEND_UNAVAILABLE:
            return "backend unavailable";
        case TL_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case TL_STATUS_RUNTIME_ERROR:
            return "runtime error";
        case TL_STATUS_UNKNOWN_ERROR:
            return "unknown error";
    }
    return "unrecognized status";
}

const char* TL_CALL tl_last_error(void) {
    return last_error.data();
}

tl_status TL_CALL tl_tokenizer_options_init(
    tl_tokenizer_options* options,
    uint64_t options_size
) {
    return guard([&] {
        if (options == nullptr) {
            throw std::invalid_argument(
                "tokenizer options must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(
                tl_tokenizer_options,
                minimum_pair_frequency
            ) +
            sizeof(std::uint64_t);
        checked_structure_size(
            options_size,
            minimum_size,
            "tokenizer options structure"
        );
        *options = {
            options_size,
            TL_TOKENIZER_METHOD_BYTE,
            0,
            512,
            2,
        };
    });
}

tl_status TL_CALL tl_tokenizer_create(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    tl_tokenizer** output
) {
    return tl_tokenizer_create_with_options(
        corpus_bytes,
        corpus_size,
        nullptr,
        output
    );
}

tl_status TL_CALL tl_tokenizer_create_with_options(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    const tl_tokenizer_options* options,
    tl_tokenizer** output
) {
    return guard([&] {
        require_output(output);
        auto result = std::make_unique<tl_tokenizer>(
            transformer_lab::make_tokenizer(
                checked_bytes(
                    corpus_bytes,
                    corpus_size,
                    "tokenizer corpus"
                ),
                checked_tokenizer_options(options)
            )
        );
        *output = result.release();
    });
}

tl_status TL_CALL tl_tokenizer_create_from_byte_vocabulary(
    const uint8_t* ordered_vocabulary,
    uint64_t vocabulary_size,
    tl_tokenizer** output
) {
    return guard([&] {
        require_output(output);
        const std::size_t count = checked_size(
            vocabulary_size,
            "tokenizer byte-vocabulary size"
        );
        if (count != 0 && ordered_vocabulary == nullptr) {
            throw std::invalid_argument(
                "tokenizer byte vocabulary must not be null"
            );
        }
        auto result = std::make_unique<tl_tokenizer>(
            std::make_unique<transformer_lab::ByteTokenizer>(
                std::span<const std::uint8_t>(
                    ordered_vocabulary,
                    count
                )
            )
        );
        *output = result.release();
    });
}

tl_status TL_CALL tl_tokenizer_create_from_bpe_merges(
    const tl_bpe_merge_rule* ordered_merge_rules,
    uint64_t merge_count,
    tl_tokenizer** output
) {
    return guard([&] {
        require_output(output);
        const std::size_t count = checked_size(
            merge_count,
            "BPE merge-rule count"
        );
        if (count != 0 && ordered_merge_rules == nullptr) {
            throw std::invalid_argument(
                "BPE merge rules must not be null"
            );
        }
        std::vector<transformer_lab::BpeMergeRule> rules;
        rules.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto& rule = ordered_merge_rules[index];
            rules.push_back({
                rule.left,
                rule.right,
                rule.result,
            });
        }
        auto result = std::make_unique<tl_tokenizer>(
            std::make_unique<transformer_lab::BytePairTokenizer>(
                std::span<const transformer_lab::BpeMergeRule>(
                    rules.data(),
                    rules.size()
                )
            )
        );
        *output = result.release();
    });
}

void TL_CALL tl_tokenizer_release(tl_tokenizer* tokenizer) {
    delete tokenizer;
}

tl_status TL_CALL tl_tokenizer_get_method(
    const tl_tokenizer* tokenizer,
    tl_tokenizer_method* output
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (output == nullptr) {
            throw std::invalid_argument(
                "tokenizer method output must not be null"
            );
        }
        *output = c_tokenizer_method(
            tokenizer->value->method()
        );
    });
}

tl_status TL_CALL tl_tokenizer_vocabulary_size(
    const tl_tokenizer* tokenizer,
    uint64_t* output
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (output == nullptr) {
            throw std::invalid_argument(
                "tokenizer vocabulary-size output must not be null"
            );
        }
        *output = checked_u64(
            tokenizer->value->vocab_size(),
            "tokenizer vocabulary size"
        );
    });
}

tl_status TL_CALL tl_tokenizer_bpe_merge_count(
    const tl_tokenizer* tokenizer,
    uint64_t* output
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (output == nullptr) {
            throw std::invalid_argument(
                "BPE merge-count output must not be null"
            );
        }
        const auto* byte_pair =
            dynamic_cast<const transformer_lab::BytePairTokenizer*>(
                tokenizer->value.get()
            );
        if (byte_pair == nullptr) {
            throw std::invalid_argument(
                "BPE merge rules require a BPE tokenizer"
            );
        }
        *output = checked_u64(
            byte_pair->merge_rules().size(),
            "BPE merge-rule count"
        );
    });
}

tl_status TL_CALL tl_tokenizer_bpe_merge_rule(
    const tl_tokenizer* tokenizer,
    uint64_t index,
    tl_bpe_merge_rule* output
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (output == nullptr) {
            throw std::invalid_argument(
                "BPE merge-rule output must not be null"
            );
        }
        const auto* byte_pair =
            dynamic_cast<const transformer_lab::BytePairTokenizer*>(
                tokenizer->value.get()
            );
        if (byte_pair == nullptr) {
            throw std::invalid_argument(
                "BPE merge rules require a BPE tokenizer"
            );
        }
        const std::size_t native_index = checked_size(
            index,
            "BPE merge-rule index"
        );
        const auto rules = byte_pair->merge_rules();
        if (native_index >= rules.size()) {
            throw std::out_of_range(
                "BPE merge-rule index is outside the tokenizer"
            );
        }
        const auto& rule = rules[native_index];
        *output = {
            rule.left,
            rule.right,
            rule.result,
        };
    });
}

tl_status TL_CALL tl_tokenizer_vocabulary(
    const tl_tokenizer* tokenizer,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (tokenizer->value->method() !=
            transformer_lab::TokenizerMethod::CorpusByte) {
            throw std::invalid_argument(
                "tokenizer vocabulary is only available for "
                "the corpus-byte method; use "
                "tl_tokenizer_token_bytes for BPE"
            );
        }

        std::vector<std::uint8_t> vocabulary;
        vocabulary.reserve(tokenizer->value->vocab_size());
        for (std::size_t index = 0;
             index < tokenizer->value->vocab_size();
             ++index) {
            const auto bytes = tokenizer->value->token_bytes(
                static_cast<transformer_lab::TokenId>(index)
            );
            if (bytes.size() != 1) {
                throw std::logic_error(
                    "corpus-byte tokenizer produced a non-byte token"
                );
            }
            vocabulary.push_back(bytes.front());
        }
        copy_sized_output(
            std::span<const std::uint8_t>(
                vocabulary.data(),
                vocabulary.size()
            ),
            output,
            capacity,
            required_count,
            "tokenizer vocabulary"
        );
    });
}

tl_status TL_CALL tl_tokenizer_token_bytes(
    const tl_tokenizer* tokenizer,
    uint32_t token_id,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        const auto bytes = tokenizer->value->token_bytes(token_id);
        copy_sized_output(
            bytes,
            output,
            capacity,
            required_count,
            "tokenizer token bytes"
        );
    });
}

tl_status TL_CALL tl_tokenizer_encode(
    const tl_tokenizer* tokenizer,
    const uint8_t* text,
    uint64_t text_size,
    uint32_t* output,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        const auto tokens = tokenizer->value->encode(
            checked_bytes(
                text,
                text_size,
                "tokenizer text"
            )
        );
        copy_sized_output(
            std::span<const transformer_lab::TokenId>(
                tokens.data(),
                tokens.size()
            ),
            output,
            capacity,
            required_count,
            "encoded token"
        );
    });
}

tl_status TL_CALL tl_tokenizer_decode(
    const tl_tokenizer* tokenizer,
    const uint32_t* tokens,
    uint64_t token_count,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        const std::string text = tokenizer->value->decode(
            checked_token_ids(
                tokens,
                token_count,
                "encoded token"
            )
        );
        copy_sized_output(
            std::span<const char>(
                text.data(),
                text.size()
            ),
            output,
            capacity,
            required_count,
            "decoded byte"
        );
    });
}

tl_status TL_CALL tl_backend_is_available(
    tl_backend backend,
    int32_t* available
) {
    return guard([&] {
        if (available == nullptr) {
            throw std::invalid_argument(
                "availability output must not be null"
            );
        }
        transformer_lab::ExecutionBackend native;
        switch (backend) {
            case TL_BACKEND_CPU:
                native =
                    transformer_lab::ExecutionBackend::Cpu;
                break;
            case TL_BACKEND_METAL:
                native =
                    transformer_lab::ExecutionBackend::Metal;
                break;
            default:
                throw std::invalid_argument(
                    "unknown C API backend"
                );
        }
        *available =
            transformer_lab::execution_backend_available(native)
                ? 1
                : 0;
    });
}

tl_status TL_CALL tl_context_create(
    tl_backend backend,
    tl_context** output
) {
    return guard([&] {
        require_output(output);
        const auto native = checked_backend(backend);
        auto result = std::make_unique<tl_context>(
            tl_context{native}
        );
        *output = result.release();
    });
}

void TL_CALL tl_context_release(tl_context* context) {
    delete context;
}

tl_status TL_CALL tl_context_backend(
    const tl_context* context,
    tl_backend* output
) {
    return guard([&] {
        require_context(context);
        if (output == nullptr) {
            throw std::invalid_argument(
                "backend output must not be null"
            );
        }
        *output = c_backend(context->backend);
    });
}

tl_status TL_CALL tl_tensor_create_f32(
    const tl_context* context,
    const uint64_t* shape,
    uint64_t rank,
    const float* values,
    uint64_t value_count,
    tl_tensor** output
) {
    return guard([&] {
        require_output(output);
        require_context(context);
        const std::size_t native_count =
            checked_size(value_count, "tensor value count");
        if (native_count != 0 && values == nullptr) {
            throw std::invalid_argument(
                "tensor values pointer must not be null"
            );
        }

        std::vector<float> owned_values;
        owned_values.reserve(native_count);
        if (native_count != 0) {
            owned_values.assign(
                values,
                values + native_count
            );
        }
        auto result = std::make_unique<tl_tensor>(
            tl_tensor{
                transformer_lab::Tensor(
                    checked_shape(shape, rank),
                    std::move(owned_values),
                    context->backend
                ),
            }
        );
        *output = result.release();
    });
}

tl_status TL_CALL tl_tensor_zeros_f32(
    const tl_context* context,
    const uint64_t* shape,
    uint64_t rank,
    tl_tensor** output
) {
    return guard([&] {
        require_output(output);
        require_context(context);
        auto result = std::make_unique<tl_tensor>(
            tl_tensor{
                transformer_lab::Tensor::zeros(
                    checked_shape(shape, rank),
                    context->backend
                ),
            }
        );
        *output = result.release();
    });
}

void TL_CALL tl_tensor_release(tl_tensor* tensor) {
    delete tensor;
}

tl_status TL_CALL tl_tensor_backend(
    const tl_tensor* tensor,
    tl_backend* output
) {
    return guard([&] {
        require_tensor(tensor);
        if (output == nullptr) {
            throw std::invalid_argument(
                "backend output must not be null"
            );
        }
        *output = c_backend(tensor->value.backend());
    });
}

tl_status TL_CALL tl_tensor_rank(
    const tl_tensor* tensor,
    uint64_t* output
) {
    return guard([&] {
        require_tensor(tensor);
        if (output == nullptr) {
            throw std::invalid_argument(
                "rank output must not be null"
            );
        }
        *output = checked_u64(
            tensor->value.rank(),
            "tensor rank"
        );
    });
}

tl_status TL_CALL tl_tensor_shape(
    const tl_tensor* tensor,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
) {
    return guard([&] {
        require_tensor(tensor);
        copy_tensor_shape(
            tensor->value,
            output_dimensions,
            dimension_capacity
        );
    });
}

tl_status TL_CALL tl_tensor_numel(
    const tl_tensor* tensor,
    uint64_t* output
) {
    return guard([&] {
        require_tensor(tensor);
        if (output == nullptr) {
            throw std::invalid_argument(
                "element-count output must not be null"
            );
        }
        *output = checked_u64(
            tensor->value.numel(),
            "tensor element count"
        );
    });
}

tl_status TL_CALL tl_tensor_copy_to_host_f32(
    const tl_tensor* tensor,
    float* output_values,
    uint64_t value_capacity
) {
    return guard([&] {
        require_tensor(tensor);
        copy_tensor_values(
            tensor->value,
            output_values,
            value_capacity
        );
    });
}

tl_status TL_CALL tl_tensor_matmul(
    const tl_tensor* left,
    const tl_tensor* right,
    tl_tensor** output
) {
    return guard([&] {
        require_output(output);
        require_tensor(left);
        require_tensor(right);
        if (left->value.backend() != right->value.backend()) {
            throw std::invalid_argument(
                "matmul tensors must use the same backend"
            );
        }
        auto result = std::make_unique<tl_tensor>(
            tl_tensor{
                transformer_lab::tensor_ops::matmul(
                    left->value,
                    right->value
                ),
            }
        );
        *output = result.release();
    });
}

tl_status TL_CALL tl_transformer_config_init(
    tl_transformer_config* config,
    uint64_t config_size
) {
    return guard([&] {
        if (config == nullptr) {
            throw std::invalid_argument(
                "transformer config must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(
                tl_transformer_config,
                layer_norm_epsilon
            ) +
            sizeof(float);
        checked_structure_size(
            config_size,
            minimum_size,
            "transformer config structure"
        );
        *config = {
            config_size,
            0,
            0,
            0,
            0,
            0,
            0,
            5489U,
            1.0e-5F,
        };
    });
}

tl_status TL_CALL tl_decode_session_options_init(
    tl_decode_session_options* options,
    uint64_t options_size
) {
    return guard([&] {
        if (options == nullptr) {
            throw std::invalid_argument(
                "decode-session options must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(tl_decode_session_options, block_size) +
            sizeof(std::uint64_t);
        checked_structure_size(
            options_size,
            minimum_size,
            "decode-session options structure"
        );
        *options = {
            options_size,
            TL_KV_CACHE_PAGED,
            0,
            16,
        };
    });
}

tl_status TL_CALL tl_model_create(
    const tl_transformer_config* config,
    tl_model** output
) {
    return guard([&] {
        require_output(output);
        if (config == nullptr) {
            throw std::invalid_argument(
                "transformer config must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(
                tl_transformer_config,
                layer_norm_epsilon
            ) +
            sizeof(float);
        checked_structure_size(
            config->struct_size,
            minimum_size,
            "transformer config structure"
        );

        const transformer_lab::TransformerDimensions dimensions{
            checked_size(
                config->vocabulary_size,
                "vocabulary size"
            ),
            checked_size(
                config->maximum_context,
                "maximum context"
            ),
            checked_size(config->model_width, "model width"),
            checked_size(config->head_count, "head count"),
            checked_size(config->block_count, "block count"),
            checked_size(
                config->feed_forward_width,
                "feed-forward width"
            ),
        };
        std::mt19937 random(config->random_seed);
        const transformer_lab::ScopedExecutionBackend
            construction_backend(
                transformer_lab::ExecutionBackend::Cpu
            );
        auto state = std::make_shared<ModelState>(
            dimensions,
            random,
            config->layer_norm_epsilon
        );
        auto result = std::make_unique<tl_model>(
            tl_model{std::move(state)}
        );
        *output = result.release();
    });
}

void TL_CALL tl_model_release(tl_model* model) {
    delete model;
}

tl_status TL_CALL tl_model_to(
    tl_model* model,
    tl_backend backend
) {
    return guard([&] {
        require_model(model);
        if (model->state->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot move a model while variable graphs are alive"
            );
        }
        if (model->state->active_optimizers.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot move a model while optimizers are alive"
            );
        }
        require_no_active_decode_sessions(
            *model->state,
            "move a model"
        );
        require_epoch_increment_available(*model->state);
        model->state->value.to(checked_backend(backend));
        model->state->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
    });
}

tl_status TL_CALL tl_model_backend(
    const tl_model* model,
    tl_backend* output
) {
    return guard([&] {
        require_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "backend output must not be null"
            );
        }
        *output = c_backend(model_backend(*model->state));
    });
}

tl_status TL_CALL tl_model_set_full_sequence_attention(
    tl_model* model,
    tl_full_sequence_attention_kind kind
) {
    return guard([&] {
        require_model(model);
        model->state->value.set_full_sequence_attention_kind(
            checked_full_sequence_attention(kind)
        );
    });
}

tl_status TL_CALL tl_model_full_sequence_attention(
    const tl_model* model,
    tl_full_sequence_attention_kind* output
) {
    return guard([&] {
        require_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "full-sequence attention output must not be null"
            );
        }
        *output = c_full_sequence_attention(
            model->state->value.full_sequence_attention_kind()
        );
    });
}

tl_status TL_CALL tl_model_set_activation_checkpointing(
    tl_model* model,
    tl_activation_checkpointing_kind kind
) {
    return guard([&] {
        require_model(model);
        model->state->value.set_activation_checkpointing_kind(
            checked_activation_checkpointing(kind)
        );
    });
}

tl_status TL_CALL tl_model_activation_checkpointing(
    const tl_model* model,
    tl_activation_checkpointing_kind* output
) {
    return guard([&] {
        require_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "activation checkpointing output must not be null"
            );
        }
        *output = c_activation_checkpointing(
            model->state->value.activation_checkpointing_kind()
        );
    });
}

tl_status TL_CALL tl_lora_config_init(
    tl_lora_config* config,
    uint64_t config_size
) {
    return guard([&] {
        if (config == nullptr) {
            throw std::invalid_argument(
                "LoRA config must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(tl_lora_config, reserved) +
            sizeof(std::uint64_t);
        checked_structure_size(
            config_size,
            minimum_size,
            "LoRA config structure"
        );
        const transformer_lab::LoraConfig defaults;
        *config = {
            config_size,
            checked_u64(defaults.rank, "LoRA rank"),
            defaults.alpha,
            defaults.random_seed,
            static_cast<tl_lora_target_mask>(defaults.targets),
            0,
        };
    });
}

tl_status TL_CALL tl_model_attach_lora(
    tl_model* model,
    const tl_lora_config* config
) {
    return guard([&] {
        require_model(model);
        if (model->state->value.has_lora()) {
            throw std::invalid_argument(
                "model already has an attached LoRA adapter"
            );
        }
        if (model->state->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot attach LoRA while variable graphs are alive"
            );
        }
        if (model->state->active_optimizers.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot attach LoRA while optimizers are alive"
            );
        }
        require_no_active_decode_sessions(
            *model->state,
            "attach LoRA"
        );
        require_epoch_increment_available(*model->state);
        model->state->value.attach_lora(
            checked_lora_config(config)
        );
        model->state->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
    });
}

tl_status TL_CALL tl_model_has_lora(
    const tl_model* model,
    int32_t* output
) {
    return guard([&] {
        require_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "LoRA state output must not be null"
            );
        }
        *output = model->state->value.has_lora() ? 1 : 0;
    });
}

tl_status TL_CALL tl_model_lora_config(
    const tl_model* model,
    tl_lora_config* output
) {
    return guard([&] {
        require_model(model);
        if (!model->state->value.has_lora()) {
            throw std::invalid_argument(
                "model has no attached LoRA adapter"
            );
        }
        write_lora_config(
            model->state->value.lora_config(),
            output
        );
    });
}

tl_status TL_CALL tl_model_forward(
    const tl_model* model,
    const uint32_t* token_ids,
    uint64_t token_count,
    uint64_t batch_size,
    uint64_t sequence_length,
    tl_variable** output
) {
    return guard([&] {
        require_output(output);
        require_model(model);
        const std::size_t native_batch =
            checked_size(batch_size, "batch size");
        const std::size_t native_sequence =
            checked_size(sequence_length, "sequence length");
        const std::size_t expected_count = checked_product(
            native_batch,
            native_sequence,
            "token shape"
        );
        const auto values = checked_token_ids(
            token_ids,
            token_count,
            "token IDs"
        );
        if (values.size() != expected_count) {
            throw std::invalid_argument(
                "token count must match batch and sequence sizes"
            );
        }
        auto graph = std::make_shared<VariableGraphState>(
            model->state->parameter_epoch.load(
                std::memory_order_relaxed
            )
        );
        auto result = std::make_unique<tl_variable>(
            model->state,
            std::move(graph),
            model->state->value.forward(
                values,
                {native_batch, native_sequence}
            )
        );
        *output = result.release();
    });
}

tl_status TL_CALL tl_model_decode_session_create(
    const tl_model* model,
    const tl_decode_session_options* options,
    tl_decode_session** output
) {
    return guard([&] {
        require_output(output);
        require_model(model);
        const auto configured =
            checked_decode_session_options(options);
        const auto& dimensions =
            model->state->value.dimensions();
        const auto backend = model->state->value.backend();

        std::unique_ptr<transformer_lab::DecoderKeyValueCache>
            cache;
        std::size_t block_size = 0;
        if (configured.kind == TL_KV_CACHE_CONTIGUOUS) {
            const transformer_lab::stages::serving::
                ContiguousKvCacheFactory factory(
                    dimensions,
                    backend
                );
            cache = factory.create();
            block_size = dimensions.maximum_context;
        } else {
            const transformer_lab::stages::serving::
                PagedKvCachePool pool(
                    dimensions,
                    backend,
                    configured.block_size
                );
            cache = pool.create();
            block_size = pool.block_size();
        }

        auto result = std::make_unique<tl_decode_session>(
            model->state,
            std::move(cache),
            configured.kind,
            block_size,
            model->state->parameter_epoch.load(
                std::memory_order_relaxed
            )
        );
        *output = result.release();
    });
}

void TL_CALL tl_decode_session_release(
    tl_decode_session* session
) {
    delete session;
}

tl_status TL_CALL tl_decode_session_reset(
    tl_decode_session* session
) {
    return guard([&] {
        require_current_decode_session(session);
        session->cache->reset();
    });
}

tl_status TL_CALL tl_decode_session_size(
    const tl_decode_session* session,
    uint64_t* output
) {
    return guard([&] {
        require_decode_session(session);
        if (output == nullptr) {
            throw std::invalid_argument(
                "decode-session size output must not be null"
            );
        }
        *output = checked_u64(
            session->cache->size(),
            "decode-session size"
        );
    });
}

tl_status TL_CALL tl_decode_session_capacity(
    const tl_decode_session* session,
    uint64_t* output
) {
    return guard([&] {
        require_decode_session(session);
        if (output == nullptr) {
            throw std::invalid_argument(
                "decode-session capacity output must not be null"
            );
        }
        *output = checked_u64(
            session->cache->capacity(),
            "decode-session capacity"
        );
    });
}

tl_status TL_CALL tl_decode_session_cache_kind(
    const tl_decode_session* session,
    tl_kv_cache_kind* output
) {
    return guard([&] {
        require_decode_session(session);
        if (output == nullptr) {
            throw std::invalid_argument(
                "decode-session cache-kind output must not be null"
            );
        }
        *output = session->kind;
    });
}

tl_status TL_CALL tl_decode_session_block_size(
    const tl_decode_session* session,
    uint64_t* output
) {
    return guard([&] {
        require_decode_session(session);
        if (output == nullptr) {
            throw std::invalid_argument(
                "decode-session block-size output must not be null"
            );
        }
        *output = checked_u64(
            session->block_size,
            "decode-session block size"
        );
    });
}

tl_status TL_CALL tl_decode_session_step(
    tl_decode_session* session,
    uint32_t token_id,
    float* output_logits,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_decode_session(session);
        if (required_count == nullptr) {
            throw std::invalid_argument(
                "decode-session required-count output must not be null"
            );
        }
        const std::size_t vocabulary_size =
            session->owner->value.dimensions().vocabulary_size;
        *required_count = checked_u64(
            vocabulary_size,
            "decode-session logits"
        );
        require_current_decode_session(session);
        if (static_cast<std::size_t>(token_id) >=
            vocabulary_size) {
            throw std::out_of_range(
                "decode-session token is outside the model vocabulary"
            );
        }
        const std::size_t output_capacity = checked_size(
            capacity,
            "decode-session logits capacity"
        );
        if (output_logits == nullptr) {
            throw std::invalid_argument(
                "decode-session logits output must not be null"
            );
        }
        if (output_capacity < vocabulary_size) {
            throw std::out_of_range(
                "decode-session logits output capacity is too small"
            );
        }
        if (session->cache->size() >=
            session->cache->capacity()) {
            throw std::out_of_range(
                "decode-session cache capacity is exhausted"
            );
        }

        const transformer_lab::Tensor logits =
            session->owner->value.decode_token(
                token_id,
                *session->cache
            );
        const transformer_lab::Tensor::Shape expected_shape{
            1,
            1,
            vocabulary_size,
        };
        if (logits.shape() != expected_shape ||
            logits.numel() != vocabulary_size) {
            throw std::logic_error(
                "decode session returned an unexpected logit shape"
            );
        }
        std::copy(
            logits.data().begin(),
            logits.data().end(),
            output_logits
        );
    });
}

tl_status TL_CALL tl_model_parameters(
    tl_model* model,
    tl_parameter_list** output
) {
    return guard([&] {
        require_output(output);
        require_model(model);
        auto result = std::make_unique<tl_parameter_list>(
            model->state,
            model->state->value.parameters(),
            false
        );
        *output = result.release();
    });
}

tl_status TL_CALL tl_model_lora_parameters(
    tl_model* model,
    tl_parameter_list** output
) {
    return guard([&] {
        require_output(output);
        require_model(model);
        if (!model->state->value.has_lora()) {
            throw std::invalid_argument(
                "model has no attached LoRA adapter"
            );
        }
        auto result = std::make_unique<tl_parameter_list>(
            model->state,
            model->state->value.lora_parameters(),
            true
        );
        *output = result.release();
    });
}

tl_status TL_CALL tl_model_merge_lora(tl_model* model) {
    return guard([&] {
        require_model(model);
        if (!model->state->value.has_lora()) {
            throw std::invalid_argument(
                "model has no attached LoRA adapter"
            );
        }
        if (model->state->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot merge LoRA while variable graphs are alive"
            );
        }
        if (model->state->active_optimizers.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot merge LoRA while optimizers are alive"
            );
        }
        if (model->state->active_lora_parameter_lists.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot merge LoRA while adapter parameter lists "
                "are alive"
            );
        }
        require_no_active_decode_sessions(
            *model->state,
            "merge LoRA"
        );
        require_epoch_increment_available(*model->state);
        model->state->value.merge_lora();
        model->state->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
    });
}

void TL_CALL tl_parameter_list_release(
    tl_parameter_list* parameters
) {
    delete parameters;
}

tl_status TL_CALL tl_parameter_list_count(
    const tl_parameter_list* parameters,
    uint64_t* output
) {
    return guard([&] {
        require_parameter_list(parameters);
        if (output == nullptr) {
            throw std::invalid_argument(
                "parameter-count output must not be null"
            );
        }
        *output = checked_u64(
            parameters->value.size(),
            "parameter count"
        );
    });
}

tl_status TL_CALL tl_parameter_list_backend(
    const tl_parameter_list* parameters,
    tl_backend* output
) {
    return guard([&] {
        require_parameter_list(parameters);
        if (output == nullptr) {
            throw std::invalid_argument(
                "backend output must not be null"
            );
        }
        if (parameters->value.empty() ||
            parameters->value.front().parameter == nullptr) {
            throw std::logic_error(
                "parameter list must not be empty"
            );
        }
        *output = c_backend(
            parameters->value.front()
                .parameter->value().backend()
        );
    });
}

tl_status TL_CALL tl_parameter_list_name(
    const tl_parameter_list* parameters,
    uint64_t index,
    char* output_name,
    uint64_t name_capacity,
    uint64_t* required_capacity
) {
    return guard([&] {
        require_parameter_list(parameters);
        if (required_capacity == nullptr) {
            throw std::invalid_argument(
                "required name capacity output must not be null"
            );
        }
        const std::size_t native_index =
            checked_size(index, "parameter index");
        if (native_index >= parameters->value.size()) {
            throw std::out_of_range(
                "parameter index is outside the list"
            );
        }
        const std::string& name =
            parameters->value[native_index].name;
        if (name.size() ==
            std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error(
                "parameter name capacity overflows size_t"
            );
        }
        const std::size_t required = name.size() + 1;
        *required_capacity = checked_u64(
            required,
            "parameter name capacity"
        );
        const std::size_t capacity =
            checked_size(name_capacity, "name output capacity");
        if (capacity == 0 && output_name == nullptr) {
            return;
        }
        if (output_name == nullptr) {
            throw std::invalid_argument(
                "name output must not be null"
            );
        }
        if (capacity < required) {
            throw std::invalid_argument(
                "name output capacity is too small"
            );
        }
        std::copy(
            name.c_str(),
            name.c_str() + required,
            output_name
        );
    });
}

tl_status TL_CALL tl_parameter_list_rank(
    const tl_parameter_list* parameters,
    uint64_t index,
    uint64_t* output
) {
    return guard([&] {
        if (output == nullptr) {
            throw std::invalid_argument(
                "rank output must not be null"
            );
        }
        *output = checked_u64(
            checked_parameter(parameters, index)
                .parameter->value().rank(),
            "parameter rank"
        );
    });
}

tl_status TL_CALL tl_parameter_list_shape(
    const tl_parameter_list* parameters,
    uint64_t index,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
) {
    return guard([&] {
        copy_tensor_shape(
            checked_parameter(parameters, index)
                .parameter->value(),
            output_dimensions,
            dimension_capacity
        );
    });
}

tl_status TL_CALL tl_parameter_list_numel(
    const tl_parameter_list* parameters,
    uint64_t index,
    uint64_t* output
) {
    return guard([&] {
        if (output == nullptr) {
            throw std::invalid_argument(
                "element-count output must not be null"
            );
        }
        *output = checked_u64(
            checked_parameter(parameters, index)
                .parameter->value().numel(),
            "parameter element count"
        );
    });
}

tl_status TL_CALL tl_parameter_list_total_numel(
    const tl_parameter_list* parameters,
    uint64_t* output
) {
    return guard([&] {
        require_parameter_list(parameters);
        if (output == nullptr) {
            throw std::invalid_argument(
                "total element-count output must not be null"
            );
        }
        *output = checked_u64(
            transformer_lab::parameter_count(parameters->value),
            "total parameter element count"
        );
    });
}

tl_status TL_CALL tl_parameter_list_copy_to_host_f32(
    const tl_parameter_list* parameters,
    float* output_values,
    uint64_t value_capacity
) {
    return guard([&] {
        require_parameter_list(parameters);
        const std::size_t total =
            transformer_lab::parameter_count(parameters->value);
        const std::size_t capacity = checked_size(
            value_capacity,
            "parameter value output capacity"
        );
        if (capacity < total) {
            throw std::invalid_argument(
                "parameter value output capacity is too small"
            );
        }
        if (total != 0 && output_values == nullptr) {
            throw std::invalid_argument(
                "parameter value output must not be null"
            );
        }

        std::size_t offset = 0;
        for (const auto& named_parameter : parameters->value) {
            const auto values =
                named_parameter.parameter->value().data();
            std::copy(
                values.begin(),
                values.end(),
                output_values + offset
            );
            offset += values.size();
        }
    });
}

tl_status TL_CALL tl_parameter_list_load_from_host_f32(
    tl_parameter_list* parameters,
    const float* values,
    uint64_t value_count
) {
    return guard([&] {
        require_parameter_list(parameters);
        const std::size_t total =
            transformer_lab::parameter_count(parameters->value);
        const std::size_t count = checked_size(
            value_count,
            "parameter value count"
        );
        if (count != total) {
            throw std::invalid_argument(
                "parameter value count must exactly match the "
                "parameter list"
            );
        }
        if (total != 0 && values == nullptr) {
            throw std::invalid_argument(
                "parameter values must not be null"
            );
        }
        if (total != 0 &&
            !std::all_of(
                values,
                values + total,
                [](float value) {
                    return std::isfinite(value);
                }
            )) {
            throw std::invalid_argument(
                "parameter values must all be finite"
            );
        }
        if (parameters->owner->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot load parameters while variable graphs are alive"
            );
        }
        if (parameters->owner->active_optimizers.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot load parameters while optimizers are alive"
            );
        }
        require_no_active_decode_sessions(
            *parameters->owner,
            "load parameters"
        );
        require_epoch_increment_available(*parameters->owner);

        struct Replacement {
            transformer_lab::Parameter* parameter;
            transformer_lab::Tensor value;
        };
        std::vector<Replacement> replacements;
        replacements.reserve(parameters->value.size());

        std::size_t offset = 0;
        for (const auto& named_parameter : parameters->value) {
            auto* parameter = named_parameter.parameter;
            const auto& old_value = parameter->value();
            const std::size_t parameter_size = old_value.numel();
            std::vector<float> replacement_values(
                values + offset,
                values + offset + parameter_size
            );
            replacements.push_back({
                parameter,
                transformer_lab::Tensor(
                    old_value.shape(),
                    std::move(replacement_values),
                    old_value.backend()
                ),
            });
            offset += parameter_size;
        }

        // All validation and allocations have completed. Same-shape,
        // same-backend replacement is non-allocating and clears the gradient.
        for (auto& replacement : replacements) {
            replacement.parameter->set_value(
                std::move(replacement.value)
            );
        }
        parameters->owner->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
    });
}

#if defined(TRANSFORMER_LAB_C_API_TESTING)
tl_status TL_CALL tl_test_model_parameter_epoch(
    const tl_model* model,
    uint64_t* output
) {
    return guard([&] {
        require_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "parameter epoch output must not be null"
            );
        }
        *output = model->state->parameter_epoch.load(
            std::memory_order_relaxed
        );
    });
}

tl_status TL_CALL tl_test_model_set_parameter_epoch(
    tl_model* model,
    uint64_t value
) {
    return guard([&] {
        require_model(model);
        model->state->parameter_epoch.store(
            value,
            std::memory_order_relaxed
        );
    });
}
#endif

void TL_CALL tl_variable_release(tl_variable* variable) {
    delete variable;
}

tl_status TL_CALL tl_variable_backend(
    const tl_variable* variable,
    tl_backend* output
) {
    return guard([&] {
        require_variable(variable);
        if (output == nullptr) {
            throw std::invalid_argument(
                "backend output must not be null"
            );
        }
        *output = c_backend(variable->value.value().backend());
    });
}

tl_status TL_CALL tl_variable_rank(
    const tl_variable* variable,
    uint64_t* output
) {
    return guard([&] {
        require_variable(variable);
        if (output == nullptr) {
            throw std::invalid_argument(
                "rank output must not be null"
            );
        }
        *output = checked_u64(
            variable->value.value().rank(),
            "variable rank"
        );
    });
}

tl_status TL_CALL tl_variable_shape(
    const tl_variable* variable,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
) {
    return guard([&] {
        require_variable(variable);
        copy_tensor_shape(
            variable->value.value(),
            output_dimensions,
            dimension_capacity
        );
    });
}

tl_status TL_CALL tl_variable_numel(
    const tl_variable* variable,
    uint64_t* output
) {
    return guard([&] {
        require_variable(variable);
        if (output == nullptr) {
            throw std::invalid_argument(
                "element-count output must not be null"
            );
        }
        *output = checked_u64(
            variable->value.value().numel(),
            "variable element count"
        );
    });
}

tl_status TL_CALL tl_variable_copy_to_host_f32(
    const tl_variable* variable,
    float* output_values,
    uint64_t value_capacity
) {
    return guard([&] {
        require_variable(variable);
        copy_tensor_values(
            variable->value.value(),
            output_values,
            value_capacity
        );
    });
}

tl_status TL_CALL tl_variable_backward(
    const tl_variable* variable
) {
    return guard([&] {
        require_variable(variable);
        require_current_graph(variable, "backward");
        variable->value.backward();
        variable->graph->backward_consumed = true;
    });
}

tl_status TL_CALL tl_cross_entropy(
    const tl_variable* logits,
    const uint32_t* targets,
    uint64_t target_count,
    tl_variable** output
) {
    return guard([&] {
        require_output(output);
        require_variable(logits);
        require_current_graph(logits, "cross entropy");
        const auto values = checked_token_ids(
            targets,
            target_count,
            "cross-entropy targets"
        );
        auto result = std::make_unique<tl_variable>(
            logits->owner,
            logits->graph,
            transformer_lab::cross_entropy(
                logits->value,
                values
            )
        );
        *output = result.release();
    });
}

tl_status TL_CALL tl_adam_options_init(
    tl_adam_options* options,
    uint64_t options_size
) {
    return guard([&] {
        if (options == nullptr) {
            throw std::invalid_argument(
                "Adam options must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(tl_adam_options, reserved) +
            sizeof(std::uint32_t);
        checked_structure_size(
            options_size,
            minimum_size,
            "Adam options structure"
        );
        const transformer_lab::AdamOptions defaults;
        *options = {
            options_size,
            defaults.learning_rate,
            defaults.beta1,
            defaults.beta2,
            defaults.epsilon,
            defaults.maximum_gradient_norm,
            0,
        };
    });
}

tl_status TL_CALL tl_adam_create(
    const tl_parameter_list* parameters,
    const tl_adam_options* options,
    tl_adam** output
) {
    return guard([&] {
        require_output(output);
        require_parameter_list(parameters);
        auto result = std::make_unique<tl_adam>(
            parameters->owner,
            parameters->value,
            checked_adam_options(options)
        );
        *output = result.release();
    });
}

void TL_CALL tl_adam_release(tl_adam* adam) {
    delete adam;
}

tl_status TL_CALL tl_adam_backend(
    const tl_adam* adam,
    tl_backend* output
) {
    return guard([&] {
        require_adam(adam);
        if (output == nullptr) {
            throw std::invalid_argument(
                "backend output must not be null"
            );
        }
        *output = c_backend(adam->value.backend());
    });
}

tl_status TL_CALL tl_adam_step_count(
    const tl_adam* adam,
    uint64_t* output
) {
    return guard([&] {
        require_adam(adam);
        if (output == nullptr) {
            throw std::invalid_argument(
                "step-count output must not be null"
            );
        }
        *output = checked_u64(
            adam->value.step_count(),
            "Adam step count"
        );
    });
}

tl_status TL_CALL tl_adam_parameter_count(
    const tl_adam* adam,
    uint64_t* output
) {
    return guard([&] {
        require_adam(adam);
        if (output == nullptr) {
            throw std::invalid_argument(
                "parameter-count output must not be null"
            );
        }
        *output = checked_u64(
            adam->value.parameter_tensor_count(),
            "Adam parameter count"
        );
    });
}

tl_status TL_CALL tl_adam_step(
    tl_adam* adam,
    tl_adam_step_stats* output_stats
) {
    return guard([&] {
        require_adam(adam);
        if (output_stats == nullptr) {
            throw std::invalid_argument(
                "Adam step statistics must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(tl_adam_step_stats, clip_scale) +
            sizeof(double);
        checked_structure_size(
            output_stats->struct_size,
            minimum_size,
            "Adam step statistics structure"
        );
        const std::uint64_t structure_size =
            output_stats->struct_size;
        require_no_active_decode_sessions(
            *adam->owner,
            "step an optimizer"
        );
        require_epoch_increment_available(*adam->owner);
        const auto stats = adam->value.step();
        adam->owner->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
        *output_stats = {
            structure_size,
            checked_u64(stats.step, "Adam step"),
            stats.gradient_norm,
            stats.clip_scale,
        };
    });
}

tl_status TL_CALL tl_adam_zero_gradients(
    const tl_adam* adam
) {
    return guard([&] {
        require_adam(adam);
        adam->value.zero_gradients();
    });
}

}  // extern "C"
