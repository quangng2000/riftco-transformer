#include "riftco_transformer/c_api.h"

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/core/tensor.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"
#include "riftco_transformer/data/tokenizer.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/model/lora.hpp"
#include "riftco_transformer/nn/loss.hpp"
#include "riftco_transformer/nn/parameter.hpp"
#include "riftco_transformer/optim/adam.hpp"
#include "riftco_transformer/stages/serving/kv_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

static_assert(sizeof(rt_transformer_config) == 64);
static_assert(sizeof(rt_lora_config) == 40);
static_assert(sizeof(rt_quantized_memory_stats) == 96);
static_assert(sizeof(rt_decode_session_options) == 24);
static_assert(sizeof(rt_adam_options) == 48);
static_assert(sizeof(rt_adam_step_stats) == 32);
static_assert(sizeof(rt_tokenizer_options) == 32);
static_assert(sizeof(rt_bpe_merge_rule) == 12);
static_assert(sizeof(rt_activation_checkpointing_kind) == 4);
static_assert(sizeof(rt_adam_state_storage_kind) == 4);
static_assert(
    RT_LORA_TARGET_ATTENTION_QUERY ==
    riftco_transformer::kLoraAttentionQuery
);
static_assert(
    RT_LORA_TARGET_ATTENTION_KEY ==
    riftco_transformer::kLoraAttentionKey
);
static_assert(
    RT_LORA_TARGET_ATTENTION_VALUE ==
    riftco_transformer::kLoraAttentionValue
);
static_assert(
    RT_LORA_TARGET_ATTENTION_OUTPUT ==
    riftco_transformer::kLoraAttentionOutput
);
static_assert(
    RT_LORA_TARGET_FF_EXPAND ==
    riftco_transformer::kLoraFeedForwardExpand
);
static_assert(
    RT_LORA_TARGET_FF_PROJECT ==
    riftco_transformer::kLoraFeedForwardProject
);
static_assert(
    RT_LORA_TARGET_LM_HEAD ==
    riftco_transformer::kLoraLanguageModelHead
);
static_assert(
    RT_LORA_TARGET_DEFAULT ==
    riftco_transformer::kLoraDefaultTargets
);
static_assert(
    RT_LORA_TARGET_ALL_LINEAR ==
    riftco_transformer::kLoraAllTargets
);

struct rt_context {
    riftco_transformer::ExecutionBackend backend;
};

struct rt_tensor {
    riftco_transformer::Tensor value;
};

struct rt_tokenizer {
    explicit rt_tokenizer(
        std::unique_ptr<riftco_transformer::TokenizerStrategy> strategy
    )
        : value(std::move(strategy)) {
        if (value == nullptr) {
            throw std::invalid_argument(
                "tokenizer strategy must not be null"
            );
        }
    }

    std::unique_ptr<riftco_transformer::TokenizerStrategy> value;
};

struct ModelState {
    ModelState(
        riftco_transformer::TransformerDimensions dimensions,
        std::mt19937& random,
        float layer_norm_epsilon
    )
        : value(
              dimensions,
              random,
              layer_norm_epsilon
          ) {}

    riftco_transformer::DecoderOnlyTransformer value;
    std::atomic_size_t active_variables{0};
    std::atomic_size_t active_optimizers{0};
    std::atomic_size_t active_parameter_lists{0};
    std::atomic_size_t active_lora_parameter_lists{0};
    std::atomic_size_t active_decode_sessions{0};
    std::atomic_uint64_t parameter_epoch{0};
};

struct rt_model {
    std::shared_ptr<ModelState> state;
};

struct rt_decode_session {
    rt_decode_session(
        std::shared_ptr<ModelState> model_owner,
        std::unique_ptr<riftco_transformer::DecoderKeyValueCache>
            key_value_cache,
        rt_kv_cache_kind cache_kind,
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

    ~rt_decode_session() {
        owner->active_decode_sessions.fetch_sub(
            1,
            std::memory_order_relaxed
        );
    }

    rt_decode_session(const rt_decode_session&) = delete;
    rt_decode_session& operator=(const rt_decode_session&) = delete;
    rt_decode_session(rt_decode_session&&) = delete;
    rt_decode_session& operator=(rt_decode_session&&) = delete;

    std::shared_ptr<ModelState> owner;
    std::unique_ptr<riftco_transformer::DecoderKeyValueCache> cache;
    rt_kv_cache_kind kind;
    std::size_t block_size;
    std::uint64_t parameter_epoch;
};

struct rt_parameter_list {
    rt_parameter_list(
        std::shared_ptr<ModelState> model_owner,
        riftco_transformer::ParameterList parameters,
        bool tracks_lora_parameters
    )
        : owner(std::move(model_owner)),
          value(std::move(parameters)),
          tracks_lora(tracks_lora_parameters) {
        owner->active_parameter_lists.fetch_add(
            1,
            std::memory_order_relaxed
        );
        if (tracks_lora) {
            owner->active_lora_parameter_lists.fetch_add(
                1,
                std::memory_order_relaxed
            );
        }
    }

    ~rt_parameter_list() {
        owner->active_parameter_lists.fetch_sub(
            1,
            std::memory_order_relaxed
        );
        if (tracks_lora) {
            owner->active_lora_parameter_lists.fetch_sub(
                1,
                std::memory_order_relaxed
            );
        }
    }

    rt_parameter_list(const rt_parameter_list&) = delete;
    rt_parameter_list& operator=(const rt_parameter_list&) = delete;
    rt_parameter_list(rt_parameter_list&&) = delete;
    rt_parameter_list& operator=(rt_parameter_list&&) = delete;

    std::shared_ptr<ModelState> owner;
    riftco_transformer::ParameterList value;
    bool tracks_lora;
};

struct VariableGraphState {
    explicit VariableGraphState(std::uint64_t epoch)
        : parameter_epoch(epoch) {}

    std::uint64_t parameter_epoch;
    bool backward_consumed = false;
};

struct rt_variable {
    rt_variable(
        std::shared_ptr<ModelState> model_owner,
        std::shared_ptr<VariableGraphState> graph_state,
        riftco_transformer::Variable variable
    )
        : owner(std::move(model_owner)),
          graph(std::move(graph_state)),
          value(std::move(variable)) {
        owner->active_variables.fetch_add(
            1,
            std::memory_order_relaxed
        );
    }

    ~rt_variable() {
        owner->active_variables.fetch_sub(
            1,
            std::memory_order_relaxed
        );
    }

    std::shared_ptr<ModelState> owner;
    std::shared_ptr<VariableGraphState> graph;
    riftco_transformer::Variable value;
};

struct rt_adam {
    rt_adam(
        std::shared_ptr<ModelState> model_owner,
        riftco_transformer::ParameterList parameters,
        riftco_transformer::AdamOptions options
    )
        : owner(std::move(model_owner)),
          value(std::move(parameters), options) {
        owner->active_optimizers.fetch_add(
            1,
            std::memory_order_relaxed
        );
    }

    ~rt_adam() {
        owner->active_optimizers.fetch_sub(
            1,
            std::memory_order_relaxed
        );
    }

    std::shared_ptr<ModelState> owner;
    riftco_transformer::Adam value;
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
rt_status guard(Function&& function) noexcept {
    try {
        clear_last_error();
        function();
        return RT_STATUS_OK;
    } catch (const BackendUnavailable& error) {
        set_last_error(error.what());
        return RT_STATUS_BACKEND_UNAVAILABLE;
    } catch (const std::domain_error& error) {
        set_last_error(error.what());
        return RT_STATUS_INVALID_ARGUMENT;
    } catch (const std::invalid_argument& error) {
        set_last_error(error.what());
        return RT_STATUS_INVALID_ARGUMENT;
    } catch (const std::out_of_range& error) {
        set_last_error(error.what());
        return RT_STATUS_OUT_OF_RANGE;
    } catch (const std::overflow_error& error) {
        set_last_error(error.what());
        return RT_STATUS_OVERFLOW;
    } catch (const std::bad_alloc& error) {
        set_last_error(error.what());
        return RT_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        set_last_error(error.what());
        return RT_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_last_error("unknown native error");
        return RT_STATUS_UNKNOWN_ERROR;
    }
}

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

riftco_transformer::AdamStateStorageKind checked_adam_state_storage(
    rt_adam_state_storage_kind kind
) {
    switch (kind) {
        case RT_ADAM_STATE_CONTIGUOUS:
            return riftco_transformer::AdamStateStorageKind::Contiguous;
        case RT_ADAM_STATE_PAGED:
            return riftco_transformer::AdamStateStorageKind::Paged;
        default:
            throw std::invalid_argument(
                "unknown C API Adam state storage kind"
            );
    }
}

rt_adam_state_storage_kind c_adam_state_storage(
    riftco_transformer::AdamStateStorageKind kind
) {
    switch (kind) {
        case riftco_transformer::AdamStateStorageKind::Contiguous:
            return RT_ADAM_STATE_CONTIGUOUS;
        case riftco_transformer::AdamStateStorageKind::Paged:
            return RT_ADAM_STATE_PAGED;
    }
    throw std::invalid_argument(
        "unknown native Adam state storage kind"
    );
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

riftco_transformer::ActivationCheckpointingKind
checked_activation_checkpointing(
    rt_activation_checkpointing_kind kind
) {
    switch (kind) {
        case RT_ACTIVATION_CHECKPOINTING_DISABLED:
            return riftco_transformer::ActivationCheckpointingKind::Disabled;
        case RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK:
            return riftco_transformer::ActivationCheckpointingKind::
                TransformerBlock;
        default:
            throw std::invalid_argument(
                "unknown C API activation checkpointing kind"
            );
    }
}

rt_activation_checkpointing_kind c_activation_checkpointing(
    riftco_transformer::ActivationCheckpointingKind kind
) {
    switch (kind) {
        case riftco_transformer::ActivationCheckpointingKind::Disabled:
            return RT_ACTIVATION_CHECKPOINTING_DISABLED;
        case riftco_transformer::ActivationCheckpointingKind::
                TransformerBlock:
            return RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK;
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

void require_tokenizer(const rt_tokenizer* tokenizer) {
    if (tokenizer == nullptr || tokenizer->value == nullptr) {
        throw std::invalid_argument(
            "tokenizer handle must not be null"
        );
    }
}

riftco_transformer::TokenizerMethod checked_tokenizer_method(
    rt_tokenizer_method method
) {
    switch (method) {
        case RT_TOKENIZER_METHOD_BYTE:
            return riftco_transformer::TokenizerMethod::CorpusByte;
        case RT_TOKENIZER_METHOD_BPE:
            return riftco_transformer::TokenizerMethod::BytePair;
        default:
            throw std::invalid_argument(
                "unknown C API tokenizer method"
            );
    }
}

rt_tokenizer_method c_tokenizer_method(
    riftco_transformer::TokenizerMethod method
) {
    switch (method) {
        case riftco_transformer::TokenizerMethod::CorpusByte:
            return RT_TOKENIZER_METHOD_BYTE;
        case riftco_transformer::TokenizerMethod::BytePair:
            return RT_TOKENIZER_METHOD_BPE;
    }
    throw std::invalid_argument("unknown native tokenizer method");
}

riftco_transformer::TokenizerOptions checked_tokenizer_options(
    const rt_tokenizer_options* options
) {
    if (options == nullptr) {
        return {};
    }
    constexpr std::size_t minimum_size =
        offsetof(rt_tokenizer_options, minimum_pair_frequency) +
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
    if (method == riftco_transformer::TokenizerMethod::CorpusByte) {
        return {
            method,
            riftco_transformer::TokenizerOptions{}.vocabulary_size,
            riftco_transformer::TokenizerOptions{}.minimum_pair_frequency,
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
            std::numeric_limits<riftco_transformer::TokenId>::max()
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
    rt_kv_cache_kind kind;
    std::size_t block_size;
};

CheckedDecodeSessionOptions checked_decode_session_options(
    const rt_decode_session_options* options
) {
    if (options == nullptr) {
        return {
            RT_KV_CACHE_PAGED,
            16,
        };
    }
    constexpr std::size_t minimum_size =
        offsetof(rt_decode_session_options, block_size) +
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
        case RT_KV_CACHE_CONTIGUOUS:
        case RT_KV_CACHE_PAGED:
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

void require_context(const rt_context* context) {
    if (context == nullptr) {
        throw std::invalid_argument(
            "context handle must not be null"
        );
    }
}

void require_tensor(const rt_tensor* tensor) {
    if (tensor == nullptr) {
        throw std::invalid_argument(
            "tensor handle must not be null"
        );
    }
}

void require_model(const rt_model* model) {
    if (model == nullptr || model->state == nullptr) {
        throw std::invalid_argument(
            "model handle must not be null"
        );
    }
}

void require_decode_session(const rt_decode_session* session) {
    if (session == nullptr ||
        session->owner == nullptr ||
        session->cache == nullptr) {
        throw std::invalid_argument(
            "decode-session handle must not be null"
        );
    }
}

void require_current_decode_session(
    const rt_decode_session* session
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

void require_variable(const rt_variable* variable) {
    if (variable == nullptr ||
        variable->owner == nullptr ||
        variable->graph == nullptr) {
        throw std::invalid_argument(
            "variable handle must not be null"
        );
    }
}

void require_current_graph(
    const rt_variable* variable,
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

void require_model_quantization_safe(rt_model* model) {
    require_model(model);
    if (model->state->active_variables.load(
            std::memory_order_relaxed
        ) != 0) {
        throw std::invalid_argument(
            "cannot quantize a model while variable graphs are alive"
        );
    }
    if (model->state->active_optimizers.load(
            std::memory_order_relaxed
        ) != 0) {
        throw std::invalid_argument(
            "cannot quantize a model while optimizers are alive"
        );
    }
    if (model->state->active_parameter_lists.load(
            std::memory_order_relaxed
        ) != 0) {
        throw std::invalid_argument(
            "cannot quantize a model while parameter lists are alive"
        );
    }
    require_no_active_decode_sessions(
        *model->state,
        "quantize a model"
    );
    require_epoch_increment_available(*model->state);
}

void require_adam(const rt_adam* adam) {
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

riftco_transformer::ExecutionBackend model_backend(
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

riftco_transformer::AdamOptions checked_adam_options(
    const rt_adam_options* options
) {
    if (options == nullptr) {
        return {};
    }
    constexpr std::size_t minimum_size =
        offsetof(rt_adam_options, reserved) +
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
    riftco_transformer::AdamOptions result{
        options->learning_rate,
        options->beta1,
        options->beta2,
        options->epsilon,
        options->maximum_gradient_norm,
    };
    if (options->struct_size >=
        offsetof(rt_adam_options, state_storage) +
            sizeof(rt_adam_state_storage_kind)) {
        result.state_storage = checked_adam_state_storage(
            options->state_storage
        );
    }
    if (options->struct_size >=
            offsetof(rt_adam_options, reserved2) +
                sizeof(std::uint32_t) &&
        options->reserved2 != 0) {
        throw std::invalid_argument(
            "Adam options second reserved field must be zero"
        );
    }
    if (options->struct_size >=
        offsetof(rt_adam_options, page_size) +
            sizeof(std::uint64_t)) {
        result.page_size = checked_size(
            options->page_size,
            "Adam state page size"
        );
    }
    return result;
}

riftco_transformer::LoraConfig checked_lora_config(
    const rt_lora_config* config
) {
    if (config == nullptr) {
        return {};
    }
    constexpr std::size_t minimum_size =
        offsetof(rt_lora_config, reserved) +
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
    if ((config->targets & ~RT_LORA_TARGET_ALL_LINEAR) != 0) {
        throw std::invalid_argument(
            "LoRA targets contain an unknown bit"
        );
    }
    return {
        checked_size(config->rank, "LoRA rank"),
        config->alpha,
        config->random_seed,
        static_cast<riftco_transformer::LoraTargetMask>(
            config->targets
        ),
    };
}

void write_lora_config(
    const riftco_transformer::LoraConfig& config,
    rt_lora_config* output
) {
    if (output == nullptr) {
        throw std::invalid_argument(
            "LoRA config output must not be null"
        );
    }
    constexpr std::size_t minimum_size =
        offsetof(rt_lora_config, reserved) +
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
        static_cast<rt_lora_target_mask>(config.targets),
        0,
    };
}

}  // namespace

extern "C" {

uint32_t RT_CALL rt_abi_version(void) {
    return RT_ABI_VERSION;
}

const char* RT_CALL rt_status_string(rt_status status) {
    switch (status) {
        case RT_STATUS_OK:
            return "ok";
        case RT_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case RT_STATUS_OUT_OF_RANGE:
            return "out of range";
        case RT_STATUS_OVERFLOW:
            return "overflow";
        case RT_STATUS_BACKEND_UNAVAILABLE:
            return "backend unavailable";
        case RT_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case RT_STATUS_RUNTIME_ERROR:
            return "runtime error";
        case RT_STATUS_UNKNOWN_ERROR:
            return "unknown error";
    }
    return "unrecognized status";
}

const char* RT_CALL rt_last_error(void) {
    return last_error.data();
}

rt_status RT_CALL rt_tokenizer_options_init(
    rt_tokenizer_options* options,
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
                rt_tokenizer_options,
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
            RT_TOKENIZER_METHOD_BYTE,
            0,
            512,
            2,
        };
    });
}

rt_status RT_CALL rt_tokenizer_create(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    rt_tokenizer** output
) {
    return rt_tokenizer_create_with_options(
        corpus_bytes,
        corpus_size,
        nullptr,
        output
    );
}

rt_status RT_CALL rt_tokenizer_create_with_options(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    const rt_tokenizer_options* options,
    rt_tokenizer** output
) {
    return guard([&] {
        require_output(output);
        auto result = std::make_unique<rt_tokenizer>(
            riftco_transformer::make_tokenizer(
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

rt_status RT_CALL rt_tokenizer_create_from_byte_vocabulary(
    const uint8_t* ordered_vocabulary,
    uint64_t vocabulary_size,
    rt_tokenizer** output
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
        auto result = std::make_unique<rt_tokenizer>(
            std::make_unique<riftco_transformer::ByteTokenizer>(
                std::span<const std::uint8_t>(
                    ordered_vocabulary,
                    count
                )
            )
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_tokenizer_create_from_bpe_merges(
    const rt_bpe_merge_rule* ordered_merge_rules,
    uint64_t merge_count,
    rt_tokenizer** output
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
        std::vector<riftco_transformer::BpeMergeRule> rules;
        rules.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto& rule = ordered_merge_rules[index];
            rules.push_back({
                rule.left,
                rule.right,
                rule.result,
            });
        }
        auto result = std::make_unique<rt_tokenizer>(
            std::make_unique<riftco_transformer::BytePairTokenizer>(
                std::span<const riftco_transformer::BpeMergeRule>(
                    rules.data(),
                    rules.size()
                )
            )
        );
        *output = result.release();
    });
}

void RT_CALL rt_tokenizer_release(rt_tokenizer* tokenizer) {
    delete tokenizer;
}

rt_status RT_CALL rt_tokenizer_get_method(
    const rt_tokenizer* tokenizer,
    rt_tokenizer_method* output
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

rt_status RT_CALL rt_tokenizer_vocabulary_size(
    const rt_tokenizer* tokenizer,
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

rt_status RT_CALL rt_tokenizer_bpe_merge_count(
    const rt_tokenizer* tokenizer,
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
            dynamic_cast<const riftco_transformer::BytePairTokenizer*>(
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

rt_status RT_CALL rt_tokenizer_bpe_merge_rule(
    const rt_tokenizer* tokenizer,
    uint64_t index,
    rt_bpe_merge_rule* output
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (output == nullptr) {
            throw std::invalid_argument(
                "BPE merge-rule output must not be null"
            );
        }
        const auto* byte_pair =
            dynamic_cast<const riftco_transformer::BytePairTokenizer*>(
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

rt_status RT_CALL rt_tokenizer_vocabulary(
    const rt_tokenizer* tokenizer,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (tokenizer->value->method() !=
            riftco_transformer::TokenizerMethod::CorpusByte) {
            throw std::invalid_argument(
                "tokenizer vocabulary is only available for "
                "the corpus-byte method; use "
                "rt_tokenizer_token_bytes for BPE"
            );
        }

        std::vector<std::uint8_t> vocabulary;
        vocabulary.reserve(tokenizer->value->vocab_size());
        for (std::size_t index = 0;
             index < tokenizer->value->vocab_size();
             ++index) {
            const auto bytes = tokenizer->value->token_bytes(
                static_cast<riftco_transformer::TokenId>(index)
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

rt_status RT_CALL rt_tokenizer_token_bytes(
    const rt_tokenizer* tokenizer,
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

rt_status RT_CALL rt_tokenizer_encode(
    const rt_tokenizer* tokenizer,
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
            std::span<const riftco_transformer::TokenId>(
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

rt_status RT_CALL rt_tokenizer_decode(
    const rt_tokenizer* tokenizer,
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

rt_status RT_CALL rt_backend_is_available(
    rt_backend backend,
    int32_t* available
) {
    return guard([&] {
        if (available == nullptr) {
            throw std::invalid_argument(
                "availability output must not be null"
            );
        }
        riftco_transformer::ExecutionBackend native;
        switch (backend) {
            case RT_BACKEND_CPU:
                native =
                    riftco_transformer::ExecutionBackend::Cpu;
                break;
            case RT_BACKEND_METAL:
                native =
                    riftco_transformer::ExecutionBackend::Metal;
                break;
            case RT_BACKEND_CUDA:
                native =
                    riftco_transformer::ExecutionBackend::Cuda;
                break;
            case RT_BACKEND_TPU:
                native =
                    riftco_transformer::ExecutionBackend::Tpu;
                break;
            default:
                throw std::invalid_argument(
                    "unknown C API backend"
                );
        }
        *available =
            riftco_transformer::execution_backend_available(native)
                ? 1
                : 0;
    });
}

rt_status RT_CALL rt_context_create(
    rt_backend backend,
    rt_context** output
) {
    return guard([&] {
        require_output(output);
        const auto native = checked_backend(backend);
        auto result = std::make_unique<rt_context>(
            rt_context{native}
        );
        *output = result.release();
    });
}

void RT_CALL rt_context_release(rt_context* context) {
    delete context;
}

rt_status RT_CALL rt_context_backend(
    const rt_context* context,
    rt_backend* output
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

rt_status RT_CALL rt_tensor_create_f32(
    const rt_context* context,
    const uint64_t* shape,
    uint64_t rank,
    const float* values,
    uint64_t value_count,
    rt_tensor** output
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
        auto result = std::make_unique<rt_tensor>(
            rt_tensor{
                riftco_transformer::Tensor(
                    checked_shape(shape, rank),
                    std::move(owned_values),
                    context->backend
                ),
            }
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_tensor_zeros_f32(
    const rt_context* context,
    const uint64_t* shape,
    uint64_t rank,
    rt_tensor** output
) {
    return guard([&] {
        require_output(output);
        require_context(context);
        auto result = std::make_unique<rt_tensor>(
            rt_tensor{
                riftco_transformer::Tensor::zeros(
                    checked_shape(shape, rank),
                    context->backend
                ),
            }
        );
        *output = result.release();
    });
}

void RT_CALL rt_tensor_release(rt_tensor* tensor) {
    delete tensor;
}

rt_status RT_CALL rt_tensor_backend(
    const rt_tensor* tensor,
    rt_backend* output
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

rt_status RT_CALL rt_tensor_rank(
    const rt_tensor* tensor,
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

rt_status RT_CALL rt_tensor_shape(
    const rt_tensor* tensor,
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

rt_status RT_CALL rt_tensor_numel(
    const rt_tensor* tensor,
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

rt_status RT_CALL rt_tensor_copy_to_host_f32(
    const rt_tensor* tensor,
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

rt_status RT_CALL rt_tensor_matmul(
    const rt_tensor* left,
    const rt_tensor* right,
    rt_tensor** output
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
        auto result = std::make_unique<rt_tensor>(
            rt_tensor{
                riftco_transformer::tensor_ops::matmul(
                    left->value,
                    right->value
                ),
            }
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_transformer_config_init(
    rt_transformer_config* config,
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
                rt_transformer_config,
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

rt_status RT_CALL rt_decode_session_options_init(
    rt_decode_session_options* options,
    uint64_t options_size
) {
    return guard([&] {
        if (options == nullptr) {
            throw std::invalid_argument(
                "decode-session options must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(rt_decode_session_options, block_size) +
            sizeof(std::uint64_t);
        checked_structure_size(
            options_size,
            minimum_size,
            "decode-session options structure"
        );
        *options = {
            options_size,
            RT_KV_CACHE_PAGED,
            0,
            16,
        };
    });
}

rt_status RT_CALL rt_model_create(
    const rt_transformer_config* config,
    rt_model** output
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
                rt_transformer_config,
                layer_norm_epsilon
            ) +
            sizeof(float);
        checked_structure_size(
            config->struct_size,
            minimum_size,
            "transformer config structure"
        );

        const riftco_transformer::TransformerDimensions dimensions{
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
        const riftco_transformer::ScopedExecutionBackend
            construction_backend(
                riftco_transformer::ExecutionBackend::Cpu
            );
        auto state = std::make_shared<ModelState>(
            dimensions,
            random,
            config->layer_norm_epsilon
        );
        auto result = std::make_unique<rt_model>(
            rt_model{std::move(state)}
        );
        *output = result.release();
    });
}

void RT_CALL rt_model_release(rt_model* model) {
    delete model;
}

rt_status RT_CALL rt_model_to(
    rt_model* model,
    rt_backend backend
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

rt_status RT_CALL rt_model_backend(
    const rt_model* model,
    rt_backend* output
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

rt_status RT_CALL rt_model_set_full_sequence_attention(
    rt_model* model,
    rt_full_sequence_attention_kind kind
) {
    return guard([&] {
        require_model(model);
        model->state->value.set_full_sequence_attention_kind(
            checked_full_sequence_attention(kind)
        );
    });
}

rt_status RT_CALL rt_model_full_sequence_attention(
    const rt_model* model,
    rt_full_sequence_attention_kind* output
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

rt_status RT_CALL rt_model_set_activation_checkpointing(
    rt_model* model,
    rt_activation_checkpointing_kind kind
) {
    return guard([&] {
        require_model(model);
        model->state->value.set_activation_checkpointing_kind(
            checked_activation_checkpointing(kind)
        );
    });
}

rt_status RT_CALL rt_model_activation_checkpointing(
    const rt_model* model,
    rt_activation_checkpointing_kind* output
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

rt_status RT_CALL rt_lora_config_init(
    rt_lora_config* config,
    uint64_t config_size
) {
    return guard([&] {
        if (config == nullptr) {
            throw std::invalid_argument(
                "LoRA config must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(rt_lora_config, reserved) +
            sizeof(std::uint64_t);
        checked_structure_size(
            config_size,
            minimum_size,
            "LoRA config structure"
        );
        const riftco_transformer::LoraConfig defaults;
        *config = {
            config_size,
            checked_u64(defaults.rank, "LoRA rank"),
            defaults.alpha,
            defaults.random_seed,
            static_cast<rt_lora_target_mask>(defaults.targets),
            0,
        };
    });
}

rt_status RT_CALL rt_model_attach_lora(
    rt_model* model,
    const rt_lora_config* config
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

rt_status RT_CALL rt_model_has_lora(
    const rt_model* model,
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

rt_status RT_CALL rt_model_lora_config(
    const rt_model* model,
    rt_lora_config* output
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

rt_status RT_CALL rt_model_quantize_linear_weights_nf4(
    rt_model* model,
    uint64_t block_size
) {
    return guard([&] {
        require_model_quantization_safe(model);
        model->state->value.quantize_linear_weights_nf4(
            checked_size(block_size, "NF4 block size")
        );
        model->state->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
    });
}


rt_status RT_CALL
rt_model_quantize_linear_weights_nf4_double_quantized(
    rt_model* model,
    uint64_t block_size,
    uint64_t scale_block_size
) {
    return guard([&] {
        require_model_quantization_safe(model);
        model->state->value
            .quantize_linear_weights_nf4_double_quantized(
                checked_size(block_size, "NF4 block size"),
                checked_size(
                    scale_block_size,
                    "NF4 scale block size"
                )
            );
        model->state->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
    });
}

rt_status RT_CALL rt_model_has_quantized_linear_weights(
    const rt_model* model,
    int32_t* output
) {
    return guard([&] {
        require_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "quantized model state output must not be null"
            );
        }
        *output =
            model->state->value.has_quantized_linear_weights()
                ? 1
                : 0;
    });
}

rt_status RT_CALL rt_model_quantized_memory_stats(
    const rt_model* model,
    rt_quantized_memory_stats* output
) {
    return guard([&] {
        require_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "quantized memory statistics output must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(
                rt_quantized_memory_stats,
                fp32_equivalent_bytes
            ) +
            sizeof(std::uint64_t);
        checked_structure_size(
            output->struct_size,
            minimum_size,
            "quantized memory statistics structure"
        );
        const std::uint64_t output_size = output->struct_size;
        const auto usage =
            model->state->value.quantized_memory_usage();
        const rt_quantized_memory_stats initialized{
            output_size,
            checked_u64(
                model->state->value.quantized_linear_weight_count(),
                "quantized linear weight count"
            ),
            checked_u64(
                usage.packed_code_bytes,
                "NF4 packed code bytes"
            ),
            checked_u64(usage.scale_bytes, "NF4 scale bytes"),
            checked_u64(
                usage.logical_payload_bytes,
                "NF4 logical payload bytes"
            ),
            checked_u64(
                usage.resident_payload_bytes,
                "NF4 resident payload bytes"
            ),
            checked_u64(
                usage.fp32_equivalent_bytes,
                "NF4 FP32-equivalent bytes"
            ),
            checked_u64(
                usage.fp32_scale_bytes,
                "NF4 FP32 first-level scale bytes"
            ),
            checked_u64(
                usage.scale_code_bytes,
                "NF4 double-quantized scale-code bytes"
            ),
            checked_u64(
                usage.second_level_scale_bytes,
                "NF4 second-level scale bytes"
            ),
            checked_u64(
                usage.scale_offset_bytes,
                "NF4 scale-offset bytes"
            ),
            checked_u64(
                model->state->value
                    .double_quantized_linear_weight_count(),
                "double-quantized linear weight count"
            ),
        };
        const auto writable_size = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                output_size,
                sizeof(initialized)
            )
        );
        std::memcpy(output, &initialized, writable_size);
    });
}

rt_status RT_CALL rt_model_forward(
    const rt_model* model,
    const uint32_t* token_ids,
    uint64_t token_count,
    uint64_t batch_size,
    uint64_t sequence_length,
    rt_variable** output
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
        auto result = std::make_unique<rt_variable>(
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

rt_status RT_CALL rt_model_decode_session_create(
    const rt_model* model,
    const rt_decode_session_options* options,
    rt_decode_session** output
) {
    return guard([&] {
        require_output(output);
        require_model(model);
        const auto configured =
            checked_decode_session_options(options);
        const auto& dimensions =
            model->state->value.dimensions();
        const auto backend = model->state->value.backend();

        std::unique_ptr<riftco_transformer::DecoderKeyValueCache>
            cache;
        std::size_t block_size = 0;
        if (configured.kind == RT_KV_CACHE_CONTIGUOUS) {
            const riftco_transformer::stages::serving::
                ContiguousKvCacheFactory factory(
                    dimensions,
                    backend
                );
            cache = factory.create();
            block_size = dimensions.maximum_context;
        } else {
            const riftco_transformer::stages::serving::
                PagedKvCachePool pool(
                    dimensions,
                    backend,
                    configured.block_size
                );
            cache = pool.create();
            block_size = pool.block_size();
        }

        auto result = std::make_unique<rt_decode_session>(
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

void RT_CALL rt_decode_session_release(
    rt_decode_session* session
) {
    delete session;
}

rt_status RT_CALL rt_decode_session_reset(
    rt_decode_session* session
) {
    return guard([&] {
        require_current_decode_session(session);
        session->cache->reset();
    });
}

rt_status RT_CALL rt_decode_session_size(
    const rt_decode_session* session,
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

rt_status RT_CALL rt_decode_session_capacity(
    const rt_decode_session* session,
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

rt_status RT_CALL rt_decode_session_cache_kind(
    const rt_decode_session* session,
    rt_kv_cache_kind* output
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

rt_status RT_CALL rt_decode_session_block_size(
    const rt_decode_session* session,
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

rt_status RT_CALL rt_decode_session_step(
    rt_decode_session* session,
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

        const riftco_transformer::Tensor logits =
            session->owner->value.decode_token(
                token_id,
                *session->cache
            );
        const riftco_transformer::Tensor::Shape expected_shape{
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

rt_status RT_CALL rt_model_parameters(
    rt_model* model,
    rt_parameter_list** output
) {
    return guard([&] {
        require_output(output);
        require_model(model);
        auto result = std::make_unique<rt_parameter_list>(
            model->state,
            model->state->value.parameters(),
            false
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_model_lora_parameters(
    rt_model* model,
    rt_parameter_list** output
) {
    return guard([&] {
        require_output(output);
        require_model(model);
        if (!model->state->value.has_lora()) {
            throw std::invalid_argument(
                "model has no attached LoRA adapter"
            );
        }
        auto result = std::make_unique<rt_parameter_list>(
            model->state,
            model->state->value.lora_parameters(),
            true
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_model_merge_lora(rt_model* model) {
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

void RT_CALL rt_parameter_list_release(
    rt_parameter_list* parameters
) {
    delete parameters;
}

rt_status RT_CALL rt_parameter_list_count(
    const rt_parameter_list* parameters,
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

rt_status RT_CALL rt_parameter_list_backend(
    const rt_parameter_list* parameters,
    rt_backend* output
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

rt_status RT_CALL rt_parameter_list_name(
    const rt_parameter_list* parameters,
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

rt_status RT_CALL rt_parameter_list_rank(
    const rt_parameter_list* parameters,
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

rt_status RT_CALL rt_parameter_list_shape(
    const rt_parameter_list* parameters,
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

rt_status RT_CALL rt_parameter_list_numel(
    const rt_parameter_list* parameters,
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

rt_status RT_CALL rt_parameter_list_total_numel(
    const rt_parameter_list* parameters,
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
            riftco_transformer::parameter_count(parameters->value),
            "total parameter element count"
        );
    });
}

rt_status RT_CALL rt_parameter_list_copy_to_host_f32(
    const rt_parameter_list* parameters,
    float* output_values,
    uint64_t value_capacity
) {
    return guard([&] {
        require_parameter_list(parameters);
        const std::size_t total =
            riftco_transformer::parameter_count(parameters->value);
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

rt_status RT_CALL rt_parameter_list_load_from_host_f32(
    rt_parameter_list* parameters,
    const float* values,
    uint64_t value_count
) {
    return guard([&] {
        require_parameter_list(parameters);
        const std::size_t total =
            riftco_transformer::parameter_count(parameters->value);
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
            riftco_transformer::Parameter* parameter;
            riftco_transformer::Tensor value;
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
                riftco_transformer::Tensor(
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

#if defined(RIFTCO_TRANSFORMER_C_API_TESTING)
rt_status RT_CALL rt_test_model_parameter_epoch(
    const rt_model* model,
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

rt_status RT_CALL rt_test_model_set_parameter_epoch(
    rt_model* model,
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

void RT_CALL rt_variable_release(rt_variable* variable) {
    delete variable;
}

rt_status RT_CALL rt_variable_backend(
    const rt_variable* variable,
    rt_backend* output
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

rt_status RT_CALL rt_variable_rank(
    const rt_variable* variable,
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

rt_status RT_CALL rt_variable_shape(
    const rt_variable* variable,
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

rt_status RT_CALL rt_variable_numel(
    const rt_variable* variable,
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

rt_status RT_CALL rt_variable_copy_to_host_f32(
    const rt_variable* variable,
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

rt_status RT_CALL rt_variable_backward(
    const rt_variable* variable
) {
    return guard([&] {
        require_variable(variable);
        require_current_graph(variable, "backward");
        variable->value.backward();
        variable->graph->backward_consumed = true;
    });
}

rt_status RT_CALL rt_cross_entropy(
    const rt_variable* logits,
    const uint32_t* targets,
    uint64_t target_count,
    rt_variable** output
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
        auto result = std::make_unique<rt_variable>(
            logits->owner,
            logits->graph,
            riftco_transformer::cross_entropy(
                logits->value,
                values
            )
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_adam_options_init(
    rt_adam_options* options,
    uint64_t options_size
) {
    return guard([&] {
        if (options == nullptr) {
            throw std::invalid_argument(
                "Adam options must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(rt_adam_options, reserved) +
            sizeof(std::uint32_t);
        checked_structure_size(
            options_size,
            minimum_size,
            "Adam options structure"
        );
        const riftco_transformer::AdamOptions defaults;
        const rt_adam_options initialized{
            options_size,
            defaults.learning_rate,
            defaults.beta1,
            defaults.beta2,
            defaults.epsilon,
            defaults.maximum_gradient_norm,
            0,
            c_adam_state_storage(defaults.state_storage),
            0,
            checked_u64(defaults.page_size, "Adam state page size"),
        };
        const auto writable_size = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                options_size,
                sizeof(initialized)
            )
        );
        std::memcpy(options, &initialized, writable_size);
    });
}

rt_status RT_CALL rt_adam_create(
    const rt_parameter_list* parameters,
    const rt_adam_options* options,
    rt_adam** output
) {
    return guard([&] {
        require_output(output);
        require_parameter_list(parameters);
        auto result = std::make_unique<rt_adam>(
            parameters->owner,
            parameters->value,
            checked_adam_options(options)
        );
        *output = result.release();
    });
}

void RT_CALL rt_adam_release(rt_adam* adam) {
    delete adam;
}

rt_status RT_CALL rt_adam_backend(
    const rt_adam* adam,
    rt_backend* output
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

rt_status RT_CALL rt_adam_step_count(
    const rt_adam* adam,
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

rt_status RT_CALL rt_adam_parameter_count(
    const rt_adam* adam,
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

rt_status RT_CALL rt_adam_state_storage(
    const rt_adam* adam,
    rt_adam_state_storage_kind* output
) {
    return guard([&] {
        require_adam(adam);
        if (output == nullptr) {
            throw std::invalid_argument(
                "Adam state-storage output must not be null"
            );
        }
        *output = c_adam_state_storage(
            adam->value.state_storage_kind()
        );
    });
}

rt_status RT_CALL rt_adam_state_page_size(
    const rt_adam* adam,
    uint64_t* output
) {
    return guard([&] {
        require_adam(adam);
        if (output == nullptr) {
            throw std::invalid_argument(
                "Adam state-page-size output must not be null"
            );
        }
        *output = checked_u64(
            adam->value.state_page_size(),
            "Adam state page size"
        );
    });
}

rt_status RT_CALL rt_adam_state_page_count(
    const rt_adam* adam,
    uint64_t* output
) {
    return guard([&] {
        require_adam(adam);
        if (output == nullptr) {
            throw std::invalid_argument(
                "Adam state-page-count output must not be null"
            );
        }
        *output = checked_u64(
            adam->value.state_page_count(),
            "Adam state page count"
        );
    });
}

rt_status RT_CALL rt_adam_state_payload_bytes(
    const rt_adam* adam,
    uint64_t* output
) {
    return guard([&] {
        require_adam(adam);
        if (output == nullptr) {
            throw std::invalid_argument(
                "Adam state-payload output must not be null"
            );
        }
        *output = checked_u64(
            adam->value.state_payload_bytes(),
            "Adam state payload bytes"
        );
    });
}

rt_status RT_CALL rt_adam_step(
    rt_adam* adam,
    rt_adam_step_stats* output_stats
) {
    return guard([&] {
        require_adam(adam);
        if (output_stats == nullptr) {
            throw std::invalid_argument(
                "Adam step statistics must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(rt_adam_step_stats, clip_scale) +
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

rt_status RT_CALL rt_adam_zero_gradients(
    const rt_adam* adam
) {
    return guard([&] {
        require_adam(adam);
        adam->value.zero_gradients();
    });
}

}  // extern "C"
