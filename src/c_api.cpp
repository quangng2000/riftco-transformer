#include "riftco_transformer/c_api.h"

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/core/tensor.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"
#include "riftco_transformer/compiler/cajal/multilinear_map.hpp"
#include "riftco_transformer/data/tokenizer.hpp"
#include "riftco_transformer/lowering/strategy.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/model/llama_mistral_transformer.hpp"
#include "riftco_transformer/model/lora.hpp"
#include "riftco_transformer/nn/loss.hpp"
#include "riftco_transformer/nn/parameter.hpp"
#include "riftco_transformer/optim/adam.hpp"
#include "riftco_transformer/programmed/program_augmented_model.hpp"
#include "riftco_transformer/stages/serving/kv_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

static_assert(sizeof(rt_transformer_config) == 64);
static_assert(sizeof(rt_llama_mistral_config) == 88);
static_assert(sizeof(rt_lora_config) == 40);
static_assert(sizeof(rt_quantized_memory_stats) == 96);
static_assert(sizeof(rt_decode_session_options) == 24);
static_assert(sizeof(rt_adam_options) == 48);
static_assert(sizeof(rt_adam_step_stats) == 32);
static_assert(sizeof(rt_tokenizer_options) == 32);
static_assert(sizeof(rt_bpe_merge_rule) == 12);
static_assert(sizeof(rt_program_augmented_model_config) == 64);
static_assert(sizeof(rt_program_input_layout) == 24);
static_assert(sizeof(rt_program_branch_config) == 64);
static_assert(sizeof(rt_neural_lowering_options) == 56);
static_assert(sizeof(rt_program_input_steering) == 48);
static_assert(sizeof(rt_program_augmented_forward_options) == 64);
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

struct TrainableOwnerState {
    virtual ~TrainableOwnerState() = default;

    std::atomic_size_t active_variables{0};
    std::atomic_size_t active_optimizers{0};
    std::atomic_size_t active_parameter_lists{0};
    std::atomic_size_t active_lora_parameter_lists{0};
    std::atomic_size_t active_decode_sessions{0};
    std::atomic_uint64_t parameter_epoch{0};
};

struct ModelState final : TrainableOwnerState {
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
};

struct LlamaMistralState final : TrainableOwnerState {
    LlamaMistralState(
        riftco_transformer::LlamaMistralConfig config,
        std::mt19937& random
    ) : value(std::move(config), random) {}

    riftco_transformer::LlamaMistralTransformer value;
};

struct rt_model {
    std::shared_ptr<ModelState> state;
};

struct rt_llama_mistral_model {
    std::shared_ptr<LlamaMistralState> state;
};

struct ProgramAugmentedState final : TrainableOwnerState {
    ProgramAugmentedState(
        riftco_transformer::programmed::ProgramAugmentedModelConfig config,
        std::optional<riftco_transformer::programmed::ProgramBranch> branch
    ) : value(std::move(config), std::move(branch)) {}

    riftco_transformer::programmed::ProgramAugmentedModel value;
};

struct rt_program_augmented_model {
    std::shared_ptr<ProgramAugmentedState> state;
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
        std::shared_ptr<TrainableOwnerState> model_owner,
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

    std::shared_ptr<TrainableOwnerState> owner;
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
        std::shared_ptr<TrainableOwnerState> model_owner,
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

    std::shared_ptr<TrainableOwnerState> owner;
    std::shared_ptr<VariableGraphState> graph;
    riftco_transformer::Variable value;
};

struct rt_adam {
    rt_adam(
        std::shared_ptr<TrainableOwnerState> model_owner,
        riftco_transformer::ParameterList parameters,
        riftco_transformer::AdamOptions options
    )
        : owner(std::move(model_owner)),
          parameter_identity(parameters),
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

    std::shared_ptr<TrainableOwnerState> owner;
    riftco_transformer::ParameterList parameter_identity;
    riftco_transformer::Adam value;
};

struct rt_multilinear_map {
    explicit rt_multilinear_map(
        riftco_transformer::compiler::cajal::MultilinearMap map
    ) : value(std::move(map)) {}

    riftco_transformer::compiler::cajal::MultilinearMap value;
};

struct rt_representation_trace {
    explicit rt_representation_trace(
        riftco_transformer::analysis::RepresentationTrace trace
    ) : value(std::move(trace)) {}

    riftco_transformer::analysis::RepresentationTrace value;
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

std::string checked_lowering_strategy(rt_lowering_strategy strategy) {
    switch (strategy) {
        case RT_LOWERING_STRATEGY_AUTO:
            return riftco_transformer::lowering::kAutomaticStrategy;
        case RT_LOWERING_STRATEGY_DENSE:
            return riftco_transformer::lowering::kDenseContractionStrategy;
        case RT_LOWERING_STRATEGY_LINEAR:
            return riftco_transformer::lowering::kLinearStrategy;
        case RT_LOWERING_STRATEGY_LINEAR_ATTENTION:
            return riftco_transformer::lowering::kLinearAttentionStrategy;
        default:
            throw std::invalid_argument(
                "unknown C API neural-lowering strategy"
            );
    }
}

riftco_transformer::lowering::CoefficientPrecision
checked_coefficient_precision(rt_coefficient_precision precision) {
    switch (precision) {
        case RT_COEFFICIENT_REQUIRE_EXACT_F32:
            return riftco_transformer::lowering::CoefficientPrecision::
                RequireExactFloat32;
        case RT_COEFFICIENT_ALLOW_ROUNDED_F32:
            return riftco_transformer::lowering::CoefficientPrecision::
                AllowRoundedFloat32;
        default:
            throw std::invalid_argument(
                "unknown C API coefficient precision"
            );
    }
}

riftco_transformer::lowering::CoefficientInitialization
checked_coefficient_initialization(
    rt_coefficient_initialization initialization
) {
    switch (initialization) {
        case RT_COEFFICIENT_INITIALIZATION_COMPILED:
            return riftco_transformer::lowering::
                CoefficientInitialization::Compiled;
        case RT_COEFFICIENT_INITIALIZATION_RANDOM_UNIFORM:
            return riftco_transformer::lowering::
                CoefficientInitialization::RandomUniform;
        default:
            throw std::invalid_argument(
                "unknown C API coefficient initialization"
            );
    }
}

bool checked_boolean(std::int32_t value, const char* description) {
    if (value != 0 && value != 1) {
        throw std::invalid_argument(
            std::string(description) + " must be zero or one"
        );
    }
    return value != 0;
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

riftco_transformer::LlamaMistralArchitecture
checked_llama_mistral_architecture(
    rt_llama_mistral_architecture architecture
) {
    switch (architecture) {
        case RT_LLAMA_MISTRAL_ARCHITECTURE_LLAMA:
            return riftco_transformer::LlamaMistralArchitecture::Llama;
        case RT_LLAMA_MISTRAL_ARCHITECTURE_MISTRAL:
            return riftco_transformer::LlamaMistralArchitecture::Mistral;
        default:
            throw std::invalid_argument(
                "unknown C API Llama/Mistral architecture"
            );
    }
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

void require_multilinear_map(const rt_multilinear_map* map) {
    if (map == nullptr) {
        throw std::invalid_argument(
            "multilinear-map handle must not be null"
        );
    }
}

void require_representation_trace(const rt_representation_trace* trace) {
    if (trace == nullptr) {
        throw std::invalid_argument(
            "representation-trace handle must not be null"
        );
    }
}

const riftco_transformer::analysis::NamedRepresentation&
checked_representation(
    const rt_representation_trace* trace,
    std::uint64_t index
) {
    require_representation_trace(trace);
    const std::size_t native_index = checked_size(
        index,
        "representation index"
    );
    const auto entries = trace->value.entries();
    if (native_index >= entries.size()) {
        throw std::out_of_range(
            "representation index is outside the trace"
        );
    }
    return entries[native_index];
}

std::vector<std::size_t> representation_shape(
    const riftco_transformer::analysis::NamedRepresentation& representation
) {
    std::vector<std::size_t> result = representation.leading_shape;
    result.push_back(representation.observations.columns);
    return result;
}

void require_model(const rt_model* model) {
    if (model == nullptr || model->state == nullptr) {
        throw std::invalid_argument(
            "model handle must not be null"
        );
    }
}

void require_llama_mistral_model(
    const rt_llama_mistral_model* model
) {
    if (model == nullptr || model->state == nullptr) {
        throw std::invalid_argument(
            "Llama/Mistral model handle must not be null"
        );
    }
}

void require_program_augmented_model(
    const rt_program_augmented_model* model
) {
    if (model == nullptr || model->state == nullptr) {
        throw std::invalid_argument(
            "program-augmented model handle must not be null"
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

riftco_transformer::LlamaMistralConfig checked_llama_mistral_config(
    const rt_llama_mistral_config* config
) {
    if (config == nullptr) {
        throw std::invalid_argument(
            "Llama/Mistral config must not be null"
        );
    }
    checked_structure_size(
        config->struct_size,
        sizeof(rt_llama_mistral_config),
        "Llama/Mistral config structure"
    );
    riftco_transformer::LlamaMistralConfig result{
        checked_llama_mistral_architecture(config->architecture),
        checked_size(config->vocabulary_size, "vocabulary size"),
        checked_size(config->maximum_context, "maximum context"),
        checked_size(config->model_width, "model width"),
        checked_size(config->query_head_count, "query head count"),
        checked_size(
            config->key_value_head_count,
            "key/value head count"
        ),
        checked_size(config->block_count, "block count"),
        checked_size(
            config->feed_forward_width,
            "feed-forward width"
        ),
        config->rms_norm_epsilon,
        config->rope_theta,
        std::nullopt,
    };
    if (config->sliding_window != 0) {
        result.sliding_window = checked_size(
            config->sliding_window,
            "sliding window"
        );
    }
    return riftco_transformer::validate_llama_mistral_config(
        std::move(result)
    );
}

riftco_transformer::programmed::ProgramAugmentedModelConfig
checked_program_augmented_model_config(
    const rt_program_augmented_model_config* config
) {
    if (config == nullptr) {
        throw std::invalid_argument(
            "program-augmented model config must not be null"
        );
    }
    checked_structure_size(
        config->struct_size,
        sizeof(rt_program_augmented_model_config),
        "program-augmented model config"
    );
    riftco_transformer::programmed::ProgramAugmentedModelConfig result{
        checked_size(config->vocabulary_size, "vocabulary size"),
        checked_size(config->context_length, "context length"),
        checked_size(config->model_width, "model width"),
        checked_size(config->head_count, "head count"),
        checked_size(
            config->attention_branch_count,
            "attention branch count"
        ),
        checked_size(
            config->feed_forward_width,
            "feed-forward width"
        ),
        checked_full_sequence_attention(config->attention_kind),
        config->random_seed,
    };
    result.validate();
    return result;
}

riftco_transformer::lowering::NeuralLoweringConfig
checked_neural_lowering_options(
    const rt_neural_lowering_options* options
) {
    if (options == nullptr) {
        throw std::invalid_argument(
            "neural-lowering options must not be null"
        );
    }
    checked_structure_size(
        options->struct_size,
        sizeof(rt_neural_lowering_options),
        "neural-lowering options"
    );
    if (options->reserved != 0 || options->reserved2 != 0) {
        throw std::invalid_argument(
            "neural-lowering reserved fields must be zero"
        );
    }
    riftco_transformer::lowering::NeuralLoweringConfig result;
    result.strategy = checked_lowering_strategy(options->strategy);
    result.precision = checked_coefficient_precision(options->precision);
    result.initialization = checked_coefficient_initialization(
        options->initialization
    );
    result.trainable = checked_boolean(
        options->trainable,
        "neural-lowering trainable"
    );
    result.backend = riftco_transformer::ExecutionBackend::Cpu;
    result.seed = options->random_seed;
    result.random_scale = options->random_scale;
    result.max_coefficient_elements = checked_size(
        options->max_coefficient_elements,
        "neural-lowering coefficient limit"
    );
    if (options->attention_query_axis < -1) {
        throw std::invalid_argument(
            "neural-lowering attention query axis must be -1 or nonnegative"
        );
    }
    if (options->attention_query_axis >= 0) {
        result.attention_query_axis = checked_size(
            static_cast<std::uint64_t>(options->attention_query_axis),
            "neural-lowering attention query axis"
        );
    }
    result.validate();
    return result;
}

riftco_transformer::programmed::ProgramInputSource
checked_program_input_source(rt_program_input_source source) {
    switch (source) {
        case RT_PROGRAM_INPUT_WHOLE_SOURCE:
            return riftco_transformer::programmed::ProgramInputSource::
                WholeSource;
        case RT_PROGRAM_INPUT_SOURCE_POSITION:
            return riftco_transformer::programmed::ProgramInputSource::
                SourcePosition;
        default:
            throw std::invalid_argument(
                "unknown C API program input source"
            );
    }
}

riftco_transformer::programmed::ProgramBranch checked_program_branch(
    const rt_multilinear_map* map,
    const rt_program_branch_config* branch,
    const rt_neural_lowering_options* lowering
) {
    require_multilinear_map(map);
    if (branch == nullptr) {
        throw std::invalid_argument("program branch config must not be null");
    }
    checked_structure_size(
        branch->struct_size,
        sizeof(rt_program_branch_config),
        "program branch config"
    );
    const std::size_t input_count = checked_size(
        branch->input_count,
        "program input count"
    );
    if (input_count != 0 && branch->inputs == nullptr) {
        throw std::invalid_argument(
            "program input layouts must not be null"
        );
    }
    riftco_transformer::programmed::SequenceCoreConfig core;
    core.source_length = checked_size(
        branch->source_length,
        "program source length"
    );
    core.output_length = checked_size(
        branch->output_length,
        "program output length"
    );
    core.input_projection_bias = checked_boolean(
        branch->input_projection_bias,
        "program input projection bias"
    );
    core.inputs.reserve(input_count);
    core.input_projection_groups.reserve(input_count);
    for (std::size_t index = 0; index < input_count; ++index) {
        const rt_program_input_layout& input = branch->inputs[index];
        if (input.reserved != 0) {
            throw std::invalid_argument(
                "program input layout reserved field must be zero"
            );
        }
        core.inputs.push_back({
            checked_program_input_source(input.source),
            checked_size(input.position, "program input position"),
        });
        core.input_projection_groups.push_back(checked_size(
            input.projection_group,
            "program input projection group"
        ));
    }
    auto lowered = riftco_transformer::lowering::lower_to_neural(
        map->value,
        checked_neural_lowering_options(lowering)
    );
    return {
        checked_size(branch->source_offset, "program source offset"),
        checked_size(branch->target_offset, "program target offset"),
        std::move(core),
        std::move(lowered),
        checked_boolean(branch->merge_bias, "program merge bias"),
    };
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

bool same_parameter_identity(
    const riftco_transformer::ParameterList& first,
    const riftco_transformer::ParameterList& second
) {
    if (first.size() != second.size()) {
        return false;
    }
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (first[index].name != second[index].name ||
            first[index].parameter != second[index].parameter) {
            return false;
        }
    }
    return true;
}

struct ParameterReplacement {
    riftco_transformer::Parameter* parameter;
    riftco_transformer::Tensor value;
};

std::vector<ParameterReplacement> prepare_parameter_replacements(
    const riftco_transformer::ParameterList& parameters,
    const float* values,
    std::uint64_t value_count,
    const char* description
) {
    const std::size_t expected =
        riftco_transformer::parameter_count(parameters);
    const std::size_t count = checked_size(value_count, description);
    if (count != expected) {
        throw std::invalid_argument(
            std::string(description) +
            " must exactly match the parameter list"
        );
    }
    const std::vector<float> checked = checked_f32_values(
        values,
        value_count,
        description
    );

    std::vector<ParameterReplacement> replacements;
    replacements.reserve(parameters.size());
    std::size_t offset = 0;
    for (const auto& named_parameter : parameters) {
        auto* parameter = named_parameter.parameter;
        if (parameter == nullptr) {
            throw std::invalid_argument(
                "parameter list contains a null parameter"
            );
        }
        const auto& old_value = parameter->value();
        const std::size_t parameter_size = old_value.numel();
        replacements.push_back({
            parameter,
            riftco_transformer::Tensor(
                old_value.shape(),
                std::vector<float>(
                    checked.begin() + static_cast<std::ptrdiff_t>(offset),
                    checked.begin() + static_cast<std::ptrdiff_t>(
                        offset + parameter_size
                    )
                ),
                old_value.backend()
            ),
        });
        offset += parameter_size;
    }
    return replacements;
}

void commit_parameter_replacements(
    std::vector<ParameterReplacement> replacements
) {
    // All validation and allocations have completed. Same-shape,
    // same-backend replacement is non-allocating and clears gradients.
    for (auto& replacement : replacements) {
        replacement.parameter->set_value(std::move(replacement.value));
    }
}

riftco_transformer::programmed::ProgramAugmentedForwardOptions
checked_program_augmented_forward_options(
    const rt_program_augmented_forward_options* options
) {
    if (options == nullptr) {
        return {};
    }
    checked_structure_size(
        options->struct_size,
        sizeof(rt_program_augmented_forward_options),
        "program-augmented forward options"
    );
    if (options->reserved != 0) {
        throw std::invalid_argument(
            "program-augmented forward reserved field must be zero"
        );
    }
    riftco_transformer::programmed::ProgramAugmentedForwardOptions result;
    result.capture_representations = checked_boolean(
        options->capture_representations,
        "representation capture"
    );
    result.batch_roll_attention = checked_boolean(
        options->batch_roll_learned_attention,
        "learned-attention batch roll"
    );
    result.batch_roll_shift = checked_size(
        options->batch_roll_shift,
        "batch-roll shift"
    );

    const std::size_t steering_count = checked_size(
        options->steering_count,
        "program steering count"
    );
    if (steering_count != 0 && options->steering == nullptr) {
        throw std::invalid_argument(
            "program steering records must not be null"
        );
    }
    result.program.steering.reserve(steering_count);
    for (std::size_t index = 0; index < steering_count; ++index) {
        const rt_program_input_steering& steering = options->steering[index];
        result.program.steering.push_back({
            checked_size(steering.input_index, "program steering input"),
            {checked_size(steering.position, "program steering position")},
            checked_f32_values(
                steering.scales,
                steering.scale_count,
                "program steering scales"
            ),
            checked_f32_values(
                steering.offsets,
                steering.offset_count,
                "program steering offsets"
            ),
        });
    }

    const std::size_t ablated_count = checked_size(
        options->ablated_program_input_count,
        "ablated program input count"
    );
    if (ablated_count != 0 && options->ablated_program_inputs == nullptr) {
        throw std::invalid_argument(
            "ablated program input indices must not be null"
        );
    }
    std::vector<std::size_t> ablated_inputs;
    ablated_inputs.reserve(ablated_count);
    for (std::size_t index = 0; index < ablated_count; ++index) {
        ablated_inputs.push_back(checked_size(
            options->ablated_program_inputs[index],
            "ablated program input index"
        ));
    }
    const bool ablate_output = checked_boolean(
        options->ablate_program_output,
        "program-output ablation"
    );
    if (!ablated_inputs.empty() || ablate_output) {
        result.program.ablation =
            riftco_transformer::programmed::BatchRollAblation{
                std::move(ablated_inputs),
                ablate_output,
                result.batch_roll_shift,
            };
    }
    return result;
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

constexpr std::array<std::uint8_t, 8> kPackedModelStateMagic{
    'R', 'T', 'N', 'F', '4', 'S', '1', '\0'
};

void append_u32_le(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_u64_le(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_f32_le(std::vector<std::uint8_t>& output, float value) {
    append_u32_le(output, std::bit_cast<std::uint32_t>(value));
}

void append_f32_values_le(
    std::vector<std::uint8_t>& output,
    std::span<const float> values
) {
    for (const float value : values) {
        append_f32_le(output, value);
    }
}

std::vector<std::uint8_t> packed_model_state_bytes(
    const riftco_transformer::DecoderOnlyTransformer& model
) {
    const auto weights = model.packed_linear_weight_state();
    std::vector<std::uint8_t> output(
        kPackedModelStateMagic.begin(),
        kPackedModelStateMagic.end()
    );
    append_u64_le(
        output,
        checked_u64(weights.size(), "packed model weight count")
    );
    for (const auto& weight : weights) {
        const auto& payload = weight.payload;
        const bool double_quantized =
            payload.double_quantized_scales.has_value();
        const auto scale_code_count = double_quantized
                                          ? payload.double_quantized_scales
                                                ->scale_codes.size()
                                          : 0U;
        const auto second_scale_count = double_quantized
                                            ? payload
                                                  .double_quantized_scales
                                                  ->second_level_scales.size()
                                            : 0U;
        const auto scale_block_size = double_quantized
                                          ? payload.double_quantized_scales
                                                ->scale_block_size
                                          : 0U;
        const float offset = double_quantized
                                 ? payload.double_quantized_scales->offset
                                 : 0.0F;
        append_u64_le(output, checked_u64(weight.shape.size(), "NF4 rank"));
        append_u64_le(
            output,
            checked_u64(weight.block_size, "NF4 block size")
        );
        append_u64_le(
            output,
            checked_u64(payload.packed_codes.size(), "NF4 packed code count")
        );
        append_u64_le(
            output,
            checked_u64(payload.block_scales.size(), "NF4 block scale count")
        );
        append_u64_le(
            output,
            checked_u64(scale_code_count, "NF4 scale code count")
        );
        append_u64_le(
            output,
            checked_u64(second_scale_count, "NF4 second-level scale count")
        );
        append_u64_le(
            output,
            checked_u64(scale_block_size, "NF4 scale block size")
        );
        append_f32_le(output, offset);
        append_u32_le(output, double_quantized ? 1U : 0U);
        for (const std::size_t dimension : weight.shape) {
            append_u64_le(output, checked_u64(dimension, "NF4 dimension"));
        }
        output.insert(
            output.end(),
            payload.packed_codes.begin(),
            payload.packed_codes.end()
        );
        append_f32_values_le(output, payload.block_scales);
        if (double_quantized) {
            const auto& nested = *payload.double_quantized_scales;
            output.insert(
                output.end(),
                nested.scale_codes.begin(),
                nested.scale_codes.end()
            );
            append_f32_values_le(output, nested.second_level_scales);
        }
    }
    return output;
}

class PackedModelStateReader {
public:
    explicit PackedModelStateReader(std::span<const std::uint8_t> bytes)
        : remaining_(bytes) {}

    [[nodiscard]] std::span<const std::uint8_t> take(
        std::size_t count,
        const char* description
    ) {
        if (count > remaining_.size()) {
            throw std::invalid_argument(
                std::string("truncated packed model state: ") + description
            );
        }
        const auto result = remaining_.first(count);
        remaining_ = remaining_.subspan(count);
        return result;
    }

    [[nodiscard]] std::uint32_t u32(const char* description) {
        const auto bytes = take(4, description);
        std::uint32_t value = 0;
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            value |= static_cast<std::uint32_t>(bytes[index]) << (index * 8U);
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64(const char* description) {
        const auto bytes = take(8, description);
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
        }
        return value;
    }

    [[nodiscard]] float f32(const char* description) {
        return std::bit_cast<float>(u32(description));
    }

    [[nodiscard]] bool empty() const noexcept { return remaining_.empty(); }
    [[nodiscard]] std::size_t remaining_size() const noexcept {
        return remaining_.size();
    }

private:
    std::span<const std::uint8_t> remaining_;
};

std::vector<riftco_transformer::PackedLinearWeightState>
parse_packed_model_state(std::span<const std::uint8_t> bytes) {
    PackedModelStateReader reader(bytes);
    const auto magic = reader.take(
        kPackedModelStateMagic.size(),
        "magic"
    );
    if (!std::equal(
            magic.begin(), magic.end(), kPackedModelStateMagic.begin()
        )) {
        throw std::invalid_argument("unknown packed model state format");
    }
    const std::size_t count = checked_size(
        reader.u64("weight count"),
        "packed model weight count"
    );
    constexpr std::size_t minimum_entry_bytes = 80;
    if (count == 0 || count > reader.remaining_size() / minimum_entry_bytes) {
        throw std::invalid_argument(
            "packed model state weight count is inconsistent with its size"
        );
    }
    std::vector<riftco_transformer::PackedLinearWeightState> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t rank = checked_size(
            reader.u64("rank"), "NF4 rank"
        );
        if (rank != 2) {
            throw std::invalid_argument(
                "packed decoder Linear weights must have rank two"
            );
        }
        const std::size_t block_size = checked_size(
            reader.u64("block size"), "NF4 block size"
        );
        const std::size_t packed_count = checked_size(
            reader.u64("packed code count"), "NF4 packed code count"
        );
        const std::size_t block_scale_count = checked_size(
            reader.u64("block scale count"), "NF4 block scale count"
        );
        const std::size_t scale_code_count = checked_size(
            reader.u64("scale code count"), "NF4 scale code count"
        );
        const std::size_t second_scale_count = checked_size(
            reader.u64("second-level scale count"),
            "NF4 second-level scale count"
        );
        const std::size_t scale_block_size = checked_size(
            reader.u64("scale block size"), "NF4 scale block size"
        );
        const std::uint32_t offset_bits = reader.u32("scale offset");
        const float offset = std::bit_cast<float>(offset_bits);
        const std::uint32_t flags = reader.u32("flags");
        if (flags > 1U) {
            throw std::invalid_argument("packed model state has unknown flags");
        }
        const bool double_quantized = flags == 1U;
        if ((!double_quantized &&
             (scale_code_count != 0 || second_scale_count != 0 ||
              scale_block_size != 0 || offset_bits != 0U)) ||
            (double_quantized &&
             (block_scale_count != 0 || scale_block_size == 0))) {
            throw std::invalid_argument(
                "packed model state has inconsistent NF4 scale metadata"
            );
        }

        std::size_t required_payload_bytes = rank * sizeof(std::uint64_t);
        const auto add_required = [&](std::size_t value,
                                      std::size_t element_size) {
            if (value >
                (std::numeric_limits<std::size_t>::max() -
                 required_payload_bytes) /
                    element_size) {
                throw std::overflow_error(
                    "packed model state entry size exceeds addressable memory"
                );
            }
            required_payload_bytes += value * element_size;
        };
        add_required(packed_count, sizeof(std::uint8_t));
        add_required(block_scale_count, sizeof(float));
        add_required(scale_code_count, sizeof(std::uint8_t));
        add_required(second_scale_count, sizeof(float));
        if (required_payload_bytes > reader.remaining_size()) {
            throw std::invalid_argument(
                "truncated packed model state: weight payload"
            );
        }

        riftco_transformer::QuantizedWeight::Shape shape;
        shape.reserve(rank);
        for (std::size_t dimension = 0; dimension < rank; ++dimension) {
            shape.push_back(checked_size(
                reader.u64("dimension"), "NF4 dimension"
            ));
        }
        const auto packed = reader.take(packed_count, "packed codes");
        riftco_transformer::Nf4Payload payload;
        payload.packed_codes.assign(packed.begin(), packed.end());
        payload.block_scales.reserve(block_scale_count);
        for (std::size_t scale = 0; scale < block_scale_count; ++scale) {
            payload.block_scales.push_back(reader.f32("block scale"));
        }
        if (double_quantized) {
            riftco_transformer::Nf4DoubleQuantizedScales nested;
            const auto scale_codes = reader.take(
                scale_code_count, "scale codes"
            );
            nested.scale_codes.assign(scale_codes.begin(), scale_codes.end());
            nested.second_level_scales.reserve(second_scale_count);
            for (std::size_t scale = 0; scale < second_scale_count; ++scale) {
                nested.second_level_scales.push_back(
                    reader.f32("second-level scale")
                );
            }
            nested.scale_block_size = scale_block_size;
            nested.offset = offset;
            payload.double_quantized_scales.emplace(std::move(nested));
        }
        result.push_back({
            std::move(shape), block_size, std::move(payload)
        });
    }
    if (!reader.empty()) {
        throw std::invalid_argument(
            "packed model state contains trailing bytes"
        );
    }
    return result;
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

rt_status RT_CALL rt_multilinear_map_create_dense(
    const uint64_t* input_dimensions,
    uint64_t input_count,
    uint64_t output_dimension,
    const double* coefficients,
    uint64_t coefficient_count,
    rt_multilinear_map** output
) {
    return guard([&] {
        require_output(output);
        const auto dimensions = checked_shape(
            input_dimensions,
            input_count
        );
        const std::size_t count = checked_size(
            coefficient_count,
            "multilinear coefficient count"
        );
        if (count != 0 && coefficients == nullptr) {
            throw std::invalid_argument(
                "multilinear coefficients must not be null"
            );
        }
        std::vector<double> copied;
        if (count != 0) {
            copied.assign(coefficients, coefficients + count);
        }
        auto result = std::make_unique<rt_multilinear_map>(
            riftco_transformer::compiler::cajal::MultilinearMap(
                dimensions,
                checked_size(
                    output_dimension,
                    "multilinear output dimension"
                ),
                std::move(copied)
            )
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_multilinear_map_create_sparse(
    const uint64_t* input_dimensions,
    uint64_t input_count,
    uint64_t output_dimension,
    const uint64_t* flat_indices,
    const double* values,
    uint64_t nonzero_count,
    rt_multilinear_map** output
) {
    return guard([&] {
        require_output(output);
        const auto dimensions = checked_shape(
            input_dimensions,
            input_count
        );
        const std::size_t count = checked_size(
            nonzero_count,
            "multilinear sparse nonzero count"
        );
        if (count != 0 && (flat_indices == nullptr || values == nullptr)) {
            throw std::invalid_argument(
                "multilinear sparse indices and values must not be null"
            );
        }
        std::vector<std::size_t> indices;
        std::vector<double> copied_values;
        indices.reserve(count);
        copied_values.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            indices.push_back(checked_size(
                flat_indices[index],
                "multilinear sparse flat index"
            ));
            copied_values.push_back(values[index]);
        }
        auto result = std::make_unique<rt_multilinear_map>(
            riftco_transformer::compiler::cajal::MultilinearMap::from_sparse(
                dimensions,
                checked_size(
                    output_dimension,
                    "multilinear output dimension"
                ),
                indices,
                copied_values
            )
        );
        *output = result.release();
    });
}

void RT_CALL rt_multilinear_map_release(rt_multilinear_map* map) {
    delete map;
}

rt_status RT_CALL rt_program_augmented_model_config_init(
    rt_program_augmented_model_config* config,
    uint64_t config_size
) {
    return guard([&] {
        if (config == nullptr) {
            throw std::invalid_argument(
                "program-augmented model config must not be null"
            );
        }
        checked_structure_size(
            config_size,
            sizeof(rt_program_augmented_model_config),
            "program-augmented model config"
        );
        *config = {
            config_size,
            512,
            128,
            64,
            4,
            2,
            256,
            42,
            RT_FULL_SEQUENCE_ATTENTION_MATERIALIZED,
        };
    });
}

rt_status RT_CALL rt_program_branch_config_init(
    rt_program_branch_config* config,
    uint64_t config_size
) {
    return guard([&] {
        if (config == nullptr) {
            throw std::invalid_argument(
                "program branch config must not be null"
            );
        }
        checked_structure_size(
            config_size,
            sizeof(rt_program_branch_config),
            "program branch config"
        );
        *config = {
            config_size,
            0,
            1,
            1,
            1,
            nullptr,
            0,
            0,
            0,
        };
    });
}

rt_status RT_CALL rt_neural_lowering_options_init(
    rt_neural_lowering_options* options,
    uint64_t options_size
) {
    return guard([&] {
        if (options == nullptr) {
            throw std::invalid_argument(
                "neural-lowering options must not be null"
            );
        }
        checked_structure_size(
            options_size,
            sizeof(rt_neural_lowering_options),
            "neural-lowering options"
        );
        const riftco_transformer::lowering::NeuralLoweringConfig defaults;
        *options = {
            options_size,
            RT_LOWERING_STRATEGY_AUTO,
            RT_COEFFICIENT_REQUIRE_EXACT_F32,
            RT_COEFFICIENT_INITIALIZATION_COMPILED,
            defaults.trainable ? 1 : 0,
            defaults.seed,
            0,
            defaults.random_scale,
            0,
            checked_u64(
                defaults.max_coefficient_elements,
                "neural-lowering coefficient limit"
            ),
            -1,
        };
    });
}

rt_status RT_CALL rt_program_augmented_forward_options_init(
    rt_program_augmented_forward_options* options,
    uint64_t options_size
) {
    return guard([&] {
        if (options == nullptr) {
            throw std::invalid_argument(
                "program-augmented forward options must not be null"
            );
        }
        checked_structure_size(
            options_size,
            sizeof(rt_program_augmented_forward_options),
            "program-augmented forward options"
        );
        *options = {
            options_size,
            0,
            0,
            1,
            nullptr,
            0,
            nullptr,
            0,
            0,
            0,
        };
    });
}

rt_status RT_CALL rt_program_augmented_model_create(
    const rt_program_augmented_model_config* config,
    const rt_multilinear_map* map,
    const rt_program_branch_config* branch,
    const rt_neural_lowering_options* lowering,
    rt_program_augmented_model** output
) {
    return guard([&] {
        require_output(output);
        const bool has_map = map != nullptr;
        if (has_map != (branch != nullptr) ||
            has_map != (lowering != nullptr)) {
            throw std::invalid_argument(
                "program map, branch config, and lowering options must be "
                "provided together"
            );
        }
        auto native_config = checked_program_augmented_model_config(config);
        std::optional<riftco_transformer::programmed::ProgramBranch>
            native_branch;
        const riftco_transformer::ScopedExecutionBackend construction_backend(
            riftco_transformer::ExecutionBackend::Cpu
        );
        if (has_map) {
            native_branch.emplace(
                checked_program_branch(map, branch, lowering)
            );
        }
        auto state = std::make_shared<ProgramAugmentedState>(
            std::move(native_config),
            std::move(native_branch)
        );
        auto result = std::make_unique<rt_program_augmented_model>(
            rt_program_augmented_model{std::move(state)}
        );
        *output = result.release();
    });
}

void RT_CALL rt_program_augmented_model_release(
    rt_program_augmented_model* model
) {
    delete model;
}

rt_status RT_CALL rt_program_augmented_model_to(
    rt_program_augmented_model* model,
    rt_backend backend
) {
    return guard([&] {
        require_program_augmented_model(model);
        if (model->state->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot move a program-augmented model while variable "
                "graphs are alive"
            );
        }
        if (model->state->active_optimizers.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot move a program-augmented model while optimizers "
                "are alive"
            );
        }
        require_epoch_increment_available(*model->state);
        model->state->value.to(checked_backend(backend));
        model->state->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
    });
}

rt_status RT_CALL rt_program_augmented_model_backend(
    const rt_program_augmented_model* model,
    rt_backend* output
) {
    return guard([&] {
        require_program_augmented_model(model);
        if (output == nullptr) {
            throw std::invalid_argument("backend output must not be null");
        }
        *output = c_backend(model->state->value.backend());
    });
}

rt_status RT_CALL rt_program_augmented_model_has_program(
    const rt_program_augmented_model* model,
    int32_t* output
) {
    return guard([&] {
        require_program_augmented_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "program-presence output must not be null"
            );
        }
        *output = model->state->value.has_program() ? 1 : 0;
    });
}

rt_status RT_CALL rt_program_augmented_model_parameters(
    rt_program_augmented_model* model,
    rt_parameter_list** output
) {
    return guard([&] {
        require_output(output);
        require_program_augmented_model(model);
        auto result = std::make_unique<rt_parameter_list>(
            model->state,
            model->state->value.parameters(),
            false
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_program_augmented_model_forward(
    const rt_program_augmented_model* model,
    const uint32_t* token_ids,
    uint64_t token_count,
    uint64_t batch_size,
    uint64_t context_length,
    const rt_program_augmented_forward_options* options,
    rt_variable** output_logits,
    rt_representation_trace** output_trace
) {
    return guard([&] {
        require_output(output_logits);
        if (output_trace != nullptr) {
            *output_trace = nullptr;
        }
        require_program_augmented_model(model);
        const auto configured =
            checked_program_augmented_forward_options(options);
        if (configured.capture_representations && output_trace == nullptr) {
            throw std::invalid_argument(
                "capturing representations requires a trace output"
            );
        }
        const std::size_t native_batch = checked_size(
            batch_size,
            "program-augmented batch size"
        );
        const std::size_t native_context = checked_size(
            context_length,
            "program-augmented context length"
        );
        const auto values = checked_token_ids(
            token_ids,
            token_count,
            "program-augmented token IDs"
        );
        if (values.size() != checked_product(
                native_batch,
                native_context,
                "program-augmented token shape"
            )) {
            throw std::invalid_argument(
                "token count must match batch and context sizes"
            );
        }
        if (native_context != model->state->value.config().context_length) {
            throw std::invalid_argument(
                "context length must match the program-augmented model"
            );
        }
        auto graph = std::make_shared<VariableGraphState>(
            model->state->parameter_epoch.load(std::memory_order_relaxed)
        );
        auto forward_result = model->state->value.forward(
            values,
            native_batch,
            configured
        );
        auto logits = std::make_unique<rt_variable>(
            model->state,
            std::move(graph),
            std::move(forward_result.logits)
        );
        std::unique_ptr<rt_representation_trace> trace;
        if (configured.capture_representations) {
            trace = std::make_unique<rt_representation_trace>(
                std::move(forward_result.representations)
            );
        }
        *output_logits = logits.release();
        if (output_trace != nullptr) {
            *output_trace = trace.release();
        }
    });
}

void RT_CALL rt_representation_trace_release(
    rt_representation_trace* trace
) {
    delete trace;
}

rt_status RT_CALL rt_representation_trace_count(
    const rt_representation_trace* trace,
    uint64_t* output
) {
    return guard([&] {
        require_representation_trace(trace);
        if (output == nullptr) {
            throw std::invalid_argument(
                "representation-count output must not be null"
            );
        }
        *output = checked_u64(
            trace->value.entries().size(),
            "representation count"
        );
    });
}

rt_status RT_CALL rt_representation_trace_name(
    const rt_representation_trace* trace,
    uint64_t index,
    char* output_name,
    uint64_t name_capacity,
    uint64_t* required_capacity
) {
    return guard([&] {
        const auto& representation = checked_representation(trace, index);
        if (required_capacity == nullptr) {
            throw std::invalid_argument(
                "required representation-name capacity must not be null"
            );
        }
        if (representation.name.size() ==
            std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error(
                "representation name capacity overflows size_t"
            );
        }
        const std::size_t required = representation.name.size() + 1;
        *required_capacity = checked_u64(
            required,
            "representation name capacity"
        );
        const std::size_t capacity = checked_size(
            name_capacity,
            "representation name capacity"
        );
        if (capacity == 0 && output_name == nullptr) {
            return;
        }
        if (output_name == nullptr) {
            throw std::invalid_argument(
                "representation name output must not be null"
            );
        }
        if (capacity < required) {
            throw std::invalid_argument(
                "representation name output capacity is too small"
            );
        }
        std::copy(
            representation.name.c_str(),
            representation.name.c_str() + required,
            output_name
        );
    });
}

rt_status RT_CALL rt_representation_trace_rank(
    const rt_representation_trace* trace,
    uint64_t index,
    uint64_t* output
) {
    return guard([&] {
        if (output == nullptr) {
            throw std::invalid_argument(
                "representation-rank output must not be null"
            );
        }
        *output = checked_u64(
            representation_shape(checked_representation(trace, index)).size(),
            "representation rank"
        );
    });
}

rt_status RT_CALL rt_representation_trace_shape(
    const rt_representation_trace* trace,
    uint64_t index,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
) {
    return guard([&] {
        const auto shape = representation_shape(
            checked_representation(trace, index)
        );
        const std::size_t capacity = checked_size(
            dimension_capacity,
            "representation shape capacity"
        );
        if (capacity < shape.size()) {
            throw std::invalid_argument(
                "representation shape capacity is too small"
            );
        }
        if (!shape.empty() && output_dimensions == nullptr) {
            throw std::invalid_argument(
                "representation shape output must not be null"
            );
        }
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            output_dimensions[axis] = checked_u64(
                shape[axis],
                "representation dimension"
            );
        }
    });
}

rt_status RT_CALL rt_representation_trace_numel(
    const rt_representation_trace* trace,
    uint64_t index,
    uint64_t* output
) {
    return guard([&] {
        if (output == nullptr) {
            throw std::invalid_argument(
                "representation-element output must not be null"
            );
        }
        *output = checked_u64(
            checked_representation(trace, index).observations.values.size(),
            "representation element count"
        );
    });
}

rt_status RT_CALL rt_representation_trace_copy_to_host_f32(
    const rt_representation_trace* trace,
    uint64_t index,
    float* output_values,
    uint64_t value_capacity
) {
    return guard([&] {
        const auto& values = checked_representation(
            trace,
            index
        ).observations.values;
        const std::size_t capacity = checked_size(
            value_capacity,
            "representation value capacity"
        );
        if (capacity < values.size()) {
            throw std::invalid_argument(
                "representation value capacity is too small"
            );
        }
        if (!values.empty() && output_values == nullptr) {
            throw std::invalid_argument(
                "representation value output must not be null"
            );
        }
        std::copy(values.begin(), values.end(), output_values);
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

rt_status RT_CALL rt_llama_mistral_config_init(
    rt_llama_mistral_config* config,
    uint64_t config_size
) {
    return guard([&] {
        if (config == nullptr) {
            throw std::invalid_argument(
                "Llama/Mistral config must not be null"
            );
        }
        checked_structure_size(
            config_size,
            sizeof(rt_llama_mistral_config),
            "Llama/Mistral config structure"
        );
        *config = {
            config_size,
            RT_LLAMA_MISTRAL_ARCHITECTURE_LLAMA,
            5489U,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1.0e-5F,
            10000.0F,
            0,
        };
    });
}

rt_status RT_CALL rt_llama_mistral_model_create(
    const rt_llama_mistral_config* config,
    rt_llama_mistral_model** output
) {
    return guard([&] {
        require_output(output);
        auto native_config = checked_llama_mistral_config(config);
        std::mt19937 random(config->random_seed);
        const riftco_transformer::ScopedExecutionBackend
            construction_backend(
                riftco_transformer::ExecutionBackend::Cpu
            );
        auto state = std::make_shared<LlamaMistralState>(
            std::move(native_config),
            random
        );
        auto result = std::make_unique<rt_llama_mistral_model>(
            rt_llama_mistral_model{std::move(state)}
        );
        *output = result.release();
    });
}

void RT_CALL rt_llama_mistral_model_release(
    rt_llama_mistral_model* model
) {
    delete model;
}

rt_status RT_CALL rt_llama_mistral_model_to(
    rt_llama_mistral_model* model,
    rt_backend backend
) {
    return guard([&] {
        require_llama_mistral_model(model);
        if (model->state->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot move a Llama/Mistral model while variable graphs "
                "are alive"
            );
        }
        if (model->state->active_optimizers.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot move a Llama/Mistral model while optimizers are alive"
            );
        }
        require_epoch_increment_available(*model->state);
        model->state->value.to(checked_backend(backend));
        model->state->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
    });
}

rt_status RT_CALL rt_llama_mistral_model_backend(
    const rt_llama_mistral_model* model,
    rt_backend* output
) {
    return guard([&] {
        require_llama_mistral_model(model);
        if (output == nullptr) {
            throw std::invalid_argument("backend output must not be null");
        }
        *output = c_backend(model->state->value.backend());
    });
}

rt_status RT_CALL rt_llama_mistral_model_forward(
    const rt_llama_mistral_model* model,
    const uint32_t* token_ids,
    uint64_t token_count,
    uint64_t batch_size,
    uint64_t sequence_length,
    rt_variable** output
) {
    return guard([&] {
        require_output(output);
        require_llama_mistral_model(model);
        const std::size_t native_batch = checked_size(
            batch_size,
            "batch size"
        );
        const std::size_t native_sequence = checked_size(
            sequence_length,
            "sequence length"
        );
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

rt_status RT_CALL rt_llama_mistral_model_parameters(
    rt_llama_mistral_model* model,
    rt_parameter_list** output
) {
    return guard([&] {
        require_output(output);
        require_llama_mistral_model(model);
        auto result = std::make_unique<rt_parameter_list>(
            model->state,
            model->state->value.parameters(),
            false
        );
        *output = result.release();
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

rt_status RT_CALL rt_model_packed_state_size(
    const rt_model* model,
    uint64_t* output
) {
    return guard([&] {
        require_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "packed model state size output must not be null"
            );
        }
        const auto bytes = packed_model_state_bytes(model->state->value);
        *output = checked_u64(bytes.size(), "packed model state size");
    });
}

rt_status RT_CALL rt_model_packed_state_copy(
    const rt_model* model,
    uint8_t* output,
    uint64_t output_size
) {
    return guard([&] {
        require_model(model);
        const auto bytes = packed_model_state_bytes(model->state->value);
        const std::size_t native_size = checked_size(
            output_size, "packed model state output size"
        );
        if (native_size != bytes.size()) {
            throw std::invalid_argument(
                "packed model state output size does not match"
            );
        }
        if (output == nullptr && native_size != 0) {
            throw std::invalid_argument(
                "packed model state output must not be null"
            );
        }
        std::copy(bytes.begin(), bytes.end(), output);
    });
}

rt_status RT_CALL rt_model_packed_state_load(
    rt_model* model,
    const uint8_t* state,
    uint64_t state_size
) {
    return guard([&] {
        require_model(model);
        const std::size_t native_size = checked_size(
            state_size, "packed model state size"
        );
        if (state == nullptr && native_size != 0) {
            throw std::invalid_argument(
                "packed model state input must not be null"
            );
        }
        if (model->state->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot restore packed model state while variable graphs are alive"
            );
        }
        require_no_active_decode_sessions(
            *model->state,
            "restore packed model state"
        );
        const auto parsed = parse_packed_model_state(
            std::span<const std::uint8_t>(state, native_size)
        );
        model->state->value.load_packed_linear_weight_state(parsed);
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

rt_status RT_CALL rt_model_frozen_parameters_load_from_host_f32(
    rt_model* model,
    const rt_adam* adapter_optimizer,
    const float* values,
    uint64_t value_count
) {
    return guard([&] {
        require_model(model);
        require_adam(adapter_optimizer);
        if (adapter_optimizer->owner != model->state) {
            throw std::invalid_argument(
                "adapter Adam and model must share the same owner"
            );
        }
        if (!model->state->value.has_lora()) {
            throw std::invalid_argument(
                "frozen-parameter restore requires an attached LoRA adapter"
            );
        }
        if (model->state->active_optimizers.load(
                std::memory_order_relaxed
            ) != 1) {
            throw std::invalid_argument(
                "frozen-parameter restore requires the supplied Adam to be "
                "the model's sole live optimizer"
            );
        }
        const auto adapter_parameters =
            model->state->value.lora_parameters();
        if (!same_parameter_identity(
                adapter_optimizer->parameter_identity,
                adapter_parameters
            )) {
            throw std::invalid_argument(
                "Adam must own the model's complete LoRA parameter list"
            );
        }
        if (model->state->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot restore frozen parameters while variable graphs are alive"
            );
        }
        require_no_active_decode_sessions(
            *model->state,
            "restore frozen parameters"
        );
        require_epoch_increment_available(*model->state);

        auto base_parameters = model->state->value.parameters();
        auto replacements = prepare_parameter_replacements(
            base_parameters,
            values,
            value_count,
            "frozen parameter value count"
        );
        commit_parameter_replacements(std::move(replacements));
        model->state->parameter_epoch.fetch_add(
            1,
            std::memory_order_relaxed
        );
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
        auto replacements = prepare_parameter_replacements(
            parameters->value,
            values,
            value_count,
            "parameter value count"
        );
        commit_parameter_replacements(std::move(replacements));
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

rt_status RT_CALL rt_cross_entropy_time_range(
    const rt_variable* logits,
    const uint32_t* targets,
    uint64_t target_count,
    uint64_t time_offset,
    uint64_t time_count,
    rt_variable** output
) {
    return guard([&] {
        require_output(output);
        require_variable(logits);
        require_current_graph(logits, "time-range cross entropy");
        const auto values = checked_token_ids(
            targets,
            target_count,
            "time-range cross-entropy targets"
        );
        auto result = std::make_unique<rt_variable>(
            logits->owner,
            logits->graph,
            riftco_transformer::cross_entropy_time_range(
                logits->value,
                values,
                checked_size(time_offset, "cross-entropy time offset"),
                checked_size(time_count, "cross-entropy time count")
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

rt_status RT_CALL rt_adam_state_get(
    const rt_adam* adam,
    rt_adam_state* output_state
) {
    return guard([&] {
        require_adam(adam);
        if (output_state == nullptr) {
            throw std::invalid_argument(
                "Adam state output must not be null"
            );
        }
        checked_structure_size(
            output_state->struct_size,
            sizeof(rt_adam_state),
            "Adam state structure"
        );
        const std::uint64_t structure_size =
            output_state->struct_size;
        *output_state = {
            structure_size,
            checked_u64(
                adam->value.step_count(), "Adam state step count"
            ),
            adam->value.beta1_power(),
            adam->value.beta2_power(),
            checked_u64(
                adam->value.state_value_count(),
                "Adam state value count"
            ),
        };
    });
}

rt_status RT_CALL rt_adam_state_copy_to_host_f32(
    const rt_adam* adam,
    float* output_parameter_values,
    float* output_first_moments,
    float* output_second_moments,
    uint64_t value_capacity
) {
    return guard([&] {
        require_adam(adam);
        if (adam->owner->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot capture Adam state while variable graphs are alive"
            );
        }
        const std::size_t capacity = checked_size(
            value_capacity, "Adam state output capacity"
        );
        const std::size_t required =
            adam->value.state_value_count();
        if (capacity < required) {
            throw std::invalid_argument(
                "Adam state output capacity is too small"
            );
        }
        if (required != 0 &&
            (output_parameter_values == nullptr ||
             output_first_moments == nullptr ||
             output_second_moments == nullptr)) {
            throw std::invalid_argument(
                "Adam state outputs must not be null"
            );
        }
        const riftco_transformer::AdamState state =
            adam->value.state();
        std::copy(
            state.parameter_values.begin(),
            state.parameter_values.end(),
            output_parameter_values
        );
        std::copy(
            state.first_moments.begin(),
            state.first_moments.end(),
            output_first_moments
        );
        std::copy(
            state.second_moments.begin(),
            state.second_moments.end(),
            output_second_moments
        );
    });
}

rt_status RT_CALL rt_adam_state_load_from_host_f32(
    rt_adam* adam,
    const rt_adam_state* state,
    const float* parameter_values,
    const float* first_moments,
    const float* second_moments,
    uint64_t value_count
) {
    return guard([&] {
        require_adam(adam);
        if (state == nullptr) {
            throw std::invalid_argument(
                "Adam state input must not be null"
            );
        }
        checked_structure_size(
            state->struct_size,
            sizeof(rt_adam_state),
            "Adam state structure"
        );
        const std::size_t count = checked_size(
            value_count, "Adam state value count"
        );
        if (checked_size(
                state->value_count,
                "Adam state declared value count"
            ) != count ||
            count != adam->value.state_value_count()) {
            throw std::invalid_argument(
                "Adam state value count does not match the optimizer"
            );
        }
        if (adam->owner->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot restore Adam state while variable graphs are alive"
            );
        }
        require_no_active_decode_sessions(
            *adam->owner, "restore Adam state"
        );
        require_epoch_increment_available(*adam->owner);

        riftco_transformer::AdamState native_state;
        native_state.step_count = checked_size(
            state->step_count, "Adam state step count"
        );
        native_state.beta1_power = state->beta1_power;
        native_state.beta2_power = state->beta2_power;
        native_state.parameter_values = checked_f32_values(
            parameter_values,
            value_count,
            "Adam state parameter values"
        );
        native_state.first_moments = checked_f32_values(
            first_moments,
            value_count,
            "Adam state first moments"
        );
        native_state.second_moments = checked_f32_values(
            second_moments,
            value_count,
            "Adam state second moments"
        );
        adam->value.load_state(std::move(native_state));
        adam->owner->parameter_epoch.fetch_add(
            1, std::memory_order_relaxed
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
