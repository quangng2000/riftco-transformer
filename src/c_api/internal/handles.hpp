#pragma once

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
