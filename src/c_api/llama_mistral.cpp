#include "internal/bridge.hpp"

using namespace riftco_transformer::c_api::detail;

namespace {

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

void require_llama_mistral_model(
    const rt_llama_mistral_model* model
) {
    if (model == nullptr || model->state == nullptr) {
        throw std::invalid_argument(
            "Llama/Mistral model handle must not be null"
        );
    }
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

}  // namespace


extern "C" {

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

}  // extern "C"
