#include "internal/bridge.hpp"

using namespace riftco_transformer::c_api::detail;

namespace {

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

}  // extern "C"
