#include "internal/bridge.hpp"

using namespace riftco_transformer::c_api::detail;

namespace {

struct ParameterReplacement {
    riftco_transformer::Parameter* parameter;
    riftco_transformer::Tensor value;
};

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

}  // namespace


extern "C" {

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

}  // extern "C"
