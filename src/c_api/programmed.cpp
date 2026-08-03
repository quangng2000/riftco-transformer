#include "internal/bridge.hpp"

using namespace riftco_transformer::c_api::detail;

namespace {

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

void require_program_augmented_model(
    const rt_program_augmented_model* model
) {
    if (model == nullptr || model->state == nullptr) {
        throw std::invalid_argument(
            "program-augmented model handle must not be null"
        );
    }
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

}  // namespace


extern "C" {

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

}  // extern "C"
