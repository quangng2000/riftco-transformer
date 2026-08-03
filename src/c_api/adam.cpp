#include "internal/bridge.hpp"

using namespace riftco_transformer::c_api::detail;

namespace {

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

}  // namespace


extern "C" {

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
