#include "internal/bridge.hpp"

using namespace riftco_transformer::c_api::detail;

namespace {

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

}  // namespace


extern "C" {

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

}  // extern "C"
