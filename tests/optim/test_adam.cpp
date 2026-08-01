#include "core/backend/optim/adam/metal/diagnostics.hpp"
#include "core/backend/optim/adam/dispatch.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/nn/loss.hpp"
#include "riftco_transformer/optim/adam.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using riftco_transformer::Adam;
using riftco_transformer::AdamOptions;
using riftco_transformer::AdamStepStats;
using riftco_transformer::DecoderOnlyTransformer;
using riftco_transformer::ExecutionBackend;
using riftco_transformer::NamedParameter;
using riftco_transformer::Parameter;
using riftco_transformer::ParameterList;
using riftco_transformer::Tensor;
using riftco_transformer::TokenId;
using riftco_transformer::Variable;
using riftco_transformer::cross_entropy;
using riftco_transformer::global_gradient_norm;
using riftco_transformer::move_parameters_to;
using riftco_transformer::backend_detail::metal_adam_path_counts;
using riftco_transformer::backend_detail::reset_metal_adam_path_counts;
namespace backend = riftco_transformer::backend_detail;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    double actual,
    double expected,
    const std::string& message,
    double absolute_tolerance = 1.0e-6,
    double relative_tolerance = 1.0e-6
) {
    const double tolerance =
        absolute_tolerance +
        relative_tolerance * std::fabs(expected);
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void require_tensor_close(
    const Tensor& actual,
    const Tensor& expected,
    const std::string& message,
    double tolerance = 1.0e-6
) {
    require(
        actual.shape() == expected.shape(),
        message + ": shape mismatch"
    );
    for (std::size_t index = 0;
         index < actual.numel();
         ++index) {
        require_close(
            static_cast<double>(actual.flat(index)),
            static_cast<double>(expected.flat(index)),
            message + " at flat index " + std::to_string(index),
            tolerance,
            tolerance
        );
    }
}

void require_tensor_finite(
    const Tensor& tensor,
    const std::string& message
) {
    for (std::size_t index = 0;
         index < tensor.numel();
         ++index) {
        require(
            std::isfinite(tensor.flat(index)),
            message + " at flat index " + std::to_string(index)
        );
    }
}

void require_zero_gradient(
    const Parameter& parameter,
    const std::string& message
) {
    for (std::size_t index = 0;
         index < parameter.gradient().numel();
         ++index) {
        require_close(
            static_cast<double>(parameter.gradient().flat(index)),
            0.0,
            message + " at flat index " + std::to_string(index),
            0.0,
            0.0
        );
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

void seed_gradient(Parameter& parameter, Tensor gradient) {
    parameter.variable().backward(gradient);
}

template <typename Function>
void for_each_available_accelerator(Function&& function) {
    constexpr std::array<ExecutionBackend, 3> backends{
        ExecutionBackend::Metal,
        ExecutionBackend::Cuda,
        ExecutionBackend::Tpu,
    };
    for (const auto backend : backends) {
        if (!riftco_transformer::execution_backend_available(backend)) {
            continue;
        }
        function(
            backend,
            std::string(riftco_transformer::execution_backend_name(backend))
        );
    }
}

AdamOptions hand_reference_options() {
    AdamOptions options;
    options.learning_rate = 0.1F;
    options.beta1 = 0.5F;
    options.beta2 = 0.5F;
    options.epsilon = 1.0F;
    options.maximum_gradient_norm = 100.0F;
    return options;
}

void test_adam_dispatch_rejects_candidate_aliases() {
    Tensor value({1}, 1.0F, ExecutionBackend::Cpu);
    Tensor gradient({1}, 2.0F, ExecutionBackend::Cpu);
    Tensor first_moment({1}, 0.0F, ExecutionBackend::Cpu);
    Tensor second_moment({1}, 0.0F, ExecutionBackend::Cpu);
    Tensor next_value({1}, 0.0F, ExecutionBackend::Cpu);
    Tensor next_first_moment({1}, 0.0F, ExecutionBackend::Cpu);
    Tensor next_second_moment({1}, 0.0F, ExecutionBackend::Cpu);

    const auto require_alias_rejected = [](
        std::span<const backend::AdamTensorUpdate> updates,
        const std::string& message
    ) {
        try {
            backend::dispatch_adam_update(
                ExecutionBackend::Cpu,
                {
                    updates,
                    0.1F,
                    0.9F,
                    0.999F,
                    1.0e-8F,
                    1.0,
                    0.1,
                    0.001,
                }
            );
        } catch (const std::invalid_argument&) {
            return;
        }
        throw std::runtime_error(message);
    };

    const std::array<backend::AdamTensorUpdate, 1> live_alias{{
        {
            backend::tensor_storage(value),
            backend::tensor_storage(gradient),
            backend::tensor_storage(first_moment),
            backend::tensor_storage(second_moment),
            backend::tensor_storage(value),
            backend::tensor_storage(next_first_moment),
            backend::tensor_storage(next_second_moment),
        },
    }};
    require_alias_rejected(
        live_alias,
        "Adam dispatch must reject a candidate that aliases live state"
    );

    const std::array<backend::AdamTensorUpdate, 1> candidate_alias{{
        {
            backend::tensor_storage(value),
            backend::tensor_storage(gradient),
            backend::tensor_storage(first_moment),
            backend::tensor_storage(second_moment),
            backend::tensor_storage(next_value),
            backend::tensor_storage(next_value),
            backend::tensor_storage(next_second_moment),
        },
    }};
    require_alias_rejected(
        candidate_alias,
        "Adam dispatch must reject aliased candidate outputs"
    );

    Tensor other_gradient({1}, -1.0F, ExecutionBackend::Cpu);
    Tensor other_first_moment({1}, 0.0F, ExecutionBackend::Cpu);
    Tensor other_second_moment({1}, 0.0F, ExecutionBackend::Cpu);
    Tensor other_next_value({1}, 0.0F, ExecutionBackend::Cpu);
    Tensor other_next_first_moment({1}, 0.0F, ExecutionBackend::Cpu);
    Tensor other_next_second_moment({1}, 0.0F, ExecutionBackend::Cpu);
    const std::array<backend::AdamTensorUpdate, 2> cross_tensor_alias{{
        {
            backend::tensor_storage(value),
            backend::tensor_storage(gradient),
            backend::tensor_storage(first_moment),
            backend::tensor_storage(second_moment),
            backend::tensor_storage(next_value),
            backend::tensor_storage(next_first_moment),
            backend::tensor_storage(next_second_moment),
        },
        {
            backend::tensor_storage(next_value),
            backend::tensor_storage(other_gradient),
            backend::tensor_storage(other_first_moment),
            backend::tensor_storage(other_second_moment),
            backend::tensor_storage(other_next_value),
            backend::tensor_storage(other_next_first_moment),
            backend::tensor_storage(other_next_second_moment),
        },
    }};
    require_alias_rejected(
        cross_tensor_alias,
        "Adam dispatch must reject aliases across a candidate batch"
    );

    require_close(
        value.flat(0),
        1.0,
        "Adam alias validation leaves live state unchanged",
        0.0,
        0.0
    );
    require_close(
        next_value.flat(0),
        0.0,
        "Adam alias validation leaves candidates unchanged",
        0.0,
        0.0
    );
}

void test_parameter_handle_lifetime_and_moved_identity() {
    std::unique_ptr<Adam> optimizer;
    Parameter* retained_parameter = nullptr;
    {
        Parameter parameter(Tensor({1}, 2.0F));
        seed_gradient(parameter, Tensor({1}, 3.0F));
        ParameterList parameters{{"weight", &parameter}};
        retained_parameter = parameters.front().parameter;
        optimizer = std::make_unique<Adam>(
            parameters,
            hand_reference_options()
        );
    }

    const AdamStepStats statistics = optimizer->step();
    require(
        statistics.step == 1,
        "Adam owns parameter state after wrapper destruction"
    );
    require_close(
        retained_parameter->value().flat(0),
        1.925,
        "Adam updates retained canonical parameter",
        1.0e-6,
        1.0e-6
    );
    require_zero_gradient(
        *retained_parameter,
        "retained canonical parameter gradient is cleared"
    );

    Parameter original(Tensor({1}, 4.0F));
    const ParameterList before_move{{"original", &original}};
    Parameter moved(std::move(original));
    const ParameterList after_move{{"moved", &moved}};
    require(
        before_move.front().parameter == after_move.front().parameter,
        "moved wrapper resolves to the same canonical parameter"
    );
    require_throws(
        [&] {
            Adam duplicate_optimizer({
                {"original", before_move.front().parameter},
                {"moved", after_move.front().parameter},
            });
        },
        "Adam deduplicates aliases by canonical parameter identity"
    );
}

void require_backend(
    const Parameter& parameter,
    ExecutionBackend backend,
    const std::string& message
) {
    require(
        parameter.value().backend() == backend,
        message + ": value backend"
    );
    require(
        parameter.gradient().backend() == backend,
        message + ": gradient backend"
    );
}

bool tensors_differ(const Tensor& left, const Tensor& right) {
    require(
        left.shape() == right.shape(),
        "tensor difference check requires identical shapes"
    );
    for (std::size_t index = 0; index < left.numel(); ++index) {
        if (left.flat(index) != right.flat(index)) {
            return true;
        }
    }
    return false;
}

struct AdamBackendTrace {
    std::vector<AdamStepStats> statistics;
    std::vector<Tensor> first_values;
    std::vector<Tensor> second_values;
};

AdamBackendTrace run_multi_tensor_adam(ExecutionBackend backend) {
    Parameter first(
        Tensor(
            {3},
            {2.0F, -1.0F, 0.5F},
            backend
        )
    );
    Parameter second(
        Tensor(
            {2},
            {-3.0F, 4.0F},
            backend
        )
    );
    ParameterList parameters{
        {"first", &first},
        {"second", &second},
    };
    AdamOptions options;
    options.learning_rate = 0.05F;
    options.beta1 = 0.8F;
    options.beta2 = 0.9F;
    options.epsilon = 0.01F;
    options.maximum_gradient_norm = 6.0F;
    Adam optimizer(parameters, options);
    require(
        optimizer.backend() == backend,
        "Adam should capture the parameter storage backend"
    );

    const std::vector<std::vector<float>> first_gradients{
        {3.0F, 4.0F, 0.0F},
        {-1.0F, 2.0F, 0.5F},
        {},
        {0.25F, -0.5F, 1.0F},
    };
    const std::vector<std::vector<float>> second_gradients{
        {0.0F, 12.0F},
        {3.0F, -4.0F},
        {},
        {-2.0F, 0.75F},
    };

    AdamBackendTrace trace;
    trace.statistics.reserve(first_gradients.size());
    trace.first_values.reserve(first_gradients.size());
    trace.second_values.reserve(first_gradients.size());
    for (std::size_t step = 0;
         step < first_gradients.size();
         ++step) {
        require_backend(first, backend, "first parameter before step");
        require_backend(second, backend, "second parameter before step");
        if (!first_gradients[step].empty()) {
            seed_gradient(
                first,
                Tensor({3}, first_gradients[step], backend)
            );
            seed_gradient(
                second,
                Tensor({2}, second_gradients[step], backend)
            );
        }

        const AdamStepStats statistics = optimizer.step();
        trace.statistics.push_back(statistics);
        trace.first_values.push_back(
            first.value().to(ExecutionBackend::Cpu)
        );
        trace.second_values.push_back(
            second.value().to(ExecutionBackend::Cpu)
        );

        require(
            statistics.step == step + 1,
            "multi-tensor Adam step number"
        );
        require_backend(first, backend, "first parameter after step");
        require_backend(second, backend, "second parameter after step");
        require_zero_gradient(
            first,
            "multi-tensor step clears first gradient"
        );
        require_zero_gradient(
            second,
            "multi-tensor step clears second gradient"
        );
    }

    require_close(
        trace.statistics[0].gradient_norm,
        13.0,
        "multi-tensor clipped gradient norm"
    );
    require_close(
        trace.statistics[0].clip_scale,
        6.0 / 13.0,
        "multi-tensor non-power-of-two clip scale"
    );
    require_close(
        trace.statistics[2].gradient_norm,
        0.0,
        "zero-gradient momentum-tail norm",
        0.0,
        0.0
    );
    require_close(
        trace.statistics[2].clip_scale,
        1.0,
        "zero-gradient momentum-tail clip scale",
        0.0,
        0.0
    );
    require(
        tensors_differ(
            trace.first_values[2],
            trace.first_values[1]
        ) ||
            tensors_differ(
                trace.second_values[2],
                trace.second_values[1]
            ),
        "Adam must retain a momentum tail on a current zero gradient"
    );
    return trace;
}

void test_parameter_transfer_and_gradient_backend() {
    Parameter first(
        Tensor(
            {2},
            {1.5F, -2.5F},
            ExecutionBackend::Cpu
        )
    );
    Parameter second(
        Tensor(
            {1},
            4.0F,
            ExecutionBackend::Cpu
        )
    );
    const ParameterList parameters{
        {"first", &first},
        {"second", &second},
    };

    seed_gradient(
        first,
        Tensor(
            {2},
            {3.0F, -4.0F},
            ExecutionBackend::Cpu
        )
    );
    first.to(ExecutionBackend::Cpu);
    require_close(
        first.gradient().flat(0),
        3.0,
        "same-backend Parameter::to should be a no-op",
        0.0,
        0.0
    );

    require_throws(
        [&] {
            move_parameters_to(
                {
                    {"first", &first},
                    {"again", &first},
                },
                ExecutionBackend::Cpu
            );
        },
        "bulk transfer should reject duplicate parameters"
    );
    require_throws(
        [] {
            move_parameters_to(
                {{"missing", nullptr}},
                ExecutionBackend::Cpu
            );
        },
        "bulk transfer should reject null parameters"
    );

    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    const Tensor first_before = first.value();
    const Tensor second_before = second.value();
    move_parameters_to(parameters, ExecutionBackend::Metal);
    require_backend(
        first,
        ExecutionBackend::Metal,
        "bulk-transferred first parameter"
    );
    require_backend(
        second,
        ExecutionBackend::Metal,
        "bulk-transferred second parameter"
    );
    require_tensor_close(
        first.value(),
        first_before,
        "bulk transfer preserves first value",
        0.0
    );
    require_tensor_close(
        second.value(),
        second_before,
        "bulk transfer preserves second value",
        0.0
    );
    require_zero_gradient(
        first,
        "backend-changing transfer resets the first gradient"
    );

    const Variable objective = riftco_transformer::sum(
        first.variable() * first.variable()
    );
    require(
        objective.value().backend() == ExecutionBackend::Metal,
        "autograd output should stay on the parameter backend"
    );
    objective.backward();
    require(
        first.gradient().backend() == ExecutionBackend::Metal,
        "autograd gradient should stay on the parameter backend"
    );

    move_parameters_to(parameters, ExecutionBackend::Cpu);
    require_backend(
        first,
        ExecutionBackend::Cpu,
        "round-trip first parameter"
    );
    require_backend(
        second,
        ExecutionBackend::Cpu,
        "round-trip second parameter"
    );
    require_zero_gradient(
        first,
        "round-trip transfer resets the first gradient"
    );
    require_tensor_close(
        first.value(),
        first_before,
        "round-trip transfer preserves first value",
        0.0
    );
}

void test_adam_backend_validation_if_metal_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    Parameter cpu_parameter(
        Tensor({1}, 1.0F, ExecutionBackend::Cpu)
    );
    Parameter metal_parameter(
        Tensor({1}, 2.0F, ExecutionBackend::Metal)
    );
    require_throws(
        [&] {
            Adam mixed({
                {"cpu", &cpu_parameter},
                {"metal", &metal_parameter},
            });
            static_cast<void>(mixed);
        },
        "Adam should reject a mixed-backend parameter list"
    );

    Adam optimizer({{"weight", &cpu_parameter}});
    require(
        optimizer.backend() == ExecutionBackend::Cpu,
        "optimizer should retain its construction backend"
    );
    cpu_parameter.to(ExecutionBackend::Metal);
    const Tensor moved_value = cpu_parameter.value();
    require_throws(
        [&] {
            static_cast<void>(optimizer.step());
        },
        "Adam should reject a parameter moved after construction"
    );
    require(optimizer.step_count() == 0,
            "backend-mismatch failure must not advance Adam");
    require_tensor_close(
        cpu_parameter.value(), moved_value,
        "backend-mismatch failure must not change the parameter", 0.0);

    cpu_parameter.to(ExecutionBackend::Cpu);
    seed_gradient(cpu_parameter, Tensor({1}, 1.0F, ExecutionBackend::Cpu));
    require(optimizer.step().step == 1,
            "Adam should remain usable after restoring its backend");
}

void test_accelerator_adam_parity_if_available() {
    const AdamBackendTrace cpu = run_multi_tensor_adam(ExecutionBackend::Cpu);
    for_each_available_accelerator([&](ExecutionBackend backend,
                                       const std::string& name) {
        const AdamBackendTrace accelerator = run_multi_tensor_adam(backend);
        const std::string prefix = "CPU/" + name + " Adam ";
        require(accelerator.statistics.size() == cpu.statistics.size(),
                prefix + "statistic count");
        for (std::size_t step = 0; step < cpu.statistics.size(); ++step) {
            require(accelerator.statistics[step].step ==
                        cpu.statistics[step].step,
                    prefix + "step parity");
            require_close(accelerator.statistics[step].gradient_norm,
                          cpu.statistics[step].gradient_norm,
                          prefix + "gradient-norm parity", 1.0e-12, 1.0e-12);
            require_close(accelerator.statistics[step].clip_scale,
                          cpu.statistics[step].clip_scale,
                          prefix + "clip-scale parity", 1.0e-12, 1.0e-12);
            require_tensor_close(accelerator.first_values[step],
                                 cpu.first_values[step],
                                 prefix + "first-parameter parity", 5.0e-5);
            require_tensor_close(accelerator.second_values[step],
                                 cpu.second_values[step],
                                 prefix + "second-parameter parity", 5.0e-5);
        }
    });
}

void test_metal_fused_path_if_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal)) {
        return;
    }

    reset_metal_adam_path_counts();
    Parameter parameter(Tensor({1}, 1.0F, ExecutionBackend::Metal));
    Adam optimizer({{"weight", &parameter}}, hand_reference_options());
    seed_gradient(parameter, Tensor({1}, 2.0F, ExecutionBackend::Metal));
    require(
        optimizer.step().step == 1,
        "ordinary Metal Adam step"
    );

    const auto counts = metal_adam_path_counts();
    require(
        counts.fused_batches == 1,
        "ordinary Metal Adam must use the fused kernel"
    );
    require(
        counts.reference_batches == 0,
        "ordinary Metal Adam must not use the safety retry"
    );
}

struct ExtremeAdamResult {
    AdamStepStats statistics;
    float value;
};

ExtremeAdamResult run_extreme_adam(
    ExecutionBackend backend,
    float gradient,
    float maximum_gradient_norm,
    float epsilon
) {
    Parameter parameter(Tensor({1}, 0.0F, backend));
    AdamOptions options;
    options.learning_rate = 0.25F;
    options.beta1 = std::numeric_limits<float>::min();
    options.beta2 = std::numeric_limits<float>::min();
    options.epsilon = epsilon;
    options.maximum_gradient_norm = maximum_gradient_norm;
    Adam optimizer({{"weight", &parameter}}, options);

    seed_gradient(
        parameter,
        Tensor({1}, gradient, backend)
    );
    const AdamStepStats statistics = optimizer.step();
    require_backend(
        parameter,
        backend,
        "extreme Adam result"
    );
    require_zero_gradient(
        parameter,
        "extreme Adam clears its gradient"
    );
    return {statistics, parameter.value().flat(0)};
}

void test_metal_extreme_clipping_if_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    constexpr float learning_rate = 0.25F;
    const float minimum = std::numeric_limits<float>::min();
    const float maximum = std::numeric_limits<float>::max();
    const double expected_scale =
        static_cast<double>(minimum) /
        static_cast<double>(maximum);
    require(
        static_cast<float>(expected_scale) == 0.0F,
        "extreme Adam probe requires a clip scale below float range"
    );
    reset_metal_adam_path_counts();

    const auto verify_case =
        [&](float gradient,
            float maximum_gradient_norm,
            float epsilon,
            double expected_clip_scale,
            const std::string& description) {
            const ExtremeAdamResult cpu = run_extreme_adam(
                ExecutionBackend::Cpu,
                gradient,
                maximum_gradient_norm,
                epsilon
            );
            const ExtremeAdamResult metal = run_extreme_adam(
                ExecutionBackend::Metal,
                gradient,
                maximum_gradient_norm,
                epsilon
            );

            const double expected_norm =
                std::fabs(static_cast<double>(gradient));
            require_close(
                cpu.statistics.gradient_norm,
                expected_norm,
                description + " CPU gradient norm",
                0.0,
                0.0
            );
            require_close(
                metal.statistics.gradient_norm,
                expected_norm,
                description + " Metal gradient norm",
                0.0,
                0.0
            );
            require_close(
                cpu.statistics.clip_scale,
                expected_clip_scale,
                description + " CPU clip scale",
                0.0,
                4.0e-15
            );
            require_close(
                metal.statistics.clip_scale,
                expected_clip_scale,
                description + " Metal clip scale",
                0.0,
                4.0e-15
            );

            const double clipped_gradient =
                static_cast<double>(gradient) *
                expected_clip_scale;
            const double expected_update =
                static_cast<double>(learning_rate) *
                clipped_gradient /
                (
                    std::fabs(clipped_gradient) +
                    static_cast<double>(epsilon)
                );
            const float expected_value =
                static_cast<float>(-expected_update);
            require(
                cpu.value != 0.0F && metal.value != 0.0F,
                description + " must retain a nonzero update"
            );
            require_close(
                cpu.value,
                expected_value,
                description + " CPU reference value",
                0.0,
                2.0e-6
            );
            require_close(
                metal.value,
                expected_value,
                description + " Metal value",
                0.0,
                2.0e-6
            );
            require_close(
                metal.value,
                cpu.value,
                description + " CPU/Metal parity",
                0.0,
                0.0
            );
        };

    verify_case(
        maximum,
        minimum,
        1.0e-8F,
        expected_scale,
        "extreme clip with ordinary epsilon"
    );
    verify_case(
        maximum,
        minimum,
        minimum,
        expected_scale,
        "extreme clip with minimum-normal epsilon"
    );
    verify_case(
        maximum,
        minimum,
        std::numeric_limits<float>::denorm_min(),
        expected_scale,
        "extreme clip with subnormal epsilon"
    );
    verify_case(
        minimum,
        minimum,
        1.0e-8F,
        1.0,
        "unclipped minimum-normal gradient"
    );

    const auto counts = metal_adam_path_counts();
    require(
        counts.fused_batches == 0,
        "unsafe extreme batches must not commit fused candidates"
    );
    require(
        counts.reference_batches == 4,
        "each extreme batch must use one wide-reference retry"
    );
}

float run_cancellation_adam(ExecutionBackend backend) {
    Parameter parameter(Tensor({1}, 0.0F, backend));
    AdamOptions options;
    options.learning_rate = 1.0e20F;
    options.beta1 = 0x1.c3af6ep-1F;
    options.beta2 = 0.5F;
    options.epsilon = 0x1p-58F;
    options.maximum_gradient_norm = 1.0F;
    Adam optimizer({{"weight", &parameter}}, options);

    seed_gradient(
        parameter,
        Tensor({1}, 0x1.ddb17ep-58F, backend)
    );
    static_cast<void>(optimizer.step());

    // Preserve the optimizer moments while isolating the second update.
    parameter.set_value(Tensor({1}, 0.0F, backend));
    seed_gradient(
        parameter,
        Tensor({1}, -0x1.a56b84p-58F, backend)
    );
    static_cast<void>(optimizer.step());
    return parameter.value().flat(0);
}

void test_metal_cancellation_retry_if_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    const float cpu =
        run_cancellation_adam(ExecutionBackend::Cpu);
    reset_metal_adam_path_counts();
    const float metal =
        run_cancellation_adam(ExecutionBackend::Metal);

    require(
        cpu < 0.0F && metal < 0.0F,
        "cancellation-sensitive Adam update must retain its sign"
    );
    require_close(
        metal,
        cpu,
        "cancellation-sensitive CPU/Metal Adam parity",
        0.0,
        2.0e-6
    );
    const auto counts = metal_adam_path_counts();
    require(
        counts.fused_batches == 1,
        "well-conditioned setup step should remain fused"
    );
    require(
        counts.reference_batches == 1,
        "ill-conditioned update should use one reference retry"
    );
}

struct TwoParameterValues {
    float ordinary;
    float square_overflow;
};

TwoParameterValues run_mixed_safety_adam(
    ExecutionBackend backend
) {
    Parameter ordinary(Tensor({1}, 1.0F, backend));
    Parameter square_overflow(Tensor({1}, 0.0F, backend));
    AdamOptions options;
    options.learning_rate = 0.25F;
    options.beta1 = 0.5F;
    options.beta2 = std::nextafter(1.0F, 0.0F);
    options.epsilon = 1.0F;
    options.maximum_gradient_norm =
        std::numeric_limits<float>::max();
    Adam optimizer(
        {
            {"ordinary", &ordinary},
            {"square_overflow", &square_overflow},
        },
        options
    );
    seed_gradient(
        ordinary,
        Tensor({1}, 2.0F, backend)
    );
    seed_gradient(
        square_overflow,
        Tensor({1}, 1.0e20F, backend)
    );
    static_cast<void>(optimizer.step());
    return {
        ordinary.value().flat(0),
        square_overflow.value().flat(0),
    };
}

void test_metal_batch_reference_retry_if_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    const TwoParameterValues cpu =
        run_mixed_safety_adam(ExecutionBackend::Cpu);
    reset_metal_adam_path_counts();
    const TwoParameterValues metal =
        run_mixed_safety_adam(ExecutionBackend::Metal);

    require_close(
        metal.ordinary,
        cpu.ordinary,
        "batch retry ordinary-tensor parity",
        0.0,
        0.0
    );
    require_close(
        metal.square_overflow,
        cpu.square_overflow,
        "batch retry square-overflow parity",
        0.0,
        0.0
    );
    const auto counts = metal_adam_path_counts();
    require(
        counts.fused_batches == 0,
        "one unsafe tensor must prevent a partial fused commit"
    );
    require(
        counts.reference_batches == 1,
        "one unsafe tensor must retry the whole batch once"
    );
}

void test_update_overflow_failure_is_atomic(
    ExecutionBackend backend
) {
    const float maximum = std::numeric_limits<float>::max();
    Parameter first(Tensor({1}, 1.0F, backend));
    Parameter second(Tensor({1}, maximum, backend));
    AdamOptions options = hand_reference_options();
    options.learning_rate = maximum;
    Adam optimizer(
        {
            {"first", &first},
            {"second", &second},
        },
        options
    );

    seed_gradient(first, Tensor({1}, 1.0F, backend));
    seed_gradient(second, Tensor({1}, -1.0F, backend));
    require_throws(
        [&] {
            static_cast<void>(optimizer.step());
        },
        "overflowing Adam candidate should fail"
    );
    require(
        optimizer.step_count() == 0,
        "overflow failure must not advance Adam"
    );
    require_close(
        first.value().flat(0),
        1.0,
        "overflow failure must not partially commit first tensor",
        0.0,
        0.0
    );
    require_close(
        second.value().flat(0),
        static_cast<double>(maximum),
        "overflow failure must not change overflowing tensor",
        0.0,
        0.0
    );
    require_close(
        first.gradient().flat(0),
        1.0,
        "overflow failure retains the first gradient",
        0.0,
        0.0
    );
    require_close(
        second.gradient().flat(0),
        -1.0,
        "overflow failure retains the second gradient",
        0.0,
        0.0
    );

    seed_gradient(second, Tensor({1}, 1.0F, backend));
    const AdamStepStats retry = optimizer.step();
    require(retry.step == 1, "overflow retry is the first step");
    require_tensor_finite(first.value(), "finite first retry value");
    require_tensor_finite(second.value(), "finite second retry value");
}

void test_constant_gradient_bias_correction() {
    Parameter parameter(Tensor({2}, {1.0F, -1.0F}));
    AdamOptions options;
    options.learning_rate = 0.1F;
    options.beta1 = 0.8F;
    options.beta2 = 0.9F;
    options.epsilon = 1.0F;
    options.maximum_gradient_norm = 100.0F;
    Adam optimizer({{"weights", &parameter}}, options);

    require(optimizer.step_count() == 0, "initial Adam step count");
    require(
        optimizer.parameter_tensor_count() == 1,
        "Adam parameter tensor count"
    );
    require_close(
        optimizer.options().learning_rate,
        options.learning_rate,
        "Adam learning-rate getter",
        0.0,
        0.0
    );
    require_close(
        optimizer.options().beta1,
        options.beta1,
        "Adam beta1 getter",
        0.0,
        0.0
    );
    require_close(
        optimizer.options().beta2,
        options.beta2,
        "Adam beta2 getter",
        0.0,
        0.0
    );

    // With a constant gradient, bias correction makes m_hat equal g and
    // v_hat equal g^2 on every step. The updates are therefore exactly
    // 0.1 * 2 / (2 + 1) and 0.1 * -3 / (3 + 1).
    for (std::size_t step = 1; step <= 3; ++step) {
        seed_gradient(
            parameter,
            Tensor({2}, {2.0F, -3.0F})
        );
        require_close(
            global_gradient_norm({{"weights", &parameter}}),
            std::sqrt(13.0),
            "constant-gradient global norm"
        );

        const AdamStepStats stats = optimizer.step();
        require(stats.step == step, "reported Adam step");
        require(
            optimizer.step_count() == step,
            "stored Adam step count"
        );
        require_close(
            stats.gradient_norm,
            std::sqrt(13.0),
            "constant-gradient step norm"
        );
        require_close(
            stats.clip_scale,
            1.0,
            "unclipped constant-gradient scale",
            0.0,
            0.0
        );

        const double expected_positive =
            1.0 - static_cast<double>(step) / 15.0;
        const double expected_negative =
            -1.0 + static_cast<double>(step) * 0.075;
        require_close(
            parameter.value().flat(0),
            expected_positive,
            "constant-gradient positive coordinate",
            2.0e-6,
            2.0e-6
        );
        require_close(
            parameter.value().flat(1),
            expected_negative,
            "constant-gradient negative coordinate",
            2.0e-6,
            2.0e-6
        );
        require_zero_gradient(
            parameter,
            "step clears consumed gradients"
        );
    }
}

void test_varying_gradient_moments_and_zero_gradient_tail() {
    Parameter parameter(Tensor({1}, 1.0F));
    Adam optimizer(
        {{"scalar", &parameter}},
        hand_reference_options()
    );
    const std::vector<float> gradients{2.0F, 1.0F, 0.0F};
    const std::vector<double> expected_values{
        0.9333333333333333,
        0.8781048583502540,
        0.8484328982592746,
    };

    for (std::size_t index = 0;
         index < gradients.size();
         ++index) {
        seed_gradient(
            parameter,
            Tensor({1}, gradients[index])
        );
        const AdamStepStats stats = optimizer.step();
        require_close(
            stats.gradient_norm,
            std::fabs(static_cast<double>(gradients[index])),
            "varying-gradient norm"
        );
        require_close(
            parameter.value().flat(0),
            expected_values[index],
            "hand-calculated varying-gradient Adam value",
            2.0e-6,
            2.0e-6
        );
    }

    // Standard Adam still moves on a current zero gradient when a previous
    // nonzero gradient left first-moment history.
    require(
        expected_values[2] < expected_values[1],
        "reference momentum tail must be nonzero"
    );
    require(optimizer.step_count() == 3, "three Adam steps completed");
}

void test_fresh_zero_gradient_and_explicit_zeroing() {
    Parameter parameter(Tensor({2}, {4.0F, -2.0F}));
    const Tensor initial = parameter.value();
    Adam optimizer(
        {{"weights", &parameter}},
        hand_reference_options()
    );

    const AdamStepStats fresh_zero = optimizer.step();
    require_close(
        fresh_zero.gradient_norm,
        0.0,
        "fresh zero-gradient norm",
        0.0,
        0.0
    );
    require_close(
        fresh_zero.clip_scale,
        1.0,
        "fresh zero-gradient clip scale",
        0.0,
        0.0
    );
    require_tensor_close(
        parameter.value(),
        initial,
        "fresh zero gradients leave parameters unchanged",
        0.0
    );

    seed_gradient(
        parameter,
        Tensor({2}, {3.0F, -4.0F})
    );
    require_close(
        global_gradient_norm({{"weights", &parameter}}),
        5.0,
        "gradient before explicit zeroing"
    );
    optimizer.zero_gradients();
    require_zero_gradient(parameter, "explicit Adam zero_gradients");
    require(
        optimizer.step_count() == 1,
        "zero_gradients must not advance the optimizer"
    );
    require_tensor_close(
        parameter.value(),
        initial,
        "zero_gradients must not change values",
        0.0
    );

    const AdamStepStats second_zero = optimizer.step();
    require(second_zero.step == 2, "second zero-gradient step");
    require_tensor_close(
        parameter.value(),
        initial,
        "zero moments plus zero gradients remain unchanged",
        0.0
    );
}

void test_global_gradient_clipping_across_parameters() {
    Parameter matrix(Tensor({2}, {10.0F, 20.0F}));
    Parameter scalar(Tensor({1}, 30.0F));
    ParameterList parameters{
        {"matrix", &matrix},
        {"scalar", &scalar},
    };
    AdamOptions options = hand_reference_options();
    options.maximum_gradient_norm = 6.5F;
    Adam optimizer(parameters, options);

    seed_gradient(matrix, Tensor({2}, {3.0F, 4.0F}));
    seed_gradient(scalar, Tensor({1}, 12.0F));
    require_close(
        global_gradient_norm(parameters),
        13.0,
        "global norm spans every parameter tensor"
    );

    const AdamStepStats stats = optimizer.step();
    require_close(stats.gradient_norm, 13.0, "pre-clipping norm");
    require_close(
        stats.clip_scale,
        0.5,
        "global clipping scale"
    );

    // The shared scale produces clipped gradients [1.5, 2, 6].
    require_close(
        matrix.value().flat(0),
        9.94,
        "globally clipped matrix coordinate zero",
        2.0e-6,
        2.0e-6
    );
    require_close(
        matrix.value().flat(1),
        19.933333333333333,
        "globally clipped matrix coordinate one",
        2.0e-6,
        2.0e-6
    );
    require_close(
        scalar.value().flat(0),
        29.914285714285715,
        "globally clipped scalar coordinate",
        2.0e-6,
        2.0e-6
    );
}

void test_clipping_boundary_and_large_finite_norm() {
    {
        Parameter parameter(Tensor({2}, {0.0F, 0.0F}));
        AdamOptions options = hand_reference_options();
        options.maximum_gradient_norm = 5.0F;
        Adam optimizer({{"weights", &parameter}}, options);
        seed_gradient(
            parameter,
            Tensor({2}, {3.0F, 4.0F})
        );

        const AdamStepStats stats = optimizer.step();
        require_close(stats.gradient_norm, 5.0, "boundary norm");
        require_close(
            stats.clip_scale,
            1.0,
            "gradient exactly at boundary is not scaled",
            0.0,
            0.0
        );
        require_close(
            parameter.value().flat(0),
            -0.075,
            "unclipped boundary coordinate zero"
        );
        require_close(
            parameter.value().flat(1),
            -0.08,
            "unclipped boundary coordinate one"
        );
    }

    {
        constexpr float large_gradient = 1.0e20F;
        Parameter parameter(Tensor({2}, {1.0F, -1.0F}));
        AdamOptions options = hand_reference_options();
        options.maximum_gradient_norm = 1.0F;
        Adam optimizer({{"weights", &parameter}}, options);
        seed_gradient(
            parameter,
            Tensor(
                {2},
                {large_gradient, -large_gradient}
            )
        );

        const double stored_gradient =
            static_cast<double>(large_gradient);
        const double expected_norm =
            std::hypot(stored_gradient, stored_gradient);
        require_close(
            global_gradient_norm({{"weights", &parameter}}),
            expected_norm,
            "large finite global norm",
            1.0e-6,
            1.0e-12
        );
        const AdamStepStats stats = optimizer.step();
        require_close(
            stats.gradient_norm,
            expected_norm,
            "large finite step norm",
            1.0e-6,
            1.0e-12
        );
        require(
            std::isfinite(stats.clip_scale) &&
                stats.clip_scale > 0.0,
            "large finite gradient has a finite positive clip scale"
        );

        const double clipped = 1.0 / std::sqrt(2.0);
        const double update = 0.1 * clipped / (clipped + 1.0);
        require_close(
            parameter.value().flat(0),
            1.0 - update,
            "large positive gradient clips before squaring",
            2.0e-6,
            2.0e-6
        );
        require_close(
            parameter.value().flat(1),
            -1.0 + update,
            "large negative gradient clips before squaring",
            2.0e-6,
            2.0e-6
        );
    }
}

void require_invalid_options(
    AdamOptions options,
    const std::string& message
) {
    Parameter parameter(Tensor({1}, 1.0F));
    require_throws(
        [&] {
            Adam optimizer({{"value", &parameter}}, options);
            (void)optimizer;
        },
        message
    );
}

void test_constructor_and_option_validation() {
    require_throws(
        [] {
            Adam optimizer(ParameterList{});
            (void)optimizer;
        },
        "empty Adam parameter list must be rejected"
    );
    require_throws(
        [] {
            Adam optimizer({{"missing", nullptr}});
            (void)optimizer;
        },
        "null Adam parameter pointer must be rejected"
    );
    {
        Parameter parameter(Tensor({1}, 1.0F));
        require_throws(
            [&] {
                Adam optimizer({{"", &parameter}});
                (void)optimizer;
            },
            "empty Adam parameter name must be rejected"
        );
        require_throws(
            [&] {
                Adam optimizer({
                    {"first", &parameter},
                    {"second", &parameter},
                });
                (void)optimizer;
            },
            "duplicate Adam parameter pointer must be rejected"
        );
    }
    {
        Parameter first(Tensor({1}, 1.0F));
        Parameter second(Tensor({1}, 2.0F));
        require_throws(
            [&] {
                Adam optimizer({
                    {"duplicate", &first},
                    {"duplicate", &second},
                });
                (void)optimizer;
            },
            "duplicate Adam parameter name must be rejected"
        );
    }
    {
        const float quiet_nan =
            std::numeric_limits<float>::quiet_NaN();
        Parameter invalid(Tensor({1}, quiet_nan));
        require_throws(
            [&] {
                Adam optimizer({{"invalid", &invalid}});
                (void)optimizer;
            },
            "non-finite initial parameter must be rejected"
        );
    }

    const float quiet_nan =
        std::numeric_limits<float>::quiet_NaN();
    const float infinity =
        std::numeric_limits<float>::infinity();
    AdamOptions options;

    options.learning_rate = 0.0F;
    require_invalid_options(options, "zero learning rate");
    options = AdamOptions{};
    options.learning_rate = -1.0F;
    require_invalid_options(options, "negative learning rate");
    options = AdamOptions{};
    options.learning_rate = quiet_nan;
    require_invalid_options(options, "NaN learning rate");
    options = AdamOptions{};
    options.learning_rate = infinity;
    require_invalid_options(options, "infinite learning rate");

    options = AdamOptions{};
    options.beta1 = 0.0F;
    require_invalid_options(options, "zero beta1");
    options = AdamOptions{};
    options.beta1 = -0.01F;
    require_invalid_options(options, "negative beta1");
    options = AdamOptions{};
    options.beta1 = 1.0F;
    require_invalid_options(options, "unit beta1");
    options = AdamOptions{};
    options.beta1 = quiet_nan;
    require_invalid_options(options, "NaN beta1");
    options = AdamOptions{};
    options.beta2 = 0.0F;
    require_invalid_options(options, "zero beta2");
    options = AdamOptions{};
    options.beta2 = -0.01F;
    require_invalid_options(options, "negative beta2");
    options = AdamOptions{};
    options.beta2 = 1.0F;
    require_invalid_options(options, "unit beta2");
    options = AdamOptions{};
    options.beta2 = infinity;
    require_invalid_options(options, "infinite beta2");

    options = AdamOptions{};
    options.epsilon = 0.0F;
    require_invalid_options(options, "zero Adam epsilon");
    options = AdamOptions{};
    options.epsilon = quiet_nan;
    require_invalid_options(options, "NaN Adam epsilon");
    options = AdamOptions{};
    options.maximum_gradient_norm = 0.0F;
    require_invalid_options(options, "zero clipping norm");
    options = AdamOptions{};
    options.maximum_gradient_norm = infinity;
    require_invalid_options(options, "infinite clipping norm");

}

void test_nonfinite_gradient_failure_is_atomic() {
    Parameter first(Tensor({1}, 1.0F));
    Parameter second(Tensor({1}, 2.0F));
    ParameterList parameters{
        {"first", &first},
        {"second", &second},
    };
    Adam optimizer(parameters, hand_reference_options());

    seed_gradient(first, Tensor({1}, 1.0F));
    seed_gradient(
        second,
        Tensor(
            {1},
            std::numeric_limits<float>::infinity()
        )
    );
    require_throws(
        [&] {
            (void)global_gradient_norm(parameters);
        },
        "global norm rejects a non-finite gradient"
    );
    require_throws(
        [&] {
            (void)optimizer.step();
        },
        "Adam rejects a non-finite gradient"
    );

    require(optimizer.step_count() == 0, "failed step is not counted");
    require_close(
        first.value().flat(0),
        1.0,
        "first parameter unchanged after failed step",
        0.0,
        0.0
    );
    require_close(
        second.value().flat(0),
        2.0,
        "second parameter unchanged after failed step",
        0.0,
        0.0
    );
    require_close(
        first.gradient().flat(0),
        1.0,
        "valid gradient remains after failed step",
        0.0,
        0.0
    );
    require(
        std::isinf(second.gradient().flat(0)),
        "invalid gradient remains available for diagnosis"
    );

    // Replacing only the invalid gradient and retrying must behave exactly
    // like the first Adam step, proving moments and beta powers were atomic.
    seed_gradient(second, Tensor({1}, 2.0F));
    const AdamStepStats retry = optimizer.step();
    require(retry.step == 1, "retry is the first successful step");
    require_close(
        retry.gradient_norm,
        std::sqrt(5.0),
        "retry gradient norm"
    );
    require_close(
        first.value().flat(0),
        0.95,
        "first parameter after atomic retry"
    );
    require_close(
        second.value().flat(0),
        2.0 - (0.1 * 2.0 / 3.0),
        "second parameter after atomic retry"
    );

    require_throws(
        [&] {
            (void)global_gradient_norm({
                {"first", &first},
                {"again", &first},
            });
        },
        "global norm rejects duplicate parameter pointers"
    );
    require_throws(
        [] {
            (void)global_gradient_norm({{"missing", nullptr}});
        },
        "global norm rejects null parameter pointers"
    );
    require_close(
        global_gradient_norm(ParameterList{}),
        0.0,
        "empty global norm is zero",
        0.0,
        0.0
    );
}

Parameter* find_parameter(
    const ParameterList& parameters,
    const std::string& name
) {
    for (const NamedParameter& named_parameter : parameters) {
        if (named_parameter.name == name) {
            return named_parameter.parameter;
        }
    }
    throw std::runtime_error("missing model parameter " + name);
}

float model_loss(
    DecoderOnlyTransformer& model,
    const std::vector<TokenId>& inputs,
    const std::vector<TokenId>& targets
) {
    return cross_entropy(
        model.forward(inputs, {2, 3}),
        targets
    ).value().flat(0);
}

void test_tiny_decoder_integration() {
    std::mt19937 random(313U);
    DecoderOnlyTransformer model(
        {
            5,
            4,
            4,
            2,
            1,
            6,
        },
        random
    );
    ParameterList parameters = model.parameters();
    require(
        parameters.size() == 22,
        "tiny decoder parameter tensor count"
    );
    Parameter* position_embedding = find_parameter(
        parameters,
        "position_embedding.weight"
    );
    Parameter* language_model_head = find_parameter(
        parameters,
        "language_model_head.weight"
    );
    const Tensor initial_position_embedding =
        position_embedding->value();
    const Tensor initial_language_model_head =
        language_model_head->value();

    AdamOptions options;
    options.learning_rate = 1.0e-2F;
    options.beta1 = 0.9F;
    options.beta2 = 0.999F;
    options.epsilon = 1.0e-8F;
    options.maximum_gradient_norm = 1.0F;
    Adam optimizer(parameters, options);
    require(
        optimizer.parameter_tensor_count() == 22,
        "optimizer tracks every decoder tensor"
    );

    const std::vector<TokenId> inputs{
        0, 1, 2,
        0, 1, 2,
    };
    const std::vector<TokenId> targets{
        1, 2, 3,
        1, 2, 3,
    };
    const float initial_loss = model_loss(
        model,
        inputs,
        targets
    );
    require(std::isfinite(initial_loss), "initial model loss is finite");

    constexpr std::size_t training_steps = 20;
    for (std::size_t step = 1;
         step <= training_steps;
         ++step) {
        const Variable loss = cross_entropy(
            model.forward(inputs, {2, 3}),
            targets
        );
        require(
            std::isfinite(loss.value().flat(0)),
            "training loss is finite"
        );
        loss.backward();
        const AdamStepStats stats = optimizer.step();
        require(stats.step == step, "decoder Adam step number");
        require(
            std::isfinite(stats.gradient_norm) &&
                stats.gradient_norm > 0.0,
            "decoder gradient norm is finite and positive"
        );
        require(
            std::isfinite(stats.clip_scale) &&
                stats.clip_scale > 0.0 &&
                stats.clip_scale <= 1.0,
            "decoder clip scale is in (0, 1]"
        );
        for (const NamedParameter& named_parameter : parameters) {
            require_zero_gradient(
                *named_parameter.parameter,
                "decoder step clears " + named_parameter.name
            );
            require_tensor_finite(
                named_parameter.parameter->value(),
                "finite decoder value for " + named_parameter.name
            );
        }
    }

    const float final_loss = model_loss(model, inputs, targets);
    require(
        std::isfinite(final_loss),
        "final model loss is finite"
    );
    require(
        final_loss < initial_loss * 0.5F,
        "repeated tiny-batch Adam steps halve model loss"
    );

    bool head_changed = false;
    for (std::size_t index = 0;
         index < language_model_head->value().numel();
         ++index) {
        if (language_model_head->value().flat(index) !=
            initial_language_model_head.flat(index)) {
            head_changed = true;
        }
    }
    require(head_changed, "language-model head is updated");

    // Time is three while maximum context is four. Position row three never
    // receives a gradient, so its fresh zero moments keep it bit-for-bit fixed.
    for (std::size_t channel = 0; channel < 4; ++channel) {
        require_close(
            position_embedding->value().at({3, channel}),
            initial_position_embedding.at({3, channel}),
            "unused positional row remains unchanged",
            0.0,
            0.0
        );
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    bool require_tpu = false;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--require-tpu") {
            require_tpu = true;
        } else {
            std::cerr << "usage: adam_tests [--require-tpu]\n";
            return 2;
        }
    }

    try {
        require(
            !require_tpu ||
                riftco_transformer::execution_backend_available(
                    ExecutionBackend::Tpu
                ),
            "TPU is required for this Adam test, but no TPU runtime is "
            "available"
        );
        test_adam_dispatch_rejects_candidate_aliases();
        test_parameter_handle_lifetime_and_moved_identity();
        test_parameter_transfer_and_gradient_backend();
        test_adam_backend_validation_if_metal_available();
        test_accelerator_adam_parity_if_available();
        test_metal_fused_path_if_available();
        test_metal_extreme_clipping_if_available();
        test_metal_cancellation_retry_if_available();
        test_metal_batch_reference_retry_if_available();
        test_update_overflow_failure_is_atomic(
            ExecutionBackend::Cpu
        );
        for_each_available_accelerator(
            [](ExecutionBackend backend, const std::string&) {
                test_update_overflow_failure_is_atomic(backend);
            }
        );
        test_constant_gradient_bias_correction();
        test_varying_gradient_moments_and_zero_gradient_tail();
        test_fresh_zero_gradient_and_explicit_zeroing();
        test_global_gradient_clipping_across_parameters();
        test_clipping_boundary_and_large_finite_norm();
        test_constructor_and_option_validation();
        test_nonfinite_gradient_failure_is_atomic();
        test_tiny_decoder_integration();
    } catch (const std::exception& error) {
        std::cerr << "Adam test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "Adam tests passed";
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        std::cout << " (Metal unavailable; fused parity checks skipped)";
    }
    std::cout << '\n';
    return 0;
}
