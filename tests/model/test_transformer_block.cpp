#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"
#include "riftco_transformer/model/transformer_block.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using riftco_transformer::ExecutionBackend;
using riftco_transformer::Parameter;
using riftco_transformer::Tensor;
using riftco_transformer::TransformerBlock;
using riftco_transformer::Variable;
namespace tensor_ops = riftco_transformer::tensor_ops;

static_assert(!std::is_copy_constructible_v<TransformerBlock>);
static_assert(!std::is_copy_assignable_v<TransformerBlock>);
static_assert(!std::is_move_constructible_v<TransformerBlock>);
static_assert(!std::is_move_assignable_v<TransformerBlock>);

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_parameter_backend(
    const riftco_transformer::ParameterList& parameters,
    ExecutionBackend backend,
    const std::string& message
) {
    require(!parameters.empty(), message + ": empty parameter list");
    for (const auto& named_parameter : parameters) {
        require(
            named_parameter.parameter != nullptr,
            message + ": null parameter " + named_parameter.name
        );
        require(
            named_parameter.parameter->value().backend() == backend,
            message + ": value backend for " + named_parameter.name
        );
        require(
            named_parameter.parameter->gradient().backend() == backend,
            message + ": gradient backend for " + named_parameter.name
        );
    }
}

void require_close(
    float actual,
    float expected,
    const std::string& message,
    float absolute_tolerance = 1.0e-5F,
    float relative_tolerance = 0.0F
) {
    const float scale = std::max(
        std::fabs(actual),
        std::fabs(expected)
    );
    const float tolerance =
        absolute_tolerance + relative_tolerance * scale;
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void require_tensor_exact(
    const Tensor& actual,
    const Tensor& expected,
    const std::string& message
) {
    require(
        actual.shape() == expected.shape(),
        message + ": shape mismatch"
    );
    for (std::size_t index = 0; index < expected.numel(); ++index) {
        require(
            actual.flat(index) == expected.flat(index),
            message + " at flat index " + std::to_string(index)
        );
    }
}

void require_finite_tensor(
    const Tensor& tensor,
    const std::string& message
) {
    for (const float value : tensor.data()) {
        require(std::isfinite(value), message);
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

float tensor_dot(const Tensor& left, const Tensor& right) {
    return tensor_ops::sum(
        tensor_ops::multiply(left, right)
    ).flat(0);
}

Parameter& find_parameter(
    TransformerBlock& block,
    const std::string& name
) {
    for (auto& named_parameter : block.parameters()) {
        if (named_parameter.name == name) {
            if (named_parameter.parameter == nullptr) {
                throw std::runtime_error(
                    "null transformer block parameter: " + name
                );
            }
            return *named_parameter.parameter;
        }
    }
    throw std::runtime_error(
        "missing transformer block parameter: " + name
    );
}

void set_parameter(
    TransformerBlock& block,
    const std::string& name,
    Tensor value
) {
    find_parameter(block, name).set_value(std::move(value));
}

void zero_residual_branch_projections(
    TransformerBlock& block,
    std::size_t model_width,
    std::size_t feed_forward_width
) {
    set_parameter(
        block,
        "attention.output.weight",
        Tensor::zeros({model_width, model_width})
    );
    set_parameter(
        block,
        "attention.output.bias",
        Tensor::zeros({model_width})
    );
    set_parameter(
        block,
        "feed_forward.project.weight",
        Tensor::zeros({model_width, feed_forward_width})
    );
    set_parameter(
        block,
        "feed_forward.project.bias",
        Tensor::zeros({model_width})
    );
}

void test_getters_shape_and_parameter_registration() {
    constexpr std::size_t model_width = 4;
    constexpr std::size_t head_count = 2;
    constexpr std::size_t feed_forward_width = 6;

    std::mt19937 random(17);
    TransformerBlock block(
        model_width,
        head_count,
        feed_forward_width,
        random
    );

    require(block.model_width() == model_width, "block model width");
    require(block.head_count() == head_count, "block head count");
    require(
        block.feed_forward_width() == feed_forward_width,
        "block feed-forward width"
    );

    const Tensor input_values(
        {2, 3, model_width},
        {
            0.1F, -0.2F, 0.3F, -0.4F,
            0.5F, 0.6F, -0.7F, 0.8F,
            -0.9F, 1.0F, 1.1F, -1.2F,
            1.3F, -1.4F, 1.5F, 1.6F,
            -1.7F, 1.8F, -1.9F, 2.0F,
            2.1F, -2.2F, 2.3F, -2.4F,
        }
    );
    const Variable output = block.forward(
        Variable(input_values, false)
    );
    require(
        output.value().shape() == input_values.shape(),
        "block should preserve [batch, time, model_width]"
    );
    require_finite_tensor(output.value(), "block output must be finite");

    auto parameters = block.parameters();
    const std::vector<std::string> expected_names{
        "attention_norm.scale",
        "attention_norm.bias",
        "attention.query.weight",
        "attention.query.bias",
        "attention.key.weight",
        "attention.key.bias",
        "attention.value.weight",
        "attention.value.bias",
        "attention.output.weight",
        "attention.output.bias",
        "feed_forward_norm.scale",
        "feed_forward_norm.bias",
        "feed_forward.expand.weight",
        "feed_forward.expand.bias",
        "feed_forward.project.weight",
        "feed_forward.project.bias",
    };
    const std::vector<Tensor::Shape> expected_shapes{
        {model_width},
        {model_width},
        {model_width, model_width},
        {model_width},
        {model_width, model_width},
        {model_width},
        {model_width, model_width},
        {model_width},
        {model_width, model_width},
        {model_width},
        {model_width},
        {model_width},
        {feed_forward_width, model_width},
        {feed_forward_width},
        {model_width, feed_forward_width},
        {model_width},
    };

    require(
        parameters.size() == expected_names.size(),
        "block should register exactly 16 parameters"
    );
    std::set<const Parameter*> unique_parameters;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
        require(
            parameters[index].name == expected_names[index],
            "block parameter name/order at index " +
                std::to_string(index)
        );
        require(
            parameters[index].parameter != nullptr,
            "block parameter pointer must not be null"
        );
        require(
            parameters[index].parameter->value().shape() ==
                expected_shapes[index],
            "block parameter shape for " + expected_names[index]
        );
        unique_parameters.insert(parameters[index].parameter);
    }
    require(
        unique_parameters.size() == parameters.size(),
        "block parameter pointers must be unique"
    );

    constexpr std::size_t expected_scalar_count =
        4 * model_width * model_width +
        2 * model_width * feed_forward_width +
        feed_forward_width +
        9 * model_width;
    require(
        riftco_transformer::parameter_count(parameters) ==
            expected_scalar_count,
        "block scalar parameter count"
    );
}

void test_residual_identity_and_bias_paths() {
    constexpr std::size_t model_width = 3;
    constexpr std::size_t feed_forward_width = 5;

    std::mt19937 random(23);
    TransformerBlock block(
        model_width,
        1,
        feed_forward_width,
        random
    );
    zero_residual_branch_projections(
        block,
        model_width,
        feed_forward_width
    );

    const Tensor input_values(
        {1, 2, model_width},
        {
            -1.25F, 0.5F, 2.0F,
            0.75F, -0.125F, 1.5F,
        }
    );
    const Tensor seed(
        input_values.shape(),
        {
            0.3F, -0.7F, 1.1F,
            -1.3F, 0.2F, 0.9F,
        }
    );

    {
        const Variable input(input_values);
        const Variable output = block.forward(input);
        require_tensor_exact(
            output.value(),
            input_values,
            "zero branches should make the residual block identity"
        );
        output.backward(seed);
        require_tensor_exact(
            input.gradient(),
            seed,
            "identity residual should pass input gradient unchanged"
        );
    }

    const Tensor attention_bias(
        {model_width},
        {0.25F, -0.5F, 0.75F}
    );
    const Tensor feed_forward_bias(
        {model_width},
        {-0.1F, 0.2F, 0.4F}
    );
    set_parameter(
        block,
        "attention.output.bias",
        attention_bias
    );
    set_parameter(
        block,
        "feed_forward.project.bias",
        feed_forward_bias
    );

    Tensor expected = input_values;
    for (std::size_t position = 0;
         position < input_values.numel() / model_width;
         ++position) {
        for (std::size_t channel = 0;
             channel < model_width;
             ++channel) {
            const std::size_t index =
                position * model_width + channel;
            expected.flat(index) =
                (input_values.flat(index) +
                 attention_bias.flat(channel)) +
                feed_forward_bias.flat(channel);
        }
    }
    require_tensor_exact(
        block.forward(Variable(input_values, false)).value(),
        expected,
        "attention and feed-forward biases should enter separate residuals"
    );
}

void test_fixed_seed_determinism() {
    std::mt19937 first_random(101);
    std::mt19937 second_random(101);
    TransformerBlock first(4, 2, 7, first_random);
    TransformerBlock second(4, 2, 7, second_random);

    const auto first_parameters = first.parameters();
    const auto second_parameters = second.parameters();
    require(
        first_parameters.size() == second_parameters.size(),
        "same-seed parameter list size"
    );
    for (std::size_t index = 0;
         index < first_parameters.size();
         ++index) {
        require(
            first_parameters[index].name ==
                second_parameters[index].name,
            "same-seed parameter names"
        );
        require_tensor_exact(
            first_parameters[index].parameter->value(),
            second_parameters[index].parameter->value(),
            "same-seed parameter " + first_parameters[index].name
        );
    }

    const Tensor input_values(
        {1, 3, 4},
        {
            0.2F, -0.4F, 0.6F, -0.8F,
            1.0F, 0.3F, -0.5F, 0.7F,
            -0.9F, 1.1F, 0.25F, -0.35F,
        }
    );
    require_tensor_exact(
        first.forward(Variable(input_values, false)).value(),
        second.forward(Variable(input_values, false)).value(),
        "same seed should produce identical block output"
    );
}

void test_gradients_with_centered_finite_differences() {
    std::mt19937 random(37);
    TransformerBlock block(2, 1, 3, random, 1.0e-4F);
    auto parameters = block.parameters();

    const Tensor input_values(
        {1, 2, 2},
        {
            0.2F, -0.7F,
            1.1F, 0.3F,
        }
    );
    const Tensor output_weights(
        input_values.shape(),
        {
            0.6F, -0.4F,
            1.2F, -0.8F,
        }
    );

    Tensor input_gradient(input_values.shape());
    std::vector<Tensor> parameter_values;
    std::vector<Tensor> parameter_gradients;
    parameter_values.reserve(parameters.size());
    parameter_gradients.reserve(parameters.size());
    {
        const Variable input(input_values);
        const Variable output = block.forward(input);
        riftco_transformer::sum(
            output * Variable(output_weights, false)
        ).backward();

        input_gradient = input.gradient();
        require_finite_tensor(
            input_gradient,
            "block input gradient must be finite"
        );
        for (const auto& named_parameter : parameters) {
            parameter_values.push_back(
                named_parameter.parameter->value()
            );
            parameter_gradients.push_back(
                named_parameter.parameter->gradient()
            );
            require_finite_tensor(
                parameter_gradients.back(),
                "block parameter gradient must be finite: " +
                    named_parameter.name
            );
        }
    }

    const auto evaluate = [&](const Tensor& candidate_input) {
        return tensor_dot(
            block.forward(
                Variable(candidate_input, false)
            ).value(),
            output_weights
        );
    };

    constexpr float epsilon = 1.0e-2F;
    constexpr float absolute_tolerance = 2.5e-2F;
    constexpr float relative_tolerance = 1.0e-2F;

    for (std::size_t index = 0;
         index < input_values.numel();
         ++index) {
        Tensor plus = input_values;
        Tensor minus = input_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        const float numerical =
            (evaluate(plus) - evaluate(minus)) /
            (2.0F * epsilon);
        require_close(
            input_gradient.flat(index),
            numerical,
            "block input centered finite difference at index " +
                std::to_string(index),
            absolute_tolerance,
            relative_tolerance
        );
    }

    for (std::size_t parameter_index = 0;
         parameter_index < parameters.size();
         ++parameter_index) {
        const Tensor& original = parameter_values[parameter_index];
        for (std::size_t index = 0;
             index < original.numel();
             ++index) {
            Tensor plus = original;
            Tensor minus = original;
            plus.flat(index) += epsilon;
            minus.flat(index) -= epsilon;

            parameters[parameter_index].parameter->set_value(
                std::move(plus)
            );
            const float plus_loss = evaluate(input_values);
            parameters[parameter_index].parameter->set_value(
                std::move(minus)
            );
            const float minus_loss = evaluate(input_values);
            const float numerical =
                (plus_loss - minus_loss) /
                (2.0F * epsilon);

            require_close(
                parameter_gradients[parameter_index].flat(index),
                numerical,
                "block " + parameters[parameter_index].name +
                    " centered finite difference at index " +
                    std::to_string(index),
                absolute_tolerance,
                relative_tolerance
            );
        }
        parameters[parameter_index].parameter->set_value(original);
    }
}

void test_causal_future_isolation() {
    std::mt19937 random(59);
    TransformerBlock block(4, 2, 7, random);

    const Tensor baseline(
        {1, 4, 4},
        {
            0.1F, -0.2F, 0.3F, -0.4F,
            0.5F, 0.6F, -0.7F, 0.8F,
            -0.9F, 1.0F, 1.1F, -1.2F,
            1.3F, -1.4F, 1.5F, 1.6F,
        }
    );
    Tensor changed_future = baseline;
    for (std::size_t index = 8;
         index < changed_future.numel();
         ++index) {
        changed_future.flat(index) =
            changed_future.flat(index) * -2.0F + 0.35F;
    }

    const Tensor baseline_output = block.forward(
        Variable(baseline, false)
    ).value();
    const Tensor changed_output = block.forward(
        Variable(changed_future, false)
    ).value();
    for (std::size_t index = 0; index < 8; ++index) {
        require(
            baseline_output.flat(index) ==
                changed_output.flat(index),
            "future input must not change prefix output at index " +
                std::to_string(index)
        );
    }
    bool future_output_changed = false;
    for (std::size_t index = 8;
         index < baseline_output.numel();
         ++index) {
        if (baseline_output.flat(index) !=
            changed_output.flat(index)) {
            future_output_changed = true;
        }
    }
    require(
        future_output_changed,
        "causality comparison must perturb a visible future output"
    );

    const Variable differentiable_input(baseline);
    const Variable output = block.forward(differentiable_input);
    Tensor prefix_seed(output.value().shape(), 0.0F);
    for (std::size_t index = 0; index < 8; ++index) {
        prefix_seed.flat(index) =
            index % 2 == 0 ? 0.75F : -0.5F;
    }
    output.backward(prefix_seed);
    require_finite_tensor(
        differentiable_input.gradient(),
        "causal input gradient must be finite"
    );
    for (std::size_t index = 8;
         index < differentiable_input.gradient().numel();
         ++index) {
        require_close(
            differentiable_input.gradient().flat(index),
            0.0F,
            "prefix output must have zero future-input gradient at index " +
                std::to_string(index),
            1.0e-6F
        );
    }
}

void test_invalid_dimensions_and_shapes() {
    require_throws(
        [] {
            std::mt19937 random(1);
            static_cast<void>(TransformerBlock(0, 1, 3, random));
        },
        "block should reject zero model width"
    );
    require_throws(
        [] {
            std::mt19937 random(1);
            static_cast<void>(TransformerBlock(4, 0, 3, random));
        },
        "block should reject zero head count"
    );
    require_throws(
        [] {
            std::mt19937 random(1);
            static_cast<void>(TransformerBlock(4, 2, 0, random));
        },
        "block should reject zero feed-forward width"
    );
    require_throws(
        [] {
            std::mt19937 random(1);
            static_cast<void>(TransformerBlock(5, 2, 7, random));
        },
        "block should reject indivisible head width"
    );
    require_throws(
        [] {
            std::mt19937 random(1);
            static_cast<void>(
                TransformerBlock(4, 2, 7, random, 0.0F)
            );
        },
        "block should reject zero layer-norm epsilon"
    );
    require_throws(
        [] {
            std::mt19937 random(1);
            static_cast<void>(TransformerBlock(
                4,
                2,
                7,
                random,
                std::numeric_limits<float>::infinity()
            ));
        },
        "block should reject non-finite layer-norm epsilon"
    );

    std::mt19937 random(3);
    TransformerBlock block(4, 2, 7, random);
    require_throws(
        [&] {
            static_cast<void>(block.forward(
                Variable(Tensor({2, 4}), false)
            ));
        },
        "block should reject rank-two input"
    );
    require_throws(
        [&] {
            static_cast<void>(block.forward(
                Variable(Tensor({1, 2, 3}), false)
            ));
        },
        "block should reject wrong final width"
    );
    require_throws(
        [&] {
            static_cast<void>(block.forward(
                Variable(Tensor({1, 1, 2, 4}), false)
            ));
        },
        "block should reject rank-four input"
    );
}

void test_module_device_transfer() {
    const riftco_transformer::ScopedExecutionBackend cpu_backend(
        ExecutionBackend::Cpu
    );
    std::mt19937 random(227U);
    TransformerBlock block(4, 2, 7, random);

    block.to(ExecutionBackend::Cpu);
    require_parameter_backend(
        block.parameters(),
        ExecutionBackend::Cpu,
        "transformer block CPU transfer"
    );

    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    block.to(ExecutionBackend::Metal);
    require_parameter_backend(
        block.parameters(),
        ExecutionBackend::Metal,
        "transformer block Metal transfer"
    );
    block.to(ExecutionBackend::Cpu);
    require_parameter_backend(
        block.parameters(),
        ExecutionBackend::Cpu,
        "transformer block CPU round trip"
    );
}

}  // namespace

int main() {
    try {
        test_getters_shape_and_parameter_registration();
        test_residual_identity_and_bias_paths();
        test_fixed_seed_determinism();
        test_gradients_with_centered_finite_differences();
        test_causal_future_isolation();
        test_invalid_dimensions_and_shapes();
        test_module_device_transfer();
        std::cout << "transformer block tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
