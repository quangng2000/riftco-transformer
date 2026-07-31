#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/nn/linear.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using transformer_lab::ExecutionBackend;
using transformer_lab::Linear;
using transformer_lab::Parameter;
using transformer_lab::ParameterList;
using transformer_lab::Tensor;
using transformer_lab::Variable;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    const std::string& message,
    float tolerance = 1.0e-5F
) {
    const float scale = std::max(
        {1.0F, std::fabs(actual), std::fabs(expected)}
    );
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance * scale) {
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
    float tolerance = 1.0e-5F
) {
    require(
        actual.shape() == expected.shape(),
        message + ": shape mismatch"
    );
    for (std::size_t index = 0;
         index < actual.numel();
         ++index) {
        require_close(
            actual.flat(index),
            expected.flat(index),
            message + " at index " + std::to_string(index),
            tolerance
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

Parameter* parameter_named(
    const ParameterList& parameters,
    const std::string& name
) {
    for (const auto& named_parameter : parameters) {
        if (named_parameter.name == name) {
            return named_parameter.parameter;
        }
    }
    throw std::runtime_error("missing parameter " + name);
}

Linear make_linear() {
    return Linear(
        Tensor(
            {2, 3},
            {
                1.0F, 2.0F, 3.0F,
                -1.0F, 0.0F, 2.0F,
            }
        ),
        Tensor({2}, {0.5F, -0.5F})
    );
}

void test_registration_initialization_and_validation() {
    Linear first = make_linear();
    Linear second = make_linear();
    const Variable input(
        Tensor({3}, {1.0F, 2.0F, 3.0F}),
        false
    );
    const Tensor base_output = first.forward(input).value();

    std::mt19937 first_random(71U);
    std::mt19937 second_random(71U);
    first.attach_lora(2, 4.0F, first_random);
    second.attach_lora(2, 4.0F, second_random);

    require(first.has_lora(), "linear reports active LoRA");
    const ParameterList base_parameters = first.parameters();
    require(
        base_parameters.size() == 2 &&
            base_parameters[0].name == "weight" &&
            base_parameters[1].name == "bias",
        "base registration remains exact"
    );

    const ParameterList first_lora = first.lora_parameters();
    const ParameterList second_lora = second.lora_parameters();
    require(first_lora.size() == 2, "LoRA registers two tensors");
    require(
        first_lora[0].name == "lora_a.weight" &&
            first_lora[1].name == "lora_b.weight",
        "LoRA uses stable parameter suffixes"
    );
    require(
        first_lora[0].parameter->value().shape() ==
            Tensor::Shape({2, 3}),
        "LoRA A has [rank,input] shape"
    );
    require(
        first_lora[1].parameter->value().shape() ==
            Tensor::Shape({2, 2}),
        "LoRA B has [output,rank] shape"
    );
    require_tensor_close(
        first_lora[0].parameter->value(),
        second_lora[0].parameter->value(),
        "same seed initializes A deterministically",
        0.0F
    );
    for (const float value :
         first_lora[1].parameter->value().data()) {
        require(value == 0.0F, "LoRA B starts at exact zero");
    }
    require_tensor_close(
        first.forward(input).value(),
        base_output,
        "zero B makes attachment output-neutral",
        0.0F
    );
    require_throws(
        [&] { first.attach_lora(1, 1.0F, first_random); },
        "a Linear rejects a second attachment"
    );

    {
        Linear invalid = make_linear();
        std::mt19937 random(1U);
        require_throws(
            [&] { invalid.attach_lora(0, 1.0F, random); },
            "zero rank is rejected"
        );
        require(!invalid.has_lora(), "invalid rank is transactional");
    }
    {
        Linear invalid = make_linear();
        std::mt19937 random(1U);
        require_throws(
            [&] { invalid.attach_lora(3, 1.0F, random); },
            "rank larger than a projection is rejected"
        );
        require(!invalid.has_lora(), "oversized rank is transactional");
    }
    {
        Linear invalid = make_linear();
        std::mt19937 random(1U);
        require_throws(
            [&] {
                invalid.attach_lora(
                    1,
                    std::numeric_limits<float>::infinity(),
                    random
                );
            },
            "nonfinite alpha is rejected"
        );
        require(!invalid.has_lora(), "invalid alpha is transactional");
        require_throws(
            [&] { invalid.merge_lora(); },
            "merge before attach is rejected"
        );
    }
}

void test_forward_backward_merge_and_retained_storage() {
    Linear linear = make_linear();
    std::mt19937 random(83U);
    linear.attach_lora(2, 4.0F, random);
    ParameterList lora = linear.lora_parameters();
    Parameter* stale_a = parameter_named(lora, "lora_a.weight");
    Parameter* stale_b = parameter_named(lora, "lora_b.weight");
    stale_a->set_value(
        Tensor(
            {2, 3},
            {
                1.0F, 0.0F, 0.0F,
                0.0F, 1.0F, 0.0F,
            }
        )
    );
    stale_b->set_value(
        Tensor(
            {2, 2},
            {
                1.0F, 2.0F,
                -1.0F, 3.0F,
            }
        )
    );

    const Variable input(
        Tensor({3}, {1.0F, 2.0F, 3.0F}),
        true
    );
    const Variable output = linear.forward(input);
    require_tensor_close(
        output.value(),
        Tensor({2}, {24.5F, 14.5F}),
        "LoRA composes its scaled low-rank delta"
    );
    output.backward(Tensor({2}, {1.0F, -0.5F}));
    for (const auto& named_parameter : lora) {
        bool has_nonzero_gradient = false;
        for (const float value :
             named_parameter.parameter->gradient().data()) {
            require(
                std::isfinite(value),
                "LoRA gradient must be finite"
            );
            has_nonzero_gradient =
                has_nonzero_gradient || value != 0.0F;
        }
        require(
            has_nonzero_gradient,
            named_parameter.name + " receives a gradient"
        );
    }

    const Tensor before_merge = linear.forward(input).value();
    linear.merge_lora();
    require(!linear.has_lora(), "merge deactivates LoRA");
    require(
        linear.lora_parameters().empty(),
        "merged adapter is absent from public registration"
    );
    require_tensor_close(
        linear.forward(input).value(),
        before_merge,
        "merge preserves the forward result"
    );
    require(
        stale_a->value().shape() == Tensor::Shape({2, 3}) &&
            stale_b->value().shape() == Tensor::Shape({2, 2}),
        "merged adapter storage remains alive"
    );
    require_throws(
        [&] { linear.merge_lora(); },
        "double merge is rejected"
    );
    require_throws(
        [&] { linear.attach_lora(1, 1.0F, random); },
        "reattachment after merge is rejected"
    );

    if (transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        transformer_lab::Module& module = linear;
        module.to(ExecutionBackend::Metal);
        require(
            stale_a->value().backend() == ExecutionBackend::Metal &&
                stale_b->value().backend() == ExecutionBackend::Metal,
            "post-merge transfer reaches retained adapter storage"
        );
        const Tensor metal_output = linear.forward(
            Variable(
                Tensor(
                    {3},
                    {1.0F, 2.0F, 3.0F},
                    ExecutionBackend::Metal
                ),
                false
            )
        ).value();
        require_tensor_close(
            metal_output,
            before_merge,
            "merged Linear has Metal parity",
            5.0e-4F
        );
    }
}

}  // namespace

int main() {
    try {
        test_registration_initialization_and_validation();
        test_forward_backward_merge_and_retained_storage();
        std::cout << "LoRA layer tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LoRA layer test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
