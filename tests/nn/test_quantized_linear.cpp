#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/quantized_weight.hpp"
#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/quantized_linear.hpp"
#include "riftco_transformer/optim/adam.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using riftco_transformer::Adam;
using riftco_transformer::ExecutionBackend;
using riftco_transformer::Linear;
using riftco_transformer::Module;
using riftco_transformer::Nf4Payload;
using riftco_transformer::QuantizedWeight;
using riftco_transformer::Tensor;
using riftco_transformer::Variable;
using riftco_transformer::quantized_linear;
using riftco_transformer::sum;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    const std::string& message,
    float tolerance = 1.0e-6F
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
    float tolerance = 1.0e-6F
) {
    require(actual.shape() == expected.shape(), message + ": shape");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
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

Tensor reference_forward(
    const Tensor& input,
    const Tensor& weight
) {
    const std::size_t rows = input.shape()[0];
    const std::size_t input_width = input.shape()[1];
    const std::size_t output_width = weight.shape()[0];
    Tensor output(
        {rows, output_width},
        input.backend()
    );
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t output_column = 0;
             output_column < output_width;
             ++output_column) {
            float total = 0.0F;
            for (std::size_t input_column = 0;
                 input_column < input_width;
                 ++input_column) {
                total += input.at({row, input_column}) *
                    weight.at({output_column, input_column});
            }
            output.at({row, output_column}) = total;
        }
    }
    return output;
}

Tensor reference_input_gradient(
    const Tensor& upstream,
    const Tensor& weight
) {
    const std::size_t rows = upstream.shape()[0];
    const std::size_t output_width = upstream.shape()[1];
    const std::size_t input_width = weight.shape()[1];
    Tensor gradient(
        {rows, input_width},
        upstream.backend()
    );
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t input_column = 0;
             input_column < input_width;
             ++input_column) {
            float total = 0.0F;
            for (std::size_t output_column = 0;
                 output_column < output_width;
                 ++output_column) {
                total += upstream.at({row, output_column}) *
                    weight.at({output_column, input_column});
            }
            gradient.at({row, input_column}) = total;
        }
    }
    return gradient;
}

void test_forward_backward_and_graph_memory() {
    const Tensor dense_weight(
        {2, 3},
        {-1.0F, 0.0F, 1.0F, 0.5F, -0.5F, 0.25F},
        ExecutionBackend::Cpu
    );
    const QuantizedWeight weight =
        QuantizedWeight::quantize_nf4(dense_weight, 32);
    const Tensor dequantized = weight.dequantize();
    const Tensor input_value(
        {2, 3},
        {1.0F, 2.0F, 3.0F, -2.0F, 0.5F, 4.0F},
        ExecutionBackend::Cpu
    );
    const Variable input(input_value, true);
    const Variable output = quantized_linear(input, weight);

    require_tensor_close(
        output.value(),
        reference_forward(input_value, dequantized),
        "quantized-linear forward"
    );
    const auto graph = output.graph_statistics();
    require(
        graph.node_count == 2,
        "quantized linear should add one node with input as its only parent"
    );
    require(
        graph.node_tensor_elements == 20,
        "autograd graph must not contain a full-size weight or gradient"
    );

    const Tensor upstream(
        {2, 2},
        {0.25F, -1.0F, 2.0F, 0.5F},
        ExecutionBackend::Cpu
    );
    output.backward(upstream);
    require_tensor_close(
        input.gradient(),
        reference_input_gradient(upstream, dequantized),
        "quantized-linear input gradient"
    );
}

void test_linear_qlora_composition_and_export() {
    const Tensor dense_weight(
        {2, 3},
        {-1.0F, 0.0F, 1.0F, 0.5F, -0.5F, 0.25F},
        ExecutionBackend::Cpu
    );
    Linear linear(
        dense_weight,
        Tensor({2}, {0.1F, -0.2F}, ExecutionBackend::Cpu)
    );
    {
        const auto retained = linear.parameters();
        require_throws(
            [&] { linear.quantize_weight_nf4(32); },
            "quantization must reject retained dense Parameter handles"
        );
        require(retained.size() == 2, "dense Linear parameter count");
    }

    linear.quantize_weight_nf4(32);
    require(linear.has_quantized_weight(), "Linear owns packed weight");
    require_throws(
        [&] { static_cast<void>(linear.weight()); },
        "packed Linear should not expose a trainable weight Parameter"
    );
    const auto frozen_parameters = linear.parameters();
    require(
        frozen_parameters.size() == 1 &&
            frozen_parameters.front().name == "bias",
        "packed weight must be absent from the base Parameter list"
    );
    const Nf4Payload packed_before =
        linear.quantized_weight().copy_payload_to_host();

    std::mt19937 random(457U);
    linear.attach_lora(2, 4.0F, random);
    const auto adapters = linear.lora_parameters();
    require(adapters.size() == 2, "QLoRA exposes only A and B adapters");
    Adam optimizer(adapters);
    require(
        optimizer.parameter_tensor_count() == 2,
        "Adam state is allocated only for floating-point adapters"
    );

    const Tensor input_value(
        {2, 3},
        {1.0F, 2.0F, 3.0F, -2.0F, 0.5F, 4.0F},
        ExecutionBackend::Cpu
    );
    const Variable input(input_value, true);
    const Variable loss = sum(linear.forward(input));
    loss.backward();
    static_cast<void>(optimizer.step());

    const Nf4Payload packed_after =
        linear.quantized_weight().copy_payload_to_host();
    require(
        packed_after.packed_codes == packed_before.packed_codes &&
            packed_after.block_scales == packed_before.block_scales,
        "adapter training must leave the packed base weight bit-exact"
    );

    const Variable before_merge_input(input_value, false);
    const Tensor output_before_merge =
        linear.forward(before_merge_input).value();
    linear.merge_lora();
    require(
        !linear.has_quantized_weight(),
        "explicit export merge materializes the dense weight"
    );
    const auto exported_parameters = linear.parameters();
    require(
        exported_parameters.size() == 2 &&
            exported_parameters.front().name == "weight",
        "export merge restores the ordinary parameter schema"
    );
    require_tensor_close(
        linear.forward(Variable(input_value, false)).value(),
        output_before_merge,
        "materialized QLoRA export",
        5.0e-5F
    );
}

void test_polymorphic_packed_transfer_if_metal_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    Linear linear(
        Tensor(
            {2, 3},
            {-1.0F, 0.0F, 1.0F, 0.5F, -0.5F, 0.25F},
            ExecutionBackend::Cpu
        ),
        Tensor({2}, {0.1F, -0.2F}, ExecutionBackend::Cpu)
    );
    linear.quantize_weight_nf4(32);
    std::mt19937 random(479U);
    linear.attach_lora(2, 4.0F, random);

    Module& module = linear;
    module.to(ExecutionBackend::Metal);
    require(
        linear.quantized_weight().backend() == ExecutionBackend::Metal &&
            linear.bias().value().backend() == ExecutionBackend::Metal,
        "virtual Module transfer must move packed and ordinary Linear state"
    );
    for (const auto& adapter : linear.lora_parameters()) {
        require(
            adapter.parameter->value().backend() == ExecutionBackend::Metal,
            "virtual Module transfer must move QLoRA adapters"
        );
    }
    const Tensor input(
        {1, 3},
        {1.0F, 2.0F, 3.0F},
        ExecutionBackend::Metal
    );
    require(
        linear.forward(Variable(input, false)).value().backend() ==
            ExecutionBackend::Metal,
        "transferred packed Linear must execute on Metal"
    );

    module.to(ExecutionBackend::Cpu);
    require(
        linear.quantized_weight().backend() == ExecutionBackend::Cpu &&
            linear.bias().value().backend() == ExecutionBackend::Cpu,
        "virtual Module transfer must move packed Linear state back to CPU"
    );
}

}  // namespace

int main() {
    try {
        test_forward_backward_and_graph_memory();
        test_linear_qlora_composition_and_export();
        test_polymorphic_packed_transfer_if_metal_available();
        std::cout << "quantized Linear tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "quantized Linear test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
