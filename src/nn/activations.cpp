#include "riftco_transformer/nn/activations.hpp"

#include "riftco_transformer/core/tensor_ops.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace riftco_transformer {
namespace {

Tensor relu_forward(const Tensor &input) {
  Tensor output(input.shape(), input.backend());
  for (std::size_t index = 0; index < input.numel(); ++index) {
    const float value = input.flat(index);
    output.flat(index) = std::isnan(value) ? value : std::fmax(0.0F, value);
  }
  return output;
}

Tensor relu_input_gradient(const Tensor &input, const Tensor &upstream) {
  if (input.shape() != upstream.shape() ||
      input.backend() != upstream.backend()) {
    throw std::invalid_argument("ReLU input and upstream gradient must match");
  }
  Tensor gradient(input.shape(), input.backend());
  for (std::size_t index = 0; index < input.numel(); ++index) {
    const float value = input.flat(index);
    gradient.flat(index) = std::isnan(value)
                               ? std::numeric_limits<float>::quiet_NaN()
                               : (value > 0.0F ? upstream.flat(index) : 0.0F);
  }
  return gradient;
}

float stable_sigmoid(float value) {
    if (value >= 0.0F) {
        const float exponential = std::exp(-value);
        return 1.0F / (1.0F + exponential);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

Tensor silu_forward(const Tensor& input) {
    Tensor output(input.shape(), input.backend());
    for (std::size_t index = 0; index < input.numel(); ++index) {
        const float value = input.flat(index);
        if (value == -std::numeric_limits<float>::infinity()) {
            output.flat(index) = -0.0F;
            continue;
        }
        output.flat(index) = value * stable_sigmoid(value);
    }
    return output;
}

Tensor silu_input_gradient(const Tensor& input, const Tensor& upstream) {
    if (input.shape() != upstream.shape() ||
        input.backend() != upstream.backend()) {
        throw std::invalid_argument("SiLU input and upstream gradient must match");
    }
    Tensor gradient(input.shape(), input.backend());
    for (std::size_t index = 0; index < input.numel(); ++index) {
        const float value = input.flat(index);
        if (value == -std::numeric_limits<float>::infinity()) {
            gradient.flat(index) = 0.0F;
            continue;
        }
        if (value == std::numeric_limits<float>::infinity()) {
            gradient.flat(index) = upstream.flat(index);
            continue;
        }
        const float sigmoid = stable_sigmoid(value);
        gradient.flat(index) = upstream.flat(index) *
            (sigmoid + value * sigmoid * (1.0F - sigmoid));
    }
    return gradient;
}

} // namespace

Variable gelu(const Variable& input) {
    const auto input_node = input.node_;
    return Variable::from_operation(
        tensor_ops::gelu(input.value()),
        {input_node},
        [input_node, input](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::gelu_backward(input.value(), upstream)
            );
        }
    );
}

Variable relu(const Variable &input) {
  const auto input_node = input.node_;
  return Variable::from_operation(
      relu_forward(input.value()), {input_node},
      [input_node, input](const Tensor &upstream) {
        Variable::accumulate_gradient(
            input_node, relu_input_gradient(input.value(), upstream));
      });
}

Variable silu(const Variable& input) {
    const std::array inputs{input};
    return custom_gradient(
        silu_forward(input.value()),
        inputs,
        [input](const Tensor& upstream) {
            return std::vector<Tensor>{silu_input_gradient(
                input.value(),
                upstream
            )};
        }
    );
}

Variable softmax(const Variable& input, std::size_t axis) {
    const Tensor probabilities = tensor_ops::softmax(input.value(), axis);
    const auto input_node = input.node_;
    return Variable::from_operation(
        probabilities,
        {input_node},
        [input_node, probabilities, axis](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::softmax_backward(probabilities, upstream, axis)
            );
        }
    );
}

}  // namespace riftco_transformer
