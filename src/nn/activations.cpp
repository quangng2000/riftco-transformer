#include "riftco_transformer/nn/activations.hpp"

#include "riftco_transformer/core/tensor_ops.hpp"

namespace riftco_transformer {

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
