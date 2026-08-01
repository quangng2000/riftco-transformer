#include "core/autograd/detail/node.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer {

Variable custom_gradient(
    Tensor output,
    std::span<const Variable> inputs,
    CustomGradientFunction vector_jacobian_product
) {
    if (!vector_jacobian_product) {
        throw std::invalid_argument(
            "custom gradient operation requires a VJP callback"
        );
    }
    if (inputs.empty()) {
        throw std::invalid_argument(
            "custom gradient operation requires at least one input"
        );
    }

    const ExecutionBackend output_backend = output.backend();
    std::vector<std::shared_ptr<Variable::Node>> parent_nodes;
    std::vector<Tensor::Shape> input_shapes;
    std::vector<ExecutionBackend> input_backends;
    parent_nodes.reserve(inputs.size());
    input_shapes.reserve(inputs.size());
    input_backends.reserve(inputs.size());
    for (const Variable& input : inputs) {
        if (input.value().backend() != output_backend) {
            throw std::invalid_argument(
                "custom gradient inputs and output must use the same backend"
            );
        }
        parent_nodes.push_back(input.node_);
        input_shapes.push_back(input.value().shape());
        input_backends.push_back(input.value().backend());
    }

    auto backward_nodes = parent_nodes;
    return Variable::from_operation(
        std::move(output),
        std::move(parent_nodes),
        [
            nodes = std::move(backward_nodes),
            shapes = std::move(input_shapes),
            backends = std::move(input_backends),
            vector_jacobian_product =
                std::move(vector_jacobian_product)
        ](const Tensor& upstream) {
            std::vector<Tensor> contributions =
                vector_jacobian_product(upstream);
            if (contributions.size() != nodes.size()) {
                throw std::invalid_argument(
                    "custom gradient VJP must return one Tensor per input"
                );
            }

            // Validate the complete callback result before accumulating any
            // contribution. evaluate_gradients() also keeps all accumulation
            // isolated until the full backward pass succeeds.
            for (std::size_t index = 0;
                 index < contributions.size();
                 ++index) {
                if (contributions[index].shape() != shapes[index]) {
                    throw std::invalid_argument(
                        "custom gradient VJP Tensor shape must match its input"
                    );
                }
                if (contributions[index].backend() != backends[index]) {
                    throw std::invalid_argument(
                        "custom gradient VJP Tensor backend must match its input"
                    );
                }
            }
            for (std::size_t index = 0;
                 index < contributions.size();
                 ++index) {
                Variable::accumulate_gradient(
                    nodes[index],
                    contributions[index]
                );
            }
        }
    );
}

}  // namespace riftco_transformer
