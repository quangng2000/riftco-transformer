#include "core/autograd/detail/node.hpp"

#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace riftco_transformer {

Variable checkpoint(
    const Variable& input,
    std::span<const Variable> dependencies,
    std::function<Variable(const Variable&)> recompute
) {
    if (!recompute) {
        throw std::invalid_argument(
            "checkpoint recompute function must not be empty"
        );
    }
    if (!input.requires_gradient() && dependencies.empty()) {
        throw std::invalid_argument(
            "checkpoint requires a differentiable input or dependency"
        );
    }

    std::vector<std::shared_ptr<Variable::Node>> dependency_nodes;
    dependency_nodes.reserve(dependencies.size());
    std::unordered_set<const Variable::Node*> declared;
    declared.reserve(dependencies.size());
    for (const Variable& dependency : dependencies) {
        if (!dependency.requires_gradient() || !dependency.is_leaf()) {
            throw std::invalid_argument(
                "checkpoint dependencies must be differentiable leaves"
            );
        }
        if (dependency.value().backend() != input.value().backend()) {
            throw std::invalid_argument(
                "checkpoint dependencies must use the input backend"
            );
        }
        if (dependency.node_.get() == input.node_.get() ||
            !declared.insert(dependency.node_.get()).second) {
            throw std::invalid_argument(
                "checkpoint input and dependencies must be unique"
            );
        }
        dependency_nodes.push_back(dependency.node_);
    }

    const auto validate_replay_graph =
        [&](const Variable& output,
            const Variable& replay_input) {
            if (!output.requires_gradient()) {
                throw std::logic_error(
                    "checkpoint recompute output must require a gradient"
                );
            }
            if (output.value().backend() != input.value().backend()) {
                throw std::invalid_argument(
                    "checkpoint recompute output must use the input backend"
                );
            }

            bool saw_input = false;
            std::unordered_set<const Variable::Node*> saw_dependencies;
            std::unordered_set<const Variable::Node*> visited;
            std::function<void(
                const std::shared_ptr<Variable::Node>&
            )> visit = [&](const auto& current) {
                if (!visited.insert(current.get()).second) {
                    return;
                }
                if (current.get() == replay_input.node_.get()) {
                    saw_input = true;
                    return;
                }
                if (declared.contains(current.get())) {
                    saw_dependencies.insert(current.get());
                    return;
                }
                if (current->sequence < replay_input.node_->sequence) {
                    throw std::invalid_argument(
                        "checkpoint recompute captured undeclared "
                        "external graph state"
                    );
                }
                if (current->parents.empty() &&
                    current->requires_gradient) {
                    throw std::invalid_argument(
                        "checkpoint recompute created an undeclared "
                        "differentiable leaf"
                    );
                }
                for (const auto& parent : current->parents) {
                    visit(parent);
                }
            };
            visit(output.node_);
            if (!saw_input) {
                throw std::invalid_argument(
                    "checkpoint recompute output must depend on its input"
                );
            }
            if (saw_dependencies.size() != dependency_nodes.size()) {
                throw std::invalid_argument(
                    "checkpoint recompute did not use every dependency"
                );
            }
        };

    Variable forward_replay_input(input.value(), input.requires_gradient());
    Variable forward_replay_output = recompute(forward_replay_input);
    validate_replay_graph(forward_replay_output, forward_replay_input);

    const Tensor::Shape output_shape = forward_replay_output.value().shape();
    const ExecutionBackend output_backend =
        forward_replay_output.value().backend();
    Tensor output_value = forward_replay_output.value();
    std::vector<std::shared_ptr<Variable::Node>> parents;
    parents.reserve(dependency_nodes.size() + 1);
    parents.push_back(input.node_);
    parents.insert(
        parents.end(),
        dependency_nodes.begin(),
        dependency_nodes.end()
    );

    const auto input_node = input.node_;
    return Variable::from_operation(
        std::move(output_value),
        std::move(parents),
        [
            input_node,
            dependency_nodes = std::move(dependency_nodes),
            declared = std::move(declared),
            output_shape,
            output_backend,
            recompute = std::move(recompute)
        ](const Tensor& upstream) {
            Variable backward_replay_input(
                input_node->value,
                input_node->requires_gradient
            );
            Variable backward_replay_output =
                recompute(backward_replay_input);
            if (backward_replay_output.value().shape() != output_shape ||
                backward_replay_output.value().backend() != output_backend ||
                !backward_replay_output.requires_gradient()) {
                throw std::logic_error(
                    "checkpoint recompute contract changed after forward"
                );
            }

            bool saw_input = false;
            std::unordered_set<const Variable::Node*> saw_dependencies;
            std::unordered_set<const Variable::Node*> visited;
            std::function<void(
                const std::shared_ptr<Variable::Node>&
            )> visit = [&](const auto& current) {
                if (!visited.insert(current.get()).second) {
                    return;
                }
                if (current.get() == backward_replay_input.node_.get()) {
                    saw_input = true;
                    return;
                }
                if (declared.contains(current.get())) {
                    saw_dependencies.insert(current.get());
                    return;
                }
                if (current->sequence <
                    backward_replay_input.node_->sequence) {
                    throw std::logic_error(
                        "checkpoint recompute captured different "
                        "external graph state"
                    );
                }
                if (current->parents.empty() &&
                    current->requires_gradient) {
                    throw std::logic_error(
                        "checkpoint recompute created a differentiable "
                        "leaf outside its declared dependencies"
                    );
                }
                for (const auto& parent : current->parents) {
                    visit(parent);
                }
            };
            visit(backward_replay_output.node_);
            if (!saw_input ||
                saw_dependencies.size() != dependency_nodes.size()) {
                throw std::logic_error(
                    "checkpoint recompute dependencies changed "
                    "after forward"
                );
            }

            std::vector<Variable> boundaries;
            boundaries.reserve(dependency_nodes.size() + 1);
            boundaries.push_back(backward_replay_input);
            for (const auto& dependency : dependency_nodes) {
                boundaries.push_back(Variable(dependency));
            }
            std::vector<Tensor> gradients =
                backward_replay_output.vector_jacobian_product(
                    upstream,
                    boundaries
                );
            Variable::accumulate_gradient(
                input_node,
                gradients.front()
            );
            for (std::size_t index = 0;
                 index < dependency_nodes.size();
                 ++index) {
                Variable::accumulate_gradient(
                    dependency_nodes[index],
                    gradients[index + 1]
                );
            }
        }
    );
}

}  // namespace riftco_transformer
