#include "core/autograd/detail/node.hpp"

#include "riftco_transformer/core/tensor_ops.hpp"

#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace riftco_transformer {
namespace {

std::atomic_uint64_t next_node_sequence{0};

std::uint64_t allocate_node_sequence() {
    std::uint64_t current =
        next_node_sequence.load(std::memory_order_relaxed);
    while (true) {
        if (current == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("autograd node sequence overflow");
        }
        if (next_node_sequence.compare_exchange_weak(
                current,
                current + 1,
                std::memory_order_relaxed,
                std::memory_order_relaxed
            )) {
            return current;
        }
    }
}

}  // namespace

Variable::Node::Node(Tensor node_value, bool node_requires_gradient)
    : value(std::move(node_value)),
      gradient(Tensor::zeros(value.shape(), value.backend())),
      requires_gradient(node_requires_gradient),
      sequence(allocate_node_sequence()) {}

struct Variable::GradientContext {
    explicit GradientContext(
        const std::vector<std::shared_ptr<Node>>& graph_nodes
    ) : nodes(graph_nodes) {
        indices.reserve(graph_nodes.size());
        gradients.reserve(graph_nodes.size());
        for (std::size_t index = 0; index < graph_nodes.size(); ++index) {
            indices.emplace(graph_nodes[index].get(), index);
            gradients.push_back(Tensor::zeros(
                graph_nodes[index]->value.shape(),
                graph_nodes[index]->value.backend()
            ));
        }
    }

    [[nodiscard]] Tensor& at(const std::shared_ptr<Node>& node) {
        const auto found = indices.find(node.get());
        if (found == indices.end()) {
            throw std::logic_error(
                "autograd gradient context is missing a graph node"
            );
        }
        return gradients[found->second];
    }

    [[nodiscard]] const Tensor& at(
        const std::shared_ptr<Node>& node
    ) const {
        const auto found = indices.find(node.get());
        if (found == indices.end()) {
            throw std::logic_error(
                "autograd gradient context is missing a graph node"
            );
        }
        return gradients[found->second];
    }

    std::vector<std::shared_ptr<Node>> nodes;
    std::unordered_map<const Node*, std::size_t> indices;
    std::vector<Tensor> gradients;
};

thread_local Variable::GradientContext*
    Variable::active_gradient_context_ = nullptr;

Variable::Variable(Tensor value, bool requires_gradient)
    : node_(std::make_shared<Node>(std::move(value), requires_gradient)) {}

Variable::Variable(std::shared_ptr<Node> node) : node_(std::move(node)) {}

Variable Variable::scalar(float value, bool requires_gradient) {
    return Variable(Tensor(Tensor::Shape{}, value), requires_gradient);
}

const Tensor& Variable::value() const noexcept {
    return node_->value;
}

const Tensor& Variable::gradient() const noexcept {
    return node_->gradient;
}

bool Variable::requires_gradient() const noexcept {
    return node_->requires_gradient;
}

bool Variable::is_leaf() const noexcept {
    return node_->parents.empty();
}

AutogradGraphStatistics Variable::graph_statistics() const {
    AutogradGraphStatistics result;
    std::unordered_set<const Node*> visited;
    std::function<void(const std::shared_ptr<Node>&)> visit =
        [&](const std::shared_ptr<Node>& current) {
            if (!visited.insert(current.get()).second) {
                return;
            }
            if (result.node_count ==
                std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error(
                    "autograd graph node count overflow"
                );
            }
            ++result.node_count;
            for (const std::size_t owned_elements :
                 {current->value.numel(),
                  current->gradient.numel()}) {
                if (result.node_tensor_elements >
                    std::numeric_limits<std::size_t>::max() -
                        owned_elements) {
                    throw std::overflow_error(
                        "autograd graph Tensor element count overflow"
                    );
                }
                result.node_tensor_elements += owned_elements;
            }
            for (const auto& parent : current->parents) {
                visit(parent);
            }
        };
    visit(node_);
    return result;
}

void Variable::zero_gradient() const {
    std::fill(
        node_->gradient.data().begin(),
        node_->gradient.data().end(),
        0.0F
    );
}

void Variable::replace_leaf_value(Tensor value) {
    if (!node_->parents.empty()) {
        throw std::logic_error("only leaf Variables can replace their values");
    }
    if (value.shape() != node_->value.shape()) {
        throw std::invalid_argument(
            "replacement Variable value must keep the same shape"
        );
    }
    if (value.backend() == node_->value.backend()) {
        std::fill(
            node_->gradient.data().begin(),
            node_->gradient.data().end(),
            0.0F
        );
        node_->value = std::move(value);
        ++node_->value_version;
        return;
    }
    Tensor replacement_gradient = Tensor::zeros(value.shape(), value.backend());
    node_->value = std::move(value);
    node_->gradient = std::move(replacement_gradient);
    ++node_->value_version;
}

void Variable::replace_leaf_state(Tensor value, Tensor gradient) noexcept {
    node_->value = std::move(value);
    node_->gradient = std::move(gradient);
    ++node_->value_version;
}

Variable Variable::from_operation(
    Tensor value,
    std::vector<std::shared_ptr<Node>> parents,
    BackwardFunction backward
) {
    const bool requires_gradient =
        std::any_of(parents.begin(), parents.end(), [](const auto& parent) {
            return parent->requires_gradient;
        });
    auto node = std::make_shared<Node>(std::move(value), requires_gradient);
    if (requires_gradient) {
        node->parent_value_versions.reserve(parents.size());
        for (const auto& parent : parents) {
            node->parent_value_versions.push_back(parent->value_version);
        }
        node->parents = std::move(parents);
        node->backward = std::move(backward);
    }
    return Variable(std::move(node));
}

void Variable::accumulate_gradient(
    const std::shared_ptr<Node>& node,
    const Tensor& contribution
) {
    if (!node->requires_gradient) {
        return;
    }
    if (node->gradient.shape() != contribution.shape()) {
        throw std::invalid_argument(
            "gradient contribution shape does not match graph node"
        );
    }
    if (node->value.backend() != contribution.backend()) {
        throw std::invalid_argument(
            "gradient contribution backend does not match graph node"
        );
    }
    if (active_gradient_context_ != nullptr) {
        Tensor& destination = active_gradient_context_->at(node);
        destination = tensor_ops::add(destination, contribution);
        return;
    }
    node->gradient = tensor_ops::add(node->gradient, contribution);
}

void Variable::backward() const {
    if (node_->value.rank() != 0) {
        throw std::invalid_argument(
            "implicit backward seed requires a scalar output"
        );
    }
    backward(Tensor(Tensor::Shape{}, 1.0F, node_->value.backend()));
}

void Variable::backward(const Tensor& seed_gradient) const {
    auto result = evaluate_gradients(seed_gradient);
    for (std::size_t index = 0;
         index < result->nodes.size();
         ++index) {
        result->nodes[index]->gradient =
            std::move(result->gradients[index]);
    }
}

std::unique_ptr<Variable::GradientContext>
Variable::evaluate_gradients(const Tensor& seed_gradient) const {
    if (!node_->requires_gradient) {
        throw std::logic_error(
            "cannot call backward on a value that requires no gradient"
        );
    }
    if (seed_gradient.shape() != node_->value.shape()) {
        throw std::invalid_argument(
            "backward seed shape must match the output shape"
        );
    }
    if (seed_gradient.backend() != node_->value.backend()) {
        throw std::invalid_argument(
            "backward seed backend must match the output backend"
        );
    }
    const Tensor seed = seed_gradient;

    std::vector<std::shared_ptr<Node>> topological_order;
    std::unordered_set<const Node*> visited;
    std::function<void(const std::shared_ptr<Node>&)> visit =
        [&](const std::shared_ptr<Node>& current) {
            if (!visited.insert(current.get()).second) {
                return;
            }
            for (const auto& parent : current->parents) {
                visit(parent);
            }
            topological_order.push_back(current);
    };
    visit(node_);

    // Backward closures may read their parents' forward values. Validate the
    // entire graph before resetting any gradients so a parameter mutation
    // between forward and backward cannot silently change the derivative or
    // leave the graph partially processed.
    for (const auto& current : topological_order) {
        if (current->parent_value_versions.size() !=
            current->parents.size()) {
            throw std::logic_error(
                "autograd graph is missing parent value versions"
            );
        }
        for (std::size_t parent_index = 0;
             parent_index < current->parents.size();
             ++parent_index) {
            if (current->parents[parent_index]->value_version !=
                current->parent_value_versions[parent_index]) {
                throw std::logic_error(
                    "cannot call backward after a graph input value changed"
                );
            }
        }
    }

    auto result =
        std::make_unique<GradientContext>(topological_order);
    result->at(node_) = seed;

    GradientContext* const previous_context =
        active_gradient_context_;
    active_gradient_context_ = result.get();
    try {
        for (auto current = topological_order.rbegin();
             current != topological_order.rend();
             ++current) {
            if ((*current)->backward) {
                (*current)->backward(result->at(*current));
            }
        }
    } catch (...) {
        active_gradient_context_ = previous_context;
        throw;
    }
    active_gradient_context_ = previous_context;
    return result;
}

std::vector<Tensor> Variable::vector_jacobian_product(
    const Tensor& seed_gradient,
    std::span<const Variable> boundaries
) const {
    std::unordered_set<const Node*> seen;
    seen.reserve(boundaries.size());
    for (const Variable& boundary : boundaries) {
        if (!seen.insert(boundary.node_.get()).second) {
            throw std::invalid_argument(
                "VJP boundaries must be unique"
            );
        }
    }

    auto result = evaluate_gradients(seed_gradient);
    std::vector<Tensor> gradients;
    gradients.reserve(boundaries.size());
    for (const Variable& boundary : boundaries) {
        const auto found = result->indices.find(boundary.node_.get());
        if (found == result->indices.end()) {
            throw std::invalid_argument(
                "VJP boundary is not reachable from the output"
            );
        }
        gradients.push_back(
            std::move(result->gradients[found->second])
        );
    }
    return gradients;
}

}  // namespace riftco_transformer
