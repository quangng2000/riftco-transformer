#include "transformer_lab/core/autograd.hpp"
#include "transformer_lab/core/tensor_ops.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace transformer_lab {
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

struct Variable::Node {
    Tensor value;
    Tensor gradient;
    bool requires_gradient;
    std::uint64_t sequence;
    std::uint64_t value_version = 0;
    std::vector<std::shared_ptr<Node>> parents;
    std::vector<std::uint64_t> parent_value_versions;
    BackwardFunction backward;

    Node(Tensor node_value, bool node_requires_gradient)
        : value(std::move(node_value)),
          gradient(Tensor::zeros(value.shape(), value.backend())),
          requires_gradient(node_requires_gradient),
          sequence(allocate_node_sequence()) {}
};

namespace {

Variable constant_like(const Variable& value, float scalar) {
    return Variable(
        Tensor::full(value.value().shape(), scalar, value.value().backend()),
        false
    );
}

void require_differentiable_sqrt_domain(const Tensor& value) {
    for (const float element : value.data()) {
        if (element <= 0.0F) {
            throw std::domain_error(
                "sqrt autograd requires strictly positive values"
            );
        }
    }
}

Tensor::Shape swap_last_two_axes(std::size_t rank) {
    Tensor::Shape axes;
    axes.reserve(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
        axes.push_back(axis);
    }
    std::swap(axes[rank - 2], axes[rank - 1]);
    return axes;
}

}  // namespace

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

Variable operator+(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    return Variable::from_operation(
        tensor_ops::add(left.value(), right.value()),
        {left_node, right_node},
        [left_node, right_node](const Tensor& upstream) {
            Variable::accumulate_gradient(left_node, upstream);
            Variable::accumulate_gradient(right_node, upstream);
        }
    );
}

Variable operator-(const Variable& value) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::negate(value.value()),
        {input_node},
        [input_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::negate(upstream)
            );
        }
    );
}

Variable operator-(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    return Variable::from_operation(
        tensor_ops::subtract(left.value(), right.value()),
        {left_node, right_node},
        [left_node, right_node](const Tensor& upstream) {
            Variable::accumulate_gradient(left_node, upstream);
            Variable::accumulate_gradient(
                right_node,
                tensor_ops::negate(upstream)
            );
        }
    );
}

Variable operator*(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    return Variable::from_operation(
        tensor_ops::multiply(left.value(), right.value()),
        {left_node, right_node},
        [left_node, right_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                left_node,
                tensor_ops::multiply(upstream, right_node->value)
            );
            Variable::accumulate_gradient(
                right_node,
                tensor_ops::multiply(upstream, left_node->value)
            );
        }
    );
}

Variable operator/(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    return Variable::from_operation(
        tensor_ops::divide(left.value(), right.value()),
        {left_node, right_node},
        [left_node, right_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                left_node,
                tensor_ops::divide(upstream, right_node->value)
            );
            const Tensor denominator_squared =
                tensor_ops::multiply(right_node->value, right_node->value);
            const Tensor numerator =
                tensor_ops::multiply(upstream, left_node->value);
            Variable::accumulate_gradient(
                right_node,
                tensor_ops::negate(
                    tensor_ops::divide(numerator, denominator_squared)
                )
            );
        }
    );
}

Variable operator+(const Variable& value, float scalar) {
    return value + constant_like(value, scalar);
}

Variable operator+(float scalar, const Variable& value) {
    return value + scalar;
}

Variable operator-(const Variable& value, float scalar) {
    return value - constant_like(value, scalar);
}

Variable operator-(float scalar, const Variable& value) {
    return constant_like(value, scalar) - value;
}

Variable operator*(const Variable& value, float scalar) {
    return value * constant_like(value, scalar);
}

Variable operator*(float scalar, const Variable& value) {
    return value * scalar;
}

Variable operator/(const Variable& value, float scalar) {
    return value / constant_like(value, scalar);
}

Variable operator/(float scalar, const Variable& value) {
    return constant_like(value, scalar) / value;
}

Variable matmul(const Variable& left, const Variable& right) {
    const auto left_node = left.node_;
    const auto right_node = right.node_;
    const auto backend = left.value().backend();
    return Variable::from_operation(
        tensor_ops::matmul(left.value(), right.value(), backend),
        {left_node, right_node},
        [left_node, right_node, backend](const Tensor& upstream) {
            const auto right_transpose_axes =
                swap_last_two_axes(right_node->value.rank());
            Variable::accumulate_gradient(
                left_node,
                tensor_ops::matmul(
                    upstream,
                    tensor_ops::permute(
                        right_node->value,
                        right_transpose_axes
                    ),
                    backend
                )
            );
            const auto left_transpose_axes =
                swap_last_two_axes(left_node->value.rank());
            Variable::accumulate_gradient(
                right_node,
                tensor_ops::matmul(
                    tensor_ops::permute(left_node->value, left_transpose_axes),
                    upstream,
                    backend
                )
            );
        }
    );
}

Variable permute(const Variable& value, Tensor::Shape axes) {
    Tensor::Shape inverse_axes(axes.size(), 0);
    for (std::size_t output_axis = 0; output_axis < axes.size();
         ++output_axis) {
        const auto input_axis = axes[output_axis];
        if (input_axis < inverse_axes.size()) {
            inverse_axes[input_axis] = output_axis;
        }
    }

    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::permute(value.value(), std::move(axes)),
        {input_node},
        [input_node, inverse_axes](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::permute(upstream, inverse_axes)
            );
        }
    );
}

Variable sum(const Variable& value) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::sum(value.value()),
        {input_node},
        [input_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::broadcast_to(upstream, input_node->value.shape())
            );
        }
    );
}

Variable sum(const Variable& value, std::size_t axis, bool keep_dimensions) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::sum(value.value(), axis, keep_dimensions),
        {input_node},
        [input_node, axis, keep_dimensions](const Tensor& upstream) {
            Tensor broadcast_source = upstream;
            if (!keep_dimensions) {
                auto kept_shape = input_node->value.shape();
                kept_shape[axis] = 1;
                broadcast_source = upstream.reshape(std::move(kept_shape));
            }
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::broadcast_to(
                    broadcast_source,
                    input_node->value.shape()
                )
            );
        }
    );
}

Variable mean(const Variable& value) {
    const auto element_count = value.value().numel();
    const auto input_shape = value.value().shape();
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::mean(value.value()),
        {input_node},
        [input_node, input_shape, element_count](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::broadcast_to(
                    tensor_ops::scale(
                        upstream,
                        1.0F / static_cast<float>(element_count)
                    ),
                    input_shape
                )
            );
        }
    );
}

Variable mean(const Variable& value, std::size_t axis, bool keep_dimensions) {
    if (axis >= value.value().rank()) {
        throw std::out_of_range("mean axis is outside tensor rank");
    }
    const auto input_shape = value.value().shape();
    const auto width = value.value().shape()[axis];
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::mean(value.value(), axis, keep_dimensions),
        {input_node},
        [input_node, input_shape, axis, keep_dimensions, width](
            const Tensor& upstream
        ) {
            Tensor broadcast_source =
                tensor_ops::scale(upstream, 1.0F / static_cast<float>(width));
            if (!keep_dimensions) {
                auto kept_shape = input_shape;
                kept_shape[axis] = 1;
                broadcast_source =
                    broadcast_source.reshape(std::move(kept_shape));
            }
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::broadcast_to(broadcast_source, input_shape)
            );
        }
    );
}

Variable exp(const Variable& value) {
    const Tensor output = tensor_ops::exp(value.value());
    const Tensor derivative = output;
    const auto input_node = value.node_;
    return Variable::from_operation(
        output,
        {input_node},
        [input_node, derivative](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::multiply(upstream, derivative)
            );
        }
    );
}

Variable log(const Variable& value) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::log(value.value()),
        {input_node},
        [input_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::divide(upstream, input_node->value)
            );
        }
    );
}

Variable sqrt(const Variable& value) {
    require_differentiable_sqrt_domain(value.value());
    const Tensor output = tensor_ops::sqrt(value.value());
    const Tensor derivative_denominator = tensor_ops::scale(output, 2.0F);
    const auto input_node = value.node_;
    return Variable::from_operation(
        output,
        {input_node},
        [input_node, derivative_denominator](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::divide(upstream, derivative_denominator)
            );
        }
    );
}

Variable erf(const Variable& value) {
    const Tensor input_squared =
        tensor_ops::multiply(value.value(), value.value());
    const Tensor derivative = tensor_ops::scale(
        tensor_ops::exp(tensor_ops::negate(input_squared)),
        2.0F / std::sqrt(std::numbers::pi_v<float>)
    );
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::erf(value.value()),
        {input_node},
        [input_node, derivative](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::multiply(upstream, derivative)
            );
        }
    );
}

Variable reshape(const Variable& value, Tensor::Shape new_shape) {
    const auto original_shape = value.value().shape();
    const auto input_node = value.node_;
    return Variable::from_operation(
        value.value().reshape(std::move(new_shape)),
        {input_node},
        [input_node, original_shape](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                upstream.reshape(original_shape)
            );
        }
    );
}

Variable transpose_2d(const Variable& value) {
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::transpose_2d(value.value()),
        {input_node},
        [input_node](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::transpose_2d(upstream)
            );
        }
    );
}

Variable broadcast_to(const Variable& value, Tensor::Shape output_shape) {
    const auto input_shape = value.value().shape();
    const auto input_node = value.node_;
    return Variable::from_operation(
        tensor_ops::broadcast_to(value.value(), std::move(output_shape)),
        {input_node},
        [input_node, input_shape](const Tensor& upstream) {
            Variable::accumulate_gradient(
                input_node,
                tensor_ops::sum_to_shape(upstream, input_shape)
            );
        }
    );
}

Variable gather_rows(
    const Variable& table,
    std::span<const std::size_t> row_indices,
    Tensor::Shape index_shape
) {
    const Tensor output = tensor_ops::gather_rows(
        table.value(),
        row_indices,
        std::move(index_shape)
    );
    const auto table_shape = table.value().shape();
    const std::vector<std::size_t> owned_indices(
        row_indices.begin(),
        row_indices.end()
    );
    const auto table_node = table.node_;

    return Variable::from_operation(
        output,
        {table_node},
        [table_node, table_shape, owned_indices](const Tensor& upstream) {
            Variable::accumulate_gradient(
                table_node,
                tensor_ops::scatter_add_rows(
                    upstream,
                    owned_indices,
                    table_shape[0]
                )
            );
        }
    );
}

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

    Variable replay_input(input.value(), input.requires_gradient());
    Variable replay_output = recompute(replay_input);
    validate_replay_graph(replay_output, replay_input);

    const Tensor::Shape output_shape = replay_output.value().shape();
    const ExecutionBackend output_backend =
        replay_output.value().backend();
    Tensor output_value = replay_output.value();
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
            Variable replay_input(
                input_node->value,
                input_node->requires_gradient
            );
            Variable replay_output = recompute(replay_input);
            if (replay_output.value().shape() != output_shape ||
                replay_output.value().backend() != output_backend ||
                !replay_output.requires_gradient()) {
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
                if (current.get() == replay_input.node_.get()) {
                    saw_input = true;
                    return;
                }
                if (declared.contains(current.get())) {
                    saw_dependencies.insert(current.get());
                    return;
                }
                if (current->sequence < replay_input.node_->sequence) {
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
            visit(replay_output.node_);
            if (!saw_input ||
                saw_dependencies.size() != dependency_nodes.size()) {
                throw std::logic_error(
                    "checkpoint recompute dependencies changed "
                    "after forward"
                );
            }

            std::vector<Variable> boundaries;
            boundaries.reserve(dependency_nodes.size() + 1);
            boundaries.push_back(replay_input);
            for (const auto& dependency : dependency_nodes) {
                boundaries.push_back(Variable(dependency));
            }
            std::vector<Tensor> gradients =
                replay_output.vector_jacobian_product(
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

}  // namespace transformer_lab
