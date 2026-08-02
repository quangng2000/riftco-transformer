#pragma once

#include "riftco_transformer/core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace riftco_transformer {

class Parameter;

// Counts only graph nodes and Tensor elements owned directly by those nodes.
// Backend workspaces and Tensors captured by backward closures are not part of
// this diagnostic.
struct AutogradGraphStatistics {
  std::size_t node_count = 0;
  std::size_t node_tensor_elements = 0;
};

// A custom operation receives the gradient arriving at its output and returns
// one gradient contribution for each positional input.
using CustomGradientFunction =
    std::function<std::vector<Tensor>(const Tensor &)>;

// A Variable pairs a Tensor value with a node in a reverse-mode
// automatic-differentiation graph. Copies share the same graph node.
class Variable {
public:
  explicit Variable(Tensor value, bool requires_gradient = true);

  [[nodiscard]] static Variable scalar(float value,
                                       bool requires_gradient = true);

  [[nodiscard]] const Tensor &value() const noexcept;
  [[nodiscard]] const Tensor &gradient() const noexcept;
  [[nodiscard]] bool requires_gradient() const noexcept;
  [[nodiscard]] bool is_leaf() const noexcept;
  [[nodiscard]] AutogradGraphStatistics graph_statistics() const;

  // Clears only this node. backward() clears every reachable node before
  // computing a fresh gradient pass.
  void zero_gradient() const;

  // An implicit seed of one is allowed only for scalar outputs.
  void backward() const;
  void backward(const Tensor &seed_gradient) const;

private:
  struct Node;
  struct GradientContext;
  using BackwardFunction = std::function<void(const Tensor &)>;

  std::shared_ptr<Node> node_;
  static thread_local GradientContext *active_gradient_context_;

  explicit Variable(std::shared_ptr<Node> node);

  [[nodiscard]] static Variable
  from_operation(Tensor value, std::vector<std::shared_ptr<Node>> parents,
                 BackwardFunction backward);
  static void accumulate_gradient(const std::shared_ptr<Node> &node,
                                  const Tensor &contribution);
  [[nodiscard]] std::vector<Tensor>
  vector_jacobian_product(const Tensor &seed_gradient,
                          std::span<const Variable> boundaries) const;
  [[nodiscard]] std::unique_ptr<GradientContext>
  evaluate_gradients(const Tensor &seed_gradient) const;
  void replace_leaf_value(Tensor value);
  void replace_leaf_state(Tensor value, Tensor gradient) noexcept;

  friend class Parameter;
  friend Variable operator+(const Variable &, const Variable &);
  friend Variable operator-(const Variable &, const Variable &);
  friend Variable operator-(const Variable &);
  friend Variable operator*(const Variable &, const Variable &);
  friend Variable operator/(const Variable &, const Variable &);
  friend Variable matmul(const Variable &, const Variable &);
  friend Variable permute(const Variable &, Tensor::Shape);
  friend Variable sum(const Variable &);
  friend Variable sum(const Variable &, std::size_t, bool);
  friend Variable mean(const Variable &);
  friend Variable mean(const Variable &, std::size_t, bool);
  friend Variable exp(const Variable &);
  friend Variable log(const Variable &);
  friend Variable sqrt(const Variable &);
  friend Variable erf(const Variable &);
  friend Variable reshape(const Variable &, Tensor::Shape);
  friend Variable transpose_2d(const Variable &);
  friend Variable concatenate_last_axis(std::span<const Variable>);
  friend Variable broadcast_to(const Variable &, Tensor::Shape);
  friend Variable gather_rows(const Variable &, std::span<const std::size_t>,
                              Tensor::Shape);
  friend Variable custom_gradient(Tensor, std::span<const Variable>,
                                  CustomGradientFunction);
  friend Variable checkpoint(
      const Variable &, std::span<const Variable>,
      std::function<Variable(const Variable &)>);
  friend Variable softmax(const Variable &, std::size_t);
  friend Variable gelu(const Variable &);
  friend Variable relu(const Variable &);
  friend Variable layer_norm(const Variable &, const Variable &,
                             const Variable &, float);
  friend Variable cross_entropy(const Variable &,
                                std::span<const std::uint32_t>);
};

[[nodiscard]] Variable operator+(const Variable &left, const Variable &right);
[[nodiscard]] Variable operator-(const Variable &value);
[[nodiscard]] Variable operator-(const Variable &left, const Variable &right);
[[nodiscard]] Variable operator*(const Variable &left, const Variable &right);
[[nodiscard]] Variable operator/(const Variable &left, const Variable &right);

[[nodiscard]] Variable operator+(const Variable &value, float scalar);
[[nodiscard]] Variable operator+(float scalar, const Variable &value);
[[nodiscard]] Variable operator-(const Variable &value, float scalar);
[[nodiscard]] Variable operator-(float scalar, const Variable &value);
[[nodiscard]] Variable operator*(const Variable &value, float scalar);
[[nodiscard]] Variable operator*(float scalar, const Variable &value);
[[nodiscard]] Variable operator/(const Variable &value, float scalar);
[[nodiscard]] Variable operator/(float scalar, const Variable &value);

[[nodiscard]] Variable matmul(const Variable &left, const Variable &right);
[[nodiscard]] Variable permute(const Variable &value, Tensor::Shape axes);

// sum(value) reduces every element to a scalar. The axis overload removes the
// selected axis unless keep_dimensions is true.
[[nodiscard]] Variable sum(const Variable &value);
[[nodiscard]] Variable sum(const Variable &value, std::size_t axis,
                           bool keep_dimensions = false);
[[nodiscard]] Variable mean(const Variable &value);
[[nodiscard]] Variable mean(const Variable &value, std::size_t axis,
                            bool keep_dimensions = false);

[[nodiscard]] Variable exp(const Variable &value);
[[nodiscard]] Variable log(const Variable &value);
[[nodiscard]] Variable sqrt(const Variable &value);
[[nodiscard]] Variable erf(const Variable &value);
[[nodiscard]] Variable reshape(const Variable &value, Tensor::Shape new_shape);
[[nodiscard]] Variable transpose_2d(const Variable &value);
// Concatenates rank-one-or-greater Variables along their final axis. Backward
// slices the arriving gradient back into each input's original width.
[[nodiscard]] Variable concatenate_last_axis(std::span<const Variable> inputs);
[[nodiscard]] Variable concatenate_last_axis(const Variable &left,
                                             const Variable &right);
[[nodiscard]] Variable broadcast_to(const Variable &value,
                                    Tensor::Shape output_shape);
[[nodiscard]] Variable gather_rows(const Variable &table,
                                   std::span<const std::size_t> row_indices,
                                   Tensor::Shape index_shape);

// Creates one graph node for an externally computed forward result without
// exposing graph internals. The VJP is called during backward only when at
// least one input requires a gradient. It must return exactly one Tensor per
// positional input, with the corresponding input's shape and backend.
[[nodiscard]] Variable custom_gradient(
    Tensor output, std::span<const Variable> inputs,
    CustomGradientFunction vector_jacobian_product);

// Evaluates recompute(input) once now, then discards its internal graph. During
// backward the function is evaluated again and its VJP is accumulated at the
// input and declared differentiable leaf dependencies. The callable must be
// deterministic and replay-safe; stateful side effects are not supported.
[[nodiscard]] Variable checkpoint(
    const Variable &input, std::span<const Variable> dependencies,
    std::function<Variable(const Variable &)> recompute);

} // namespace riftco_transformer
