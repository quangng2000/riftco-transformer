#include "transformer_lab/core/autograd.hpp"
#include "transformer_lab/core/tensor_ops.hpp"
#include "transformer_lab/nn/parameter.hpp"
#include "transformer_lab/optim/adam.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using transformer_lab::Tensor;
using transformer_lab::Variable;
using transformer_lab::Adam;
using transformer_lab::ExecutionBackend;
using transformer_lab::Parameter;
using transformer_lab::move_parameters_to;
namespace tensor_ops = transformer_lab::tensor_ops;

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
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void require_tensor_close(
    const Tensor& actual,
    const Tensor::Shape& expected_shape,
    const std::vector<float>& expected,
    const std::string& message,
    float tolerance = 1.0e-5F
) {
    require(actual.shape() == expected_shape, message + ": shape mismatch");
    require(actual.numel() == expected.size(), message + ": size mismatch");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_close(
            actual.flat(index),
            expected[index],
            message + " at flat index " + std::to_string(index),
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

template <typename Function>
void require_logic_error(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, message);
}

float tensor_sum(const Tensor& tensor) {
    float result = 0.0F;
    for (const float value : tensor.data()) {
        result += value;
    }
    return result;
}

float tensor_dot(const Tensor& left, const Tensor& right) {
    return tensor_sum(tensor_ops::multiply(left, right));
}

float matmul_loss(const Tensor& left, const Tensor& right) {
    return tensor_sum(tensor_ops::matmul(left, right));
}

template <typename Function>
float finite_difference(
    Function&& function,
    float point,
    float epsilon = 1.0e-3F
) {
    return (
        function(point + epsilon) -
        function(point - epsilon)
    ) / (2.0F * epsilon);
}

void test_scalar_chain_rule_and_gradient_accumulation() {
    const Variable x = Variable::scalar(2.0F);
    const Variable y = Variable::scalar(3.0F);
    const Variable result = x * y + x;

    require_close(result.value().flat(0), 8.0F, "forward scalar result");
    result.backward();

    // x reaches the result through both multiplication and addition:
    // d(x*y + x)/dx = y + 1.
    require_close(x.gradient().flat(0), 4.0F, "accumulated x gradient");
    require_close(y.gradient().flat(0), 2.0F, "y gradient");

    result.backward();
    require_close(
        x.gradient().flat(0),
        4.0F,
        "backward should recompute rather than double gradients"
    );
}

void test_custom_seed_gradient() {
    const Variable x(Tensor({2}, {1.0F, 2.0F}));
    const Variable squared = x * x;
    squared.backward(Tensor({2}, {2.0F, 3.0F}));

    require_tensor_close(
        x.gradient(),
        {2},
        {4.0F, 12.0F},
        "custom vector-Jacobian seed"
    );
}

void test_public_custom_gradient_operation() {
    const Variable left(Tensor({2}, {2.0F, 3.0F}));
    const Variable right(Tensor({2}, {4.0F, 5.0F}));
    const Tensor left_value = left.value();
    const Tensor right_value = right.value();
    const std::vector<Variable> inputs{left, right};

    const Variable product = transformer_lab::custom_gradient(
        tensor_ops::multiply(left_value, right_value),
        inputs,
        [left_value, right_value](const Tensor& upstream) {
            return std::vector<Tensor>{
                tensor_ops::multiply(upstream, right_value),
                tensor_ops::multiply(upstream, left_value),
            };
        }
    );
    require_tensor_close(
        product.value(),
        {2},
        {8.0F, 15.0F},
        "custom gradient forward value"
    );
    product.backward(Tensor({2}, {2.0F, 3.0F}));
    require_tensor_close(
        left.gradient(),
        {2},
        {8.0F, 15.0F},
        "custom gradient left VJP"
    );
    require_tensor_close(
        right.gradient(),
        {2},
        {4.0F, 9.0F},
        "custom gradient right VJP"
    );

    // Inputs are positional, so the same graph node may be used more than
    // once and receives one contribution for each occurrence.
    const Variable repeated = Variable::scalar(3.0F);
    const Tensor repeated_value = repeated.value();
    const std::vector<Variable> repeated_inputs{repeated, repeated};
    const Variable square = transformer_lab::custom_gradient(
        tensor_ops::multiply(repeated_value, repeated_value),
        repeated_inputs,
        [repeated_value](const Tensor& upstream) {
            const Tensor contribution =
                tensor_ops::multiply(upstream, repeated_value);
            return std::vector<Tensor>{contribution, contribution};
        }
    );
    square.backward();
    require_close(
        repeated.gradient().flat(0),
        6.0F,
        "custom gradient repeated-input accumulation"
    );
}

void test_custom_gradient_validation_and_atomic_failure() {
    const Variable left(Tensor({2}, {2.0F, 3.0F}));
    const Variable right(Tensor({2}, {4.0F, 5.0F}));
    const std::vector<Variable> inputs{left, right};

    transformer_lab::sum(left * 2.0F + right * 3.0F).backward();
    require_tensor_close(
        left.gradient(), {2}, {2.0F, 2.0F},
        "custom gradient atomic baseline left"
    );
    require_tensor_close(
        right.gradient(), {2}, {3.0F, 3.0F},
        "custom gradient atomic baseline right"
    );

    require_throws(
        [&] {
            static_cast<void>(transformer_lab::custom_gradient(
                tensor_ops::add(left.value(), right.value()),
                inputs,
                {}
            ));
        },
        "custom gradient should reject an empty VJP callback"
    );
    require_throws(
        [&] {
            const std::vector<Variable> no_inputs;
            static_cast<void>(transformer_lab::custom_gradient(
                Tensor(Tensor::Shape{}, 1.0F),
                no_inputs,
                [](const Tensor&) {
                    return std::vector<Tensor>{};
                }
            ));
        },
        "custom gradient should reject an empty input list"
    );

    const Variable wrong_count = transformer_lab::custom_gradient(
        tensor_ops::add(left.value(), right.value()),
        inputs,
        [](const Tensor& upstream) {
            return std::vector<Tensor>{upstream};
        }
    );
    require_throws(
        [&] { transformer_lab::sum(wrong_count).backward(); },
        "custom gradient should reject the wrong VJP result count"
    );
    require_tensor_close(
        left.gradient(), {2}, {2.0F, 2.0F},
        "wrong VJP count must preserve the left gradient"
    );
    require_tensor_close(
        right.gradient(), {2}, {3.0F, 3.0F},
        "wrong VJP count must preserve the right gradient"
    );

    const Variable wrong_shape = transformer_lab::custom_gradient(
        tensor_ops::add(left.value(), right.value()),
        inputs,
        [](const Tensor& upstream) {
            return std::vector<Tensor>{
                upstream,
                Tensor({1}, 1.0F, upstream.backend()),
            };
        }
    );
    require_throws(
        [&] { transformer_lab::sum(wrong_shape).backward(); },
        "custom gradient should reject a mismatched VJP Tensor shape"
    );
    require_tensor_close(
        left.gradient(), {2}, {2.0F, 2.0F},
        "wrong VJP shape must preserve the left gradient"
    );
    require_tensor_close(
        right.gradient(), {2}, {3.0F, 3.0F},
        "wrong VJP shape must preserve the right gradient"
    );
    require_tensor_close(
        wrong_shape.gradient(), {2}, {0.0F, 0.0F},
        "failed custom gradient output must remain uncommitted"
    );

    const Variable throwing = transformer_lab::custom_gradient(
        tensor_ops::add(left.value(), right.value()),
        inputs,
        [](const Tensor&) -> std::vector<Tensor> {
            throw std::runtime_error("injected custom VJP failure");
        }
    );
    require_throws(
        [&] { transformer_lab::sum(throwing).backward(); },
        "custom gradient should propagate a VJP callback failure"
    );
    require_tensor_close(
        left.gradient(), {2}, {2.0F, 2.0F},
        "throwing VJP must preserve the left gradient"
    );
    require_tensor_close(
        right.gradient(), {2}, {3.0F, 3.0F},
        "throwing VJP must preserve the right gradient"
    );

    if (transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        const Variable metal_input(left.value().to(ExecutionBackend::Metal));
        const std::vector<Variable> metal_inputs{metal_input};
        require_throws(
            [&] {
                static_cast<void>(transformer_lab::custom_gradient(
                    left.value(),
                    metal_inputs,
                    [](const Tensor& upstream) {
                        return std::vector<Tensor>{upstream};
                    }
                ));
            },
            "custom gradient should reject mixed forward backends"
        );

        const std::vector<Variable> cpu_inputs{left};
        const Variable wrong_backend = transformer_lab::custom_gradient(
            left.value(),
            cpu_inputs,
            [](const Tensor& upstream) {
                return std::vector<Tensor>{
                    upstream.to(ExecutionBackend::Metal),
                };
            }
        );
        require_throws(
            [&] { transformer_lab::sum(wrong_backend).backward(); },
            "custom gradient should reject a mismatched VJP backend"
        );
        require_tensor_close(
            left.gradient(), {2}, {2.0F, 2.0F},
            "wrong VJP backend must preserve the input gradient"
        );
    }
}

void test_branches_and_temporary_lifetime() {
    const Variable branch_input = Variable::scalar(2.0F);
    const Variable square = branch_input * branch_input;
    const Variable shifted = branch_input + 1.0F;
    const Variable diamond = square * shifted;
    diamond.backward();

    // d(x^2 * (x + 1))/dx = 3x^2 + 2x = 16 at x=2.
    require_close(
        branch_input.gradient().flat(0),
        16.0F,
        "diamond-graph gradient"
    );

    const Variable shared_input = Variable::scalar(2.0F);
    const Variable shared_intermediate = shared_input * shared_input;
    const Variable shared_loss =
        shared_intermediate * shared_intermediate + shared_intermediate;
    shared_loss.backward();

    // a=x^2 and loss=a^2+a. Both downstream paths must reach a before a's
    // backward rule runs: dloss/dx=(2a+1)*2x=36 at x=2.
    require_close(
        shared_input.gradient().flat(0),
        36.0F,
        "shared-intermediate gradient scheduling"
    );

    const Variable surviving_leaf = Variable::scalar(3.0F);
    const Variable output = [&] {
        const Variable temporary = surviving_leaf * 2.0F;
        return temporary * temporary + 1.0F;
    }();
    output.backward();

    // The temporary handle is gone, but the graph owns its node.
    require_close(
        surviving_leaf.gradient().flat(0),
        24.0F,
        "temporary graph-node lifetime"
    );
}

void test_arithmetic_and_elementwise_functions() {
    const Variable numerator = Variable::scalar(6.0F);
    const Variable denominator = Variable::scalar(3.0F);
    const Variable quotient = numerator / denominator;
    quotient.backward();

    require_close(quotient.value().flat(0), 2.0F, "division forward value");
    require_close(
        numerator.gradient().flat(0),
        1.0F / 3.0F,
        "division numerator gradient"
    );
    require_close(
        denominator.gradient().flat(0),
        -2.0F / 3.0F,
        "division denominator gradient"
    );

    const Variable square_root_input = Variable::scalar(4.0F);
    const Variable square_root =
        transformer_lab::sqrt(square_root_input);
    square_root.backward();
    require_close(
        square_root_input.gradient().flat(0),
        0.25F,
        "square-root gradient"
    );

    const Variable identity_input = Variable::scalar(1.25F);
    const Variable identity = transformer_lab::log(
        transformer_lab::exp(identity_input)
    );
    identity.backward();
    require_close(identity.value().flat(0), 1.25F, "log(exp(x)) value");
    require_close(identity_input.gradient().flat(0), 1.0F,
                  "log(exp(x)) gradient");

    const Variable x = Variable::scalar(1.4F);
    const Variable y = Variable::scalar(0.8F);
    const Variable expression = transformer_lab::log(
        transformer_lab::exp(
            x * y + x / y - y + 2.0F + (-x) * 0.25F
        )
    );
    expression.backward();

    const auto plain_expression = [](float x_value, float y_value) {
        return std::log(std::exp(
            x_value * y_value +
            x_value / y_value -
            y_value +
            2.0F -
            x_value * 0.25F
        ));
    };
    const float numerical_x = finite_difference(
        [&](float candidate) {
            return plain_expression(candidate, 0.8F);
        },
        1.4F
    );
    const float numerical_y = finite_difference(
        [&](float candidate) {
            return plain_expression(1.4F, candidate);
        },
        0.8F
    );
    require_close(
        x.gradient().flat(0),
        numerical_x,
        "elementwise-chain x finite difference",
        3.0e-3F
    );
    require_close(
        y.gradient().flat(0),
        numerical_y,
        "elementwise-chain y finite difference",
        3.0e-3F
    );

    const float numerical_sqrt = finite_difference(
        [](float candidate) { return std::sqrt(candidate); },
        4.0F
    );
    require_close(
        square_root_input.gradient().flat(0),
        numerical_sqrt,
        "sqrt finite difference",
        1.0e-3F
    );
}

void test_matrix_multiplication_gradients() {
    const Tensor left_values(
        {2, 3},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    );
    const Tensor right_values(
        {3, 2},
        {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}
    );
    const Variable left(left_values);
    const Variable right(right_values);
    const Variable loss = transformer_lab::sum(
        transformer_lab::matmul(left, right)
    );
    loss.backward();

    require_tensor_close(
        left.gradient(),
        {2, 3},
        {15.0F, 19.0F, 23.0F, 15.0F, 19.0F, 23.0F},
        "left matmul gradient"
    );
    require_tensor_close(
        right.gradient(),
        {3, 2},
        {5.0F, 5.0F, 7.0F, 7.0F, 9.0F, 9.0F},
        "right matmul gradient"
    );

    // A 1e-2 perturbation avoids cancellation in our deliberately float-only
    // tensor implementation while remaining small relative to these values.
    constexpr float epsilon = 1.0e-2F;
    constexpr float tolerance = 5.0e-3F;
    for (std::size_t index = 0; index < left_values.numel(); ++index) {
        Tensor plus = left_values;
        Tensor minus = left_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        const float numerical =
            (matmul_loss(plus, right_values) -
             matmul_loss(minus, right_values)) /
            (2.0F * epsilon);
        require_close(
            left.gradient().flat(index),
            numerical,
            "left finite-difference gradient",
            tolerance
        );
    }
    for (std::size_t index = 0; index < right_values.numel(); ++index) {
        Tensor plus = right_values;
        Tensor minus = right_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        const float numerical =
            (matmul_loss(left_values, plus) -
             matmul_loss(left_values, minus)) /
            (2.0F * epsilon);
        require_close(
            right.gradient().flat(index),
            numerical,
            "right finite-difference gradient",
            tolerance
        );
    }
}

void test_permute_and_batched_matmul_gradients() {
    const Variable layout(Tensor(
        {2, 3, 2},
        {
            0.0F, 1.0F,
            2.0F, 3.0F,
            4.0F, 5.0F,
            6.0F, 7.0F,
            8.0F, 9.0F,
            10.0F, 11.0F,
        }
    ));
    transformer_lab::permute(layout, {1, 0, 2}).backward(Tensor(
        {3, 2, 2},
        {
            1.0F, 2.0F, 3.0F, 4.0F,
            5.0F, 6.0F, 7.0F, 8.0F,
            9.0F, 10.0F, 11.0F, 12.0F,
        }
    ));
    require_tensor_close(
        layout.gradient(),
        {2, 3, 2},
        {
            1.0F, 2.0F, 5.0F, 6.0F, 9.0F, 10.0F,
            3.0F, 4.0F, 7.0F, 8.0F, 11.0F, 12.0F,
        },
        "inverse-permutation gradient"
    );

    const Tensor left_values(
        {1, 2, 2, 3},
        {
            0.2F, -0.3F, 0.5F,
            1.0F, 0.4F, -0.2F,
            -0.6F, 0.8F, 0.15F,
            0.7F, -0.1F, -0.9F,
        }
    );
    const Tensor right_values(
        {1, 2, 3, 2},
        {
            0.5F, -0.25F,
            0.75F, -0.4F,
            0.6F, 0.2F,
            -0.3F, 0.9F,
            0.4F, -0.8F,
            1.1F, 0.25F,
        }
    );
    const Tensor output_weights(
        {1, 2, 2, 2},
        {0.5F, -1.0F, 1.5F, 0.25F, -0.75F, 0.6F, 0.4F, -1.2F}
    );
    const Variable left(left_values);
    const Variable right(right_values);
    transformer_lab::sum(
        transformer_lab::matmul(left, right) *
        Variable(output_weights, false)
    ).backward();

    const auto evaluate = [&](
        const Tensor& candidate_left,
        const Tensor& candidate_right
    ) {
        return tensor_dot(
            tensor_ops::matmul(candidate_left, candidate_right),
            output_weights
        );
    };

    constexpr float epsilon = 1.0e-2F;
    constexpr float tolerance = 5.0e-3F;
    for (std::size_t index = 0; index < left_values.numel(); ++index) {
        Tensor plus = left_values;
        Tensor minus = left_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        require_close(
            left.gradient().flat(index),
            (
                evaluate(plus, right_values) -
                evaluate(minus, right_values)
            ) / (2.0F * epsilon),
            "batched matmul left finite difference",
            tolerance
        );
    }
    for (std::size_t index = 0; index < right_values.numel(); ++index) {
        Tensor plus = right_values;
        Tensor minus = right_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        require_close(
            right.gradient().flat(index),
            (
                evaluate(left_values, plus) -
                evaluate(left_values, minus)
            ) / (2.0F * epsilon),
            "batched matmul right finite difference",
            tolerance
        );
    }
}

void test_reductions_reshape_and_transpose() {
    const Variable rows(
        Tensor({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})
    );
    const Variable row_sums = transformer_lab::sum(rows, 1);
    require(row_sums.value().shape() == Tensor::Shape({2}),
            "axis reduction shape");
    require_tensor_close(
        row_sums.value(),
        {2},
        {6.0F, 15.0F},
        "row sums"
    );
    row_sums.backward(Tensor({2}, {10.0F, 20.0F}));
    require_tensor_close(
        rows.gradient(),
        {2, 3},
        {10.0F, 10.0F, 10.0F, 20.0F, 20.0F, 20.0F},
        "axis-reduction gradient broadcast"
    );

    const Variable kept_rows(
        Tensor({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})
    );
    const Variable kept_row_sums =
        transformer_lab::sum(kept_rows, 1, true);
    require(
        kept_row_sums.value().shape() == Tensor::Shape({2, 1}),
        "kept axis reduction shape"
    );
    kept_row_sums.backward(Tensor({2, 1}, {3.0F, 4.0F}));
    require_tensor_close(
        kept_rows.gradient(),
        {2, 3},
        {3.0F, 3.0F, 3.0F, 4.0F, 4.0F, 4.0F},
        "kept-axis reduction gradient broadcast"
    );

    const Variable columns(
        Tensor({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})
    );
    const Variable column_means = transformer_lab::mean(columns, 0);
    require_tensor_close(
        column_means.value(),
        {3},
        {2.5F, 3.5F, 4.5F},
        "column means"
    );
    transformer_lab::sum(column_means).backward();
    require_tensor_close(
        columns.gradient(),
        {2, 3},
        {0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F},
        "axis-mean gradient"
    );

    const Variable layout(
        Tensor({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})
    );
    const Variable transformed = transformer_lab::transpose_2d(
        transformer_lab::reshape(layout, {3, 2})
    );
    require(
        transformed.value().shape() == Tensor::Shape({2, 3}),
        "reshape/transpose output shape"
    );
    transformer_lab::sum(transformed).backward();
    require_tensor_close(
        layout.gradient(),
        {2, 3},
        {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F},
        "reshape/transpose gradient"
    );

    const Tensor layout_values(
        {2, 3},
        {0.5F, -1.0F, 2.0F, 3.0F, -0.25F, 4.0F}
    );
    const Tensor weight_values(
        {2, 3},
        {1.0F, -2.0F, 3.0F, -4.0F, 5.0F, -6.0F}
    );
    const Variable layout_input(layout_values);
    const Variable weights(weight_values, false);
    const Variable layout_result = transformer_lab::sum(
        transformer_lab::mean(
            transformer_lab::transpose_2d(
                transformer_lab::reshape(layout_input, {3, 2})
            ) * weights,
            1
        )
    );
    layout_result.backward();

    const auto plain_layout_loss = [&](const Tensor& input) {
        const Tensor reshaped = input.reshape({3, 2});
        Tensor transposed({2, 3});
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 2; ++column) {
                transposed.at({column, row}) =
                    reshaped.at({row, column});
            }
        }
        return tensor_sum(
            tensor_ops::multiply(transposed, weight_values)
        ) / 3.0F;
    };

    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < layout_values.numel(); ++index) {
        Tensor plus = layout_values;
        Tensor minus = layout_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        const float numerical =
            (plain_layout_loss(plus) - plain_layout_loss(minus)) /
            (2.0F * epsilon);
        require_close(
            layout_input.gradient().flat(index),
            numerical,
            "layout/reduction finite difference",
            3.0e-3F
        );
    }
}

void test_broadcast_gather_and_erf_gradients() {
    const Variable columns(Tensor({3}, {1.0F, 2.0F, 3.0F}));
    const Variable expanded_columns =
        transformer_lab::broadcast_to(columns, {2, 3});
    expanded_columns.backward(Tensor(
        {2, 3},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    ));
    require_tensor_close(
        columns.gradient(),
        {3},
        {5.0F, 7.0F, 9.0F},
        "prepended broadcast gradient"
    );

    const Variable rows(Tensor({2, 1}, {10.0F, 20.0F}));
    const Variable expanded_rows =
        transformer_lab::broadcast_to(rows, {2, 3});
    expanded_rows.backward(Tensor(
        {2, 3},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    ));
    require_tensor_close(
        rows.gradient(),
        {2, 1},
        {6.0F, 15.0F},
        "singleton broadcast gradient"
    );

    const Variable scalar = Variable::scalar(2.0F);
    transformer_lab::broadcast_to(scalar, {2, 3}).backward(
        Tensor({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})
    );
    require_tensor_close(
        scalar.gradient(),
        {},
        {21.0F},
        "scalar broadcast gradient"
    );

    const Variable table(Tensor(
        {3, 2},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    ));
    const std::vector<std::size_t> row_indices{2, 0, 2};
    const Variable gathered = transformer_lab::gather_rows(
        table,
        row_indices,
        {3}
    );
    require(
        gathered.value().shape() == Tensor::Shape({3, 2}),
        "gathered Variable shape"
    );
    gathered.backward(Tensor(
        {3, 2},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    ));
    require_tensor_close(
        table.gradient(),
        {3, 2},
        {3.0F, 4.0F, 0.0F, 0.0F, 6.0F, 8.0F},
        "gather scatter-add gradient"
    );

    const Variable erf_input = Variable::scalar(0.75F);
    transformer_lab::erf(erf_input).backward();
    const float numerical = finite_difference(
        [](float candidate) { return std::erf(candidate); },
        0.75F
    );
    require_close(
        erf_input.gradient().flat(0),
        numerical,
        "erf finite-difference gradient",
        1.0e-3F
    );

    require_throws(
        [&] {
            static_cast<void>(
                transformer_lab::broadcast_to(columns, {2, 2})
            );
        },
        "invalid differentiable broadcast should throw"
    );
    require_throws(
        [&] {
            const std::vector<std::size_t> invalid_rows{3};
            static_cast<void>(transformer_lab::gather_rows(
                table,
                invalid_rows,
                {1}
            ));
        },
        "invalid differentiable gather should throw"
    );
}

void test_constants_and_errors() {
    const Variable trainable(Tensor({2}, {2.0F, 3.0F}));
    const Variable constant(Tensor({2}, {5.0F, 7.0F}), false);
    transformer_lab::sum(trainable * constant).backward();

    require(trainable.requires_gradient(), "trainable flag");
    require(!constant.requires_gradient(), "constant flag");
    require_tensor_close(
        trainable.gradient(),
        {2},
        {5.0F, 7.0F},
        "gradient through a constant"
    );
    require_tensor_close(
        constant.gradient(),
        {2},
        {0.0F, 0.0F},
        "constant should not accumulate gradients"
    );

    require_throws(
        [&] {
            const Variable vector(Tensor({2}, 1.0F));
            vector.backward();
        },
        "non-scalar backward without a seed should throw"
    );
    require_throws(
        [&] {
            const Variable vector(Tensor({2}, 1.0F));
            vector.backward(Tensor({3}, 1.0F));
        },
        "seed shape mismatch should throw"
    );
    require_throws(
        [] {
            const Variable value = Variable::scalar(0.0F);
            static_cast<void>(transformer_lab::log(value));
        },
        "log of zero should throw"
    );
    require_throws(
        [] {
            const Variable value = Variable::scalar(0.0F);
            static_cast<void>(transformer_lab::sqrt(value));
        },
        "sqrt derivative at zero should throw"
    );
    require_throws(
        [] {
            const Variable numerator = Variable::scalar(1.0F);
            const Variable denominator = Variable::scalar(0.0F);
            static_cast<void>(numerator / denominator);
        },
        "division by zero should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(transformer_lab::sum(trainable, 1));
        },
        "out-of-range reduction axis should throw"
    );
    require_throws(
        [] {
            const Variable constant_value =
                Variable::scalar(1.0F, false);
            constant_value.backward();
        },
        "backward on a constant should throw"
    );
}

void test_stale_graphs_are_rejected_before_gradients_change() {
    Parameter parameter(Tensor(Tensor::Shape{}, 2.0F));
    const Variable stale_loss =
        parameter.variable() * parameter.variable();

    parameter.set_value(Tensor(Tensor::Shape{}, 3.0F));
    const Variable current_loss =
        parameter.variable() * parameter.variable();
    current_loss.backward();
    require_close(
        parameter.gradient().flat(0),
        6.0F,
        "current graph gradient before stale backward"
    );

    require_logic_error(
        [&] { stale_loss.backward(); },
        "set_value should invalidate an older graph"
    );
    require_close(
        parameter.gradient().flat(0),
        6.0F,
        "stale backward must not reset an existing gradient"
    );

    Parameter optimized(Tensor(Tensor::Shape{}, 2.0F));
    const Variable pre_update_loss =
        optimized.variable() * optimized.variable();
    const Variable training_loss =
        optimized.variable() * optimized.variable();
    training_loss.backward();
    Adam optimizer({{"weight", &optimized}});
    static_cast<void>(optimizer.step());

    require_logic_error(
        [&] { pre_update_loss.backward(); },
        "Adam should invalidate graphs built before its parameter update"
    );

    if (transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        Parameter transferred(Tensor(Tensor::Shape{}, 2.0F));
        const Variable pre_transfer_loss =
            transferred.variable() * transferred.variable();
        move_parameters_to(
            {{"weight", &transferred}},
            ExecutionBackend::Metal
        );

        require_logic_error(
            [&] { pre_transfer_loss.backward(); },
            "backend transfer should invalidate its existing graph"
        );
        move_parameters_to(
            {{"weight", &transferred}},
            ExecutionBackend::Cpu
        );
    }
}

void test_checkpoint_recomputation_and_graph_reduction() {
    const Variable input(Tensor({2}, {2.0F, 3.0F}));
    Parameter weight(Tensor({2}, {4.0F, 5.0F}));
    const std::vector<Variable> dependencies{weight.variable()};
    std::size_t call_count = 0;
    const auto function =
        [parameter = weight.variable(), &call_count](
            const Variable& value
        ) {
            ++call_count;
            return value * parameter + parameter;
        };

    const Variable regular = function(input);
    call_count = 0;
    const Variable checkpointed = transformer_lab::checkpoint(
        input,
        dependencies,
        function
    );
    require(
        call_count == 1,
        "checkpoint should execute once during forward"
    );
    require_tensor_close(
        checkpointed.value(),
        regular.value().shape(),
        std::vector<float>(
            regular.value().data().begin(),
            regular.value().data().end()
        ),
        "checkpoint forward parity"
    );

    const auto regular_graph = regular.graph_statistics();
    const auto checkpoint_graph = checkpointed.graph_statistics();
    require(
        checkpoint_graph.node_count < regular_graph.node_count,
        "checkpoint should retain fewer autograd nodes"
    );
    require(
        checkpoint_graph.node_tensor_elements <
            regular_graph.node_tensor_elements,
        "checkpoint should retain fewer node-owned Tensor elements"
    );

    const Variable outside =
        transformer_lab::sum(weight.variable() * 2.0F);
    const Variable loss =
        transformer_lab::sum(checkpointed) + outside;
    loss.backward();
    require(
        call_count == 2,
        "checkpoint should recompute exactly once during backward"
    );
    require_tensor_close(
        input.gradient(),
        {2},
        {4.0F, 5.0F},
        "checkpoint input gradient"
    );
    require_tensor_close(
        weight.gradient(),
        {2},
        {5.0F, 6.0F},
        "checkpoint dependency and outside-branch gradient"
    );

    loss.backward();
    require(
        call_count == 3,
        "a repeated backward should replay the checkpoint again"
    );
    require_tensor_close(
        weight.gradient(),
        {2},
        {5.0F, 6.0F},
        "repeated checkpoint backward starts a fresh gradient pass"
    );
}

void test_checkpoint_shared_dependencies_and_atomic_failure() {
    const Variable input = Variable::scalar(2.0F);
    Parameter weight(Tensor(Tensor::Shape{}, 3.0F));
    const std::vector<Variable> dependencies{weight.variable()};

    const Variable product = transformer_lab::checkpoint(
        input,
        dependencies,
        [parameter = weight.variable()](const Variable& value) {
            return value * parameter;
        }
    );
    const Variable sum_path = transformer_lab::checkpoint(
        input,
        dependencies,
        [parameter = weight.variable()](const Variable& value) {
            return value + parameter;
        }
    );
    (product + sum_path).backward();
    require_close(
        input.gradient().flat(0),
        4.0F,
        "two checkpoints accumulate their shared input gradient"
    );
    require_close(
        weight.gradient().flat(0),
        3.0F,
        "two checkpoints accumulate their shared parameter gradient"
    );

    std::size_t call_count = 0;
    const Variable failing = transformer_lab::checkpoint(
        input,
        dependencies,
        [parameter = weight.variable(), &call_count](
            const Variable& value
        ) {
            ++call_count;
            if (call_count == 2) {
                throw std::runtime_error(
                    "injected checkpoint replay failure"
                );
            }
            return value * parameter;
        }
    );
    require_throws(
        [&] { failing.backward(); },
        "checkpoint should propagate a replay failure"
    );
    require_close(
        weight.gradient().flat(0),
        3.0F,
        "failed checkpoint backward must not partially commit gradients"
    );
}

void test_checkpoint_replay_contract_validation() {
    const Variable input = Variable::scalar(2.0F);
    Parameter weight(Tensor(Tensor::Shape{}, 3.0F));
    const std::vector<Variable> dependencies{weight.variable()};

    (input * weight.variable()).backward();
    require_close(
        input.gradient().flat(0),
        3.0F,
        "checkpoint replay contract baseline input gradient"
    );
    require_close(
        weight.gradient().flat(0),
        2.0F,
        "checkpoint replay contract baseline dependency gradient"
    );

    std::size_t shape_call_count = 0;
    const Variable shape_changing = transformer_lab::checkpoint(
        input,
        dependencies,
        [parameter = weight.variable(), &shape_call_count](
            const Variable& value
        ) {
            ++shape_call_count;
            const Variable result = value * parameter;
            return shape_call_count == 1
                       ? result
                       : transformer_lab::reshape(result, {1});
        }
    );
    require_logic_error(
        [&] { shape_changing.backward(); },
        "checkpoint replay should reject an output shape change"
    );
    require_close(
        input.gradient().flat(0),
        3.0F,
        "shape-changing replay must preserve the input gradient"
    );
    require_close(
        weight.gradient().flat(0),
        2.0F,
        "shape-changing replay must preserve the dependency gradient"
    );

    std::size_t dependency_call_count = 0;
    const Variable dependency_changing = transformer_lab::checkpoint(
        input,
        dependencies,
        [parameter = weight.variable(), &dependency_call_count](
            const Variable& value
        ) {
            ++dependency_call_count;
            return dependency_call_count == 1
                       ? value * parameter
                       : value + 1.0F;
        }
    );
    require_logic_error(
        [&] { dependency_changing.backward(); },
        "checkpoint replay should reject a missing dependency"
    );
    require_close(
        input.gradient().flat(0),
        3.0F,
        "dependency-changing replay must preserve the input gradient"
    );
    require_close(
        weight.gradient().flat(0),
        2.0F,
        "dependency-changing replay must preserve the dependency gradient"
    );
}

void test_checkpoint_contract_validation_and_stale_dependencies() {
    const Variable input = Variable::scalar(2.0F);
    Parameter weight(Tensor(Tensor::Shape{}, 3.0F));
    const std::vector<Variable> dependencies{weight.variable()};

    require_throws(
        [&] {
            static_cast<void>(transformer_lab::checkpoint(
                input,
                dependencies,
                {}
            ));
        },
        "checkpoint should reject an empty callable"
    );
    require_throws(
        [&] {
            const std::vector<Variable> duplicate_dependencies{
                weight.variable(),
                weight.variable(),
            };
            static_cast<void>(transformer_lab::checkpoint(
                input,
                duplicate_dependencies,
                [parameter = weight.variable()](
                    const Variable& value
                ) {
                    return value * parameter;
                }
            ));
        },
        "checkpoint should reject duplicate dependencies"
    );
    require_throws(
        [&] {
            static_cast<void>(transformer_lab::checkpoint(
                input,
                {},
                [parameter = weight.variable()](
                    const Variable& value
                ) {
                    return value * parameter;
                }
            ));
        },
        "checkpoint should reject an undeclared trainable leaf"
    );

    const Variable external = weight.variable() * 2.0F;
    require_throws(
        [&] {
            static_cast<void>(transformer_lab::checkpoint(
                input,
                dependencies,
                [external](const Variable& value) {
                    return value + external;
                }
            ));
        },
        "checkpoint should reject captured external non-leaf state"
    );

    const Variable stale = transformer_lab::checkpoint(
        input,
        dependencies,
        [parameter = weight.variable()](const Variable& value) {
            return value * parameter;
        }
    );
    weight.set_value(Tensor(Tensor::Shape{}, 4.0F));
    (weight.variable() * weight.variable()).backward();
    require_close(
        weight.gradient().flat(0),
        8.0F,
        "current dependency gradient before stale checkpoint"
    );
    require_logic_error(
        [&] { stale.backward(); },
        "checkpoint should reject a mutated dependency"
    );
    require_close(
        weight.gradient().flat(0),
        8.0F,
        "stale checkpoint rejection must preserve gradients"
    );
}

}  // namespace

int main() {
    try {
        test_scalar_chain_rule_and_gradient_accumulation();
        test_custom_seed_gradient();
        test_public_custom_gradient_operation();
        test_custom_gradient_validation_and_atomic_failure();
        test_branches_and_temporary_lifetime();
        test_arithmetic_and_elementwise_functions();
        test_matrix_multiplication_gradients();
        test_permute_and_batched_matmul_gradients();
        test_reductions_reshape_and_transpose();
        test_broadcast_gather_and_erf_gradients();
        test_constants_and_errors();
        test_stale_graphs_are_rejected_before_gradients_change();
        test_checkpoint_recomputation_and_graph_reduction();
        test_checkpoint_shared_dependencies_and_atomic_failure();
        test_checkpoint_replay_contract_validation();
        test_checkpoint_contract_validation_and_stale_dependencies();
        std::cout << "autograd tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
