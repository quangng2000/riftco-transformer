#include "riftco_transformer/core/tensor_ops.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using riftco_transformer::Tensor;
namespace tensor_ops = riftco_transformer::tensor_ops;

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
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(actual)
        );
    }
}

void require_tensor_close(
    const Tensor& actual,
    const Tensor::Shape& expected_shape,
    const std::vector<float>& expected_values,
    const std::string& message,
    float tolerance = 1.0e-6F
) {
    require(actual.shape() == expected_shape, message + ": shape mismatch");
    require(
        actual.numel() == expected_values.size(),
        message + ": value count mismatch"
    );
    for (std::size_t index = 0; index < expected_values.size(); ++index) {
        require_close(
            actual.flat(index),
            expected_values[index],
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

void test_elementwise_operations() {
    const Tensor left({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    const Tensor right({2, 2}, {5.0F, 4.0F, 3.0F, 2.0F});

    require_tensor_close(
        tensor_ops::add(left, right),
        {2, 2},
        {6.0F, 6.0F, 6.0F, 6.0F},
        "elementwise add"
    );
    require_tensor_close(
        tensor_ops::subtract(left, right),
        {2, 2},
        {-4.0F, -2.0F, 0.0F, 2.0F},
        "elementwise subtract"
    );
    require_tensor_close(
        tensor_ops::multiply(left, right),
        {2, 2},
        {5.0F, 8.0F, 9.0F, 8.0F},
        "elementwise multiply"
    );
    require_tensor_close(
        tensor_ops::divide(left, right),
        {2, 2},
        {0.2F, 0.5F, 1.0F, 2.0F},
        "elementwise divide"
    );
    require_tensor_close(
        tensor_ops::negate(left),
        {2, 2},
        {-1.0F, -2.0F, -3.0F, -4.0F},
        "negate"
    );
    require_tensor_close(
        tensor_ops::scale(left, 0.5F),
        {2, 2},
        {0.5F, 1.0F, 1.5F, 2.0F},
        "scale"
    );

    require_throws(
        [&] { static_cast<void>(tensor_ops::add(left, Tensor({4}, 1.0F))); },
        "elementwise shape mismatch should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(tensor_ops::divide(left, Tensor({2, 2}, 0.0F)));
        },
        "division by zero should throw"
    );
}

void test_matrix_multiplication_and_transpose() {
    const Tensor left({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const Tensor right({3, 2}, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

    require_tensor_close(
        tensor_ops::matmul(left, right),
        {2, 2},
        {58.0F, 64.0F, 139.0F, 154.0F},
        "matrix multiplication"
    );
    require_tensor_close(
        tensor_ops::transpose_2d(left),
        {3, 2},
        {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F},
        "matrix transpose"
    );

    require_throws(
        [&] { static_cast<void>(tensor_ops::matmul(left, Tensor({4, 2}))); },
        "matmul inner-dimension mismatch should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(tensor_ops::matmul(Tensor({1, 2, 3}), right));
        },
        "matmul should reject rank mismatch"
    );
    require_throws(
        [] { static_cast<void>(tensor_ops::transpose_2d(Tensor({2}, 1.0F))); },
        "transpose should reject non-matrices"
    );
}

void test_permutation_and_batched_matrix_multiplication() {
    const Tensor values(
        {2, 3, 2},
        {
            0.0F,
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
            7.0F,
            8.0F,
            9.0F,
            10.0F,
            11.0F,
        }
    );
    require_tensor_close(
        tensor_ops::permute(values, {1, 0, 2}),
        {3, 2, 2},
        {
            0.0F,
            1.0F,
            6.0F,
            7.0F,
            2.0F,
            3.0F,
            8.0F,
            9.0F,
            4.0F,
            5.0F,
            10.0F,
            11.0F,
        },
        "rank-three permutation"
    );

    const Tensor left(
        {2, 2, 3},
        {
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
            -1.0F,
            2.0F,
            0.0F,
            3.0F,
            1.0F,
            2.0F,
        }
    );
    const Tensor right(
        {2, 3, 2},
        {
            1.0F,
            0.0F,
            0.0F,
            1.0F,
            1.0F,
            1.0F,
            2.0F,
            1.0F,
            1.0F,
            -1.0F,
            0.0F,
            2.0F,
        }
    );
    require_tensor_close(
        tensor_ops::matmul(left, right),
        {2, 2, 2},
        {4.0F, 5.0F, 10.0F, 11.0F, 0.0F, -3.0F, 7.0F, 6.0F},
        "batched matrix multiplication"
    );

    require_throws(
        [&] { static_cast<void>(tensor_ops::permute(values, {0, 1})); },
        "permutation axis-count mismatch should throw"
    );
    require_throws(
        [&] { static_cast<void>(tensor_ops::permute(values, {0, 0, 2})); },
        "duplicate permutation axis should throw"
    );
    require_throws(
        [&] { static_cast<void>(tensor_ops::permute(values, {0, 1, 3})); },
        "out-of-range permutation axis should throw"
    );
    require_throws(
        [] {
            static_cast<void>(
                tensor_ops::matmul(Tensor({2, 2, 3}), Tensor({3, 3, 2}))
            );
        },
        "matmul batch-dimension mismatch should throw"
    );
    require_throws(
        [] {
            static_cast<void>(
                tensor_ops::matmul(Tensor({2, 2, 3}), Tensor({2, 4, 2}))
            );
        },
        "batched matmul inner-dimension mismatch should throw"
    );
}

void test_broadcasting() {
    require_tensor_close(
        tensor_ops::broadcast_to(Tensor({}, 2.0F), {2, 3}),
        {2, 3},
        {2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F},
        "scalar broadcast"
    );
    require_tensor_close(
        tensor_ops::broadcast_to(Tensor({3}, {1.0F, 2.0F, 3.0F}), {2, 3}),
        {2, 3},
        {1.0F, 2.0F, 3.0F, 1.0F, 2.0F, 3.0F},
        "right-aligned row broadcast"
    );
    require_tensor_close(
        tensor_ops::broadcast_to(Tensor({2, 1}, {10.0F, 20.0F}), {2, 3}),
        {2, 3},
        {10.0F, 10.0F, 10.0F, 20.0F, 20.0F, 20.0F},
        "singleton-dimension broadcast"
    );
    require_tensor_close(
        tensor_ops::broadcast_to(
            Tensor({2, 1, 2}, {1.0F, 2.0F, 3.0F, 4.0F}),
            {2, 3, 2}
        ),
        {2, 3, 2},
        {
            1.0F,
            2.0F,
            1.0F,
            2.0F,
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            3.0F,
            4.0F,
            3.0F,
            4.0F,
        },
        "rank-three broadcast"
    );
    require_tensor_close(
        tensor_ops::broadcast_to(
            Tensor({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}),
            {2, 2}
        ),
        {2, 2},
        {1.0F, 2.0F, 3.0F, 4.0F},
        "non-scalar identity broadcast"
    );
    require_tensor_close(
        tensor_ops::broadcast_to(Tensor({}, 7.0F), {}),
        {},
        {7.0F},
        "scalar identity broadcast"
    );

    require_throws(
        [] {
            static_cast<void>(tensor_ops::broadcast_to(Tensor({2, 3}), {3}));
        },
        "broadcast should reject a smaller output rank"
    );
    require_throws(
        [] {
            static_cast<void>(tensor_ops::broadcast_to(Tensor({2, 2}), {2, 3}));
        },
        "broadcast should reject incompatible dimensions"
    );
}

void test_sum_to_shape_and_gather() {
    const Tensor value({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    require_tensor_close(
        tensor_ops::sum_to_shape(value, {3}),
        {3},
        {5.0F, 7.0F, 9.0F},
        "sum prepended dimension"
    );
    require_tensor_close(
        tensor_ops::sum_to_shape(value, {2, 1}),
        {2, 1},
        {6.0F, 15.0F},
        "sum singleton dimension"
    );
    require_tensor_close(
        tensor_ops::sum_to_shape(value, {}),
        {},
        {21.0F},
        "sum to scalar"
    );
    require_tensor_close(
        tensor_ops::sum_to_shape(value, {2, 3}),
        {2, 3},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
        "sum-to-shape identity"
    );

    const Tensor table({3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const std::vector<std::size_t> rows{2, 0, 2};
    require_tensor_close(
        tensor_ops::gather_rows(table, rows, {1, 3}),
        {1, 3, 2},
        {5.0F, 6.0F, 1.0F, 2.0F, 5.0F, 6.0F},
        "row gather"
    );
    require_tensor_close(
        tensor_ops::scatter_add_rows(
            Tensor({1, 3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}),
            rows,
            3
        ),
        {3, 2},
        {3.0F, 4.0F, 0.0F, 0.0F, 6.0F, 8.0F},
        "row scatter-add with repeated indices"
    );

    require_throws(
        [&] { static_cast<void>(tensor_ops::sum_to_shape(value, {2, 2})); },
        "incompatible sum-to-shape should throw"
    );
    require_throws(
        [&] { static_cast<void>(tensor_ops::gather_rows(table, rows, {2})); },
        "gather index shape mismatch should throw"
    );
    require_throws(
        [&] {
            const std::vector<std::size_t> invalid_rows{3};
            static_cast<void>(
                tensor_ops::gather_rows(table, invalid_rows, {1})
            );
        },
        "out-of-range gather row should throw"
    );
}

void test_reductions() {
    const Tensor value({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

    require_tensor_close(tensor_ops::sum(value), {}, {21.0F}, "full sum");
    require_tensor_close(tensor_ops::mean(value), {}, {3.5F}, "full mean");
    require_tensor_close(
        tensor_ops::sum(value, 0),
        {3},
        {5.0F, 7.0F, 9.0F},
        "column sums"
    );
    require_tensor_close(
        tensor_ops::sum(value, 1),
        {2},
        {6.0F, 15.0F},
        "row sums"
    );
    require_tensor_close(
        tensor_ops::sum(value, 1, true),
        {2, 1},
        {6.0F, 15.0F},
        "row sums with retained dimension"
    );
    require_tensor_close(
        tensor_ops::mean(value, 0),
        {3},
        {2.5F, 3.5F, 4.5F},
        "column means"
    );
    require_tensor_close(
        tensor_ops::mean(value, 1, true),
        {2, 1},
        {2.0F, 5.0F},
        "row means with retained dimension"
    );

    const Tensor vector({3}, {2.0F, 4.0F, 6.0F});
    require_tensor_close(
        tensor_ops::sum(vector, 0),
        {},
        {12.0F},
        "rank-one axis sum"
    );
    require_tensor_close(
        tensor_ops::mean(vector, 0, true),
        {1},
        {4.0F},
        "rank-one retained-axis mean"
    );

    const Tensor rank_three(
        {2, 2, 3},
        {
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
            7.0F,
            8.0F,
            9.0F,
            10.0F,
            11.0F,
            12.0F,
        }
    );
    require_tensor_close(
        tensor_ops::sum(rank_three, 1),
        {2, 3},
        {5.0F, 7.0F, 9.0F, 17.0F, 19.0F, 21.0F},
        "rank-three middle-axis sum"
    );
    require_tensor_close(
        tensor_ops::sum(rank_three, 1, true),
        {2, 1, 3},
        {5.0F, 7.0F, 9.0F, 17.0F, 19.0F, 21.0F},
        "rank-three retained middle-axis sum"
    );

    require_throws(
        [&] { static_cast<void>(tensor_ops::sum(value, 2)); },
        "sum should reject an out-of-range axis"
    );
    require_throws(
        [&] { static_cast<void>(tensor_ops::mean(value, 2)); },
        "mean should reject an out-of-range axis"
    );
}

void test_unary_functions_and_domains() {
    const float e = std::exp(1.0F);
    require_tensor_close(
        tensor_ops::exp(Tensor({3}, {-1.0F, 0.0F, 1.0F})),
        {3},
        {1.0F / e, 1.0F, e},
        "exponential",
        1.0e-5F
    );
    require_tensor_close(
        tensor_ops::log(Tensor({3}, {1.0F, e, e * e})),
        {3},
        {0.0F, 1.0F, 2.0F},
        "logarithm",
        1.0e-5F
    );
    require_tensor_close(
        tensor_ops::sqrt(Tensor({3}, {0.0F, 4.0F, 9.0F})),
        {3},
        {0.0F, 2.0F, 3.0F},
        "square root"
    );
    require_tensor_close(
        tensor_ops::erf(Tensor({3}, {-1.0F, 0.0F, 1.0F})),
        {3},
        {-std::erf(1.0F), 0.0F, std::erf(1.0F)},
        "error function"
    );

    require_throws(
        [] { static_cast<void>(tensor_ops::log(Tensor({}, 0.0F))); },
        "log should reject zero"
    );
    require_throws(
        [] { static_cast<void>(tensor_ops::sqrt(Tensor({}, -1.0F))); },
        "sqrt should reject negative values"
    );
}

void test_softmax_and_backward() {
    const Tensor logits({2, 3}, {0.0F, 0.0F, 0.0F, 1.0F, 2.0F, 3.0F});
    const Tensor probabilities = tensor_ops::softmax(logits, 1);
    const float denominator = std::exp(-2.0F) + std::exp(-1.0F) + 1.0F;
    require_tensor_close(
        probabilities,
        {2, 3},
        {
            1.0F / 3.0F,
            1.0F / 3.0F,
            1.0F / 3.0F,
            std::exp(-2.0F) / denominator,
            std::exp(-1.0F) / denominator,
            1.0F / denominator,
        },
        "softmax values",
        1.0e-6F
    );
    require_close(
        probabilities.at({0, 0}) + probabilities.at({0, 1}) +
            probabilities.at({0, 2}),
        1.0F,
        "first softmax row sum"
    );

    const Tensor shifted =
        tensor_ops::softmax(Tensor({3}, {10000.0F, 10001.0F, 10002.0F}), 0);
    const Tensor centered =
        tensor_ops::softmax(Tensor({3}, {0.0F, 1.0F, 2.0F}), 0);
    for (std::size_t index = 0; index < 3; ++index) {
        require_close(
            shifted.flat(index),
            centered.flat(index),
            "softmax translation invariance"
        );
    }

    const float negative_infinity = -std::numeric_limits<float>::infinity();
    require_tensor_close(
        tensor_ops::softmax(Tensor({2}, {0.0F, negative_infinity}), 0),
        {2},
        {1.0F, 0.0F},
        "masked softmax"
    );

    require_tensor_close(
        tensor_ops::softmax_backward(
            Tensor({2}, {0.25F, 0.75F}),
            Tensor({2}, {2.0F, -1.0F}),
            0
        ),
        {2},
        {0.5625F, -0.5625F},
        "softmax backward"
    );

    require_throws(
        [negative_infinity] {
            static_cast<void>(tensor_ops::softmax(
                Tensor({2}, {negative_infinity, negative_infinity}),
                0
            ));
        },
        "fully masked softmax should throw"
    );
    require_throws(
        [] {
            static_cast<void>(tensor_ops::softmax(
                Tensor(
                    {2},
                    {
                        0.0F,
                        std::numeric_limits<float>::infinity(),
                    }
                ),
                0
            ));
        },
        "positive-infinity softmax should throw"
    );
    require_throws(
        [] { static_cast<void>(tensor_ops::softmax(Tensor({2}, 1.0F), 1)); },
        "out-of-range softmax axis should throw"
    );
}

void test_causal_softmax_and_backward() {
    const Tensor scores(
        {1, 1, 3, 3},
        {
            0.0F,
            1000.0F,
            -1000.0F,
            0.0F,
            2.0F,
            1000.0F,
            -2.0F,
            0.0F,
            2.0F,
        }
    );
    constexpr float score_scale = 0.5F;
    const Tensor probabilities =
        tensor_ops::causal_softmax(scores, score_scale);
    const float row_one_denominator = 1.0F + std::exp(1.0F);
    const float row_two_denominator = std::exp(-1.0F) + 1.0F + std::exp(1.0F);
    require_tensor_close(
        probabilities,
        {1, 1, 3, 3},
        {
            1.0F,
            0.0F,
            0.0F,
            1.0F / row_one_denominator,
            std::exp(1.0F) / row_one_denominator,
            0.0F,
            std::exp(-1.0F) / row_two_denominator,
            1.0F / row_two_denominator,
            std::exp(1.0F) / row_two_denominator,
        },
        "causal softmax values",
        1.0e-6F
    );
    require(
        probabilities.at({0, 0, 0, 1}) == 0.0F &&
            probabilities.at({0, 0, 0, 2}) == 0.0F &&
            probabilities.at({0, 0, 1, 2}) == 0.0F,
        "causal softmax future values must be exact zero"
    );

    const Tensor upstream(
        {1, 1, 3, 3},
        {
            0.5F,
            99.0F,
            -99.0F,
            -1.0F,
            2.0F,
            77.0F,
            0.25F,
            -0.75F,
            1.5F,
        }
    );
    const Tensor gradient = tensor_ops::causal_softmax_backward(
        probabilities,
        upstream,
        score_scale
    );
    const auto objective = [&](const Tensor& candidate_scores) {
        const Tensor candidate_probabilities =
            tensor_ops::causal_softmax(candidate_scores, score_scale);
        float result = 0.0F;
        for (std::size_t index = 0; index < candidate_probabilities.numel();
             ++index) {
            result +=
                candidate_probabilities.flat(index) * upstream.flat(index);
        }
        return result;
    };

    constexpr float epsilon = 1.0e-3F;
    for (std::size_t query = 0; query < 3; ++query) {
        for (std::size_t key = 0; key < 3; ++key) {
            const auto index = query * 3 + key;
            if (key > query) {
                require(
                    gradient.flat(index) == 0.0F,
                    "causal softmax future gradient must be exact zero"
                );
                continue;
            }
            Tensor plus = scores;
            Tensor minus = scores;
            plus.flat(index) += epsilon;
            minus.flat(index) -= epsilon;
            require_close(
                gradient.flat(index),
                (objective(plus) - objective(minus)) / (2.0F * epsilon),
                "causal softmax finite difference",
                2.0e-3F
            );
        }
    }

    require_throws(
        [] {
            static_cast<void>(tensor_ops::causal_softmax(Tensor({2, 3}, 0.0F)));
        },
        "causal softmax should reject non-rank-four scores"
    );
}

}  // namespace

int main() {
    try {
        test_elementwise_operations();
        test_matrix_multiplication_and_transpose();
        test_permutation_and_batched_matrix_multiplication();
        test_broadcasting();
        test_sum_to_shape_and_gather();
        test_reductions();
        test_unary_functions_and_domains();
        test_softmax_and_backward();
        test_causal_softmax_and_backward();
        std::cout << "tensor_ops tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
