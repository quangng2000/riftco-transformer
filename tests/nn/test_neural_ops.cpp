#include "transformer_lab/nn/activations.hpp"
#include "transformer_lab/nn/layer_norm.hpp"
#include "transformer_lab/nn/loss.hpp"
#include "transformer_lab/core/tensor_ops.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using transformer_lab::LayerNorm;
using transformer_lab::Tensor;
using transformer_lab::TokenId;
using transformer_lab::Variable;
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
    const std::vector<float>& expected_values,
    const std::string& message,
    float tolerance = 1.0e-5F
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

float tensor_dot(const Tensor& left, const Tensor& right) {
    return tensor_ops::sum(
        tensor_ops::multiply(left, right)
    ).flat(0);
}

float exact_gelu(float value) {
    return 0.5F * value *
           (1.0F + std::erf(
               value * 0.70710678118654752440F
           ));
}

void test_gelu_forward_and_gradients() {
    const Tensor input_values(
        {5},
        {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F}
    );
    const Tensor weights(
        {5},
        {0.5F, -1.0F, 2.0F, 0.25F, -0.75F}
    );
    const Variable input(input_values);
    const Variable output = transformer_lab::gelu(input);

    require_tensor_close(
        output.value(),
        {5},
        {
            exact_gelu(-2.0F),
            exact_gelu(-1.0F),
            0.0F,
            exact_gelu(1.0F),
            exact_gelu(2.0F),
        },
        "GELU forward",
        1.0e-6F
    );
    transformer_lab::sum(
        output * Variable(weights, false)
    ).backward();

    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < input_values.numel(); ++index) {
        Tensor plus = input_values;
        Tensor minus = input_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        const float plus_loss = tensor_dot(
            transformer_lab::gelu(
                Variable(plus, false)
            ).value(),
            weights
        );
        const float minus_loss = tensor_dot(
            transformer_lab::gelu(
                Variable(minus, false)
            ).value(),
            weights
        );
        require_close(
            input.gradient().flat(index),
            (plus_loss - minus_loss) / (2.0F * epsilon),
            "GELU finite difference",
            2.0e-3F
        );
    }
}

void test_layer_norm_forward_and_parameters() {
    const Variable input(Tensor(
        {2, 3},
        {1.0F, 2.0F, 3.0F, -2.0F, 0.0F, 2.0F}
    ));
    const Variable scale(Tensor({3}, 1.0F));
    const Variable bias(Tensor({3}, 0.0F));
    const Variable output = transformer_lab::layer_norm(
        input,
        scale,
        bias
    );
    require(
        output.value().shape() == Tensor::Shape({2, 3}),
        "layer norm shape"
    );
    for (std::size_t row = 0; row < 2; ++row) {
        float mean = 0.0F;
        float square_mean = 0.0F;
        for (std::size_t column = 0; column < 3; ++column) {
            const float value = output.value().at({row, column});
            mean += value / 3.0F;
            square_mean += value * value / 3.0F;
        }
        require_close(mean, 0.0F, "normalized row mean", 1.0e-6F);
        require_close(
            square_mean,
            row == 0 ? 0.999985F : 0.999996F,
            "normalized row variance",
            2.0e-5F
        );
    }

    const Variable constant_input(Tensor({2, 3}, 4.0F));
    const Variable affine_scale(
        Tensor({3}, {2.0F, 0.5F, -1.0F})
    );
    const Variable affine_bias(
        Tensor({3}, {0.1F, -0.2F, 0.3F})
    );
    require_tensor_close(
        transformer_lab::layer_norm(
            constant_input,
            affine_scale,
            affine_bias
        ).value(),
        {2, 3},
        {0.1F, -0.2F, 0.3F, 0.1F, -0.2F, 0.3F},
        "constant layer norm slice"
    );

    LayerNorm layer(3);
    const auto parameters = layer.parameters();
    require(parameters.size() == 2, "layer norm parameter count");
    require(
        transformer_lab::parameter_count(parameters) == 6,
        "layer norm scalar parameter count"
    );

    require_throws(
        [&] {
            static_cast<void>(transformer_lab::layer_norm(
                Variable::scalar(1.0F),
                scale,
                bias
            ));
        },
        "layer norm scalar input should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(transformer_lab::layer_norm(
                input,
                Variable(Tensor({2}, 1.0F)),
                bias
            ));
        },
        "layer norm parameter shape mismatch should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(transformer_lab::layer_norm(
                input,
                scale,
                bias,
                -1.0F
            ));
        },
        "negative layer norm epsilon should throw"
    );
}

void test_layer_norm_finite_differences() {
    const Tensor input_values(
        {2, 3},
        {0.2F, -0.7F, 1.3F, 2.0F, -1.0F, 0.4F}
    );
    const Tensor scale_values({3}, {1.2F, 0.8F, -0.5F});
    const Tensor bias_values({3}, {0.1F, -0.2F, 0.3F});
    const Tensor weights(
        {2, 3},
        {0.5F, -1.0F, 0.25F, 1.5F, 0.75F, -0.4F}
    );

    const Variable input(input_values);
    const Variable scale(scale_values);
    const Variable bias(bias_values);
    transformer_lab::sum(
        transformer_lab::layer_norm(
            input,
            scale,
            bias
        ) * Variable(weights, false)
    ).backward();

    const auto evaluate = [&](
        const Tensor& candidate_input,
        const Tensor& candidate_scale,
        const Tensor& candidate_bias
    ) {
        return tensor_dot(
            transformer_lab::layer_norm(
                Variable(candidate_input, false),
                Variable(candidate_scale, false),
                Variable(candidate_bias, false)
            ).value(),
            weights
        );
    };

    constexpr float epsilon = 1.0e-2F;
    for (std::size_t index = 0; index < input_values.numel(); ++index) {
        Tensor plus = input_values;
        Tensor minus = input_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        require_close(
            input.gradient().flat(index),
            (
                evaluate(plus, scale_values, bias_values) -
                evaluate(minus, scale_values, bias_values)
            ) / (2.0F * epsilon),
            "layer norm input finite difference",
            5.0e-3F
        );
    }
    for (std::size_t index = 0; index < scale_values.numel(); ++index) {
        Tensor plus = scale_values;
        Tensor minus = scale_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        require_close(
            scale.gradient().flat(index),
            (
                evaluate(input_values, plus, bias_values) -
                evaluate(input_values, minus, bias_values)
            ) / (2.0F * epsilon),
            "layer norm scale finite difference",
            5.0e-3F
        );
    }
    for (std::size_t index = 0; index < bias_values.numel(); ++index) {
        Tensor plus = bias_values;
        Tensor minus = bias_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        require_close(
            bias.gradient().flat(index),
            (
                evaluate(input_values, scale_values, plus) -
                evaluate(input_values, scale_values, minus)
            ) / (2.0F * epsilon),
            "layer norm bias finite difference",
            5.0e-3F
        );
    }
}

void test_softmax_forward_and_gradients() {
    const Tensor logits_values(
        {2, 3},
        {0.2F, -0.5F, 1.0F, 2.0F, 0.3F, -1.0F}
    );
    const Tensor weights(
        {2, 3},
        {0.5F, -1.0F, 2.0F, -0.25F, 1.5F, 0.75F}
    );
    const Variable logits(logits_values);
    const Variable probabilities =
        transformer_lab::softmax(logits, 1);
    for (std::size_t row = 0; row < 2; ++row) {
        float total = 0.0F;
        for (std::size_t column = 0; column < 3; ++column) {
            total += probabilities.value().at({row, column});
        }
        require_close(total, 1.0F, "softmax row sum");
    }
    transformer_lab::sum(
        probabilities * Variable(weights, false)
    ).backward();

    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < logits_values.numel(); ++index) {
        Tensor plus = logits_values;
        Tensor minus = logits_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        const float plus_loss = tensor_dot(
            transformer_lab::softmax(
                Variable(plus, false),
                1
            ).value(),
            weights
        );
        const float minus_loss = tensor_dot(
            transformer_lab::softmax(
                Variable(minus, false),
                1
            ).value(),
            weights
        );
        require_close(
            logits.gradient().flat(index),
            (plus_loss - minus_loss) / (2.0F * epsilon),
            "softmax finite difference",
            2.0e-3F
        );
    }

    const Tensor middle_values(
        {2, 3, 2},
        {
            0.0F, 1.0F,
            2.0F, -0.5F,
            -1.0F, 3.0F,
            4.0F, -2.0F,
            0.25F, 5.0F,
            1.5F, 0.75F,
        }
    );
    const Tensor middle_weights(
        {2, 3, 2},
        {
            0.5F, -1.0F,
            1.5F, 0.25F,
            -0.75F, 0.6F,
            0.4F, -1.2F,
            0.9F, 0.3F,
            -0.2F, 1.1F,
        }
    );
    const Variable middle_axis(middle_values);
    const Variable middle_probabilities =
        transformer_lab::softmax(middle_axis, 1);
    for (std::size_t outer = 0; outer < 2; ++outer) {
        for (std::size_t inner = 0; inner < 2; ++inner) {
            require_close(
                middle_probabilities.value().at({outer, 0, inner}) +
                    middle_probabilities.value().at({outer, 1, inner}) +
                    middle_probabilities.value().at({outer, 2, inner}),
                1.0F,
                "middle-axis softmax sum"
            );
        }
    }
    transformer_lab::sum(
        middle_probabilities * Variable(middle_weights, false)
    ).backward();
    for (std::size_t index = 0; index < middle_values.numel(); ++index) {
        Tensor plus = middle_values;
        Tensor minus = middle_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        const float plus_loss = tensor_dot(
            transformer_lab::softmax(
                Variable(plus, false),
                1
            ).value(),
            middle_weights
        );
        const float minus_loss = tensor_dot(
            transformer_lab::softmax(
                Variable(minus, false),
                1
            ).value(),
            middle_weights
        );
        require_close(
            middle_axis.gradient().flat(index),
            (plus_loss - minus_loss) / (2.0F * epsilon),
            "middle-axis softmax finite difference",
            2.0e-3F
        );
    }

    const float negative_infinity =
        -std::numeric_limits<float>::infinity();
    const Variable masked_logits(
        Tensor({2}, {0.0F, negative_infinity})
    );
    const Variable masked =
        transformer_lab::softmax(masked_logits, 0);
    masked.backward(Tensor({2}, {2.0F, 7.0F}));
    require_tensor_close(
        masked.value(),
        {2},
        {1.0F, 0.0F},
        "masked differentiable softmax"
    );
    require_tensor_close(
        masked_logits.gradient(),
        {2},
        {0.0F, 0.0F},
        "masked softmax gradient"
    );
}

void test_cross_entropy_forward_and_gradients() {
    const std::vector<TokenId> simple_targets{0, 2};
    const Variable simple_logits(Tensor({2, 3}, 0.0F));
    const Variable simple_loss =
        transformer_lab::cross_entropy(
            simple_logits,
            simple_targets
        );
    require_close(
        simple_loss.value().flat(0),
        std::log(3.0F),
        "zero-logit cross entropy"
    );
    simple_loss.backward();
    require_tensor_close(
        simple_logits.gradient(),
        {2, 3},
        {
            -1.0F / 3.0F, 1.0F / 6.0F, 1.0F / 6.0F,
            1.0F / 6.0F, 1.0F / 6.0F, -1.0F / 3.0F,
        },
        "cross entropy closed-form gradient"
    );

    const Tensor logits_values(
        {2, 2, 3},
        {
            0.2F, -0.4F, 0.8F,
            1.0F, 0.5F, -0.5F,
            -0.3F, 0.7F, 0.1F,
            0.6F, -0.2F, 0.4F,
        }
    );
    const std::vector<TokenId> targets{0, 2, 1, 0};
    const Variable logits(logits_values);
    transformer_lab::cross_entropy(logits, targets).backward();

    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < logits_values.numel(); ++index) {
        Tensor plus = logits_values;
        Tensor minus = logits_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        const float plus_loss = transformer_lab::cross_entropy(
            Variable(plus, false),
            targets
        ).value().flat(0);
        const float minus_loss = transformer_lab::cross_entropy(
            Variable(minus, false),
            targets
        ).value().flat(0);
        require_close(
            logits.gradient().flat(index),
            (plus_loss - minus_loss) / (2.0F * epsilon),
            "cross entropy finite difference",
            2.0e-3F
        );
    }
    for (std::size_t position = 0; position < 4; ++position) {
        float gradient_sum = 0.0F;
        for (std::size_t token = 0; token < 3; ++token) {
            gradient_sum += logits.gradient().flat(
                position * 3 + token
            );
        }
        require_close(
            gradient_sum,
            0.0F,
            "cross entropy row gradient sum"
        );
    }

    require_close(
        transformer_lab::cross_entropy(
            Variable(
                Tensor({3}, {10000.0F, 0.0F, -10000.0F}),
                false
            ),
            std::vector<TokenId>{0}
        ).value().flat(0),
        0.0F,
        "large confident cross entropy",
        1.0e-6F
    );
    require_close(
        transformer_lab::cross_entropy(
            Variable(
                Tensor({3}, {10000.0F, 0.0F, -10000.0F}),
                false
            ),
            std::vector<TokenId>{2}
        ).value().flat(0),
        20000.0F,
        "large wrong cross entropy",
        1.0e-2F
    );

    require_throws(
        [&] {
            static_cast<void>(transformer_lab::cross_entropy(
                Variable(Tensor({2, 3}, 0.0F)),
                std::vector<TokenId>{0}
            ));
        },
        "cross entropy target count mismatch should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(transformer_lab::cross_entropy(
                Variable(Tensor({3}, 0.0F)),
                std::vector<TokenId>{3}
            ));
        },
        "cross entropy out-of-range target should throw"
    );
}

}  // namespace

int main() {
    try {
        test_gelu_forward_and_gradients();
        test_layer_norm_forward_and_parameters();
        test_layer_norm_finite_differences();
        test_softmax_forward_and_gradients();
        test_cross_entropy_forward_and_gradients();
        std::cout << "neural operation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
