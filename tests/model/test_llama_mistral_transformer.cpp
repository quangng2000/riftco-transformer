#include "riftco_transformer/model/grouped_query_attention.hpp"
#include "riftco_transformer/model/llama_mistral_transformer.hpp"
#include "riftco_transformer/model/rotary_embedding.hpp"
#include "riftco_transformer/nn/activations.hpp"
#include "riftco_transformer/nn/rms_norm.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using riftco_transformer::LlamaMistralArchitecture;
using riftco_transformer::LlamaMistralConfig;
using riftco_transformer::LlamaMistralTransformer;
using riftco_transformer::GroupedQueryAttention;
using riftco_transformer::RMSNorm;
using riftco_transformer::Tensor;
using riftco_transformer::TokenId;
using riftco_transformer::Variable;
using riftco_transformer::apply_rotary_position_embedding;
using riftco_transformer::repeat_key_value_heads;
using riftco_transformer::silu;
using riftco_transformer::sum;
using riftco_transformer::validate_llama_mistral_config;

constexpr LlamaMistralConfig kConfig{
    LlamaMistralArchitecture::Llama,
    11,
    4,
    8,
    4,
    2,
    1,
    12,
    1.0e-5F,
    10000.0F,
    std::nullopt,
};

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
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
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

void require_finite(const Tensor& tensor, const std::string& message) {
    for (const float value : tensor.data()) {
        require(std::isfinite(value), message);
    }
}

void test_rms_norm_formula_and_gradients() {
    const std::vector<float> input_values{1.0F, -2.0F, 3.0F, -4.0F};
    const std::vector<float> scale_values{0.5F, 1.0F, 1.5F, 2.0F};
    constexpr float epsilon = 1.0e-5F;
    RMSNorm norm(Tensor({4}, scale_values), epsilon);
    Variable input(Tensor({1, 1, 4}, input_values));
    const Variable output = norm.forward(input);

    double mean_square = 0.0;
    double dot = 0.0;
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        mean_square += static_cast<double>(input_values[index]) *
                       static_cast<double>(input_values[index]);
        dot += static_cast<double>(input_values[index]) *
               static_cast<double>(scale_values[index]);
    }
    mean_square /= static_cast<double>(input_values.size());
    const double root = std::sqrt(mean_square + epsilon);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        require_close(
            output.value().flat(index),
            static_cast<float>(
                input_values[index] * scale_values[index] / root
            ),
            "RMSNorm forward"
        );
    }

    sum(output).backward();
    const double root_cubed = root * root * root;
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        const double expected_input_gradient =
            static_cast<double>(scale_values[index]) / root -
            static_cast<double>(input_values[index]) * dot /
                (static_cast<double>(input_values.size()) * root_cubed);
        require_close(
            input.gradient().flat(index),
            static_cast<float>(expected_input_gradient),
            "RMSNorm input gradient",
            3.0e-5F
        );
        require_close(
            norm.scale().gradient().flat(index),
            static_cast<float>(input_values[index] / root),
            "RMSNorm scale gradient"
        );
    }
    require(norm.parameters().size() == 1, "RMSNorm must have no bias");
}

void test_silu_and_rotary_embedding() {
    Variable activation_input(Tensor(
        {5},
        {-1000.0F, -1.0F, 0.0F, 2.0F, 1000.0F}
    ));
    const Variable activated = silu(activation_input);
    for (std::size_t index = 0; index < 5; ++index) {
        const float value = activation_input.value().flat(index);
        const double sigmoid = value >= 0.0F
            ? 1.0 / (1.0 + std::exp(-static_cast<double>(value)))
            : std::exp(static_cast<double>(value)) /
                (1.0 + std::exp(static_cast<double>(value)));
        require_close(
            activated.value().flat(index),
            static_cast<float>(static_cast<double>(value) * sigmoid),
            "stable SiLU formula"
        );
    }
    sum(activated).backward();
    require_finite(activated.value(), "SiLU extreme outputs must be finite");
    require_finite(
        activation_input.gradient(),
        "SiLU extreme gradients must be finite"
    );

    Variable rotary_input(Tensor(
        {1, 1, 2, 4},
        {1.0F, 2.0F, 3.0F, 4.0F,
         5.0F, 6.0F, 7.0F, 8.0F}
    ));
    const Variable rotated = apply_rotary_position_embedding(rotary_input);
    for (std::size_t index = 0; index < 4; ++index) {
        require_close(
            rotated.value().flat(index),
            rotary_input.value().flat(index),
            "RoPE position zero"
        );
    }
    const float cosine_zero = std::cos(1.0F);
    const float sine_zero = std::sin(1.0F);
    const float cosine_one = std::cos(0.01F);
    const float sine_one = std::sin(0.01F);
    require_close(
        rotated.value().flat(4),
        5.0F * cosine_zero - 7.0F * sine_zero,
        "RoPE first half channel zero"
    );
    require_close(
        rotated.value().flat(6),
        7.0F * cosine_zero + 5.0F * sine_zero,
        "RoPE second half channel zero"
    );
    require_close(
        rotated.value().flat(5),
        6.0F * cosine_one - 8.0F * sine_one,
        "RoPE first half channel one"
    );
    require_close(
        rotated.value().flat(7),
        8.0F * cosine_one + 6.0F * sine_one,
        "RoPE second half channel one"
    );

    Tensor rotary_seed(
        {1, 1, 2, 4},
        {0.2F, -0.1F, 0.4F, 0.7F,
         -0.3F, 0.5F, 0.8F, -0.2F}
    );
    rotated.backward(rotary_seed);
    require_close(
        rotary_input.gradient().flat(4),
        -0.3F * cosine_zero + 0.8F * sine_zero,
        "RoPE transpose gradient first channel"
    );
    require_close(
        rotary_input.gradient().flat(6),
        0.3F * sine_zero + 0.8F * cosine_zero,
        "RoPE transpose gradient second channel"
    );

    constexpr float step = 1.0e-2F;
    const auto rotary_objective = [&](float perturbed) {
        std::vector<float> values{
            1.0F, 2.0F, 3.0F, 4.0F,
            5.0F, 6.0F, 7.0F, 8.0F,
        };
        values[4] = perturbed;
        const Variable candidate = apply_rotary_position_embedding(
            Variable(Tensor({1, 1, 2, 4}, values), false)
        );
        float objective = 0.0F;
        for (std::size_t index = 0; index < values.size(); ++index) {
            objective += candidate.value().flat(index) *
                         rotary_seed.flat(index);
        }
        return objective;
    };
    const float numerical =
        (rotary_objective(5.0F + step) -
         rotary_objective(5.0F - step)) /
        (2.0F * step);
    require_close(
        rotary_input.gradient().flat(4),
        numerical,
        "RoPE finite-difference gradient",
        5.0e-4F
    );
}

void test_grouped_head_repetition_forward_and_backward() {
    Variable input(Tensor({1, 2, 1, 2}, {1.0F, 2.0F, 3.0F, 4.0F}));
    const Variable repeated = repeat_key_value_heads(input, 2);
    const std::vector<float> expected{
        1.0F, 2.0F,
        1.0F, 2.0F,
        3.0F, 4.0F,
        3.0F, 4.0F,
    };
    require(
        repeated.value().shape() == Tensor::Shape({1, 4, 1, 2}),
        "repeated key/value shape"
    );
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_close(
            repeated.value().flat(index),
            expected[index],
            "repeated key/value value"
        );
    }
    sum(repeated).backward();
    for (const float gradient : input.gradient().data()) {
        require_close(gradient, 2.0F, "repeated key/value gradient");
    }
}

void test_grouped_query_attention_finite_difference() {
    std::mt19937 random(991U);
    GroupedQueryAttention attention(8, 4, 2, 10000.0F, random);
    std::vector<float> values(16);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = (static_cast<float>(index) - 7.0F) / 17.0F;
    }
    Variable input(Tensor({1, 2, 8}, values));
    const Variable output = attention.forward(input);
    std::vector<float> seed_values(16);
    for (std::size_t index = 0; index < seed_values.size(); ++index) {
        seed_values[index] =
            (static_cast<float>((index * 5) % 11) - 5.0F) / 13.0F;
    }
    Tensor seed({1, 2, 8}, seed_values);
    output.backward(seed);

    constexpr std::size_t differentiated_index = 6;
    constexpr float step = 1.0e-3F;
    const auto objective = [&](float perturbed) {
        std::vector<float> candidate_values = values;
        candidate_values[differentiated_index] = perturbed;
        const Variable candidate = attention.forward(
            Variable(Tensor({1, 2, 8}, candidate_values), false)
        );
        float total = 0.0F;
        for (std::size_t index = 0; index < seed_values.size(); ++index) {
            total += candidate.value().flat(index) * seed_values[index];
        }
        return total;
    };
    const float center = values[differentiated_index];
    const float numerical =
        (objective(center + step) - objective(center - step)) /
        (2.0F * step);
    require_close(
        input.gradient().flat(differentiated_index),
        numerical,
        "GQA finite-difference input gradient",
        4.0e-3F
    );
}

void test_config_validation_and_dense_mistral_policy() {
    static_cast<void>(validate_llama_mistral_config(kConfig));

    LlamaMistralConfig invalid = kConfig;
    invalid.query_head_count = 3;
    require_throws(
        [&] { static_cast<void>(validate_llama_mistral_config(invalid)); },
        "config should reject model/query head mismatch"
    );
    invalid = kConfig;
    invalid.key_value_head_count = 3;
    require_throws(
        [&] { static_cast<void>(validate_llama_mistral_config(invalid)); },
        "config should reject query/key-value head mismatch"
    );
    invalid = kConfig;
    invalid.model_width = 12;
    invalid.query_head_count = 4;
    require_throws(
        [&] { static_cast<void>(validate_llama_mistral_config(invalid)); },
        "config should reject odd RoPE head width"
    );
    invalid = kConfig;
    invalid.rms_norm_epsilon = 0.0F;
    require_throws(
        [&] { static_cast<void>(validate_llama_mistral_config(invalid)); },
        "config should reject zero RMSNorm epsilon"
    );
    invalid = kConfig;
    invalid.rope_theta = std::numeric_limits<float>::infinity();
    require_throws(
        [&] { static_cast<void>(validate_llama_mistral_config(invalid)); },
        "config should reject infinite RoPE theta"
    );

    LlamaMistralConfig mistral = kConfig;
    mistral.architecture = LlamaMistralArchitecture::Mistral;
    mistral.sliding_window = mistral.maximum_context;
    static_cast<void>(validate_llama_mistral_config(mistral));
    mistral.sliding_window = mistral.maximum_context - 1;
    require_throws(
        [&] { static_cast<void>(validate_llama_mistral_config(mistral)); },
        "narrow Mistral sliding window must not become dense attention"
    );
}

void test_model_schema_forward_causality_and_backward() {
    std::mt19937 random(123U);
    LlamaMistralTransformer model(kConfig, random);
    const auto parameters = model.parameters();
    const std::vector<std::string> expected_names{
        "token_embedding.weight",
        "blocks.0.attention_norm.scale",
        "blocks.0.attention.query.weight",
        "blocks.0.attention.key.weight",
        "blocks.0.attention.value.weight",
        "blocks.0.attention.output.weight",
        "blocks.0.feed_forward_norm.scale",
        "blocks.0.feed_forward.gate.weight",
        "blocks.0.feed_forward.up.weight",
        "blocks.0.feed_forward.down.weight",
        "final_norm.scale",
        "language_model_head.weight",
    };
    require(
        parameters.size() == expected_names.size(),
        "Llama/Mistral parameter count"
    );
    for (std::size_t index = 0; index < expected_names.size(); ++index) {
        require(
            parameters[index].name == expected_names[index],
            "Llama/Mistral parameter order at " +
                std::to_string(index)
        );
        require(
            parameters[index].name.find("bias") == std::string::npos,
            "Llama/Mistral projections and norms must be bias-free"
        );
    }
    require(
        parameters[3].parameter->value().shape() == Tensor::Shape({4, 8}),
        "GQA key projection shape"
    );
    require(
        parameters[4].parameter->value().shape() == Tensor::Shape({4, 8}),
        "GQA value projection shape"
    );

    const std::vector<TokenId> tokens{1, 2, 3, 4, 5, 6};
    const Variable logits = model.forward(tokens, {2, 3});
    require(
        logits.value().shape() == Tensor::Shape({2, 3, 11}),
        "Llama/Mistral logits shape"
    );
    require_finite(logits.value(), "Llama/Mistral logits must be finite");
    sum(logits * logits).backward();
    for (const auto& parameter : parameters) {
        require_finite(
            parameter.parameter->gradient(),
            "Llama/Mistral gradient must be finite: " + parameter.name
        );
    }

    std::mt19937 causal_random(321U);
    LlamaMistralTransformer causal_model(kConfig, causal_random);
    const Variable first = causal_model.forward(
        std::vector<TokenId>{1, 2, 3},
        {1, 3}
    );
    const Variable second = causal_model.forward(
        std::vector<TokenId>{1, 8, 9},
        {1, 3}
    );
    for (std::size_t vocabulary = 0;
         vocabulary < kConfig.vocabulary_size;
         ++vocabulary) {
        require_close(
            first.value().flat(vocabulary),
            second.value().flat(vocabulary),
            "future tokens must not affect first-position logits"
        );
    }
}

}  // namespace

int main() {
    try {
        test_rms_norm_formula_and_gradients();
        test_silu_and_rotary_embedding();
        test_grouped_head_repetition_forward_and_backward();
        test_grouped_query_attention_finite_difference();
        test_config_validation_and_dense_mistral_policy();
        test_model_schema_forward_causality_and_backward();
        std::cout << "Llama/Mistral transformer tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
