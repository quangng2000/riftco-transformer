#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"
#include "riftco_transformer/model/feed_forward.hpp"
#include "riftco_transformer/nn/embedding.hpp"
#include "riftco_transformer/nn/layer_norm.hpp"
#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/parameter.hpp"
#include "riftco_transformer/optim/adam.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using riftco_transformer::Embedding;
using riftco_transformer::ExecutionBackend;
using riftco_transformer::FeedForward;
using riftco_transformer::FeedForwardActivation;
using riftco_transformer::LayerNorm;
using riftco_transformer::Linear;
using riftco_transformer::Parameter;
using riftco_transformer::Tensor;
using riftco_transformer::TokenId;
using riftco_transformer::Variable;
namespace tensor_ops = riftco_transformer::tensor_ops;

static_assert(!std::is_assignable_v<
              decltype((std::declval<
                  riftco_transformer::NamedParameter&>().parameter)),
              Parameter*>);

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

void require_finite_tensor(
    const Tensor& tensor,
    const std::string& message
) {
    for (const float value : tensor.data()) {
        require(std::isfinite(value), message);
    }
}

void require_parameter_backend(
    const riftco_transformer::ParameterList& parameters,
    ExecutionBackend backend,
    const std::string& message
) {
    require(!parameters.empty(), message + ": empty parameter list");
    for (const auto& named_parameter : parameters) {
        require(
            named_parameter.parameter != nullptr,
            message + ": null parameter " + named_parameter.name
        );
        require(
            named_parameter.parameter->value().backend() == backend,
            message + ": value backend for " + named_parameter.name
        );
        require(
            named_parameter.parameter->gradient().backend() == backend,
            message + ": gradient backend for " + named_parameter.name
        );
    }
}

void test_parameter_and_registration() {
    Parameter parameter(Tensor({2}, {1.0F, 2.0F}));
    riftco_transformer::sum(
        parameter.variable() * parameter.variable()
    ).backward();
    require_tensor_close(
        parameter.gradient(),
        {2},
        {2.0F, 4.0F},
        "parameter gradient"
    );

    parameter.set_value(Tensor({2}, {3.0F, 4.0F}));
    require_tensor_close(
        parameter.value(),
        {2},
        {3.0F, 4.0F},
        "parameter replacement"
    );
    require_tensor_close(
        parameter.gradient(),
        {2},
        {0.0F, 0.0F},
        "parameter replacement should clear gradient"
    );
    require_throws(
        [&] { parameter.set_value(Tensor({1}, 3.0F)); },
        "parameter replacement should preserve shape"
    );

    riftco_transformer::ParameterList list{
        {"example", &parameter},
    };
    require(
        riftco_transformer::parameter_count(list) == 2,
        "parameter count mismatch"
    );
    require_throws(
        [] {
            const riftco_transformer::ParameterList invalid{
                {"missing", nullptr},
            };
            static_cast<void>(
                riftco_transformer::parameter_count(invalid)
            );
        },
        "null registered parameter should throw"
    );
}

void test_parameter_handle_lifetime_move_and_identity() {
    riftco_transformer::ParameterList retained;
    Parameter* canonical_parameter = nullptr;
    {
        Parameter original(Tensor({2}, {1.0F, 2.0F}));
        retained = {{"value", &original}};
        const riftco_transformer::ParameterList copied = retained;
        canonical_parameter = retained.front().parameter;
        require(
            canonical_parameter != nullptr &&
                copied.front().parameter == canonical_parameter,
            "copied parameter handles preserve canonical identity"
        );

        Parameter moved(std::move(original));
        const riftco_transformer::ParameterList after_move{
            {"value", &moved},
        };
        require(
            after_move.front().parameter == canonical_parameter,
            "parameter move preserves registered identity"
        );
        moved.set_value(Tensor({2}, {3.0F, 4.0F}));
        require_tensor_close(
            retained.front().parameter->value(),
            {2},
            {3.0F, 4.0F},
            "retained handle observes moved-wrapper mutation"
        );

        Parameter rebound(Tensor({1}, 9.0F));
        const riftco_transformer::ParameterList rebound_state{
            {"old", &rebound},
        };
        rebound = std::move(moved);
        require(
            rebound.handle() == canonical_parameter,
            "move assignment adopts the canonical parameter identity"
        );
        require_tensor_close(
            rebound_state.front().parameter->value(),
            {1},
            {9.0F},
            "move assignment does not invalidate old registered state"
        );
    }

    require(
        retained.front().parameter == canonical_parameter,
        "handle survives originating wrapper destruction"
    );
    retained.front().parameter->set_value(
        Tensor({2}, {5.0F, 6.0F})
    );
    require_tensor_close(
        retained.front().parameter->value(),
        {2},
        {5.0F, 6.0F},
        "retained handle owns parameter state"
    );
    Parameter replacement(Tensor({2}, {7.0F, 8.0F}));
    require_throws(
        [&] {
            *retained.front().parameter = std::move(replacement);
        },
        "canonical parameter compatibility view cannot be rebound"
    );
    require_tensor_close(
        retained.front().parameter->value(),
        {2},
        {5.0F, 6.0F},
        "rejected canonical rebind preserves retained state"
    );

    riftco_transformer::ParameterList module_parameters;
    Parameter* module_identity = nullptr;
    {
        Embedding embedding(
            Tensor(
                {2, 2},
                {0.0F, 1.0F, 2.0F, 3.0F}
            )
        );
        module_parameters = embedding.parameters();
        module_identity = module_parameters.front().parameter;

        require(
            embedding.parameters().front().parameter == module_identity,
            "module registration reuses parameter identity"
        );
    }
    require_tensor_close(
        module_parameters.front().parameter->value(),
        {2, 2},
        {0.0F, 1.0F, 2.0F, 3.0F},
        "parameter list survives originating module destruction"
    );
}

void test_embedding_forward_and_gradients() {
    const Tensor table(
        {3, 2},
        {0.1F, 0.2F, 1.0F, 2.0F, 3.0F, 4.0F}
    );
    Embedding embedding(table);
    const std::vector<TokenId> ids{2, 0, 2, 1};
    const Variable output = embedding.forward(ids, {2, 2});
    require_tensor_close(
        output.value(),
        {2, 2, 2},
        {3.0F, 4.0F, 0.1F, 0.2F, 3.0F, 4.0F, 1.0F, 2.0F},
        "embedding lookup"
    );

    const Tensor seed(
        {2, 2, 2},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F}
    );
    output.backward(seed);
    require_tensor_close(
        embedding.weight().gradient(),
        {3, 2},
        {3.0F, 4.0F, 7.0F, 8.0F, 6.0F, 8.0F},
        "embedding scatter-add gradient"
    );

    const Tensor analytical = embedding.weight().gradient();
    auto parameters = embedding.parameters();
    require(parameters.size() == 1, "embedding parameter count");
    require(parameters[0].name == "weight", "embedding parameter name");
    constexpr float epsilon = 1.0e-2F;
    for (std::size_t index = 0; index < table.numel(); ++index) {
        Tensor plus = table;
        Tensor minus = table;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;

        parameters[0].parameter->set_value(plus);
        const float plus_loss = tensor_dot(
            embedding.forward(ids, {2, 2}).value(),
            seed
        );
        parameters[0].parameter->set_value(minus);
        const float minus_loss = tensor_dot(
            embedding.forward(ids, {2, 2}).value(),
            seed
        );
        const float numerical =
            (plus_loss - minus_loss) / (2.0F * epsilon);
        require_close(
            analytical.flat(index),
            numerical,
            "embedding finite difference",
            2.0e-3F
        );
    }
    parameters[0].parameter->set_value(table);

    require_throws(
        [&] {
            static_cast<void>(embedding.forward(ids, {3}));
        },
        "embedding token shape mismatch should throw"
    );
    require_throws(
        [&] {
            const std::vector<TokenId> invalid_ids{3};
            static_cast<void>(embedding.forward(invalid_ids, {1}));
        },
        "embedding out-of-vocabulary ID should throw"
    );
}

void test_linear_forward_shapes() {
    const Tensor weight(
        {2, 3},
        {1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 2.0F}
    );
    const Tensor bias({2}, {0.5F, -0.5F});
    Linear linear(weight, bias);

    require_tensor_close(
        linear.forward(
            Variable(Tensor({3}, {1.0F, 2.0F, 3.0F}), false)
        ).value(),
        {2},
        {14.5F, 4.5F},
        "rank-one linear forward"
    );
    require_tensor_close(
        linear.forward(Variable(
            Tensor(
                {1, 2, 3},
                {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
            ),
            false
        )).value(),
        {1, 2, 2},
        {14.5F, 4.5F, 32.5F, 7.5F},
        "rank-three linear forward"
    );

    require_throws(
        [&] {
            static_cast<void>(linear.forward(
                Variable(Tensor({2}, 1.0F), false)
            ));
        },
        "linear input width mismatch should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(Linear(weight, Tensor({3}, 0.0F)));
        },
        "linear bias shape mismatch should throw"
    );
}

void test_bias_free_linear() {
  const Tensor weight({2, 3}, {1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 2.0F});
  Linear linear(weight);
  require(!linear.has_bias(), "explicit bias-free Linear state");
  require_tensor_close(
      linear.forward(Variable(Tensor({3}, {1.0F, 2.0F, 3.0F}), false)).value(),
      {2}, {14.0F, 5.0F}, "bias-free Linear forward");
  const auto parameters = linear.parameters();
  require(parameters.size() == 1 && parameters.front().name == "weight",
          "bias-free Linear registers only its weight");
  require_tensor_close(
      linear.bias().value(), {2}, {0.0F, 0.0F},
      "legacy bias accessor remains safe for bias-free Linear");
  {
    riftco_transformer::Adam optimizer(linear.parameters());
    require(optimizer.parameter_tensor_count() == 1 &&
                optimizer.state_payload_bytes() ==
                    2 * weight.numel() * sizeof(float),
            "bias-free Linear allocates Adam state only for its weight");
  }
  linear.to(ExecutionBackend::Cpu);
  require(linear.bias().value().backend() == ExecutionBackend::Cpu,
          "inert Linear bias follows an explicit CPU transfer");
  if (riftco_transformer::execution_backend_available(
          ExecutionBackend::Metal)) {
    linear.to(ExecutionBackend::Metal);
    require(linear.parameters().size() == 1 &&
                linear.parameters().front().parameter->value().backend() ==
                    ExecutionBackend::Metal &&
                linear.bias().value().backend() == ExecutionBackend::Metal,
            "bias-free Linear transfers weight and inert bias to Metal");
    linear.to(ExecutionBackend::Cpu);
  }

  std::mt19937 no_bias_random(91U);
  Linear initialized_without_bias(3, 2, no_bias_random, false);
  require(!initialized_without_bias.has_bias() &&
              initialized_without_bias.parameters().size() == 1,
          "initialized bias-free Linear schema");
  std::mt19937 legacy_random(91U);
  Linear legacy(3, 2, legacy_random);
  require(legacy.has_bias() && legacy.parameters().size() == 2,
          "legacy Linear constructor remains biased");

  Linear quantized_without_bias(Tensor::zeros({1, 32}));
  quantized_without_bias.quantize_weight_nf4(32);
  require(quantized_without_bias.has_quantized_weight() &&
              !quantized_without_bias.has_bias() &&
              quantized_without_bias.parameters().empty(),
          "bias-free quantized Linear has no floating-point base parameter");
  std::mt19937 adapter_random(92U);
  quantized_without_bias.attach_lora(1, 2.0F, adapter_random);
  {
    riftco_transformer::Adam adapter_optimizer(
        quantized_without_bias.lora_parameters());
    require(adapter_optimizer.parameter_tensor_count() == 2,
            "bias-free QLoRA Adam state contains only adapter tensors");
  }
  quantized_without_bias.merge_lora();
  require(!quantized_without_bias.has_quantized_weight() &&
              !quantized_without_bias.has_bias() &&
              quantized_without_bias.parameters().size() == 1 &&
              quantized_without_bias.parameters().front().name == "weight",
          "bias-free Linear preserves its schema across LoRA merge");
}

void test_linear_finite_differences() {
    const Tensor input_values(
        {2, 2, 3},
        {
            0.2F, -0.3F, 0.5F,
            1.0F, 0.4F, -0.2F,
            -0.6F, 0.8F, 0.15F,
            0.7F, -0.1F, -0.9F,
        }
    );
    const Tensor weight_values(
        {2, 3},
        {0.5F, -0.25F, 0.75F, -0.4F, 0.6F, 0.2F}
    );
    const Tensor bias_values({2}, {0.1F, -0.2F});
    const Tensor output_weights(
        {2, 2, 2},
        {0.5F, -1.0F, 1.5F, 0.25F, -0.75F, 0.6F, 0.4F, -1.2F}
    );

    Linear linear(weight_values, bias_values);
    const Variable input(input_values);
    const Variable loss = riftco_transformer::sum(
        linear.forward(input) * Variable(output_weights, false)
    );
    loss.backward();
    const Tensor input_gradient = input.gradient();
    auto parameters = linear.parameters();
    const Tensor weight_gradient = parameters[0].parameter->gradient();
    const Tensor bias_gradient = parameters[1].parameter->gradient();

    const auto evaluate = [&](const Tensor& candidate_input) {
        return tensor_dot(
            linear.forward(Variable(candidate_input, false)).value(),
            output_weights
        );
    };

    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < input_values.numel(); ++index) {
        Tensor plus = input_values;
        Tensor minus = input_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        const float numerical =
            (evaluate(plus) - evaluate(minus)) /
            (2.0F * epsilon);
        require_close(
            input_gradient.flat(index),
            numerical,
            "rank-three linear input finite difference",
            2.0e-3F
        );
    }

    for (std::size_t index = 0; index < weight_values.numel(); ++index) {
        Tensor plus = weight_values;
        Tensor minus = weight_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        parameters[0].parameter->set_value(plus);
        const float plus_loss = evaluate(input_values);
        parameters[0].parameter->set_value(minus);
        const float minus_loss = evaluate(input_values);
        require_close(
            weight_gradient.flat(index),
            (plus_loss - minus_loss) / (2.0F * epsilon),
            "rank-three linear weight finite difference",
            2.0e-3F
        );
    }
    parameters[0].parameter->set_value(weight_values);

    for (std::size_t index = 0; index < bias_values.numel(); ++index) {
        Tensor plus = bias_values;
        Tensor minus = bias_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        parameters[1].parameter->set_value(plus);
        const float plus_loss = evaluate(input_values);
        parameters[1].parameter->set_value(minus);
        const float minus_loss = evaluate(input_values);
        require_close(
            bias_gradient.flat(index),
            (plus_loss - minus_loss) / (2.0F * epsilon),
            "rank-three linear bias finite difference",
            2.0e-3F
        );
    }
    parameters[1].parameter->set_value(bias_values);
}

void test_initialization_and_feed_forward() {
    std::mt19937 first_random(7);
    std::mt19937 second_random(7);
    Linear first(3, 4, first_random);
    Linear second(3, 4, second_random);
    require_tensor_close(
        first.weight().value(),
        second.weight().value().shape(),
        std::vector<float>(
            second.weight().value().data().begin(),
            second.weight().value().data().end()
        ),
        "seeded linear initialization"
    );

    std::mt19937 random(42);
    FeedForward feed_forward(3, 5, random);
    const Variable input(Tensor(
        {2, 2, 3},
        {
            -1.0F, -0.5F, 0.0F,
            0.5F, 1.0F, 1.5F,
            2.0F, -1.5F, 0.25F,
            -0.25F, 0.75F, -0.75F,
        }
    ));
    const Variable output = feed_forward.forward(input);
    const Tensor output_weights(
        {2, 2, 3},
        {
            0.5F, -1.0F, 0.25F,
            1.5F, 0.75F, -0.4F,
            -0.6F, 0.2F, 1.25F,
            0.9F, -1.1F, 0.35F,
        }
    );
    require(
        output.value().shape() == Tensor::Shape({2, 2, 3}),
        "feed-forward should preserve model shape"
    );
    riftco_transformer::sum(
        output * Variable(output_weights, false)
    ).backward();
    require_finite_tensor(
        input.gradient(),
        "feed-forward input gradient should be finite"
    );
    const Tensor input_gradient = input.gradient();

    auto parameters = feed_forward.parameters();
    require(parameters.size() == 4, "feed-forward parameter count");
    const std::set<std::string> expected_names{
        "expand.weight",
        "expand.bias",
        "project.weight",
        "project.bias",
    };
    std::set<std::string> actual_names;
    for (const auto& named_parameter : parameters) {
        actual_names.insert(named_parameter.name);
        require_finite_tensor(
            named_parameter.parameter->gradient(),
            "feed-forward parameter gradient should be finite"
        );
    }
    require(
        actual_names == expected_names,
        "feed-forward registration names"
    );
    require(
        riftco_transformer::parameter_count(parameters) == 38,
        "feed-forward scalar parameter count"
    );

    std::vector<Tensor> parameter_values;
    std::vector<Tensor> parameter_gradients;
    parameter_values.reserve(parameters.size());
    parameter_gradients.reserve(parameters.size());
    for (const auto& named_parameter : parameters) {
        parameter_values.push_back(named_parameter.parameter->value());
        parameter_gradients.push_back(
            named_parameter.parameter->gradient()
        );
    }

    const auto evaluate = [&](const Tensor& candidate_input) {
        return tensor_dot(
            feed_forward.forward(
                Variable(candidate_input, false)
            ).value(),
            output_weights
        );
    };

    constexpr float epsilon = 1.0e-2F;
    const Tensor input_values = input.value();
    for (std::size_t index = 0; index < input_values.numel(); ++index) {
        Tensor plus = input_values;
        Tensor minus = input_values;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        require_close(
            input_gradient.flat(index),
            (evaluate(plus) - evaluate(minus)) /
                (2.0F * epsilon),
            "feed-forward input finite difference",
            5.0e-3F
        );
    }

    for (std::size_t parameter_index = 0;
         parameter_index < parameters.size();
         ++parameter_index) {
        const Tensor& original = parameter_values[parameter_index];
        for (std::size_t index = 0; index < original.numel(); ++index) {
            Tensor plus = original;
            Tensor minus = original;
            plus.flat(index) += epsilon;
            minus.flat(index) -= epsilon;

            parameters[parameter_index].parameter->set_value(plus);
            const float plus_loss = evaluate(input_values);
            parameters[parameter_index].parameter->set_value(minus);
            const float minus_loss = evaluate(input_values);
            require_close(
                parameter_gradients[parameter_index].flat(index),
                (plus_loss - minus_loss) / (2.0F * epsilon),
                "feed-forward " +
                    parameters[parameter_index].name +
                    " finite difference",
                5.0e-3F
            );
        }
        parameters[parameter_index].parameter->set_value(original);
    }
}

void test_relu_feed_forward() {
  std::mt19937 random(123U);
  FeedForward feed_forward(2, 2, random, FeedForwardActivation::Relu);
  require(feed_forward.activation() == FeedForwardActivation::Relu,
          "feed-forward ReLU configuration");
  auto parameters = feed_forward.parameters();
  for (const auto &named_parameter : parameters) {
    if (named_parameter.name == "expand.weight" ||
        named_parameter.name == "project.weight") {
      named_parameter.parameter->set_value(
          Tensor({2, 2}, {1.0F, 0.0F, 0.0F, 1.0F}));
    } else {
      named_parameter.parameter->set_value(Tensor({2}, 0.0F));
    }
  }

  const Variable input(Tensor({1, 2}, {-1.0F, 2.0F}));
  const Variable output = feed_forward.forward(input);
  require_tensor_close(output.value(), {1, 2}, {0.0F, 2.0F},
                       "feed-forward exact ReLU output");
  riftco_transformer::sum(output).backward();
  require_tensor_close(input.gradient(), {1, 2}, {0.0F, 1.0F},
                       "feed-forward exact ReLU gradient");

  std::mt19937 default_random(456U);
  FeedForward default_feed_forward(2, 3, default_random);
  require(default_feed_forward.activation() == FeedForwardActivation::Gelu,
          "legacy feed-forward constructor defaults to GELU");
  require_throws(
      [] {
        std::mt19937 invalid_random(789U);
        static_cast<void>(FeedForward(2, 3, invalid_random,
                                      static_cast<FeedForwardActivation>(255)));
      },
      "feed-forward should reject an unknown activation");
}

void test_module_device_transfer() {
    const riftco_transformer::ScopedExecutionBackend cpu_backend(
        ExecutionBackend::Cpu
    );
    std::mt19937 random(211U);
    Linear linear(3, 2, random);
    Embedding embedding(5, 3, random);
    LayerNorm layer_norm(3);
    FeedForward feed_forward(3, 5, random);

    const auto exercise_transfer =
        [](auto& module, const std::string& name) {
            module.to(ExecutionBackend::Cpu);
            require_parameter_backend(
                module.parameters(),
                ExecutionBackend::Cpu,
                name + " CPU transfer"
            );

            if (!riftco_transformer::execution_backend_available(
                    ExecutionBackend::Metal
                )) {
                return;
            }

            module.to(ExecutionBackend::Metal);
            require_parameter_backend(
                module.parameters(),
                ExecutionBackend::Metal,
                name + " Metal transfer"
            );
            module.to(ExecutionBackend::Cpu);
            require_parameter_backend(
                module.parameters(),
                ExecutionBackend::Cpu,
                name + " CPU round trip"
            );
        };

    exercise_transfer(linear, "linear");
    exercise_transfer(embedding, "embedding");
    exercise_transfer(layer_norm, "layer norm");
    exercise_transfer(feed_forward, "feed-forward");
}

}  // namespace

int main() {
    try {
        test_parameter_and_registration();
        test_parameter_handle_lifetime_move_and_identity();
        test_embedding_forward_and_gradients();
        test_linear_forward_shapes();
        test_bias_free_linear();
        test_linear_finite_differences();
        test_initialization_and_feed_forward();
        test_relu_feed_forward();
        test_module_device_transfer();
        std::cout << "layer tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
