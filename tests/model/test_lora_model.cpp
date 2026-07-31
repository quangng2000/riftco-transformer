#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/model/decoder_only_transformer.hpp"
#include "transformer_lab/model/lora.hpp"
#include "transformer_lab/nn/loss.hpp"
#include "transformer_lab/optim/adam.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using transformer_lab::Adam;
using transformer_lab::AdamOptions;
using transformer_lab::DecoderOnlyTransformer;
using transformer_lab::ExecutionBackend;
using transformer_lab::LoraConfig;
using transformer_lab::NamedParameter;
using transformer_lab::Parameter;
using transformer_lab::ParameterList;
using transformer_lab::Tensor;
using transformer_lab::TokenId;
using transformer_lab::TransformerDimensions;
using transformer_lab::Variable;
using transformer_lab::cross_entropy;
using transformer_lab::kLoraAllTargets;
using transformer_lab::kLoraAttentionQuery;
using transformer_lab::kLoraAttentionValue;
using transformer_lab::kLoraDefaultTargets;

constexpr TransformerDimensions kDimensions{
    5,
    4,
    4,
    2,
    2,
    6,
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
    const float scale = std::max(
        {1.0F, std::fabs(actual), std::fabs(expected)}
    );
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance * scale) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void require_tensor_close(
    const Tensor& actual,
    const Tensor& expected,
    const std::string& message,
    float tolerance = 1.0e-5F
) {
    require(
        actual.shape() == expected.shape(),
        message + ": shape mismatch"
    );
    for (std::size_t index = 0;
         index < actual.numel();
         ++index) {
        require_close(
            actual.flat(index),
            expected.flat(index),
            message + " at index " + std::to_string(index),
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

Parameter* parameter_named(
    const ParameterList& parameters,
    const std::string& name
) {
    for (const auto& named_parameter : parameters) {
        if (named_parameter.name == name) {
            return named_parameter.parameter;
        }
    }
    throw std::runtime_error("missing parameter " + name);
}

struct ParameterSnapshot {
    std::string name;
    Parameter* address;
    Tensor value;
};

std::vector<ParameterSnapshot> snapshot(
    const ParameterList& parameters
) {
    std::vector<ParameterSnapshot> result;
    result.reserve(parameters.size());
    for (const NamedParameter& named_parameter : parameters) {
        require(
            named_parameter.parameter != nullptr,
            "snapshot parameter must be non-null"
        );
        result.push_back({
            named_parameter.name,
            named_parameter.parameter,
            named_parameter.parameter->value(),
        });
    }
    return result;
}

void require_registration_and_values_unchanged(
    const ParameterList& current,
    const std::vector<ParameterSnapshot>& expected,
    const std::string& message
) {
    require(
        current.size() == expected.size(),
        message + ": parameter count"
    );
    for (std::size_t index = 0;
         index < current.size();
         ++index) {
        require(
            current[index].name == expected[index].name,
            message + ": name at index " + std::to_string(index)
        );
        require(
            current[index].parameter == expected[index].address,
            message + ": address at index " + std::to_string(index)
        );
        require_tensor_close(
            current[index].parameter->value(),
            expected[index].value,
            message + ": value for " + expected[index].name,
            0.0F
        );
    }
}

std::vector<std::string> default_lora_names() {
    std::vector<std::string> result;
    for (std::size_t block = 0; block < 2; ++block) {
        const std::string prefix =
            "blocks." + std::to_string(block) + ".attention.";
        result.push_back(prefix + "query.lora_a.weight");
        result.push_back(prefix + "query.lora_b.weight");
        result.push_back(prefix + "value.lora_a.weight");
        result.push_back(prefix + "value.lora_b.weight");
    }
    return result;
}

void test_default_registration_determinism_and_neutrality() {
    std::mt19937 base_random(211U);
    std::mt19937 comparison_random(997U);
    DecoderOnlyTransformer model(kDimensions, base_random);
    DecoderOnlyTransformer comparison(kDimensions, comparison_random);
    const auto base_snapshot = snapshot(model.parameters());
    const std::vector<TokenId> tokens{0, 1, 2, 3};
    const Tensor base_logits =
        model.forward(tokens, {1, 4}).value();

    const LoraConfig config{
        2,
        4.0F,
        313U,
        kLoraDefaultTargets,
    };
    model.attach_lora(config);
    comparison.attach_lora(config);

    require(model.has_lora(), "model reports active LoRA");
    require(
        model.lora_config() == config,
        "model reports its exact LoRA configuration"
    );
    require_registration_and_values_unchanged(
        model.parameters(),
        base_snapshot,
        "attachment preserves base parameters"
    );
    require_tensor_close(
        model.forward(tokens, {1, 4}).value(),
        base_logits,
        "zero-B model attachment is output-neutral",
        0.0F
    );

    const ParameterList adapters = model.lora_parameters();
    const ParameterList comparison_adapters =
        comparison.lora_parameters();
    const std::vector<std::string> expected_names =
        default_lora_names();
    require(
        adapters.size() == expected_names.size(),
        "default target tensor count"
    );
    for (std::size_t index = 0;
         index < adapters.size();
         ++index) {
        require(
            adapters[index].name == expected_names[index],
            "default target name at index " + std::to_string(index)
        );
        require_tensor_close(
            adapters[index].parameter->value(),
            comparison_adapters[index].parameter->value(),
            "adapter seed is independent of base initialization",
            0.0F
        );
        const bool is_a =
            adapters[index].name.ends_with("lora_a.weight");
        require(
            adapters[index].parameter->value().shape() ==
                (is_a
                     ? Tensor::Shape({2, 4})
                     : Tensor::Shape({4, 2})),
            "default adapter shape for " + adapters[index].name
        );
    }
}

void test_validation_is_transactional() {
    std::mt19937 random(401U);
    DecoderOnlyTransformer model(kDimensions, random);
    const auto base_snapshot = snapshot(model.parameters());

    LoraConfig invalid;
    invalid.rank = 0;
    require_throws(
        [&] { model.attach_lora(invalid); },
        "zero model LoRA rank is rejected"
    );
    invalid = {};
    invalid.rank = 5;
    require_throws(
        [&] { model.attach_lora(invalid); },
        "oversized model LoRA rank is rejected"
    );
    invalid = {};
    invalid.alpha = std::numeric_limits<float>::quiet_NaN();
    require_throws(
        [&] { model.attach_lora(invalid); },
        "nonfinite model LoRA alpha is rejected"
    );
    invalid = {};
    invalid.targets = 0;
    require_throws(
        [&] { model.attach_lora(invalid); },
        "empty model LoRA target mask is rejected"
    );
    invalid = {};
    invalid.targets =
        kLoraAllTargets |
        (transformer_lab::LoraTargetMask{1} << 40U);
    require_throws(
        [&] { model.attach_lora(invalid); },
        "unknown model LoRA target bits are rejected"
    );

    require(!model.has_lora(), "invalid attachment leaves no LoRA");
    require(
        model.lora_parameters().empty(),
        "invalid attachment leaves no adapter registration"
    );
    require_throws(
        [&] { static_cast<void>(model.lora_config()); },
        "configuration is unavailable without active LoRA"
    );
    require_registration_and_values_unchanged(
        model.parameters(),
        base_snapshot,
        "invalid attachment preserves the base model"
    );

    const LoraConfig valid{
        2,
        3.0F,
        17U,
        kLoraAttentionQuery | kLoraAttentionValue,
    };
    model.attach_lora(valid);
    require(
        model.has_lora() && model.lora_config() == valid,
        "valid attachment succeeds after rejected configurations"
    );
}

void test_all_target_registration() {
    TransformerDimensions dimensions = kDimensions;
    dimensions.block_count = 1;
    std::mt19937 random(419U);
    DecoderOnlyTransformer model(dimensions, random);
    model.attach_lora(LoraConfig{
        2,
        4.0F,
        23U,
        kLoraAllTargets,
    });

    const std::vector<std::string> projections{
        "blocks.0.attention.query",
        "blocks.0.attention.key",
        "blocks.0.attention.value",
        "blocks.0.attention.output",
        "blocks.0.feed_forward.expand",
        "blocks.0.feed_forward.project",
        "language_model_head",
    };
    std::vector<std::string> expected;
    for (const std::string& projection : projections) {
        expected.push_back(projection + ".lora_a.weight");
        expected.push_back(projection + ".lora_b.weight");
    }

    const ParameterList adapters = model.lora_parameters();
    require(
        adapters.size() == expected.size(),
        "all-target registration tensor count"
    );
    for (std::size_t index = 0;
         index < expected.size();
         ++index) {
        require(
            adapters[index].name == expected[index],
            "all-target registration order at index " +
                std::to_string(index)
        );
    }
}

bool is_merged_weight_name(const std::string& name) {
    return
        name.ends_with("attention.query.weight") ||
        name.ends_with("attention.value.weight");
}

void test_adapter_only_optimization_merge_and_transfer() {
    std::mt19937 random(443U);
    DecoderOnlyTransformer model(kDimensions, random);
    model.attach_lora(LoraConfig{
        2,
        4.0F,
        29U,
        kLoraDefaultTargets,
    });
    const auto base_before_training = snapshot(model.parameters());
    ParameterList adapters = model.lora_parameters();
    const auto adapters_before_training = snapshot(adapters);

    Adam optimizer(
        adapters,
        AdamOptions{
            5.0e-2F,
            0.9F,
            0.999F,
            1.0e-8F,
            10.0F,
        }
    );
    const std::vector<TokenId> tokens{0, 1, 2, 3};
    const std::vector<TokenId> targets{1, 2, 3, 4};
    const Variable loss = cross_entropy(
        model.forward(tokens, {1, 4}),
        targets
    );
    loss.backward();
    const auto step_stats = optimizer.step();
    require(
        step_stats.step == 1 &&
            std::isfinite(step_stats.gradient_norm),
        "adapter-only Adam completes one finite step"
    );
    require_registration_and_values_unchanged(
        model.parameters(),
        base_before_training,
        "adapter-only Adam leaves every base parameter exact"
    );
    bool adapter_changed = false;
    for (std::size_t index = 0;
         index < adapters.size();
         ++index) {
        const Tensor& current = adapters[index].parameter->value();
        const Tensor& before = adapters_before_training[index].value;
        for (std::size_t value_index = 0;
             value_index < current.numel();
             ++value_index) {
            adapter_changed =
                adapter_changed ||
                current.flat(value_index) != before.flat(value_index);
        }
    }
    require(adapter_changed, "Adam updates at least one adapter value");

    const Tensor adapted_logits =
        model.forward(tokens, {1, 4}).value();
    const auto base_before_merge = snapshot(model.parameters());
    std::vector<Parameter*> stale_adapter_addresses;
    stale_adapter_addresses.reserve(adapters.size());
    for (const auto& adapter : adapters) {
        stale_adapter_addresses.push_back(adapter.parameter);
    }

    model.merge_lora();
    require(!model.has_lora(), "model merge deactivates LoRA");
    require(
        model.lora_parameters().empty(),
        "model merge hides adapter parameters"
    );
    require_throws(
        [&] { static_cast<void>(model.lora_config()); },
        "merged model has no active LoRA configuration"
    );
    require_tensor_close(
        model.forward(tokens, {1, 4}).value(),
        adapted_logits,
        "model merge preserves adapted logits",
        2.0e-5F
    );
    optimizer.zero_gradients();
    for (Parameter* stale : stale_adapter_addresses) {
        require(
            stale != nullptr && stale->value().numel() > 0,
            "optimizer adapter pointer remains valid after merge"
        );
    }

    const ParameterList merged_base = model.parameters();
    require(
        merged_base.size() == base_before_merge.size(),
        "merge preserves base registration count"
    );
    bool merged_weight_changed = false;
    for (std::size_t index = 0;
         index < merged_base.size();
         ++index) {
        require(
            merged_base[index].name == base_before_merge[index].name &&
                merged_base[index].parameter ==
                    base_before_merge[index].address,
            "merge preserves base names and addresses"
        );
        if (is_merged_weight_name(merged_base[index].name)) {
            for (std::size_t value_index = 0;
                 value_index <
                     merged_base[index].parameter->value().numel();
                 ++value_index) {
                merged_weight_changed =
                    merged_weight_changed ||
                    merged_base[index].parameter->value().flat(
                        value_index
                    ) !=
                        base_before_merge[index].value.flat(value_index);
            }
        } else {
            require_tensor_close(
                merged_base[index].parameter->value(),
                base_before_merge[index].value,
                "merge leaves nontarget base exact: " +
                    merged_base[index].name,
                0.0F
            );
        }
    }
    require(
        merged_weight_changed,
        "merge changes at least one targeted base weight"
    );
    require_throws(
        [&] { model.merge_lora(); },
        "model rejects double merge"
    );
    require_throws(
        [&] { model.attach_lora(LoraConfig{}); },
        "model rejects reattachment after merge"
    );

    if (transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        transformer_lab::Module& module = model;
        module.to(ExecutionBackend::Metal);
        for (const auto& parameter : model.parameters()) {
            require(
                parameter.parameter->value().backend() ==
                    ExecutionBackend::Metal,
                "post-merge transfer moves a base parameter"
            );
        }
        for (Parameter* stale : stale_adapter_addresses) {
            require(
                stale->value().backend() == ExecutionBackend::Metal,
                "post-merge transfer moves retained adapter storage"
            );
        }
        const Tensor metal_logits = model.forward(
            tokens,
            {1, 4}
        ).value();
        require_tensor_close(
            metal_logits,
            adapted_logits,
            "merged model has Metal parity",
            1.0e-3F
        );
    }
}

void test_model_merge_is_transactional() {
    TransformerDimensions dimensions = kDimensions;
    dimensions.block_count = 1;
    std::mt19937 random(461U);
    DecoderOnlyTransformer model(dimensions, random);
    model.attach_lora(LoraConfig{
        2,
        4.0F,
        31U,
        kLoraDefaultTargets,
    });
    const auto base_before = snapshot(model.parameters());
    Parameter* late_adapter = parameter_named(
        model.lora_parameters(),
        "blocks.0.attention.value.lora_b.weight"
    );
    Tensor invalid_value = late_adapter->value();
    invalid_value.flat(0) =
        std::numeric_limits<float>::infinity();
    late_adapter->set_value(std::move(invalid_value));

    require_throws(
        [&] { model.merge_lora(); },
        "nonfinite late adapter rejects the whole merge"
    );
    require(model.has_lora(), "failed merge leaves LoRA active");
    require_registration_and_values_unchanged(
        model.parameters(),
        base_before,
        "failed merge commits no base weights"
    );

    late_adapter->set_value(
        Tensor::zeros(
            late_adapter->value().shape(),
            late_adapter->value().backend()
        )
    );
    model.merge_lora();
    require(!model.has_lora(), "model can merge after repair");
}

void test_active_metal_lora_training_when_available() {
    if (!transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    TransformerDimensions dimensions = kDimensions;
    dimensions.block_count = 1;
    std::mt19937 random(467U);
    DecoderOnlyTransformer model(dimensions, random);
    model.to(ExecutionBackend::Metal);
    model.attach_lora(LoraConfig{
        2,
        4.0F,
        37U,
        kLoraDefaultTargets,
    });

    const ParameterList adapters = model.lora_parameters();
    require(!adapters.empty(), "Metal LoRA registers adapters");
    for (const NamedParameter& adapter : adapters) {
        require(
            adapter.parameter != nullptr &&
                adapter.parameter->value().backend() ==
                    ExecutionBackend::Metal &&
                adapter.parameter->gradient().backend() ==
                    ExecutionBackend::Metal,
            "active LoRA values and gradients use Metal"
        );
    }

    const auto base_before = snapshot(model.parameters());
    const auto adapters_before = snapshot(adapters);
    const std::vector<TokenId> tokens{0, 1, 2, 3};
    const std::vector<TokenId> targets{1, 2, 3, 4};
    {
        Adam optimizer(
            adapters,
            AdamOptions{
                1.0e-2F,
                0.9F,
                0.999F,
                1.0e-8F,
                1.0F,
            }
        );
        const Variable loss = cross_entropy(
            model.forward(tokens, {1, 4}),
            targets
        );
        loss.backward();
        const auto statistics = optimizer.step();
        require(
            statistics.step == 1 &&
                std::isfinite(statistics.gradient_norm),
            "Metal LoRA completes one finite Adam step"
        );
    }

    require_registration_and_values_unchanged(
        model.parameters(),
        base_before,
        "Metal adapter-only Adam leaves base parameters exact"
    );
    bool adapter_changed = false;
    for (std::size_t index = 0;
         index < adapters.size();
         ++index) {
        const Tensor& current = adapters[index].parameter->value();
        const Tensor& before = adapters_before[index].value;
        for (std::size_t value_index = 0;
             value_index < current.numel();
             ++value_index) {
            adapter_changed =
                adapter_changed ||
                current.flat(value_index) != before.flat(value_index);
        }
    }
    require(adapter_changed, "Metal Adam updates LoRA factors");

    const Tensor adapted_logits =
        model.forward(tokens, {1, 4}).value();
    model.merge_lora();
    require_tensor_close(
        model.forward(tokens, {1, 4}).value(),
        adapted_logits,
        "Metal LoRA merge preserves adapted logits",
        1.0e-3F
    );
}

}  // namespace

int main() {
    try {
        test_default_registration_determinism_and_neutrality();
        test_validation_is_transactional();
        test_all_target_registration();
        test_adapter_only_optimization_merge_and_transfer();
        test_model_merge_is_transactional();
        test_active_metal_lora_training_when_available();
        std::cout << "LoRA model tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LoRA model test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
