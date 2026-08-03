#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/model/lora.hpp"
#include "riftco_transformer/nn/loss.hpp"
#include "riftco_transformer/optim/adam.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using riftco_transformer::Adam;
using riftco_transformer::AdamOptions;
using riftco_transformer::AdamStateStorageKind;
using riftco_transformer::DecoderOnlyTransformer;
using riftco_transformer::ExecutionBackend;
using riftco_transformer::LoraConfig;
using riftco_transformer::NamedParameter;
using riftco_transformer::Nf4Payload;
using riftco_transformer::PackedLinearWeightState;
using riftco_transformer::ParameterList;
using riftco_transformer::QuantizedMemoryUsage;
using riftco_transformer::Tensor;
using riftco_transformer::TokenId;
using riftco_transformer::TransformerDimensions;
using riftco_transformer::cross_entropy;
using riftco_transformer::kLoraDefaultTargets;

constexpr TransformerDimensions kDimensions{
    8,
    4,
    8,
    2,
    1,
    16,
};

const std::vector<TokenId> kTokens{0, 1, 2, 3};
const std::vector<TokenId> kTargets{1, 2, 3, 4};
const std::vector<TokenId> kHeldOutTokens{7, 6, 5, 4};
const std::vector<TokenId> kHeldOutTargets{6, 5, 4, 3};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_finite(const Tensor& tensor, const std::string& message) {
    for (std::size_t index = 0; index < tensor.numel(); ++index) {
        require(
            std::isfinite(tensor.flat(index)),
            message + " at flat index " + std::to_string(index)
        );
    }
}

std::vector<Tensor> snapshot_values(const ParameterList& parameters) {
    std::vector<Tensor> result;
    result.reserve(parameters.size());
    for (const NamedParameter& named : parameters) {
        require(named.parameter != nullptr, "parameter snapshot is null");
        result.push_back(named.parameter->value());
    }
    return result;
}

void require_values_exact(
    const ParameterList& parameters,
    const std::vector<Tensor>& expected,
    const std::string& message
) {
    require(parameters.size() == expected.size(), message + ": count");
    for (std::size_t parameter = 0;
         parameter < parameters.size();
         ++parameter) {
        const Tensor& actual = parameters[parameter].parameter->value();
        require(
            actual.shape() == expected[parameter].shape(),
            message + ": shape for " + parameters[parameter].name
        );
        for (std::size_t index = 0; index < actual.numel(); ++index) {
            require(
                actual.flat(index) == expected[parameter].flat(index),
                message + ": value for " + parameters[parameter].name
            );
        }
    }
}

void require_any_value_changed(
    const ParameterList& parameters,
    const std::vector<Tensor>& before,
    const std::string& message
) {
    require(parameters.size() == before.size(), message + ": count");
    for (std::size_t parameter = 0;
         parameter < parameters.size();
         ++parameter) {
        const Tensor& actual = parameters[parameter].parameter->value();
        require(
            actual.shape() == before[parameter].shape(),
            message + ": shape"
        );
        for (std::size_t index = 0; index < actual.numel(); ++index) {
            if (actual.flat(index) != before[parameter].flat(index)) {
                return;
            }
        }
    }
    throw std::runtime_error(message);
}

bool payloads_equal(const Nf4Payload& left, const Nf4Payload& right) {
    if (left.packed_codes != right.packed_codes ||
        left.block_scales != right.block_scales ||
        left.double_quantized_scales.has_value() !=
            right.double_quantized_scales.has_value()) {
        return false;
    }
    if (!left.double_quantized_scales.has_value()) {
        return true;
    }
    const auto& left_scales = *left.double_quantized_scales;
    const auto& right_scales = *right.double_quantized_scales;
    return left_scales.scale_codes == right_scales.scale_codes &&
        left_scales.second_level_scales ==
            right_scales.second_level_scales &&
        left_scales.scale_block_size == right_scales.scale_block_size &&
        left_scales.offset == right_scales.offset;
}

void require_packed_state_exact(
    const std::vector<PackedLinearWeightState>& actual,
    const std::vector<PackedLinearWeightState>& expected,
    const std::string& message
) {
    require(actual.size() == expected.size(), message + ": count");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(
            actual[index].shape == expected[index].shape &&
                actual[index].block_size == expected[index].block_size &&
                payloads_equal(actual[index].payload, expected[index].payload),
            message + " at packed weight " + std::to_string(index)
        );
    }
}

void require_tpu_tensor(const Tensor& tensor, const std::string& message) {
    require(
        tensor.backend() == ExecutionBackend::Tpu,
        message + " must retain TPU storage"
    );
    require_finite(tensor, message);
}

void test_full_training_and_evaluation() {
    std::mt19937 random(701U);
    DecoderOnlyTransformer model(kDimensions, random);
    model.to(ExecutionBackend::Tpu);
    ParameterList parameters = model.parameters();
    const std::vector<Tensor> parameters_before = snapshot_values(parameters);
    Adam optimizer(parameters);

    const auto logits = model.forward(kTokens, {1, kTokens.size()});
    require_tpu_tensor(logits.value(), "full-training logits");
    const auto loss = cross_entropy(logits, kTargets);
    require_tpu_tensor(loss.value(), "full-training loss");
    loss.backward();
    const auto statistics = optimizer.step();
    require(
        statistics.step == 1 && statistics.gradient_norm > 0.0 &&
            std::isfinite(statistics.gradient_norm) &&
            std::isfinite(statistics.clip_scale),
        "full-training Adam step must be finite"
    );
    require_any_value_changed(
        parameters,
        parameters_before,
        "full-training Adam step changed no parameter"
    );

    const auto evaluation_logits = model.forward(
        kHeldOutTokens,
        {1, kHeldOutTokens.size()}
    );
    require_tpu_tensor(
        evaluation_logits.value(),
        "held-out evaluation logits"
    );
    const Tensor evaluation_loss =
        cross_entropy(evaluation_logits, kHeldOutTargets).value();
    require_tpu_tensor(evaluation_loss, "held-out evaluation loss");
}

void test_lora_training_keeps_base_frozen() {
    std::mt19937 random(709U);
    DecoderOnlyTransformer model(kDimensions, random);
    model.to(ExecutionBackend::Tpu);
    model.attach_lora(LoraConfig{
        2,
        4.0F,
        719U,
        kLoraDefaultTargets,
    });

    ParameterList base = model.parameters();
    const std::vector<Tensor> base_before = snapshot_values(base);
    ParameterList adapters = model.lora_parameters();
    require(!adapters.empty(), "LoRA must expose adapter parameters");
    const std::vector<Tensor> adapters_before = snapshot_values(adapters);
    Adam optimizer(adapters);

    const auto logits = model.forward(kTokens, {1, kTokens.size()});
    require_tpu_tensor(logits.value(), "LoRA logits");
    const auto loss = cross_entropy(logits, kTargets);
    loss.backward();
    const auto statistics = optimizer.step();
    require(
        statistics.step == 1 && statistics.gradient_norm > 0.0 &&
            std::isfinite(statistics.gradient_norm),
        "LoRA Adam step must be finite"
    );
    require_values_exact(
        base,
        base_before,
        "adapter-only optimization changed a frozen base weight"
    );
    require_any_value_changed(
        adapters,
        adapters_before,
        "LoRA Adam step changed no adapter parameter"
    );
}

void test_packed_qlora_training_stays_packed() {
    std::mt19937 random(727U);
    DecoderOnlyTransformer model(kDimensions, random);
    model.quantize_linear_weights_nf4_double_quantized(32, 32);
    model.to(ExecutionBackend::Tpu);
    model.attach_lora(LoraConfig{
        2,
        4.0F,
        733U,
        kLoraDefaultTargets,
    });

    const QuantizedMemoryUsage packed_before =
        model.quantized_memory_usage();
    const std::vector<PackedLinearWeightState> packed_state_before =
        model.packed_linear_weight_state();
    require(
        model.has_quantized_linear_weights() &&
            packed_before.packed_code_bytes > 0 &&
            packed_before.fp32_scale_bytes == 0 &&
            packed_before.logical_payload_bytes <
                packed_before.fp32_equivalent_bytes,
        "QLoRA base must be resident as packed double-quantized NF4"
    );

    ParameterList adapters = model.lora_parameters();
    const std::vector<Tensor> adapters_before = snapshot_values(adapters);
    AdamOptions options;
    options.state_storage = AdamStateStorageKind::Paged;
    options.page_size = 16;
    Adam optimizer(adapters, options);

    const auto logits = model.forward(kTokens, {1, kTokens.size()});
    require_tpu_tensor(logits.value(), "QLoRA logits");
    const auto loss = cross_entropy(logits, kTargets);
    loss.backward();
    const auto statistics = optimizer.step();
    require(
        statistics.step == 1 && statistics.gradient_norm > 0.0 &&
            std::isfinite(statistics.gradient_norm),
        "QLoRA paged Adam step must be finite"
    );

    const QuantizedMemoryUsage packed_after =
        model.quantized_memory_usage();
    require(
        model.has_quantized_linear_weights() &&
            packed_after.packed_code_bytes ==
                packed_before.packed_code_bytes &&
            packed_after.scale_bytes == packed_before.scale_bytes &&
            packed_after.logical_payload_bytes ==
                packed_before.logical_payload_bytes &&
            packed_after.resident_payload_bytes ==
                packed_before.resident_payload_bytes &&
            packed_after.fp32_scale_bytes == 0,
        "QLoRA training must not materialize a persistent FP32 base"
    );
    require_packed_state_exact(
        model.packed_linear_weight_state(),
        packed_state_before,
        "QLoRA training changed an immutable packed base weight"
    );
    require_any_value_changed(
        adapters,
        adapters_before,
        "QLoRA paged Adam step changed no adapter parameter"
    );
}

}  // namespace

int main() {
    try {
        require(
            riftco_transformer::execution_backend_available(
                ExecutionBackend::Tpu
            ),
            "TPU acceptance requires an addressable PJRT TPU device"
        );
        test_full_training_and_evaluation();
        test_lora_training_keeps_base_frozen();
        test_packed_qlora_training_stays_packed();
        std::cout << "TPU training acceptance tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TPU acceptance failure: " << error.what() << '\n';
        return 1;
    }
}
