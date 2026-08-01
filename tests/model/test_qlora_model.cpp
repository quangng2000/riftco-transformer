#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/model/lora.hpp"
#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/nn/loss.hpp"
#include "riftco_transformer/optim/adam.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using riftco_transformer::ActivationCheckpointingKind;
using riftco_transformer::Adam;
using riftco_transformer::AdamOptions;
using riftco_transformer::AdamStateStorageKind;
using riftco_transformer::DecoderOnlyTransformer;
using riftco_transformer::LoraConfig;
using riftco_transformer::Module;
using riftco_transformer::ParameterList;
using riftco_transformer::QuantizedMemoryUsage;
using riftco_transformer::Tensor;
using riftco_transformer::TokenId;
using riftco_transformer::TransformerDimensions;
using riftco_transformer::cross_entropy;

constexpr TransformerDimensions kDimensions{
    5,
    3,
    4,
    2,
    1,
    8,
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
    float tolerance = 5.0e-4F
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
    const std::string& message
) {
    require(actual.shape() == expected.shape(), message + ": shape");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        require_close(
            actual.flat(index),
            expected.flat(index),
            message + " at index " + std::to_string(index)
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

std::vector<std::string> names(const ParameterList& parameters) {
    std::vector<std::string> result;
    result.reserve(parameters.size());
    for (const auto& parameter : parameters) {
        result.push_back(parameter.name);
    }
    return result;
}

void test_model_qlora_training_and_fp32_export() {
    std::mt19937 random(461U);
    DecoderOnlyTransformer model(
        kDimensions,
        random,
        1.0e-5F,
        riftco_transformer::FullSequenceAttentionKind::Materialized,
        ActivationCheckpointingKind::TransformerBlock
    );

    std::vector<std::string> dense_names;
    std::size_t dense_count = 0;
    {
        const ParameterList retained = model.parameters();
        dense_names = names(retained);
        dense_count = retained.size();
        require_throws(
            [&] {
                model.quantize_linear_weights_nf4_double_quantized();
            },
            "model quantization rejects retained dense handles"
        );
    }

    model.quantize_linear_weights_nf4_double_quantized();
    require(
        model.has_quantized_linear_weights(),
        "model reports packed Linear weights"
    );
    require(
        model.quantized_linear_weight_count() == 7,
        "one-block model has six packed block projections and one head"
    );
    require(
        model.double_quantized_linear_weight_count() == 7,
        "every packed model weight should encode its first-level scales"
    );
    const QuantizedMemoryUsage packed = model.quantized_memory_usage();
    require(
        packed.packed_code_bytes == 74 &&
            packed.scale_bytes == 63 &&
            packed.logical_payload_bytes == 137 &&
            packed.resident_payload_bytes == 137 &&
            packed.fp32_equivalent_bytes == 592 &&
            packed.fp32_scale_bytes == 0 &&
            packed.scale_code_bytes == 7 &&
            packed.second_level_scale_bytes == 28 &&
            packed.scale_offset_bytes == 28,
        "model exposes exact double-quantized NF4 memory accounting"
    );
    {
        const ParameterList frozen = model.parameters();
        require(
            frozen.size() + 7 == dense_count,
            "packed weights disappear from the base Parameter tree"
        );
    }
    require_throws(
        [&] {
            model.quantize_linear_weights_nf4_double_quantized();
        },
        "model rejects a second quantization pass"
    );

    model.attach_lora(LoraConfig{
        2,
        4.0F,
        463U,
        riftco_transformer::kLoraDefaultTargets,
    });
    ParameterList adapters = model.lora_parameters();
    require(
        adapters.size() == 4,
        "default QLoRA targets expose four adapter tensors"
    );
    AdamOptions optimizer_options;
    optimizer_options.state_storage = AdamStateStorageKind::Paged;
    optimizer_options.page_size = 3;
    Adam optimizer(adapters, optimizer_options);
    require(
        optimizer.parameter_tensor_count() == adapters.size() &&
            optimizer.state_storage_kind() ==
                AdamStateStorageKind::Paged &&
            optimizer.state_page_count() == 12 &&
            optimizer.state_payload_bytes() == 256,
        "QLoRA paged Adam owns state only for floating-point adapters"
    );

    const std::vector<TokenId> tokens{0, 1};
    const std::vector<TokenId> targets{1, 2};
    {
        const auto logits = model.forward(tokens, {1, 2});
        const auto loss = cross_entropy(logits, targets);
        loss.backward();
        static_cast<void>(optimizer.step());
    }
    const QuantizedMemoryUsage after_step =
        model.quantized_memory_usage();
    require(
        after_step.packed_code_bytes == packed.packed_code_bytes &&
            after_step.scale_bytes == packed.scale_bytes &&
            after_step.logical_payload_bytes ==
                packed.logical_payload_bytes &&
            after_step.resident_payload_bytes ==
                packed.resident_payload_bytes &&
            after_step.fp32_equivalent_bytes ==
                packed.fp32_equivalent_bytes &&
            after_step.fp32_scale_bytes == 0 &&
            after_step.scale_code_bytes == packed.scale_code_bytes &&
            after_step.second_level_scale_bytes ==
                packed.second_level_scale_bytes &&
            after_step.scale_offset_bytes == packed.scale_offset_bytes,
        "QLoRA optimizer step preserves packed base allocation sizes"
    );

    const Tensor before_export =
        model.forward(tokens, {1, 2}).value();
    model.merge_lora();
    require(
        !model.has_quantized_linear_weights() &&
            model.quantized_linear_weight_count() == 0,
        "QLoRA export releases every packed runtime weight"
    );
    require(
        model.quantized_memory_usage().logical_payload_bytes == 0,
        "QLoRA export clears packed-memory diagnostics"
    );
    const ParameterList exported = model.parameters();
    require(
        exported.size() == dense_count &&
            names(exported) == dense_names,
        "QLoRA export restores the original FP32 parameter schema"
    );
    require_tensor_close(
        model.forward(tokens, {1, 2}).value(),
        before_export,
        "QLoRA materialized export preserves model output"
    );
}

void test_polymorphic_model_transfer_if_metal_available() {
    if (!riftco_transformer::execution_backend_available(
            riftco_transformer::ExecutionBackend::Metal
        )) {
        return;
    }

    std::mt19937 random(487U);
    DecoderOnlyTransformer model(kDimensions, random);
    model.quantize_linear_weights_nf4_double_quantized(32, 32);
    model.attach_lora(LoraConfig{
        2,
        4.0F,
        491U,
        riftco_transformer::kLoraDefaultTargets,
    });
    const QuantizedMemoryUsage cpu_memory =
        model.quantized_memory_usage();

    Module& module = model;
    module.to(riftco_transformer::ExecutionBackend::Metal);
    require(
        model.backend() == riftco_transformer::ExecutionBackend::Metal,
        "virtual Module transfer must move ordinary model parameters"
    );
    require(
        model.quantized_memory_usage().logical_payload_bytes ==
            cpu_memory.logical_payload_bytes,
        "virtual Module transfer must preserve the packed model payload"
    );
    require(
        model.forward(std::vector<TokenId>{0, 1}, {1, 2})
                .value()
                .backend() ==
            riftco_transformer::ExecutionBackend::Metal,
        "virtually transferred QLoRA model must execute on Metal"
    );

    module.to(riftco_transformer::ExecutionBackend::Cpu);
    require(
        model.backend() == riftco_transformer::ExecutionBackend::Cpu,
        "virtual Module transfer must return ordinary model state to CPU"
    );
    require(
        model.has_quantized_linear_weights(),
        "virtual Module transfer must retain packed model weights"
    );
}

}  // namespace

int main() {
    try {
        test_model_qlora_training_and_fp32_export();
        test_polymorphic_model_transfer_if_metal_available();
        std::cout << "QLoRA model tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "QLoRA model test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
