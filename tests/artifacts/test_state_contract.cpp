#include "transformer_lab/artifacts/state.hpp"

#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/nn/parameter.hpp"

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

using transformer_lab::BpeMergeRule;
using transformer_lab::BytePairTokenizer;
using transformer_lab::ByteTokenizer;
using transformer_lab::DecoderOnlyTransformer;
using transformer_lab::ExecutionBackend;
using transformer_lab::LoraConfig;
using transformer_lab::Tensor;
using transformer_lab::TokenId;
using transformer_lab::TokenizerMethod;
using transformer_lab::TransformerDimensions;
using transformer_lab::artifacts::ModelState;
using transformer_lab::artifacts::TokenizerState;
using transformer_lab::artifacts::capture_model_state;
using transformer_lab::artifacts::capture_snapshot;
using transformer_lab::artifacts::capture_tokenizer_state;
using transformer_lab::artifacts::load_model_state;
using transformer_lab::artifacts::restore_tokenizer;

constexpr TransformerDimensions kDimensions{
    258,
    4,
    4,
    2,
    1,
    8,
};
constexpr float kLayerNormEpsilon = 2.0e-5F;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid_argument(
    Function&& function,
    const std::string& message
) {
    try {
        std::forward<Function>(function)();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            message + ": wrong exception: " + error.what()
        );
    }
    throw std::runtime_error(message + ": no exception");
}

template <typename Function>
void require_logic_error(
    Function&& function,
    const std::string& message
) {
    try {
        std::forward<Function>(function)();
    } catch (const std::logic_error&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            message + ": wrong exception: " + error.what()
        );
    }
    throw std::runtime_error(message + ": no exception");
}

bool same_dimensions(
    const TransformerDimensions& left,
    const TransformerDimensions& right
) {
    return
        left.vocabulary_size == right.vocabulary_size &&
        left.maximum_context == right.maximum_context &&
        left.model_width == right.model_width &&
        left.head_count == right.head_count &&
        left.block_count == right.block_count &&
        left.feed_forward_width == right.feed_forward_width;
}

void require_model_state_equal(
    const ModelState& actual,
    const ModelState& expected,
    const std::string& message
) {
    require(
        same_dimensions(actual.dimensions, expected.dimensions),
        message + ": dimensions"
    );
    require(
        actual.layer_norm_epsilon == expected.layer_norm_epsilon,
        message + ": layer normalization epsilon"
    );
    require(
        actual.parameters.size() == expected.parameters.size(),
        message + ": parameter count"
    );
    for (std::size_t index = 0;
         index < expected.parameters.size();
         ++index) {
        const auto& actual_parameter = actual.parameters[index];
        const auto& expected_parameter = expected.parameters[index];
        require(
            actual_parameter.name == expected_parameter.name,
            message + ": parameter name " + std::to_string(index)
        );
        require(
            actual_parameter.shape == expected_parameter.shape,
            message + ": parameter shape " + std::to_string(index)
        );
        require(
            actual_parameter.values == expected_parameter.values,
            message + ": parameter values " + std::to_string(index)
        );
    }
}

void require_parameter_backend(
    DecoderOnlyTransformer& model,
    ExecutionBackend backend,
    const std::string& message
) {
    const auto parameters = model.parameters();
    require(!parameters.empty(), message + ": no parameters");
    for (const auto& named_parameter : parameters) {
        require(
            named_parameter.parameter != nullptr,
            message + ": null parameter"
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

void require_tensor_close(
    const Tensor& actual,
    const Tensor& expected,
    const std::string& message
) {
    require(actual.shape() == expected.shape(), message + ": shape");
    for (std::size_t index = 0;
         index < expected.numel();
         ++index) {
        const float difference = std::fabs(
            actual.flat(index) - expected.flat(index)
        );
        require(
            std::isfinite(difference) && difference <= 1.0e-6F,
            message + ": value " + std::to_string(index)
        );
    }
}

void test_model_and_snapshot_round_trip() {
    std::mt19937 source_random(101U);
    std::mt19937 target_random(103U);
    DecoderOnlyTransformer source(
        kDimensions,
        source_random,
        kLayerNormEpsilon
    );
    DecoderOnlyTransformer target(
        kDimensions,
        target_random,
        kLayerNormEpsilon
    );
    const BytePairTokenizer tokenizer("abababab", 258, 2);

    const auto snapshot = capture_snapshot(source, tokenizer);
    require(
        snapshot.tokenizer.method == TokenizerMethod::BytePair,
        "snapshot tokenizer method"
    );
    require(
        snapshot.model.layer_norm_epsilon == kLayerNormEpsilon,
        "snapshot layer normalization epsilon"
    );

    load_model_state(target, snapshot.model);
    require_model_state_equal(
        capture_model_state(target),
        snapshot.model,
        "loaded CPU model state"
    );
    require_parameter_backend(
        target,
        ExecutionBackend::Cpu,
        "CPU backend preservation"
    );

    const std::vector<TokenId> tokens{97, 98, 256, 257};
    const Tensor source_logits =
        source.forward(tokens, {1, 4}).value();
    const Tensor target_logits =
        target.forward(tokens, {1, 4}).value();
    require_tensor_close(
        target_logits,
        source_logits,
        "loaded forward parity"
    );

    if (transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        target.to(ExecutionBackend::Metal);
        load_model_state(target, snapshot.model);
        require_parameter_backend(
            target,
            ExecutionBackend::Metal,
            "Metal backend preservation"
        );
        require_model_state_equal(
            capture_model_state(target),
            snapshot.model,
            "loaded Metal model state"
        );
    }
}

void test_active_lora_requires_merge_before_state_handoff() {
    std::mt19937 random(105U);
    DecoderOnlyTransformer model(
        kDimensions,
        random,
        kLayerNormEpsilon
    );
    const BytePairTokenizer tokenizer("abababab", 258, 2);
    const ModelState ordinary = capture_model_state(model);

    model.attach_lora(LoraConfig{
        2,
        4.0F,
        107U,
        transformer_lab::kLoraDefaultTargets,
    });
    require(model.has_lora(), "LoRA should be active before merge");
    require_logic_error(
        [&] {
            static_cast<void>(capture_model_state(model));
        },
        "active LoRA model-state capture"
    );
    require_logic_error(
        [&] {
            static_cast<void>(capture_snapshot(model, tokenizer));
        },
        "active LoRA snapshot capture"
    );
    require_logic_error(
        [&] {
            load_model_state(model, ordinary);
        },
        "ordinary state load into active LoRA model"
    );

    model.merge_lora();
    require(!model.has_lora(), "LoRA should be inactive after merge");
    const auto merged = capture_snapshot(model, tokenizer);
    require(
        merged.model.parameters.size() ==
            ordinary.parameters.size(),
        "merged state should retain the ordinary parameter schema"
    );
    for (std::size_t index = 0;
         index < ordinary.parameters.size();
         ++index) {
        require(
            merged.model.parameters[index].name ==
                ordinary.parameters[index].name,
            "merged state parameter name"
        );
        require(
            merged.model.parameters[index].shape ==
                ordinary.parameters[index].shape,
            "merged state parameter shape"
        );
    }
}

void require_rejected_without_mutation(
    DecoderOnlyTransformer& target,
    const ModelState& invalid,
    const std::string& message
) {
    const ModelState before = capture_model_state(target);
    require_invalid_argument(
        [&] {
            load_model_state(target, invalid);
        },
        message
    );
    require_model_state_equal(
        capture_model_state(target),
        before,
        message + ": target mutation"
    );
}

void test_invalid_model_state_is_transactional() {
    std::mt19937 source_random(107U);
    std::mt19937 target_random(109U);
    DecoderOnlyTransformer source(
        kDimensions,
        source_random,
        kLayerNormEpsilon
    );
    DecoderOnlyTransformer target(
        kDimensions,
        target_random,
        kLayerNormEpsilon
    );
    const ModelState valid = capture_model_state(source);

    ModelState invalid = valid;
    ++invalid.dimensions.maximum_context;
    require_rejected_without_mutation(
        target,
        invalid,
        "dimension mismatch"
    );

    invalid = valid;
    invalid.layer_norm_epsilon = 1.0e-4F;
    require_rejected_without_mutation(
        target,
        invalid,
        "epsilon mismatch"
    );

    invalid = valid;
    invalid.parameters.pop_back();
    require_rejected_without_mutation(
        target,
        invalid,
        "parameter count mismatch"
    );

    invalid = valid;
    invalid.parameters.front().name += ".changed";
    require_rejected_without_mutation(
        target,
        invalid,
        "parameter name mismatch"
    );

    invalid = valid;
    ++invalid.parameters.front().shape.front();
    require_rejected_without_mutation(
        target,
        invalid,
        "parameter shape mismatch"
    );

    invalid = valid;
    invalid.parameters.front().values.pop_back();
    require_rejected_without_mutation(
        target,
        invalid,
        "parameter value-count mismatch"
    );

    invalid = valid;
    invalid.parameters.back().values.back() =
        std::numeric_limits<float>::infinity();
    require_rejected_without_mutation(
        target,
        invalid,
        "non-finite parameter value"
    );

    invalid = valid;
    invalid.layer_norm_epsilon =
        std::numeric_limits<float>::quiet_NaN();
    require_rejected_without_mutation(
        target,
        invalid,
        "non-finite epsilon"
    );
}

void test_capture_rejects_non_finite_values() {
    std::mt19937 random(113U);
    DecoderOnlyTransformer model(
        kDimensions,
        random,
        kLayerNormEpsilon
    );
    const auto parameters = model.parameters();
    require(!parameters.empty(), "capture test parameters");
    auto* parameter = parameters.back().parameter;
    require(parameter != nullptr, "capture test parameter pointer");

    const Tensor& current = parameter->value();
    std::vector<float> values(
        current.data().begin(),
        current.data().end()
    );
    values.back() = std::numeric_limits<float>::quiet_NaN();
    parameter->set_value(Tensor(
        current.shape(),
        std::move(values),
        current.backend()
    ));

    require_invalid_argument(
        [&] {
            static_cast<void>(capture_model_state(model));
        },
        "capture non-finite parameter"
    );
}

void test_byte_tokenizer_round_trip() {
    const ByteTokenizer original("cab\ncab");
    const TokenizerState state = capture_tokenizer_state(original);
    require(
        state.method == TokenizerMethod::CorpusByte,
        "byte tokenizer method"
    );
    require(state.bpe_merges.empty(), "byte state BPE merges");

    const auto restored = restore_tokenizer(state);
    const std::string text = "cab\n";
    require(
        restored->encode(text) == original.encode(text),
        "byte tokenizer IDs"
    );
    require(
        restored->decode(restored->encode(text)) == text,
        "byte tokenizer decode"
    );

    TokenizerState mixed = state;
    mixed.bpe_merges.push_back(BpeMergeRule{0, 1, 256});
    require_invalid_argument(
        [&] {
            static_cast<void>(restore_tokenizer(mixed));
        },
        "mixed byte tokenizer state"
    );
}

void test_bpe_tokenizer_round_trip() {
    const BytePairTokenizer original("abababab", 258, 2);
    const TokenizerState state = capture_tokenizer_state(original);
    require(
        state.method == TokenizerMethod::BytePair,
        "BPE tokenizer method"
    );
    require(
        state.bpe_merges ==
            std::vector<BpeMergeRule>({
                {97, 98, 256},
                {256, 256, 257},
            }),
        "BPE merge state"
    );
    require(state.byte_vocabulary.empty(), "BPE byte vocabulary");

    const auto restored = restore_tokenizer(state);
    const std::string text("abab\0z", 6);
    require(
        restored->encode(text) == original.encode(text),
        "BPE tokenizer IDs"
    );
    require(
        restored->decode(restored->encode(text)) == text,
        "BPE tokenizer decode"
    );

    TokenizerState mixed = state;
    mixed.byte_vocabulary.push_back(0);
    require_invalid_argument(
        [&] {
            static_cast<void>(restore_tokenizer(mixed));
        },
        "mixed BPE tokenizer state"
    );
}

}  // namespace

int main() {
    try {
        test_model_and_snapshot_round_trip();
        test_active_lora_requires_merge_before_state_handoff();
        test_invalid_model_state_is_transactional();
        test_capture_rejects_non_finite_values();
        test_byte_tokenizer_round_trip();
        test_bpe_tokenizer_round_trip();
    } catch (const std::exception& error) {
        std::cerr
            << "artifact state-contract test failed: "
            << error.what() << '\n';
        return 1;
    }

    std::cout << "artifact state-contract tests passed\n";
    return 0;
}
