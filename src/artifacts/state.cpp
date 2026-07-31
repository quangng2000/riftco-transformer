#include "transformer_lab/artifacts/state.hpp"

#include "transformer_lab/nn/parameter.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>

namespace transformer_lab::artifacts {
namespace {

[[nodiscard]] bool same_dimensions(
    const TransformerDimensions& left,
    const TransformerDimensions& right
) noexcept {
    return
        left.vocabulary_size == right.vocabulary_size &&
        left.maximum_context == right.maximum_context &&
        left.model_width == right.model_width &&
        left.head_count == right.head_count &&
        left.block_count == right.block_count &&
        left.feed_forward_width == right.feed_forward_width;
}

[[nodiscard]] const ByteTokenizer& checked_byte_tokenizer(
    const TokenizerStrategy& tokenizer
) {
    const auto* concrete =
        dynamic_cast<const ByteTokenizer*>(&tokenizer);
    if (concrete == nullptr) {
        throw std::invalid_argument(
            "corpus-byte tokenizer state requires ByteTokenizer"
        );
    }
    return *concrete;
}

[[nodiscard]] const BytePairTokenizer& checked_bpe_tokenizer(
    const TokenizerStrategy& tokenizer
) {
    const auto* concrete =
        dynamic_cast<const BytePairTokenizer*>(&tokenizer);
    if (concrete == nullptr) {
        throw std::invalid_argument(
            "byte-pair tokenizer state requires BytePairTokenizer"
        );
    }
    return *concrete;
}

}  // namespace

ModelState capture_model_state(DecoderOnlyTransformer& model) {
    if (model.has_lora()) {
        throw std::logic_error(
            "cannot capture model state with active LoRA adapters; "
            "merge them first"
        );
    }
    ModelState result{
        model.dimensions(),
        model.layer_norm_epsilon(),
        {},
    };
    const ParameterList parameters = model.parameters();
    result.parameters.reserve(parameters.size());
    for (const auto& named_parameter : parameters) {
        if (named_parameter.parameter == nullptr) {
            throw std::logic_error(
                "model contains a null named parameter"
            );
        }
        const Tensor& value = named_parameter.parameter->value();
        if (!std::all_of(
                value.data().begin(),
                value.data().end(),
                [](float element) {
                    return std::isfinite(element);
                }
            )) {
            throw std::invalid_argument(
                "cannot capture non-finite model parameter values"
            );
        }
        result.parameters.push_back({
            named_parameter.name,
            value.shape(),
            std::vector<float>(
                value.data().begin(),
                value.data().end()
            ),
        });
    }
    return result;
}

void load_model_state(
    DecoderOnlyTransformer& model,
    const ModelState& state
) {
    if (model.has_lora()) {
        throw std::logic_error(
            "cannot load ordinary model state into a model with "
            "active LoRA adapters"
        );
    }
    if (!same_dimensions(model.dimensions(), state.dimensions)) {
        throw std::invalid_argument(
            "model state dimensions do not match the target model"
        );
    }
    if (!std::isfinite(state.layer_norm_epsilon) ||
        state.layer_norm_epsilon <= 0.0F) {
        throw std::invalid_argument(
            "model state layer normalization epsilon must be "
            "finite and positive"
        );
    }
    if (state.layer_norm_epsilon != model.layer_norm_epsilon()) {
        throw std::invalid_argument(
            "model state layer normalization epsilon does not "
            "match the target model"
        );
    }

    const ParameterList parameters = model.parameters();
    if (state.parameters.size() != parameters.size()) {
        throw std::invalid_argument(
            "model state parameter count does not match the target model"
        );
    }

    // Finish strict validation before allocating or mutating anything.
    for (std::size_t index = 0;
         index < parameters.size();
         ++index) {
        const NamedParameter& target = parameters[index];
        const ParameterState& source = state.parameters[index];
        if (target.parameter == nullptr) {
            throw std::logic_error(
                "target model contains a null named parameter"
            );
        }
        if (source.name != target.name) {
            throw std::invalid_argument(
                "model state parameter names or order do not match"
            );
        }
        const Tensor& target_value = target.parameter->value();
        if (source.shape != target_value.shape()) {
            throw std::invalid_argument(
                "model state parameter shape does not match"
            );
        }
        if (source.values.size() != target_value.numel()) {
            throw std::invalid_argument(
                "model state parameter value count does not match"
            );
        }
        if (!std::all_of(
                source.values.begin(),
                source.values.end(),
                [](float value) {
                    return std::isfinite(value);
                }
            )) {
            throw std::invalid_argument(
                "model state parameter values must all be finite"
            );
        }
    }

    struct Replacement {
        Parameter* parameter;
        Tensor value;
    };
    std::vector<Replacement> replacements;
    replacements.reserve(parameters.size());
    for (std::size_t index = 0;
         index < parameters.size();
         ++index) {
        Parameter* parameter = parameters[index].parameter;
        const Tensor& current = parameter->value();
        replacements.push_back({
            parameter,
            Tensor(
                current.shape(),
                state.parameters[index].values,
                current.backend()
            ),
        });
    }

    // Same-shape, same-backend Parameter replacement is non-allocating.
    // Therefore all fallible validation and allocation has completed.
    for (auto& replacement : replacements) {
        replacement.parameter->set_value(
            std::move(replacement.value)
        );
    }
}

TokenizerState capture_tokenizer_state(
    const TokenizerStrategy& tokenizer
) {
    switch (tokenizer.method()) {
        case TokenizerMethod::CorpusByte: {
            const auto vocabulary =
                checked_byte_tokenizer(tokenizer).vocabulary();
            return {
                TokenizerMethod::CorpusByte,
                std::vector<std::uint8_t>(
                    vocabulary.begin(),
                    vocabulary.end()
                ),
                {},
            };
        }
        case TokenizerMethod::BytePair: {
            const auto merges =
                checked_bpe_tokenizer(tokenizer).merge_rules();
            return {
                TokenizerMethod::BytePair,
                {},
                std::vector<BpeMergeRule>(
                    merges.begin(),
                    merges.end()
                ),
            };
        }
    }
    throw std::invalid_argument(
        "cannot capture an unknown tokenizer method"
    );
}

std::unique_ptr<TokenizerStrategy> restore_tokenizer(
    const TokenizerState& state
) {
    switch (state.method) {
        case TokenizerMethod::CorpusByte:
            if (!state.bpe_merges.empty()) {
                throw std::invalid_argument(
                    "corpus-byte tokenizer state must not contain "
                    "BPE merges"
                );
            }
            return std::make_unique<ByteTokenizer>(
                std::span<const std::uint8_t>(
                    state.byte_vocabulary
                )
            );
        case TokenizerMethod::BytePair:
            if (!state.byte_vocabulary.empty()) {
                throw std::invalid_argument(
                    "byte-pair tokenizer state must not contain "
                    "a byte vocabulary"
                );
            }
            return std::make_unique<BytePairTokenizer>(
                std::span<const BpeMergeRule>(state.bpe_merges)
            );
    }
    throw std::invalid_argument(
        "cannot restore an unknown tokenizer method"
    );
}

ModelSnapshot capture_snapshot(
    DecoderOnlyTransformer& model,
    const TokenizerStrategy& tokenizer
) {
    return {
        capture_model_state(model),
        capture_tokenizer_state(tokenizer),
    };
}

}  // namespace transformer_lab::artifacts
