#include "riftco_transformer/stages/pretraining/config.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace riftco_transformer::stages::pretraining {
namespace {

void validate_backend(ExecutionBackend backend) {
    switch (backend) {
        case ExecutionBackend::Cpu:
        case ExecutionBackend::Metal:
            break;
        default:
            throw std::invalid_argument(
                "pretraining backend is not recognized"
            );
    }
    if (!execution_backend_available(backend)) {
        throw std::invalid_argument(
            "pretraining backend is unavailable"
        );
    }
}

void validate_attention(FullSequenceAttentionKind attention) {
    switch (attention) {
        case FullSequenceAttentionKind::Materialized:
        case FullSequenceAttentionKind::Flash:
            return;
    }
    throw std::invalid_argument(
        "pretraining full-sequence attention kind is not recognized"
    );
}

void validate_activation_checkpointing(
    ActivationCheckpointingKind activation_checkpointing
) {
    switch (activation_checkpointing) {
        case ActivationCheckpointingKind::Disabled:
        case ActivationCheckpointingKind::TransformerBlock:
            return;
    }
    throw std::invalid_argument(
        "pretraining activation checkpointing kind is not recognized"
    );
}

}  // namespace

void PretrainingConfig::validate() const {
    if (steps == 0 || context_size == 0 || batch_size == 0) {
        throw std::invalid_argument(
            "pretraining steps, context size, and batch size "
            "must be greater than zero"
        );
    }
    if (batch_size >
        std::numeric_limits<std::size_t>::max() / context_size) {
        throw std::overflow_error(
            "pretraining batch and context sizes overflow"
        );
    }
    if (context_size - 1 >
        static_cast<std::size_t>(
            std::numeric_limits<TokenId>::max()
        )) {
        throw std::invalid_argument(
            "pretraining context size exceeds the TokenId capacity"
        );
    }
    if (model_width == 0 || head_count == 0 || block_count == 0 ||
        feed_forward_width == 0) {
        throw std::invalid_argument(
            "pretraining model dimensions must be greater than zero"
        );
    }
    if (model_width % head_count != 0) {
        throw std::invalid_argument(
            "pretraining model width must be divisible by head count"
        );
    }
    if (!std::isfinite(layer_norm_epsilon) ||
        layer_norm_epsilon <= 0.0F) {
        throw std::invalid_argument(
            "pretraining layer normalization epsilon "
            "must be finite and positive"
        );
    }
    if (tokenizer.method != TokenizerMethod::CorpusByte &&
        tokenizer.method != TokenizerMethod::BytePair) {
        throw std::invalid_argument(
            "pretraining tokenizer method is not recognized"
        );
    }
    if (tokenizer.method == TokenizerMethod::BytePair) {
        if (tokenizer.vocabulary_size < 256 ||
            tokenizer.vocabulary_size >
                static_cast<std::size_t>(
                    std::numeric_limits<TokenId>::max()
                )) {
            throw std::invalid_argument(
                "BPE vocabulary size must be between 256 and "
                "the TokenId capacity"
            );
        }
        if (tokenizer.minimum_pair_frequency == 0) {
            throw std::invalid_argument(
                "BPE minimum pair frequency must be greater than zero"
            );
        }
    }
    validate_backend(backend);
    validate_attention(attention);
    validate_activation_checkpointing(activation_checkpointing);

    // Adam performs the same checks when constructed. Checking here keeps a
    // malformed stage configuration from partially constructing its stack.
    if (!std::isfinite(optimizer.learning_rate) ||
        optimizer.learning_rate <= 0.0F ||
        !std::isfinite(optimizer.beta1) ||
        optimizer.beta1 <= 0.0F ||
        optimizer.beta1 >= 1.0F ||
        !std::isfinite(optimizer.beta2) ||
        optimizer.beta2 <= 0.0F ||
        optimizer.beta2 >= 1.0F ||
        !std::isfinite(optimizer.epsilon) ||
        optimizer.epsilon <= 0.0F ||
        !std::isfinite(optimizer.maximum_gradient_norm) ||
        optimizer.maximum_gradient_norm <= 0.0F) {
        throw std::invalid_argument(
            "pretraining Adam options are invalid"
        );
    }
}

}  // namespace riftco_transformer::stages::pretraining
