#include "riftco_transformer/stages/post_training/config.hpp"

#include "riftco_transformer/data/tokenizer.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace riftco_transformer::stages::post_training {
namespace {

void validate_backend(ExecutionBackend backend) {
    switch (backend) {
        case ExecutionBackend::Cpu:
        case ExecutionBackend::Metal:
        case ExecutionBackend::Cuda:
        case ExecutionBackend::Tpu:
            break;
        default:
            throw std::invalid_argument(
                "post-training backend is not recognized"
            );
    }
    if (!execution_backend_available(backend)) {
        throw std::invalid_argument(
            "post-training backend is unavailable"
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
        "post-training full-sequence attention kind is not recognized"
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
        "post-training activation checkpointing kind is not recognized"
    );
}

void validate_fine_tuning(
    FineTuningMethod method,
    const LoraConfig& lora
) {
    switch (method) {
        case FineTuningMethod::Full:
            return;
        case FineTuningMethod::Lora:
            break;
        default:
            throw std::invalid_argument(
                "post-training fine-tuning method is not recognized"
            );
    }
    if (lora.rank == 0) {
        throw std::invalid_argument(
            "post-training LoRA rank must be greater than zero"
        );
    }
    if (!std::isfinite(lora.alpha) || lora.alpha <= 0.0F) {
        throw std::invalid_argument(
            "post-training LoRA alpha must be finite and positive"
        );
    }
    if (lora.targets == 0 ||
        (lora.targets & ~kLoraAllTargets) != 0) {
        throw std::invalid_argument(
            "post-training LoRA targets must be nonempty and recognized"
        );
    }
}

}  // namespace

void PostTrainingConfig::validate() const {
    if (steps == 0 || context_size == 0 || batch_size == 0) {
        throw std::invalid_argument(
            "post-training steps, context size, and batch size "
            "must be greater than zero"
        );
    }
    if (batch_size >
        std::numeric_limits<std::size_t>::max() / context_size) {
        throw std::overflow_error(
            "post-training batch and context sizes overflow"
        );
    }
    if (context_size - 1 >
        static_cast<std::size_t>(
            std::numeric_limits<TokenId>::max()
        )) {
        throw std::invalid_argument(
            "post-training context size exceeds the TokenId capacity"
        );
    }
    validate_backend(backend);
    validate_attention(attention);
    validate_activation_checkpointing(activation_checkpointing);
    validate_fine_tuning(fine_tuning_method, lora);
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
            "post-training Adam options are invalid"
        );
    }
}

}  // namespace riftco_transformer::stages::post_training
