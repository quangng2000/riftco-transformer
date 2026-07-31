#include "riftco_transformer/training/causal_language_model_trainer.hpp"

#include "riftco_transformer/nn/loss.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace riftco_transformer::training {
namespace {

ExecutionBackend model_backend(
    DecoderOnlyTransformer& model
) {
    const ParameterList parameters = model.parameters();
    if (parameters.empty()) {
        throw std::invalid_argument(
            "causal language-model trainer requires model parameters"
        );
    }
    if (parameters.front().parameter == nullptr) {
        throw std::invalid_argument(
            "causal language-model trainer found a null model parameter"
        );
    }
    const ExecutionBackend backend =
        parameters.front().parameter->value().backend();
    for (const auto& named_parameter : parameters) {
        if (named_parameter.parameter == nullptr) {
            throw std::invalid_argument(
                "causal language-model trainer found a null model parameter"
            );
        }
        if (named_parameter.parameter->value().backend() != backend ||
            named_parameter.parameter->gradient().backend() != backend) {
            throw std::invalid_argument(
                "causal language-model parameters use mixed backends"
            );
        }
    }
    return backend;
}

float checked_loss_value(
    const Variable& loss,
    const char* operation
) {
    if (loss.value().rank() != 0 || loss.value().numel() != 1) {
        throw std::logic_error(
            "causal language-model loss must be scalar"
        );
    }
    const float value = loss.value().flat(0);
    if (!std::isfinite(value)) {
        throw std::domain_error(
            std::string(operation) + " loss must be finite"
        );
    }
    return value;
}

void validate_optimizer_metrics(
    const OptimizerStepMetrics& metrics,
    std::size_t previous_step,
    std::size_t reported_step
) {
    if (metrics.step != previous_step + 1 ||
        reported_step != metrics.step) {
        throw std::logic_error(
            "optimizer step count is inconsistent with its metrics"
        );
    }
    if (!std::isfinite(metrics.gradient_norm) ||
        metrics.gradient_norm < 0.0) {
        throw std::domain_error(
            "optimizer gradient norm must be finite and nonnegative"
        );
    }
    if (!std::isfinite(metrics.clip_scale) ||
        metrics.clip_scale <= 0.0 ||
        metrics.clip_scale > 1.0) {
        throw std::domain_error(
            "optimizer clip scale must be finite and in (0, 1]"
        );
    }
}

}  // namespace

CausalLanguageModelTrainer::CausalLanguageModelTrainer(
    DecoderOnlyTransformer& model,
    OptimizerStrategy& optimizer
)
    : model_(model),
      optimizer_(optimizer) {
    require_backend_alignment();
}

TrainingStepMetrics CausalLanguageModelTrainer::train_step(
    const TokenBatch& batch
) {
    require_backend_alignment();
    validate_batch(batch);
    const std::size_t previous_step = optimizer_.step_count();
    if (previous_step == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(
            "optimizer step count cannot be incremented"
        );
    }

    const Variable logits = model_.forward(
        batch.inputs(),
        {
            batch.batch_size(),
            batch.context_size(),
        }
    );
    const Variable loss = cross_entropy(logits, batch.targets());
    const float loss_value = checked_loss_value(loss, "training");
    loss.backward();
    const OptimizerStepMetrics optimizer_metrics = optimizer_.step();
    validate_optimizer_metrics(
        optimizer_metrics,
        previous_step,
        optimizer_.step_count()
    );
    return {
        optimizer_metrics.step,
        loss_value,
        optimizer_metrics.gradient_norm,
        optimizer_metrics.clip_scale,
    };
}

float CausalLanguageModelTrainer::evaluate_loss(
    const TokenBatch& batch
) const {
    require_backend_alignment();
    validate_batch(batch);
    const Variable logits = model_.forward(
        batch.inputs(),
        {
            batch.batch_size(),
            batch.context_size(),
        }
    );
    const Variable loss = cross_entropy(logits, batch.targets());
    return checked_loss_value(loss, "evaluation");
}

DecoderOnlyTransformer&
CausalLanguageModelTrainer::model() noexcept {
    return model_;
}

const DecoderOnlyTransformer&
CausalLanguageModelTrainer::model() const noexcept {
    return model_;
}

OptimizerStrategy&
CausalLanguageModelTrainer::optimizer() noexcept {
    return optimizer_;
}

const OptimizerStrategy&
CausalLanguageModelTrainer::optimizer() const noexcept {
    return optimizer_;
}

void CausalLanguageModelTrainer::require_backend_alignment() const {
    if (model_backend(model_) != optimizer_.backend()) {
        throw std::invalid_argument(
            "model and optimizer backends must match"
        );
    }
}

void CausalLanguageModelTrainer::validate_batch(
    const TokenBatch& batch
) const {
    const TransformerDimensions& dimensions = model_.dimensions();
    if (batch.context_size() > dimensions.maximum_context) {
        throw std::invalid_argument(
            "training batch context exceeds the model maximum context"
        );
    }
    for (const TokenId token : batch.inputs()) {
        if (static_cast<std::size_t>(token) >=
            dimensions.vocabulary_size) {
            throw std::out_of_range(
                "training batch input token is outside the vocabulary"
            );
        }
    }
    for (const TokenId token : batch.targets()) {
        if (static_cast<std::size_t>(token) >=
            dimensions.vocabulary_size) {
            throw std::out_of_range(
                "training batch target token is outside the vocabulary"
            );
        }
    }
}

}  // namespace riftco_transformer::training
