#pragma once

#include "transformer_lab/data/token_batch.hpp"
#include "transformer_lab/model/decoder_only_transformer.hpp"
#include "transformer_lab/training/optimizer.hpp"

#include <cstddef>

namespace transformer_lab::training {

struct TrainingStepMetrics {
    std::size_t step;
    float loss;
    double gradient_norm;
    double clip_scale;
};

class CausalLanguageModelTrainer {
public:
    // Both referenced strategies must outlive the trainer.
    CausalLanguageModelTrainer(
        DecoderOnlyTransformer& model,
        OptimizerStrategy& optimizer
    );

    [[nodiscard]] TrainingStepMetrics train_step(
        const TokenBatch& batch
    );
    [[nodiscard]] float evaluate_loss(
        const TokenBatch& batch
    ) const;

    [[nodiscard]] DecoderOnlyTransformer& model() noexcept;
    [[nodiscard]] const DecoderOnlyTransformer& model() const noexcept;
    [[nodiscard]] OptimizerStrategy& optimizer() noexcept;
    [[nodiscard]] const OptimizerStrategy& optimizer() const noexcept;

private:
    DecoderOnlyTransformer& model_;
    OptimizerStrategy& optimizer_;

    void require_backend_alignment() const;
    void validate_batch(const TokenBatch& batch) const;
};

}  // namespace transformer_lab::training
