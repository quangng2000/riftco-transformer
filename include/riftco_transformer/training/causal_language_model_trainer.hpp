#pragma once

#include "riftco_transformer/data/token_batch.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/training/optimizer.hpp"

#include <cstddef>

namespace riftco_transformer::training {

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

}  // namespace riftco_transformer::training
