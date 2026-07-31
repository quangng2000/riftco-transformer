#pragma once

#include "riftco_transformer/artifacts/state.hpp"
#include "riftco_transformer/stages/pretraining/config.hpp"
#include "riftco_transformer/training/causal_language_model_trainer.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace riftco_transformer {
class DecoderOnlyTransformer;
class TokenizerStrategy;
}

namespace riftco_transformer::stages::pretraining {

struct PretrainingResult {
    artifacts::ModelSnapshot snapshot;
    std::vector<training::TrainingStepMetrics> metrics;
    std::size_t training_token_count;
    float first_batch_loss_before_training;
    float first_batch_loss_after_training;
};

using MetricSink =
    std::function<void(const training::TrainingStepMetrics&)>;

// Stage-1 composition root. It owns the tokenizer, model, Adam adapter,
// trainer, and random-window source for one fail-stop run attempt.
class PretrainingStack final {
public:
    PretrainingStack(
        std::string corpus,
        PretrainingConfig config = {}
    );
    ~PretrainingStack();

    PretrainingStack(const PretrainingStack&) = delete;
    PretrainingStack& operator=(const PretrainingStack&) = delete;
    PretrainingStack(PretrainingStack&&) = delete;
    PretrainingStack& operator=(PretrainingStack&&) = delete;

    [[nodiscard]] const PretrainingConfig& config() const noexcept;
    [[nodiscard]] std::size_t training_token_count() const noexcept;
    [[nodiscard]] DecoderOnlyTransformer& model() noexcept;
    [[nodiscard]] const DecoderOnlyTransformer& model() const noexcept;
    [[nodiscard]] TokenizerStrategy& tokenizer() noexcept;
    [[nodiscard]] const TokenizerStrategy& tokenizer() const noexcept;

    // A stack represents one stage attempt and therefore runs once. It is
    // fail-stop: an exception after an optimizer update does not roll that
    // update back and the stack cannot be retried.
    [[nodiscard]] PretrainingResult run(
        const MetricSink& metric_sink = {}
    );

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace riftco_transformer::stages::pretraining
