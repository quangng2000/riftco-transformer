#pragma once

#include "riftco_transformer/artifacts/state.hpp"
#include "riftco_transformer/stages/post_training/config.hpp"
#include "riftco_transformer/stages/post_training/instruction.hpp"
#include "riftco_transformer/training/causal_language_model_trainer.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace riftco_transformer {
class DecoderOnlyTransformer;
class TokenizerStrategy;
}

namespace riftco_transformer::stages::post_training {

struct PostTrainingResult {
    artifacts::ModelSnapshot snapshot;
    std::vector<training::TrainingStepMetrics> metrics;
    std::size_t example_count;
    FineTuningMethod fine_tuning_method = FineTuningMethod::Full;
};

using MetricSink =
    std::function<void(const training::TrainingStepMetrics&)>;

// Stage-2 composition root. It restores a detached copy of the base snapshot,
// formats supervised records, and owns a fresh optimizer for this stage.
class PostTrainingStack final {
public:
    PostTrainingStack(
        const artifacts::ModelSnapshot& base_snapshot,
        std::vector<InstructionExample> examples,
        const InstructionFormatter& formatter,
        PostTrainingConfig config = {}
    );
    ~PostTrainingStack();

    PostTrainingStack(const PostTrainingStack&) = delete;
    PostTrainingStack& operator=(const PostTrainingStack&) = delete;
    PostTrainingStack(PostTrainingStack&&) = delete;
    PostTrainingStack& operator=(PostTrainingStack&&) = delete;

    [[nodiscard]] const PostTrainingConfig& config() const noexcept;
    [[nodiscard]] std::size_t example_count() const noexcept;
    [[nodiscard]] DecoderOnlyTransformer& model() noexcept;
    [[nodiscard]] const DecoderOnlyTransformer& model() const noexcept;
    [[nodiscard]] TokenizerStrategy& tokenizer() noexcept;
    [[nodiscard]] const TokenizerStrategy& tokenizer() const noexcept;

    // One fail-stop attempt. Completed updates are not rolled back if a later
    // metric callback or snapshot capture throws.
    [[nodiscard]] PostTrainingResult run(
        const MetricSink& metric_sink = {}
    );

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace riftco_transformer::stages::post_training
