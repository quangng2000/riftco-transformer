#include "riftco_transformer/stages/pretraining/stack.hpp"

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/data/tokenizer.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/optim/adam.hpp"
#include "riftco_transformer/training/adam_optimizer_adapter.hpp"
#include "riftco_transformer/training/batch_source.hpp"

#include <memory>
#include <random>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::stages::pretraining {

struct PretrainingStack::Implementation {
    PretrainingConfig config;
    std::unique_ptr<TokenizerStrategy> tokenizer;
    std::size_t training_token_count = 0;
    std::unique_ptr<DecoderOnlyTransformer> model;
    std::unique_ptr<Adam> adam;
    std::unique_ptr<training::AdamOptimizerAdapter> optimizer;
    std::unique_ptr<training::CausalLanguageModelTrainer> trainer;
    std::unique_ptr<training::RandomWindowBatchSource> batches;
    bool started = false;
};

PretrainingStack::PretrainingStack(
    std::string corpus,
    PretrainingConfig config
)
    : implementation_(std::make_unique<Implementation>()) {
    config.validate();
    if (corpus.empty()) {
        throw std::invalid_argument(
            "pretraining corpus must not be empty"
        );
    }
    implementation_->config = config;
    implementation_->tokenizer = make_tokenizer(
        corpus,
        config.tokenizer
    );
    std::vector<TokenId> tokens =
        implementation_->tokenizer->encode(corpus);
    if (tokens.size() <= config.context_size) {
        throw std::invalid_argument(
            "pretraining corpus must encode to at least "
            "context_size + 1 tokens"
        );
    }
    implementation_->training_token_count = tokens.size();

    const ScopedExecutionBackend selected_backend(config.backend);
    std::mt19937 model_random(config.model_seed);
    implementation_->model =
        std::make_unique<DecoderOnlyTransformer>(
            TransformerDimensions{
                implementation_->tokenizer->vocab_size(),
                config.context_size,
                config.model_width,
                config.head_count,
                config.block_count,
                config.feed_forward_width,
            },
            model_random,
            config.layer_norm_epsilon,
            config.attention,
            config.activation_checkpointing
        );
    implementation_->adam = std::make_unique<Adam>(
        implementation_->model->parameters(),
        config.optimizer
    );
    implementation_->optimizer =
        std::make_unique<training::AdamOptimizerAdapter>(
            *implementation_->adam
        );
    implementation_->trainer =
        std::make_unique<training::CausalLanguageModelTrainer>(
            *implementation_->model,
            *implementation_->optimizer
        );
    implementation_->batches =
        std::make_unique<training::RandomWindowBatchSource>(
            std::move(tokens),
            config.batch_size,
            config.context_size,
            config.batch_seed
        );
}

PretrainingStack::~PretrainingStack() = default;

const PretrainingConfig&
PretrainingStack::config() const noexcept {
    return implementation_->config;
}

std::size_t
PretrainingStack::training_token_count() const noexcept {
    return implementation_->training_token_count;
}

DecoderOnlyTransformer& PretrainingStack::model() noexcept {
    return *implementation_->model;
}

const DecoderOnlyTransformer&
PretrainingStack::model() const noexcept {
    return *implementation_->model;
}

TokenizerStrategy& PretrainingStack::tokenizer() noexcept {
    return *implementation_->tokenizer;
}

const TokenizerStrategy&
PretrainingStack::tokenizer() const noexcept {
    return *implementation_->tokenizer;
}

PretrainingResult PretrainingStack::run(
    const MetricSink& metric_sink
) {
    if (implementation_->started) {
        throw std::logic_error(
            "a pretraining stack can only be run once"
        );
    }
    implementation_->started = true;

    const TokenBatch first_batch =
        implementation_->batches->next_batch();
    std::vector<training::TrainingStepMetrics> metrics;
    metrics.reserve(implementation_->config.steps);
    for (std::size_t index = 0;
         index < implementation_->config.steps;
         ++index) {
        const TokenBatch batch =
            index == 0
                ? first_batch
                : implementation_->batches->next_batch();
        const auto observation =
            implementation_->trainer->train_step(batch);
        metrics.push_back(observation);
        if (metric_sink) {
            metric_sink(observation);
        }
    }

    const float first_batch_loss_after_training =
        implementation_->trainer->evaluate_loss(first_batch);
    const float first_batch_loss_before_training =
        metrics.front().loss;
    return {
        artifacts::capture_snapshot(
            *implementation_->model,
            *implementation_->tokenizer
        ),
        std::move(metrics),
        implementation_->training_token_count,
        first_batch_loss_before_training,
        first_batch_loss_after_training,
    };
}

}  // namespace riftco_transformer::stages::pretraining
