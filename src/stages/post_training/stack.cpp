#include "riftco_transformer/stages/post_training/stack.hpp"

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/data/tokenizer.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/optim/adam.hpp"
#include "riftco_transformer/training/adam_optimizer_adapter.hpp"
#include "riftco_transformer/training/batch_source.hpp"

#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace riftco_transformer::stages::post_training {

struct PostTrainingStack::Implementation {
    PostTrainingConfig config;
    std::size_t example_count = 0;
    std::unique_ptr<TokenizerStrategy> tokenizer;
    std::unique_ptr<DecoderOnlyTransformer> model;
    std::unique_ptr<Adam> adam;
    std::unique_ptr<training::AdamOptimizerAdapter> optimizer;
    std::unique_ptr<training::CausalLanguageModelTrainer> trainer;
    std::unique_ptr<training::SequenceWindowBatchSource> batches;
    bool started = false;
};

PostTrainingStack::PostTrainingStack(
    const artifacts::ModelSnapshot& base_snapshot,
    std::vector<InstructionExample> examples,
    const InstructionFormatter& formatter,
    PostTrainingConfig config
)
    : implementation_(std::make_unique<Implementation>()) {
    config.validate();
    if (examples.empty()) {
        throw std::invalid_argument(
            "post-training requires at least one instruction example"
        );
    }
    if (config.context_size >
        base_snapshot.model.dimensions.maximum_context) {
        throw std::invalid_argument(
            "post-training context exceeds the base model maximum context"
        );
    }

    implementation_->config = config;
    implementation_->example_count = examples.size();
    implementation_->tokenizer =
        artifacts::restore_tokenizer(base_snapshot.tokenizer);
    if (implementation_->tokenizer->vocab_size() !=
        base_snapshot.model.dimensions.vocabulary_size) {
        throw std::invalid_argument(
            "base model and tokenizer vocabulary sizes do not match"
        );
    }

    std::vector<std::vector<TokenId>> sequences;
    sequences.reserve(examples.size());
    for (std::size_t index = 0; index < examples.size(); ++index) {
        examples[index].validate();
        const std::string formatted = formatter.format(examples[index]);
        std::vector<TokenId> sequence =
            implementation_->tokenizer->encode(formatted);
        if (sequence.size() <= config.context_size) {
            throw std::invalid_argument(
                "post-training example " + std::to_string(index) +
                " must encode to at least context_size + 1 tokens"
            );
        }
        sequences.push_back(std::move(sequence));
    }

    const ScopedExecutionBackend selected_backend(config.backend);
    std::mt19937 restoration_random(0);
    implementation_->model =
        std::make_unique<DecoderOnlyTransformer>(
            base_snapshot.model.dimensions,
            restoration_random,
            base_snapshot.model.layer_norm_epsilon,
            config.attention,
            config.activation_checkpointing
        );
    artifacts::load_model_state(
        *implementation_->model,
        base_snapshot.model
    );
    ParameterList optimizer_parameters;
    switch (config.fine_tuning_method) {
        case FineTuningMethod::Full:
            optimizer_parameters =
                implementation_->model->parameters();
            break;
        case FineTuningMethod::Lora:
            implementation_->model->attach_lora(config.lora);
            optimizer_parameters =
                implementation_->model->lora_parameters();
            break;
    }
    implementation_->adam = std::make_unique<Adam>(
        std::move(optimizer_parameters),
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
        std::make_unique<training::SequenceWindowBatchSource>(
            std::move(sequences),
            config.batch_size,
            config.context_size,
            config.batch_seed
        );
}

PostTrainingStack::~PostTrainingStack() = default;

const PostTrainingConfig&
PostTrainingStack::config() const noexcept {
    return implementation_->config;
}

std::size_t
PostTrainingStack::example_count() const noexcept {
    return implementation_->example_count;
}

DecoderOnlyTransformer& PostTrainingStack::model() noexcept {
    return *implementation_->model;
}

const DecoderOnlyTransformer&
PostTrainingStack::model() const noexcept {
    return *implementation_->model;
}

TokenizerStrategy& PostTrainingStack::tokenizer() noexcept {
    return *implementation_->tokenizer;
}

const TokenizerStrategy&
PostTrainingStack::tokenizer() const noexcept {
    return *implementation_->tokenizer;
}

PostTrainingResult PostTrainingStack::run(
    const MetricSink& metric_sink
) {
    if (implementation_->started) {
        throw std::logic_error(
            "a post-training stack can only be run once"
        );
    }
    implementation_->started = true;

    std::vector<training::TrainingStepMetrics> metrics;
    metrics.reserve(implementation_->config.steps);
    for (std::size_t index = 0;
         index < implementation_->config.steps;
         ++index) {
        const auto observation =
            implementation_->trainer->train_step(
                implementation_->batches->next_batch()
            );
        metrics.push_back(observation);
        if (metric_sink) {
            metric_sink(observation);
        }
    }
    if (implementation_->config.fine_tuning_method ==
        FineTuningMethod::Lora) {
        // Adam owns raw adapter Parameter pointers. Release every user of
        // those pointers before the one-way model merge.
        implementation_->trainer.reset();
        implementation_->optimizer.reset();
        implementation_->adam.reset();
        implementation_->model->merge_lora();
    }
    return {
        artifacts::capture_snapshot(
            *implementation_->model,
            *implementation_->tokenizer
        ),
        std::move(metrics),
        implementation_->example_count,
        implementation_->config.fine_tuning_method,
    };
}

}  // namespace riftco_transformer::stages::post_training
