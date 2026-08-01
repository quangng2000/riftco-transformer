#include "riftco_transformer/stages/post_training/stack.hpp"

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/data/tokenizer.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/optim/adam.hpp"
#include "riftco_transformer/stages/post_training/evaluation.hpp"
#include "riftco_transformer/training/adam_optimizer_adapter.hpp"
#include "riftco_transformer/training/batch_source.hpp"

#include <memory>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer::stages::post_training {
namespace {

struct PreparedInstructionSplit {
    std::vector<std::vector<TokenId>> sequences;
    std::vector<std::string> formatted_examples;
};

PreparedInstructionSplit prepare_instruction_split(
    const TokenizerStrategy& tokenizer,
    const std::vector<InstructionExample>& examples,
    const InstructionFormatter& formatter,
    std::size_t context_size,
    const char* split_name,
    bool require_training_window
) {
    if (examples.empty()) {
        throw std::invalid_argument(
            std::string("post-training ") + split_name +
            " split must not be empty"
        );
    }
    PreparedInstructionSplit result;
    result.sequences.reserve(examples.size());
    result.formatted_examples.reserve(examples.size());
    bool has_evaluation_target = false;
    for (std::size_t index = 0; index < examples.size(); ++index) {
        examples[index].validate();
        std::string formatted = formatter.format(examples[index]);
        std::vector<TokenId> sequence = tokenizer.encode(formatted);
        if (require_training_window && sequence.size() <= context_size) {
            throw std::invalid_argument(
                "post-training " + std::string(split_name) + " example " +
                std::to_string(index) +
                " must encode to at least context_size + 1 tokens"
            );
        }
        has_evaluation_target =
            has_evaluation_target || sequence.size() >= 2;
        result.formatted_examples.push_back(std::move(formatted));
        result.sequences.push_back(std::move(sequence));
    }
    if (!has_evaluation_target) {
        throw std::invalid_argument(
            "post-training " + std::string(split_name) +
            " split has no example with a target token"
        );
    }
    return result;
}

void reject_formatted_overlap(
    const std::vector<std::string>& left,
    const char* left_name,
    const std::vector<std::string>& right,
    const char* right_name
) {
    const std::set<std::string> left_values(left.begin(), left.end());
    const std::set<std::string> right_values(right.begin(), right.end());
    for (const auto& formatted : left_values) {
        if (right_values.contains(formatted)) {
            throw std::invalid_argument(
                "post-training " + std::string(left_name) + " and " +
                right_name +
                " splits overlap after instruction formatting"
            );
        }
    }
}

PostTrainingEvaluationMetrics summarize_evaluations(
    DatasetEvaluationMetrics baseline,
    DatasetEvaluationMetrics final
) {
    PostTrainingEvaluationMetrics result;
    result.train_loss_delta = final.train.loss - baseline.train.loss;
    result.validation_loss_delta =
        final.validation.loss - baseline.validation.loss;
    result.test_loss_delta = final.test.loss - baseline.test.loss;
    result.baseline_validation_generalization_gap =
        baseline.validation.loss - baseline.train.loss;
    result.final_validation_generalization_gap =
        final.validation.loss - final.train.loss;
    result.baseline_test_generalization_gap =
        baseline.test.loss - baseline.train.loss;
    result.final_test_generalization_gap =
        final.test.loss - final.train.loss;
    result.baseline = std::move(baseline);
    result.final = std::move(final);
    return result;
}

}  // namespace

struct PostTrainingStack::Implementation {
    PostTrainingConfig config;
    std::size_t example_count = 0;
    std::unique_ptr<TokenizerStrategy> tokenizer;
    std::unique_ptr<DecoderOnlyTransformer> model;
    std::unique_ptr<Adam> adam;
    std::unique_ptr<training::AdamOptimizerAdapter> optimizer;
    std::unique_ptr<training::CausalLanguageModelTrainer> trainer;
    std::unique_ptr<training::SequenceWindowBatchSource> batches;
    bool evaluation_enabled = false;
    std::optional<artifacts::ModelState> baseline_model_state;
    std::vector<std::vector<TokenId>> training_sequences;
    std::vector<std::vector<TokenId>> validation_sequences;
    std::vector<std::vector<TokenId>> test_sequences;
    bool started = false;

    Implementation(
        const artifacts::ModelSnapshot& base_snapshot,
        std::vector<InstructionExample> examples,
        const InstructionFormatter& formatter,
        PostTrainingConfig configured
    ) {
        initialize(
            base_snapshot,
            std::move(examples),
            {},
            {},
            formatter,
            configured,
            false
        );
    }

    Implementation(
        const artifacts::ModelSnapshot& base_snapshot,
        InstructionSplits splits,
        const InstructionFormatter& formatter,
        PostTrainingConfig configured
    ) {
        splits.validate();
        initialize(
            base_snapshot,
            std::move(splits.train),
            std::move(splits.validation),
            std::move(splits.test),
            formatter,
            configured,
            true
        );
    }

    void initialize(
        const artifacts::ModelSnapshot& base_snapshot,
        std::vector<InstructionExample> training_examples,
        std::vector<InstructionExample> validation_examples,
        std::vector<InstructionExample> held_out_test_examples,
        const InstructionFormatter& formatter,
        PostTrainingConfig configured,
        bool enable_evaluation
    ) {
        configured.validate();
        if (training_examples.empty()) {
            throw std::invalid_argument(
                "post-training requires at least one instruction example"
            );
        }
        if (configured.context_size >
            base_snapshot.model.dimensions.maximum_context) {
            throw std::invalid_argument(
                "post-training context exceeds the base model maximum context"
            );
        }

        config = configured;
        example_count = training_examples.size();
        evaluation_enabled = enable_evaluation;
        tokenizer = artifacts::restore_tokenizer(base_snapshot.tokenizer);
        if (tokenizer->vocab_size() !=
            base_snapshot.model.dimensions.vocabulary_size) {
            throw std::invalid_argument(
                "base model and tokenizer vocabulary sizes do not match"
            );
        }

        PreparedInstructionSplit prepared_training =
            prepare_instruction_split(
                *tokenizer,
                training_examples,
                formatter,
                config.context_size,
                "train",
                true
            );
        if (evaluation_enabled) {
            PreparedInstructionSplit prepared_validation =
                prepare_instruction_split(
                    *tokenizer,
                    validation_examples,
                    formatter,
                    config.context_size,
                    "validation",
                    false
                );
            PreparedInstructionSplit prepared_test =
                prepare_instruction_split(
                    *tokenizer,
                    held_out_test_examples,
                    formatter,
                    config.context_size,
                    "test",
                    false
                );
            reject_formatted_overlap(
                prepared_training.formatted_examples,
                "train",
                prepared_validation.formatted_examples,
                "validation"
            );
            reject_formatted_overlap(
                prepared_training.formatted_examples,
                "train",
                prepared_test.formatted_examples,
                "test"
            );
            reject_formatted_overlap(
                prepared_validation.formatted_examples,
                "validation",
                prepared_test.formatted_examples,
                "test"
            );
            training_sequences = std::move(prepared_training.sequences);
            validation_sequences =
                std::move(prepared_validation.sequences);
            test_sequences = std::move(prepared_test.sequences);
            baseline_model_state = base_snapshot.model;
        }

        const ScopedExecutionBackend selected_backend(config.backend);
        std::mt19937 restoration_random(0);
        model = std::make_unique<DecoderOnlyTransformer>(
            base_snapshot.model.dimensions,
            restoration_random,
            base_snapshot.model.layer_norm_epsilon,
            config.attention,
            config.activation_checkpointing
        );
        artifacts::load_model_state(*model, base_snapshot.model);
        ParameterList optimizer_parameters;
        AdamOptions optimizer_options = config.optimizer;
        switch (config.fine_tuning_method) {
            case FineTuningMethod::Full:
                optimizer_parameters = model->parameters();
                break;
            case FineTuningMethod::Lora:
                model->attach_lora(config.lora);
                optimizer_parameters = model->lora_parameters();
                break;
            case FineTuningMethod::Qlora:
                if (config.nf4_double_quantization) {
                    model->quantize_linear_weights_nf4_double_quantized(
                        config.nf4_block_size,
                        config.nf4_scale_block_size
                    );
                } else {
                    model->quantize_linear_weights_nf4(
                        config.nf4_block_size
                    );
                }
                model->attach_lora(config.lora);
                optimizer_parameters = model->lora_parameters();
                if (config.qlora_paged_optimizer) {
                    optimizer_options.state_storage =
                        AdamStateStorageKind::Paged;
                    optimizer_options.page_size =
                        config.qlora_optimizer_page_size;
                }
                break;
        }
        adam = std::make_unique<Adam>(
            std::move(optimizer_parameters),
            optimizer_options
        );
        optimizer = std::make_unique<training::AdamOptimizerAdapter>(*adam);
        trainer =
            std::make_unique<training::CausalLanguageModelTrainer>(
                *model,
                *optimizer
            );
        if (evaluation_enabled) {
            batches =
                std::make_unique<training::SequenceWindowBatchSource>(
                    training_sequences,
                    config.batch_size,
                    config.context_size,
                    config.batch_seed
                );
        } else {
            batches =
                std::make_unique<training::SequenceWindowBatchSource>(
                    std::move(prepared_training.sequences),
                    config.batch_size,
                    config.context_size,
                    config.batch_seed
                );
        }
    }

    [[nodiscard]] CausalEvaluationMetrics evaluate_split(
        DecoderOnlyTransformer& target_model,
        const std::vector<std::vector<TokenId>>& sequences
    ) const {
        return evaluate_causal_sequences(
            target_model,
            sequences,
            config.context_size,
            config.batch_size
        );
    }

    [[nodiscard]] std::unique_ptr<DecoderOnlyTransformer>
    restore_baseline_model() const {
        if (!baseline_model_state.has_value()) {
            throw std::logic_error(
                "post-training baseline state is unavailable"
            );
        }
        const ScopedExecutionBackend selected_backend(config.backend);
        std::mt19937 restoration_random(0);
        auto result = std::make_unique<DecoderOnlyTransformer>(
            baseline_model_state->dimensions,
            restoration_random,
            baseline_model_state->layer_norm_epsilon,
            config.attention,
            config.activation_checkpointing
        );
        artifacts::load_model_state(*result, *baseline_model_state);
        return result;
    }
};

PostTrainingStack::PostTrainingStack(
    const artifacts::ModelSnapshot& base_snapshot,
    std::vector<InstructionExample> examples,
    const InstructionFormatter& formatter,
    PostTrainingConfig config
)
    : implementation_(std::make_unique<Implementation>(
          base_snapshot,
          std::move(examples),
          formatter,
          config
      )) {}

PostTrainingStack::PostTrainingStack(
    const artifacts::ModelSnapshot& base_snapshot,
    InstructionSplits splits,
    const InstructionFormatter& formatter,
    PostTrainingConfig config
)
    : implementation_(std::make_unique<Implementation>(
          base_snapshot,
          std::move(splits),
          formatter,
          config
      )) {}

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
            FineTuningMethod::Lora ||
        implementation_->config.fine_tuning_method ==
            FineTuningMethod::Qlora) {
        // Adam owns raw adapter Parameter pointers. Release every user of
        // those pointers before the one-way model merge.
        implementation_->trainer.reset();
        implementation_->optimizer.reset();
        implementation_->adam.reset();
        implementation_->model->merge_lora();
    }
    std::optional<PostTrainingEvaluationMetrics> evaluation;
    if (implementation_->evaluation_enabled) {
        // All optimizer updates and the one-way LoRA merge are complete before
        // either model sees the test split. Evaluation performs forward and
        // cross-entropy only; it never calls backward or the optimizer.
        auto baseline_model = implementation_->restore_baseline_model();
        DatasetEvaluationMetrics baseline;
        DatasetEvaluationMetrics final;
        baseline.train = implementation_->evaluate_split(
            *baseline_model,
            implementation_->training_sequences
        );
        baseline.validation = implementation_->evaluate_split(
            *baseline_model,
            implementation_->validation_sequences
        );
        final.train = implementation_->evaluate_split(
            *implementation_->model,
            implementation_->training_sequences
        );
        final.validation = implementation_->evaluate_split(
            *implementation_->model,
            implementation_->validation_sequences
        );

        baseline.test = implementation_->evaluate_split(
            *baseline_model,
            implementation_->test_sequences
        );
        final.test = implementation_->evaluate_split(
            *implementation_->model,
            implementation_->test_sequences
        );
        evaluation = summarize_evaluations(
            std::move(baseline),
            std::move(final)
        );
    }
    return {
        artifacts::capture_snapshot(
            *implementation_->model,
            *implementation_->tokenizer
        ),
        std::move(metrics),
        implementation_->example_count,
        implementation_->config.fine_tuning_method,
        std::move(evaluation),
    };
}

}  // namespace riftco_transformer::stages::post_training
