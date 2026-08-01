#include "riftco_transformer/stages/stages.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace artifacts = riftco_transformer::artifacts;
namespace post_training =
    riftco_transformer::stages::post_training;
namespace pretraining =
    riftco_transformer::stages::pretraining;
namespace serving = riftco_transformer::stages::serving;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    double actual,
    double expected,
    const std::string& message,
    double relative_tolerance = 1.0e-12
) {
    const double scale = std::max(
        {1.0, std::abs(actual), std::abs(expected)}
    );
    require(
        std::abs(actual - expected) <= relative_tolerance * scale,
        message
    );
}

void require_model_state_equal(
    const artifacts::ModelState& actual,
    const artifacts::ModelState& expected,
    const std::string& message
) {
    require(
        actual.dimensions.vocabulary_size ==
            expected.dimensions.vocabulary_size &&
            actual.dimensions.maximum_context ==
                expected.dimensions.maximum_context &&
            actual.dimensions.model_width ==
                expected.dimensions.model_width &&
            actual.dimensions.head_count ==
                expected.dimensions.head_count &&
            actual.dimensions.block_count ==
                expected.dimensions.block_count &&
            actual.dimensions.feed_forward_width ==
                expected.dimensions.feed_forward_width,
        message + ": dimensions"
    );
    require(
        actual.layer_norm_epsilon ==
            expected.layer_norm_epsilon,
        message + ": layer normalization epsilon"
    );
    require(
        actual.parameters.size() == expected.parameters.size(),
        message + ": parameter count"
    );
    for (std::size_t index = 0;
         index < expected.parameters.size();
         ++index) {
        require(
            actual.parameters[index].name ==
                expected.parameters[index].name,
            message + ": parameter name"
        );
        require(
            actual.parameters[index].shape ==
                expected.parameters[index].shape,
            message + ": parameter shape"
        );
        require(
            actual.parameters[index].values ==
                expected.parameters[index].values,
            message + ": parameter values"
        );
    }
}

const artifacts::ParameterState& find_parameter_state(
    const artifacts::ModelState& state,
    const std::string& name
) {
    for (const auto& parameter : state.parameters) {
        if (parameter.name == name) {
            return parameter;
        }
    }
    throw std::runtime_error("missing parameter state " + name);
}

template <typename Exception, typename Function>
void require_throws(
    Function&& function,
    const std::string& message
) {
    bool threw_expected = false;
    try {
        function();
    } catch (const Exception&) {
        threw_expected = true;
    }
    require(threw_expected, message);
}

pretraining::PretrainingConfig tiny_pretraining_config() {
    pretraining::PretrainingConfig config;
    config.steps = 2;
    config.context_size = 3;
    config.batch_size = 1;
    config.model_width = 4;
    config.head_count = 2;
    config.block_count = 1;
    config.feed_forward_width = 8;
    config.tokenizer = {
        riftco_transformer::TokenizerMethod::BytePair,
        260,
        2,
    };
    config.optimizer.learning_rate = 1.0e-2F;
    config.model_seed = 101;
    config.batch_seed = 103;
    config.backend = riftco_transformer::ExecutionBackend::Cpu;
    return config;
}

class CompactInstructionFormatter final
    : public post_training::InstructionFormatter {
public:
    [[nodiscard]] std::string format(
        const post_training::InstructionExample& example
    ) const override {
        example.validate();
        return example.prompt + "|" + example.response;
    }
};

std::size_t evaluation_target_count(
    const riftco_transformer::TokenizerStrategy& tokenizer,
    const std::vector<post_training::InstructionExample>& examples,
    const post_training::InstructionFormatter& formatter
) {
    std::size_t result = 0;
    for (const auto& example : examples) {
        const auto tokens = tokenizer.encode(formatter.format(example));
        if (tokens.size() >= 2) {
            result += tokens.size() - 1;
        }
    }
    return result;
}

void require_valid_evaluation(
    const post_training::CausalEvaluationMetrics& evaluation,
    std::size_t expected_examples,
    std::size_t expected_targets,
    const std::string& message
) {
    require(
        evaluation.example_count == expected_examples,
        message + ": example count"
    );
    require(
        evaluation.usable_example_count == expected_examples &&
            evaluation.skipped_example_count == 0,
        message + ": usable example count"
    );
    require(
        evaluation.target_token_count == expected_targets,
        message + ": exhaustive target-token count"
    );
    require(
        evaluation.chunk_count > 0 &&
            evaluation.forward_batch_count > 0 &&
            evaluation.forward_batch_count <= evaluation.chunk_count,
        message + ": deterministic chunk and batch counts"
    );
    require(
        std::isfinite(evaluation.loss) && evaluation.loss > 0.0,
        message + ": finite loss"
    );
    require(
        std::isfinite(evaluation.perplexity) &&
            evaluation.perplexity > 0.0,
        message + ": finite perplexity"
    );
    require_close(
        evaluation.perplexity,
        std::exp(evaluation.loss),
        message + ": perplexity is exp(loss)"
    );
}

void require_evaluation_summary_consistent(
    const post_training::PostTrainingEvaluationMetrics& evaluation,
    const std::string& message
) {
    require_close(
        evaluation.train_loss_delta,
        evaluation.final.train.loss - evaluation.baseline.train.loss,
        message + ": train loss delta"
    );
    require_close(
        evaluation.validation_loss_delta,
        evaluation.final.validation.loss -
            evaluation.baseline.validation.loss,
        message + ": validation loss delta"
    );
    require_close(
        evaluation.test_loss_delta,
        evaluation.final.test.loss - evaluation.baseline.test.loss,
        message + ": test loss delta"
    );
    require_close(
        evaluation.baseline_validation_generalization_gap,
        evaluation.baseline.validation.loss -
            evaluation.baseline.train.loss,
        message + ": baseline validation gap"
    );
    require_close(
        evaluation.final_validation_generalization_gap,
        evaluation.final.validation.loss - evaluation.final.train.loss,
        message + ": final validation gap"
    );
    require_close(
        evaluation.baseline_test_generalization_gap,
        evaluation.baseline.test.loss - evaluation.baseline.train.loss,
        message + ": baseline test gap"
    );
    require_close(
        evaluation.final_test_generalization_gap,
        evaluation.final.test.loss - evaluation.final.train.loss,
        message + ": final test gap"
    );
}

void test_pretraining_post_training_and_serving_handoff() {
    const std::string corpus =
        "tensor vectors learn patterns. "
        "tensor vectors learn patterns. "
        "attention mixes useful context. "
        "attention mixes useful context.";
    auto pretrain_config = tiny_pretraining_config();
    pretrain_config.attention =
        riftco_transformer::FullSequenceAttentionKind::Flash;
    pretrain_config.activation_checkpointing =
        riftco_transformer::ActivationCheckpointingKind::TransformerBlock;
    pretraining::PretrainingStack pretrain(
        corpus,
        pretrain_config
    );
    require(
        pretrain.model().full_sequence_attention_kind() ==
            riftco_transformer::FullSequenceAttentionKind::Flash,
        "pretraining should compose the selected Flash attention"
    );
    require(
        pretrain.model().activation_checkpointing_kind() ==
            riftco_transformer::ActivationCheckpointingKind::
                TransformerBlock,
        "pretraining should compose block activation checkpointing"
    );
    require(
        pretrain.training_token_count() >
            pretrain.config().context_size,
        "pretraining stack should own encoded training tokens"
    );
    require(
        pretrain.tokenizer().vocab_size() ==
            pretrain.model().dimensions().vocabulary_size,
        "pretraining model vocabulary should come from its tokenizer"
    );

    std::vector<std::size_t> reported_steps;
    pretraining::PretrainingResult pretrained = pretrain.run(
        [&](const riftco_transformer::training::TrainingStepMetrics& metric) {
            reported_steps.push_back(metric.step);
        }
    );
    require(
        reported_steps == std::vector<std::size_t>{1, 2},
        "pretraining metric sink should observe every completed step"
    );
    require(
        pretrained.metrics.size() == 2,
        "pretraining result should retain every metric"
    );
    require(
        std::isfinite(pretrained.first_batch_loss_before_training) &&
            std::isfinite(
                pretrained.first_batch_loss_after_training
            ),
        "pretraining acceptance losses should be finite"
    );
    require(
        !pretrained.snapshot.model.parameters.empty(),
        "pretraining should emit model state"
    );
    require(
        pretrained.snapshot.tokenizer.method ==
            riftco_transformer::TokenizerMethod::BytePair,
        "pretraining should emit exact tokenizer state"
    );
    require_throws<std::logic_error>(
        [&] { static_cast<void>(pretrain.run()); },
        "a pretraining composition root should run only once"
    );

    const artifacts::ModelState original_base_state =
        pretrained.snapshot.model;
    post_training::PlainChatFormatter formatter;
    post_training::PostTrainingConfig post_config;
    post_config.steps = 1;
    post_config.context_size = 3;
    post_config.batch_size = 1;
    post_config.optimizer.learning_rate = 5.0e-3F;
    post_config.batch_seed = 107;
    post_config.backend = riftco_transformer::ExecutionBackend::Cpu;
    post_config.attention =
        riftco_transformer::FullSequenceAttentionKind::Flash;
    post_config.activation_checkpointing =
        riftco_transformer::ActivationCheckpointingKind::TransformerBlock;
    post_training::PostTrainingStack post_train(
        pretrained.snapshot,
        {
            {
                "What does attention use?",
                "It uses context.",
            },
            {
                "What is a tensor?",
                "A multidimensional array.",
            },
        },
        formatter,
        post_config
    );
    require(
        post_train.model().full_sequence_attention_kind() ==
            riftco_transformer::FullSequenceAttentionKind::Flash,
        "post-training should compose the selected Flash attention"
    );
    require(
        post_train.model().activation_checkpointing_kind() ==
            riftco_transformer::ActivationCheckpointingKind::
                TransformerBlock,
        "post-training should compose block activation checkpointing"
    );
    require(
        post_train.example_count() == 2,
        "post-training should own its formatted example set"
    );
    post_training::PostTrainingResult tuned = post_train.run();
    require(
        tuned.metrics.size() == 1 &&
            tuned.metrics.front().step == 1,
        "post-training should use a fresh stage optimizer"
    );
    require(
        tuned.example_count == 2,
        "post-training result should record example count"
    );
    require(
        !tuned.evaluation.has_value(),
        "legacy post-training should remain training-only"
    );
    require_model_state_equal(
        pretrained.snapshot.model,
        original_base_state,
        "full post-training input snapshot"
    );
    require_throws<std::logic_error>(
        [&] { static_cast<void>(post_train.run()); },
        "a post-training composition root should run only once"
    );

    serving::ServingConfig serving_config;
    serving_config.backend =
        riftco_transformer::ExecutionBackend::Cpu;
    serving_config.maximum_new_tokens = 2;
    serving::ServingStack service(
        tuned.snapshot,
        serving_config
    );
    require(
        service.tokenizer().vocab_size() ==
            service.model().dimensions().vocabulary_size,
        "serving should restore a compatible model and tokenizer"
    );
    const serving::GenerationResult generated =
        service.generate("tensor", serving::GenerationConfig{2});
    require(
        generated.generated_token_ids.size() == 2,
        "serving should generate up to its configured limit"
    );
    require(
        generated.token_ids.size() ==
            generated.prompt_token_ids.size() + 2,
        "serving result should retain prompt and generated tokens"
    );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                service.generate(
                    "tensor",
                    serving::GenerationConfig{3}
                )
            );
        },
        "serving should reject requests above its resource limit"
    );

    serving::ServingConfig contiguous_config = serving_config;
    contiguous_config.kv_cache_kind =
        serving::KvCacheKind::Contiguous;
    serving::ServingStack contiguous_service(
        tuned.snapshot,
        contiguous_config
    );
    require(
        contiguous_service
                .generate(
                    "tensor",
                    serving::GenerationConfig{2}
                )
                .generated_token_ids == generated.generated_token_ids,
        "serving cache strategies should preserve greedy generation"
    );

    serving::TemperatureSampler first_sampler(
        0.8F,
        std::size_t{8},
        109
    );
    serving::TemperatureSampler replay_sampler(
        0.8F,
        std::size_t{8},
        109
    );
    const auto first_sample = service.generate(
        "tensor",
        first_sampler,
        serving::GenerationConfig{1}
    );
    const auto replay_sample = service.generate(
        "tensor",
        replay_sampler,
        serving::GenerationConfig{1}
    );
    require(
        first_sample.generated_token_ids ==
            replay_sample.generated_token_ids,
        "serving should accept replayable sampling strategies"
    );
}

void test_lora_post_training_merges_serving_ready_snapshot() {
    const std::string corpus =
        "tensor vectors learn patterns. "
        "tensor vectors learn patterns. "
        "attention mixes useful context. "
        "attention mixes useful context.";
    auto pretrain_config = tiny_pretraining_config();
    pretrain_config.steps = 1;
    pretraining::PretrainingStack pretrain(
        corpus,
        pretrain_config
    );
    pretraining::PretrainingResult pretrained = pretrain.run();
    const artifacts::ModelState original_base_state =
        pretrained.snapshot.model;

    post_training::PlainChatFormatter formatter;
    post_training::PostTrainingConfig post_config;
    post_config.steps = 2;
    post_config.context_size = 3;
    post_config.batch_size = 1;
    post_config.optimizer.learning_rate = 1.0e-2F;
    post_config.batch_seed = 109;
    post_config.backend = riftco_transformer::ExecutionBackend::Cpu;
    post_config.fine_tuning_method =
        post_training::FineTuningMethod::Lora;
    post_config.activation_checkpointing =
        riftco_transformer::ActivationCheckpointingKind::TransformerBlock;
    post_config.lora = {
        2,
        4.0F,
        113U,
        riftco_transformer::kLoraDefaultTargets,
    };

    post_training::PostTrainingStack post_train(
        pretrained.snapshot,
        {
            {
                "What does attention use?",
                "It uses context.",
            },
            {
                "What is a tensor?",
                "A multidimensional array.",
            },
        },
        formatter,
        post_config
    );
    require(
        post_train.model().has_lora(),
        "LoRA should be active while the stage is prepared"
    );
    const auto adapter_parameters =
        post_train.model().lora_parameters();
    require(
        adapter_parameters.size() == 4,
        "default LoRA should optimize A/B for query and value"
    );
    require(
        adapter_parameters[0].name ==
            "blocks.0.attention.query.lora_a.weight" &&
            adapter_parameters[1].name ==
                "blocks.0.attention.query.lora_b.weight" &&
            adapter_parameters[2].name ==
                "blocks.0.attention.value.lora_a.weight" &&
            adapter_parameters[3].name ==
                "blocks.0.attention.value.lora_b.weight",
        "LoRA adapter parameter names and order"
    );

    post_training::PostTrainingResult tuned = post_train.run();
    require(
        tuned.fine_tuning_method ==
            post_training::FineTuningMethod::Lora,
        "post-training result should report LoRA"
    );
    require(
        !tuned.evaluation.has_value(),
        "legacy LoRA post-training should remain training-only"
    );
    require(
        !post_train.model().has_lora(),
        "LoRA should be merged before the stage returns"
    );
    require_model_state_equal(
        pretrained.snapshot.model,
        original_base_state,
        "LoRA post-training input snapshot"
    );
    require(
        tuned.snapshot.model.parameters.size() ==
            original_base_state.parameters.size(),
        "merged LoRA snapshot should use the ordinary parameter schema"
    );

    const std::vector<std::string> selected_weights{
        "blocks.0.attention.query.weight",
        "blocks.0.attention.value.weight",
    };
    bool selected_weight_changed = false;
    for (const auto& original : original_base_state.parameters) {
        const auto& merged = find_parameter_state(
            tuned.snapshot.model,
            original.name
        );
        require(
            merged.shape == original.shape,
            "merged LoRA parameter shape"
        );
        const bool selected =
            std::find(
                selected_weights.begin(),
                selected_weights.end(),
                original.name
            ) != selected_weights.end();
        if (selected) {
            selected_weight_changed =
                selected_weight_changed ||
                merged.values != original.values;
        } else {
            require(
                merged.values == original.values,
                "LoRA changed a non-target parameter: " +
                    original.name
            );
        }
    }
    require(
        selected_weight_changed,
        "LoRA should change at least one selected projection"
    );

    serving::ServingStack service(tuned.snapshot);
    const auto generated = service.generate(
        "tensor",
        serving::GenerationConfig{1}
    );
    require(
        generated.generated_token_ids.size() == 1,
        "merged LoRA snapshot should be serving-ready"
    );
}

void test_split_evaluation_for_full_and_lora_post_training() {
    const std::string corpus =
        "small models learn tokens. small models learn tokens. "
        "held out data checks transfer. held out data checks transfer.";
    auto pretrain_config = tiny_pretraining_config();
    pretrain_config.steps = 1;
    pretraining::PretrainingStack pretrain(corpus, pretrain_config);
    const pretraining::PretrainingResult pretrained = pretrain.run();
    const artifacts::ModelState original_base_state =
        pretrained.snapshot.model;

    const std::vector<post_training::InstructionExample> training_examples{
        {"A1", "B1"},
        {"A2", "B2"},
    };
    const std::vector<post_training::InstructionExample> validation_examples{
        {"C1", "D1"},
        {"C2", "D2"},
    };
    const std::vector<post_training::InstructionExample> first_test_examples{
        {"E1", "F1"},
        {"E2", "F2"},
    };
    const std::vector<post_training::InstructionExample> second_test_examples{
        {"long-E1", "long-F1"},
        {"long-E2", "long-F2"},
    };
    CompactInstructionFormatter formatter;

    for (const auto method : {
             post_training::FineTuningMethod::Full,
             post_training::FineTuningMethod::Lora,
         }) {
        post_training::PostTrainingConfig config;
        config.steps = 1;
        config.context_size = 2;
        config.batch_size = 2;
        config.optimizer.learning_rate = 5.0e-3F;
        config.batch_seed = 211U;
        config.backend = riftco_transformer::ExecutionBackend::Cpu;
        config.fine_tuning_method = method;
        config.lora = {
            2,
            4.0F,
            223U,
            riftco_transformer::kLoraDefaultTargets,
        };

        post_training::PostTrainingStack first(
            pretrained.snapshot,
            post_training::InstructionSplits(
                training_examples,
                validation_examples,
                first_test_examples
            ),
            formatter,
            config
        );
        post_training::PostTrainingStack changed_test(
            pretrained.snapshot,
            post_training::InstructionSplits(
                training_examples,
                validation_examples,
                second_test_examples
            ),
            formatter,
            config
        );
        const std::size_t training_targets = evaluation_target_count(
            first.tokenizer(),
            training_examples,
            formatter
        );
        const std::size_t validation_targets = evaluation_target_count(
            first.tokenizer(),
            validation_examples,
            formatter
        );
        const std::size_t first_test_targets = evaluation_target_count(
            first.tokenizer(),
            first_test_examples,
            formatter
        );
        const std::size_t second_test_targets = evaluation_target_count(
            changed_test.tokenizer(),
            second_test_examples,
            formatter
        );
        require(
            first_test_targets != second_test_targets,
            "test fixtures should exercise different held-out targets"
        );

        if (method == post_training::FineTuningMethod::Full) {
            const std::vector<std::vector<riftco_transformer::TokenId>>
                short_sequences{
                    first.tokenizer().encode(
                        formatter.format(first_test_examples.front())
                    ),
                };
            const std::vector<std::vector<riftco_transformer::TokenId>>
                long_sequences{
                    first.tokenizer().encode(
                        formatter.format(second_test_examples.front())
                    ),
                };
            const std::vector<std::vector<riftco_transformer::TokenId>>
                combined_sequences{
                    short_sequences.front(),
                    long_sequences.front(),
                };
            const artifacts::ModelState before_evaluation =
                artifacts::capture_model_state(first.model());
            const auto short_evaluation =
                post_training::evaluate_causal_sequences(
                    first.model(),
                    short_sequences,
                    config.context_size,
                    config.batch_size
                );
            const auto long_evaluation =
                post_training::evaluate_causal_sequences(
                    first.model(),
                    long_sequences,
                    config.context_size,
                    config.batch_size
                );
            const auto combined_evaluation =
                post_training::evaluate_causal_sequences(
                    first.model(),
                    combined_sequences,
                    config.context_size,
                    config.batch_size
                );
            const auto replayed_evaluation =
                post_training::evaluate_causal_sequences(
                    first.model(),
                    combined_sequences,
                    config.context_size,
                    config.batch_size
                );
            const artifacts::ModelState after_evaluation =
                artifacts::capture_model_state(first.model());
            const double weighted_loss =
                (
                    short_evaluation.loss *
                        static_cast<double>(
                            short_evaluation.target_token_count
                        ) +
                    long_evaluation.loss *
                        static_cast<double>(
                            long_evaluation.target_token_count
                        )
                ) /
                static_cast<double>(
                    short_evaluation.target_token_count +
                    long_evaluation.target_token_count
                );
            require_close(
                combined_evaluation.loss,
                weighted_loss,
                "combined evaluation should be target-token weighted",
                1.0e-6
            );
            require(
                combined_evaluation.target_token_count ==
                    short_evaluation.target_token_count +
                        long_evaluation.target_token_count,
                "combined evaluation should score every target once"
            );
            require_close(
                combined_evaluation.loss,
                replayed_evaluation.loss,
                "causal evaluation should replay deterministically"
            );
            require(
                combined_evaluation.chunk_count ==
                    replayed_evaluation.chunk_count &&
                    combined_evaluation.forward_batch_count ==
                        replayed_evaluation.forward_batch_count,
                "causal evaluation batching should replay deterministically"
            );
            require_model_state_equal(
                after_evaluation,
                before_evaluation,
                "causal evaluation must not mutate model weights"
            );
        }

        const post_training::PostTrainingResult first_result = first.run();
        const post_training::PostTrainingResult changed_test_result =
            changed_test.run();
        const std::string method_name =
            method == post_training::FineTuningMethod::Full
                ? "full"
                : "LoRA";

        require(
            first_result.fine_tuning_method == method &&
                changed_test_result.fine_tuning_method == method,
            method_name + " split evaluation should retain method identity"
        );
        require(
            first_result.evaluation.has_value() &&
                changed_test_result.evaluation.has_value(),
            method_name + " split run should report evaluation"
        );
        require_model_state_equal(
            first_result.snapshot.model,
            changed_test_result.snapshot.model,
            method_name +
                " trained weights must be independent of the test split"
        );
        require_model_state_equal(
            pretrained.snapshot.model,
            original_base_state,
            method_name + " split run must not mutate the base snapshot"
        );

        const auto& evaluation = *first_result.evaluation;
        require_valid_evaluation(
            evaluation.baseline.train,
            training_examples.size(),
            training_targets,
            method_name + " baseline train"
        );
        require_valid_evaluation(
            evaluation.baseline.validation,
            validation_examples.size(),
            validation_targets,
            method_name + " baseline validation"
        );
        require_valid_evaluation(
            evaluation.baseline.test,
            first_test_examples.size(),
            first_test_targets,
            method_name + " baseline test"
        );
        require_valid_evaluation(
            evaluation.final.train,
            training_examples.size(),
            training_targets,
            method_name + " final train"
        );
        require_valid_evaluation(
            evaluation.final.validation,
            validation_examples.size(),
            validation_targets,
            method_name + " final validation"
        );
        require_valid_evaluation(
            evaluation.final.test,
            first_test_examples.size(),
            first_test_targets,
            method_name + " final test"
        );
        require_evaluation_summary_consistent(
            evaluation,
            method_name
        );

        const auto& changed_evaluation = *changed_test_result.evaluation;
        require_close(
            evaluation.baseline.train.loss,
            changed_evaluation.baseline.train.loss,
            method_name + " baseline train is test-independent"
        );
        require_close(
            evaluation.final.train.loss,
            changed_evaluation.final.train.loss,
            method_name + " final train is test-independent"
        );
        require_close(
            evaluation.baseline.validation.loss,
            changed_evaluation.baseline.validation.loss,
            method_name + " baseline validation is test-independent"
        );
        require_close(
            evaluation.final.validation.loss,
            changed_evaluation.final.validation.loss,
            method_name + " final validation is test-independent"
        );
        require_valid_evaluation(
            changed_evaluation.final.test,
            second_test_examples.size(),
            second_test_targets,
            method_name + " changed final test"
        );
    }
}

void test_stage_rejects_incompatible_handoffs() {
    const std::string corpus =
        "one two three four five six seven eight";
    auto config = tiny_pretraining_config();
    config.steps = 1;
    pretraining::PretrainingStack pretrain(corpus, config);
    pretraining::PretrainingResult result = pretrain.run();

    artifacts::ModelSnapshot mismatched = result.snapshot;
    ++mismatched.model.dimensions.vocabulary_size;
    serving::ServingConfig serving_config;
    require_throws<std::invalid_argument>(
        [&] {
            serving::ServingStack invalid(
                mismatched,
                serving_config
            );
            static_cast<void>(invalid);
        },
        "serving should reject a model/tokenizer vocabulary mismatch"
    );

    post_training::PlainChatFormatter formatter;
    post_training::PostTrainingConfig post_config;
    post_config.steps = 1;
    post_config.context_size =
        result.snapshot.model.dimensions.maximum_context + 1;
    post_config.batch_size = 1;
    require_throws<std::invalid_argument>(
        [&] {
            post_training::PostTrainingStack invalid(
                result.snapshot,
                {{"prompt", "response"}},
                formatter,
                post_config
            );
            static_cast<void>(invalid);
        },
        "post-training should reject context beyond the base model"
    );

    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(post_training::InstructionSplits(
                {{"shared", "record"}},
                {{"shared", "record"}},
                {{"held-out", "record"}}
            ));
        },
        "post-training should reject exact cross-split leakage"
    );

    post_config.context_size = 2;
    require_throws<std::invalid_argument>(
        [&] {
            post_training::PostTrainingStack invalid(
                result.snapshot,
                post_training::InstructionSplits(
                    {{" prompt ", " response "}},
                    {{"prompt", "response"}},
                    {{"held-out", "record"}}
                ),
                formatter,
                post_config
            );
            static_cast<void>(invalid);
        },
        "post-training should reject overlap after formatting"
    );
}

}  // namespace

int main() {
    try {
        test_pretraining_post_training_and_serving_handoff();
        test_lora_post_training_merges_serving_ready_snapshot();
        test_split_evaluation_for_full_and_lora_post_training();
        test_stage_rejects_incompatible_handoffs();
        std::cout << "native stage stack tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native stage stack test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
