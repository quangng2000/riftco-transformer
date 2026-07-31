#include "transformer_lab/stages/stages.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace artifacts = transformer_lab::artifacts;
namespace post_training =
    transformer_lab::stages::post_training;
namespace pretraining =
    transformer_lab::stages::pretraining;
namespace serving = transformer_lab::stages::serving;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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
        transformer_lab::TokenizerMethod::BytePair,
        260,
        2,
    };
    config.optimizer.learning_rate = 1.0e-2F;
    config.model_seed = 101;
    config.batch_seed = 103;
    config.backend = transformer_lab::ExecutionBackend::Cpu;
    return config;
}

void test_pretraining_post_training_and_serving_handoff() {
    const std::string corpus =
        "tensor vectors learn patterns. "
        "tensor vectors learn patterns. "
        "attention mixes useful context. "
        "attention mixes useful context.";
    auto pretrain_config = tiny_pretraining_config();
    pretrain_config.attention =
        transformer_lab::FullSequenceAttentionKind::Flash;
    pretrain_config.activation_checkpointing =
        transformer_lab::ActivationCheckpointingKind::TransformerBlock;
    pretraining::PretrainingStack pretrain(
        corpus,
        pretrain_config
    );
    require(
        pretrain.model().full_sequence_attention_kind() ==
            transformer_lab::FullSequenceAttentionKind::Flash,
        "pretraining should compose the selected Flash attention"
    );
    require(
        pretrain.model().activation_checkpointing_kind() ==
            transformer_lab::ActivationCheckpointingKind::
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
        [&](const transformer_lab::training::TrainingStepMetrics& metric) {
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
            transformer_lab::TokenizerMethod::BytePair,
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
    post_config.backend = transformer_lab::ExecutionBackend::Cpu;
    post_config.attention =
        transformer_lab::FullSequenceAttentionKind::Flash;
    post_config.activation_checkpointing =
        transformer_lab::ActivationCheckpointingKind::TransformerBlock;
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
            transformer_lab::FullSequenceAttentionKind::Flash,
        "post-training should compose the selected Flash attention"
    );
    require(
        post_train.model().activation_checkpointing_kind() ==
            transformer_lab::ActivationCheckpointingKind::
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
        transformer_lab::ExecutionBackend::Cpu;
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
    post_config.backend = transformer_lab::ExecutionBackend::Cpu;
    post_config.fine_tuning_method =
        post_training::FineTuningMethod::Lora;
    post_config.activation_checkpointing =
        transformer_lab::ActivationCheckpointingKind::TransformerBlock;
    post_config.lora = {
        2,
        4.0F,
        113U,
        transformer_lab::kLoraDefaultTargets,
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
}

}  // namespace

int main() {
    try {
        test_pretraining_post_training_and_serving_handoff();
        test_lora_post_training_merges_serving_ready_snapshot();
        test_stage_rejects_incompatible_handoffs();
        std::cout << "native stage stack tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native stage stack test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
