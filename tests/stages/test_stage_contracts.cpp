#include "riftco_transformer/stages/post_training/config.hpp"
#include "riftco_transformer/stages/post_training/instruction.hpp"
#include "riftco_transformer/stages/pretraining/config.hpp"
#include "riftco_transformer/stages/serving/config.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using riftco_transformer::ExecutionBackend;
using riftco_transformer::FullSequenceAttentionKind;
using riftco_transformer::ActivationCheckpointingKind;
using riftco_transformer::TokenId;
using riftco_transformer::TokenizerMethod;
using riftco_transformer::execution_backend_available;
using riftco_transformer::stages::post_training::InstructionExample;
using riftco_transformer::stages::post_training::FineTuningMethod;
using riftco_transformer::stages::post_training::PlainChatFormatter;
using riftco_transformer::stages::post_training::PostTrainingConfig;
using riftco_transformer::stages::post_training::
    kFullSequenceCausalObjective;
using riftco_transformer::stages::pretraining::PretrainingConfig;
using riftco_transformer::stages::serving::KvCacheKind;
using riftco_transformer::stages::serving::ServingConfig;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw_expected = false;
    try {
        function();
    } catch (const Exception&) {
        threw_expected = true;
    }
    require(threw_expected, message);
}

void test_pretraining_config() {
    const PretrainingConfig valid;
    valid.validate();
    require(
        valid.activation_checkpointing ==
            ActivationCheckpointingKind::Disabled,
        "pretraining should default to disabled checkpointing"
    );

    auto config = valid;
    config.steps = 0;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject zero steps"
    );

    config = valid;
    config.context_size = 0;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject zero context"
    );

    config = valid;
    config.batch_size = std::numeric_limits<std::size_t>::max();
    config.context_size = 2;
    require_throws<std::overflow_error>(
        [&] { config.validate(); },
        "pretraining should reject batch/context overflow"
    );

    if constexpr (
        std::numeric_limits<std::size_t>::max() >
        std::numeric_limits<TokenId>::max()
    ) {
        config = valid;
        config.context_size =
            static_cast<std::size_t>(
                std::numeric_limits<TokenId>::max()
            ) +
            2;
        require_throws<std::invalid_argument>(
            [&] { config.validate(); },
            "pretraining context should fit TokenId positions"
        );
    }

    config = valid;
    config.model_width = 10;
    config.head_count = 4;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining width should be divisible by head count"
    );

    config = valid;
    config.layer_norm_epsilon =
        std::numeric_limits<float>::quiet_NaN();
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject non-finite layer norm epsilon"
    );

    config = valid;
    config.tokenizer.method = static_cast<TokenizerMethod>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject an unknown tokenizer method"
    );

    config = valid;
    config.tokenizer.vocabulary_size = 255;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject a BPE vocabulary below 256"
    );

    if constexpr (
        std::numeric_limits<std::size_t>::max() >
        std::numeric_limits<TokenId>::max()
    ) {
        config = valid;
        config.tokenizer.vocabulary_size =
            static_cast<std::size_t>(
                std::numeric_limits<TokenId>::max()
            ) +
            1;
        require_throws<std::invalid_argument>(
            [&] { config.validate(); },
            "pretraining should match the tokenizer TokenId limit"
        );
    }

    config = valid;
    config.tokenizer.minimum_pair_frequency = 0;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject zero BPE pair frequency"
    );

    config = valid;
    config.tokenizer.method = TokenizerMethod::CorpusByte;
    config.tokenizer.vocabulary_size = 1;
    config.tokenizer.minimum_pair_frequency = 0;
    config.validate();

    config = valid;
    config.optimizer.learning_rate = 0.0F;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject a zero learning rate"
    );

    config = valid;
    config.optimizer.beta1 = 1.0F;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject beta1 outside (0, 1)"
    );

    config = valid;
    config.optimizer.maximum_gradient_norm =
        std::numeric_limits<float>::infinity();
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject a non-finite gradient limit"
    );

    config = valid;
    config.attention =
        static_cast<FullSequenceAttentionKind>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject an unknown attention kind"
    );

    config = valid;
    config.activation_checkpointing =
        static_cast<ActivationCheckpointingKind>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject unknown checkpointing"
    );

    config = valid;
    config.backend = static_cast<ExecutionBackend>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "pretraining should reject an unknown backend"
    );

    config = valid;
    config.backend = ExecutionBackend::Metal;
    if (execution_backend_available(ExecutionBackend::Metal)) {
        config.validate();
    } else {
        require_throws<std::invalid_argument>(
            [&] { config.validate(); },
            "pretraining should reject unavailable Metal"
        );
    }
}

void test_post_training_config() {
    const PostTrainingConfig valid;
    valid.validate();
    require(
        valid.fine_tuning_method == FineTuningMethod::Full,
        "post-training should default to full fine-tuning"
    );
    const PostTrainingConfig legacy_positional{
        20,
        16,
        2,
        riftco_transformer::AdamOptions{},
        29,
        ExecutionBackend::Cpu,
        FineTuningMethod::Full,
        riftco_transformer::LoraConfig{},
    };
    require(
        legacy_positional.attention ==
            FullSequenceAttentionKind::Materialized,
        "appending attention should preserve aggregate initialization"
    );
    require(
        legacy_positional.activation_checkpointing ==
            ActivationCheckpointingKind::Disabled,
        "appending checkpointing should preserve aggregate initialization"
    );

    auto config = valid;
    config.steps = 0;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject zero steps"
    );

    config = valid;
    config.batch_size = std::numeric_limits<std::size_t>::max();
    config.context_size = 2;
    require_throws<std::overflow_error>(
        [&] { config.validate(); },
        "post-training should reject batch/context overflow"
    );

    if constexpr (
        std::numeric_limits<std::size_t>::max() >
        std::numeric_limits<TokenId>::max()
    ) {
        config = valid;
        config.context_size =
            static_cast<std::size_t>(
                std::numeric_limits<TokenId>::max()
            ) +
            2;
        require_throws<std::invalid_argument>(
            [&] { config.validate(); },
            "post-training context should fit token positions"
        );
    }

    config = valid;
    config.optimizer.beta2 =
        std::numeric_limits<float>::quiet_NaN();
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject non-finite Adam options"
    );

    config = valid;
    config.fine_tuning_method = FineTuningMethod::Lora;
    config.lora = {
        2,
        4.0F,
        31U,
        riftco_transformer::kLoraDefaultTargets,
    };
    config.validate();

    config.lora.rank = 0;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject zero LoRA rank"
    );

    config.lora.rank = 2;
    config.lora.alpha =
        std::numeric_limits<float>::quiet_NaN();
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject non-finite LoRA alpha"
    );

    config.lora.alpha = 4.0F;
    config.lora.targets = 0;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject empty LoRA targets"
    );

    config.lora.targets =
        riftco_transformer::kLoraAllTargets |
        (riftco_transformer::LoraTargetMask{1} << 63U);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject unknown LoRA targets"
    );

    config = valid;
    config.fine_tuning_method =
        static_cast<FineTuningMethod>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject an unknown fine-tuning method"
    );

    config = valid;
    config.attention =
        static_cast<FullSequenceAttentionKind>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject an unknown attention kind"
    );

    config = valid;
    config.activation_checkpointing =
        static_cast<ActivationCheckpointingKind>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject unknown checkpointing"
    );

    config = valid;
    config.backend = static_cast<ExecutionBackend>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "post-training should reject an unknown backend"
    );

    config = valid;
    config.backend = ExecutionBackend::Metal;
    if (execution_backend_available(ExecutionBackend::Metal)) {
        config.validate();
    } else {
        require_throws<std::invalid_argument>(
            [&] { config.validate(); },
            "post-training should reject unavailable Metal"
        );
    }
}

void test_serving_config() {
    const ServingConfig valid;
    valid.validate();
    require(
        valid.kv_cache_kind == KvCacheKind::Paged &&
            valid.kv_cache_block_size == 16 &&
            valid.kv_cache_block_count == 0,
        "serving should default to an automatic paged KV cache"
    );

    auto config = valid;
    config.maximum_new_tokens = 0;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "serving should reject a zero generation limit"
    );

    config = valid;
    config.kv_cache_kind = KvCacheKind::Contiguous;
    config.validate();

    config = valid;
    config.kv_cache_kind = static_cast<KvCacheKind>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "serving should reject an unknown KV cache kind"
    );

    config = valid;
    config.kv_cache_block_size = 0;
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "serving should reject a zero KV cache block size"
    );

    if constexpr (
        std::numeric_limits<std::size_t>::max() >
        std::numeric_limits<std::uint32_t>::max()
    ) {
        config = valid;
        config.kv_cache_block_count =
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            ) +
            2;
        require_throws<std::overflow_error>(
            [&] { config.validate(); },
            "serving KV cache block IDs should fit uint32_t"
        );
    }

    config = valid;
    config.backend = static_cast<ExecutionBackend>(99);
    require_throws<std::invalid_argument>(
        [&] { config.validate(); },
        "serving should reject an unknown backend"
    );

    config = valid;
    config.backend = ExecutionBackend::Metal;
    if (execution_backend_available(ExecutionBackend::Metal)) {
        config.validate();
    } else {
        require_throws<std::invalid_argument>(
            [&] { config.validate(); },
            "serving should reject unavailable Metal"
        );
    }
}

void test_instruction_contract() {
    const InstructionExample example{
        " \t hello \nworld \r",
        "\n answer\t here \v",
    };
    example.validate();

    const PlainChatFormatter formatter;
    require(
        formatter.format(example) ==
            "### User:\n"
            "hello \nworld\n"
            "### Assistant:\n"
            "answer\t here\n",
        "plain chat formatting should trim only the outer whitespace"
    );

    for (const std::string& blank : {
             std::string{},
             std::string{" \t\n\r\f\v"},
         }) {
        require_throws<std::invalid_argument>(
            [&] {
                InstructionExample{blank, "answer"}.validate();
            },
            "instruction prompts must reject blank ASCII whitespace"
        );
        require_throws<std::invalid_argument>(
            [&] {
                InstructionExample{"prompt", blank}.validate();
            },
            "instruction responses must reject blank ASCII whitespace"
        );
    }
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(formatter.format(
                InstructionExample{" ", "answer"}
            ));
        },
        "formatting should validate its instruction"
    );

    const std::string nonbreaking_space("\xC2\xA0", 2);
    const InstructionExample unicode_example{
        nonbreaking_space,
        "response",
    };
    unicode_example.validate();
    require(
        formatter.format(unicode_example).find(nonbreaking_space) !=
            std::string::npos,
        "UTF-8 bytes must not be classified by the process locale"
    );
    require(
        kFullSequenceCausalObjective == "full_sequence_causal_sft",
        "post-training objective identifier should remain stable"
    );
}

}  // namespace

int main() {
    try {
        test_pretraining_config();
        test_post_training_config();
        test_serving_config();
        test_instruction_contract();
        std::cout << "native stage contract tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
