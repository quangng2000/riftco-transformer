#pragma once

#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/model/activation_checkpointing.hpp"
#include "transformer_lab/model/causal_self_attention.hpp"
#include "transformer_lab/model/lora.hpp"
#include "transformer_lab/optim/adam.hpp"

#include <cstddef>
#include <cstdint>

namespace transformer_lab::stages::post_training {

enum class FineTuningMethod {
    Full,
    Lora,
};

// Post-training reuses a model artifact, so architecture and tokenizer
// choices are intentionally not repeated here.
struct PostTrainingConfig {
    std::size_t steps = 20;
    std::size_t context_size = 16;
    std::size_t batch_size = 2;
    AdamOptions optimizer{
        1.0e-3F,
        0.9F,
        0.999F,
        1.0e-8F,
        1.0F,
    };
    std::uint32_t batch_seed = 29;
    ExecutionBackend backend = ExecutionBackend::Cpu;
    FineTuningMethod fine_tuning_method = FineTuningMethod::Full;
    LoraConfig lora{};
    FullSequenceAttentionKind attention =
        FullSequenceAttentionKind::Materialized;
    ActivationCheckpointingKind activation_checkpointing =
        ActivationCheckpointingKind::Disabled;

    void validate() const;
};

}  // namespace transformer_lab::stages::post_training
