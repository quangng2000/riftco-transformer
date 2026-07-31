#pragma once

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/model/activation_checkpointing.hpp"
#include "riftco_transformer/model/causal_self_attention.hpp"
#include "riftco_transformer/model/lora.hpp"
#include "riftco_transformer/optim/adam.hpp"

#include <cstddef>
#include <cstdint>

namespace riftco_transformer::stages::post_training {

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

}  // namespace riftco_transformer::stages::post_training
