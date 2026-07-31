#pragma once

#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/data/tokenizer.hpp"
#include "transformer_lab/model/activation_checkpointing.hpp"
#include "transformer_lab/model/causal_self_attention.hpp"
#include "transformer_lab/optim/adam.hpp"

#include <cstddef>
#include <cstdint>

namespace transformer_lab::stages::pretraining {

// Owns the choices that are specific to self-supervised pretraining. The
// vocabulary size is deliberately absent: it is derived from the tokenizer
// fitted to the training corpus.
struct PretrainingConfig {
    std::size_t steps = 100;
    std::size_t context_size = 32;
    std::size_t batch_size = 4;

    std::size_t model_width = 16;
    std::size_t head_count = 4;
    std::size_t block_count = 1;
    std::size_t feed_forward_width = 32;
    float layer_norm_epsilon = 1.0e-5F;

    TokenizerOptions tokenizer{
        TokenizerMethod::BytePair,
        272,
        2,
    };
    AdamOptions optimizer{
        1.0e-2F,
        0.9F,
        0.999F,
        1.0e-8F,
        1.0F,
    };

    std::uint32_t model_seed = 7;
    std::uint32_t batch_seed = 7;
    ExecutionBackend backend = ExecutionBackend::Cpu;
    FullSequenceAttentionKind attention =
        FullSequenceAttentionKind::Materialized;
    ActivationCheckpointingKind activation_checkpointing =
        ActivationCheckpointingKind::Disabled;

    void validate() const;
};

}  // namespace transformer_lab::stages::pretraining
