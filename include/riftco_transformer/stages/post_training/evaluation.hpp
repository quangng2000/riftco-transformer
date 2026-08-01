#pragma once

#include "riftco_transformer/data/tokenizer.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace riftco_transformer::stages::post_training {

// Exhaustive, read-only full-sequence causal measurements for one split.
// Every target token after the first token in each usable example is scored
// exactly once. Each chunk starts a fresh model context at position zero. Loss
// is weighted by target-token count, not chunk count.
struct CausalEvaluationMetrics {
    std::size_t example_count = 0;
    std::size_t usable_example_count = 0;
    std::size_t skipped_example_count = 0;
    std::size_t target_token_count = 0;
    std::size_t chunk_count = 0;
    std::size_t forward_batch_count = 0;
    double loss = 0.0;
    double perplexity = 1.0;
};

struct DatasetEvaluationMetrics {
    CausalEvaluationMetrics train;
    CausalEvaluationMetrics validation;
    CausalEvaluationMetrics test;
};

// Delta signs are final minus baseline, so a negative loss delta is an
// improvement. Generalization gaps are held-out loss minus training loss.
struct PostTrainingEvaluationMetrics {
    DatasetEvaluationMetrics baseline;
    DatasetEvaluationMetrics final;

    double train_loss_delta = 0.0;
    double validation_loss_delta = 0.0;
    double test_loss_delta = 0.0;

    double baseline_validation_generalization_gap = 0.0;
    double final_validation_generalization_gap = 0.0;
    double baseline_test_generalization_gap = 0.0;
    double final_test_generalization_gap = 0.0;
};

[[nodiscard]] CausalEvaluationMetrics evaluate_causal_sequences(
    DecoderOnlyTransformer& model,
    std::span<const std::vector<TokenId>> sequences,
    std::size_t context_size,
    std::size_t batch_size
);

}  // namespace riftco_transformer::stages::post_training
