#pragma once

#include "riftco_transformer/experiments/conditional_reverse/learned_hybrid.hpp"
#include "riftco_transformer/optim/adam.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace riftco_transformer::experiments::conditional_reverse {

struct LearnedTrainingConfig {
  std::size_t epochs = 10;
  std::size_t batch_size = 64;
  std::size_t evaluation_batch_size = 256;
  bool shuffle = true;
  std::uint32_t seed = 42;
  std::optional<std::size_t> maximum_steps;
  // The paper uses ordinary Adam without gradient clipping. Core Adam keeps a
  // finite clipping contract, so the largest finite float is its practical
  // no-clipping setting while preserving overflow/NaN validation.
  AdamOptions adam = [] {
    AdamOptions options;
    options.maximum_gradient_norm = std::numeric_limits<float>::max();
    return options;
  }();

  void validate() const;
};

struct LearnedTrainingStepMetrics {
  std::size_t step = 0;
  std::size_t epoch = 0;
  float target_loss = 0.0F;
  double gradient_norm = 0.0;
  double clip_scale = 1.0;
};

struct LearnedEpochMetrics {
  std::size_t epoch = 0;
  LearnedEvaluationMetrics training;
  LearnedEvaluationMetrics validation;
};

struct LearnedTrainingHistory {
  std::vector<LearnedTrainingStepMetrics> steps;
  std::vector<LearnedEpochMetrics> epochs;
};

// Evaluates in bounded batches. When capture_representations is true, each
// named trace is concatenated across every batch in dataset order.
[[nodiscard]] LearnedEvaluationResult evaluate_learned_dataset(
    const LearnedHybrid &model, const LearnedDataset &dataset,
    std::size_t batch_size = 256, const LearnedForwardOptions &options = {});

class LearnedHybridTrainer {
public:
  LearnedHybridTrainer(LearnedHybrid &model, LearnedTrainingConfig config = {});

  [[nodiscard]] const LearnedTrainingConfig &config() const noexcept;
  [[nodiscard]] LearnedHybrid &model() noexcept;
  [[nodiscard]] const LearnedHybrid &model() const noexcept;
  [[nodiscard]] Adam &optimizer() noexcept;
  [[nodiscard]] const Adam &optimizer() const noexcept;

  [[nodiscard]] LearnedTrainingStepMetrics train_step(const LearnedBatch &batch,
                                                      std::size_t epoch = 0);
  [[nodiscard]] LearnedTrainingHistory fit(const LearnedDataset &training,
                                           const LearnedDataset &validation);

private:
  LearnedHybrid &model_;
  LearnedTrainingConfig config_;
  Adam optimizer_;
};

} // namespace riftco_transformer::experiments::conditional_reverse
