#pragma once

#include "riftco_transformer/analysis/representation.hpp"
#include "riftco_transformer/experiments/conditional_reverse/learned_dataset.hpp"
#include "riftco_transformer/lowering/config.hpp"
#include "riftco_transformer/model/causal_self_attention.hpp"
#include "riftco_transformer/model/feed_forward.hpp"
#include "riftco_transformer/nn/embedding.hpp"
#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/programmed/sequence_placement.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace riftco_transformer::experiments::conditional_reverse {

// Paper controls: F is the compiled conditional head, P is an unconditional
// reverse program, T is a shape-matched randomized program, and I omits the
// programmed branch.
enum class HybridVariant : std::uint8_t {
  F,
  P,
  T,
  I,
};

struct LearnedHybridConfig {
  LearnedProtocolConfig protocol;
  HybridVariant variant = HybridVariant::F;
  std::size_t model_width = 20;
  std::size_t head_count = 2;
  std::size_t feed_forward_width = 80;
  FullSequenceAttentionKind attention_kind =
      FullSequenceAttentionKind::Materialized;
  lowering::NeuralLoweringConfig lowering;
  std::uint32_t seed = 42;

  void validate() const;
  [[nodiscard]] std::size_t program_width() const;
};

struct LearnedSteering {
  std::size_t position = 0;
  float selected_coordinate_scale = 1.0F;
  float other_coordinate_scale = 1.0F;
};

struct LearnedForwardOptions {
  bool capture_representations = false;
  bool ablate_learned_attention = false;
  bool ablate_program_output = false;
  std::size_t batch_roll_shift = 1;
  std::vector<LearnedSteering> steering;
};

struct LearnedForwardResult {
  Variable logits;
  analysis::RepresentationTrace representations;
};

struct LearnedEvaluationMetrics {
  std::size_t example_count = 0;
  std::size_t target_token_count = 0;
  std::size_t correct_target_token_count = 0;
  std::size_t correct_sequence_count = 0;
  std::size_t reverse_example_count = 0;
  std::size_t reverse_correct_target_token_count = 0;
  std::size_t reverse_correct_sequence_count = 0;
  std::size_t copy_example_count = 0;
  std::size_t copy_correct_target_token_count = 0;
  std::size_t copy_correct_sequence_count = 0;
  float loss = 0.0F;
  double target_token_accuracy = 0.0;
  double exact_sequence_accuracy = 0.0;
  double reverse_target_token_accuracy = 0.0;
  double reverse_exact_sequence_accuracy = 0.0;
  double copy_target_token_accuracy = 0.0;
  double copy_exact_sequence_accuracy = 0.0;
};

struct LearnedEvaluationResult {
  LearnedEvaluationMetrics metrics;
  // Predictions contain only the L supervised output positions.
  std::vector<TokenId> predictions;
  std::vector<double> per_example_token_accuracy;
  std::vector<double> per_example_exact_accuracy;
  analysis::RepresentationTrace representations;
};

struct LearnedHypothesisScores {
  double copy_target_token_accuracy = 0.0;
  double copy_exact_sequence_accuracy = 0.0;
  double reverse_target_token_accuracy = 0.0;
  double reverse_exact_sequence_accuracy = 0.0;
};

// Learned components surrounding the paper's optional programmed head:
// token/position embeddings, residual ReLU MLP, two learned causal heads,
// branch merge, and vocabulary projection.
class LearnedHybrid final : public Module {
public:
  explicit LearnedHybrid(LearnedHybridConfig config = {});

  [[nodiscard]] const LearnedHybridConfig &config() const noexcept;
  [[nodiscard]] HybridVariant variant() const noexcept;
  [[nodiscard]] ExecutionBackend backend() const noexcept;
  [[nodiscard]] bool has_program() const noexcept;

  // flattened_inputs has shape [batch_size, 2 * sequence_length].
  [[nodiscard]] LearnedForwardResult
  forward(std::span<const TokenId> flattened_inputs, std::size_t batch_size,
          const LearnedForwardOptions &options = {}) const;
  [[nodiscard]] LearnedEvaluationResult
  evaluate(const LearnedBatch &batch,
           const LearnedForwardOptions &options = {}) const;
  [[nodiscard]] LearnedEvaluationResult
  evaluate(std::span<const LearnedExample> examples,
           const LearnedForwardOptions &options = {}) const;

  void to(ExecutionBackend backend_value) override;

  [[nodiscard]] programmed::ProgrammedSequenceCore *program_core() noexcept;
  [[nodiscard]] const programmed::ProgrammedSequenceCore *
  program_core() const noexcept;
  [[nodiscard]] const Linear *program_merge() const noexcept;
  [[nodiscard]] ParameterList parameters();

private:
  LearnedHybridConfig config_;
  std::mt19937 initialization_random_;
  Embedding token_embedding_;
  Embedding position_embedding_;
  FeedForward feed_forward_;
  CausalSelfAttention attention_first_;
  CausalSelfAttention attention_second_;
  Linear attention_merge_;
  Linear output_;
  std::unique_ptr<programmed::ProgrammedSequenceCore> program_core_;
  std::unique_ptr<Linear> program_merge_;
};

// Selects positions [L, 2L) from teacher-forced logits/targets and computes
// their mean cross entropy. Source-side and delimiter predictions are ignored.
[[nodiscard]] Variable
learned_target_half_loss(const Variable &logits,
                         std::span<const TokenId> flattened_targets,
                         std::size_t sequence_length);

// Scores the same output predictions against unconditional copy and reverse
// hypotheses. This is useful for verifying that steering changes the executed
// algorithm rather than merely changing aggregate task accuracy.
[[nodiscard]] LearnedHypothesisScores
score_learned_hypotheses(std::span<const LearnedExample> examples,
                         std::span<const TokenId> target_predictions,
                         const LearnedProtocolConfig &config);

} // namespace riftco_transformer::experiments::conditional_reverse
