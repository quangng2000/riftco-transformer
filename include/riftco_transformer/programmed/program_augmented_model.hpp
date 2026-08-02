#pragma once

#include "riftco_transformer/analysis/representation.hpp"
#include "riftco_transformer/model/causal_self_attention.hpp"
#include "riftco_transformer/model/feed_forward.hpp"
#include "riftco_transformer/nn/embedding.hpp"
#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/module.hpp"
#include "riftco_transformer/programmed/sequence_placement.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace riftco_transformer::programmed {

// The learned scaffold surrounding an optional compiled multilinear program.
// The sequence length is fixed so position embeddings and arbitrary program
// source/target spans have one explicit, validated layout.
struct ProgramAugmentedModelConfig {
  std::size_t vocabulary_size = 1;
  std::size_t context_length = 1;
  std::size_t model_width = 1;
  std::size_t head_count = 1;
  std::size_t attention_branch_count = 2;
  std::size_t feed_forward_width = 1;
  FullSequenceAttentionKind attention_kind =
      FullSequenceAttentionKind::Materialized;
  std::uint32_t seed = 42;

  void validate() const;
};

// Move-only construction input for the optional programmed branch. The core
// owns the lowered module after model construction. Raw program output is
// placed at target_offset and jointly merged with learned attention.
struct ProgramBranch {
  std::size_t source_offset = 0;
  std::size_t target_offset = 0;
  SequenceCoreConfig core_config;
  std::unique_ptr<lowering::LoweredMultilinearModule> program;
  bool merge_bias = false;
};

struct ProgramAugmentedForwardOptions {
  bool capture_representations = false;
  bool batch_roll_attention = false;
  std::size_t batch_roll_shift = 1;
  SequenceForwardOptions program;
};

struct ProgramAugmentedForwardResult {
  Variable logits;
  analysis::RepresentationTrace representations;
};

// Task-neutral learned/programmed composite:
//
//   x  = token_embedding + position_embedding
//   r1 = x + ReLU_FFN(x)
//   h  = merge(concat(attention_i(r1)))
//   r2 = r1 + h                                      (no program)
//   r2 = r1 + merge(concat(h, placed(program(r1))))  (program)
//
// Attention modules are independent graph branches. They are evaluated in
// deterministic construction order; "parallel" describes the topology, not
// concurrent host execution.
class ProgramAugmentedModel final : public Module {
public:
  explicit ProgramAugmentedModel(
      ProgramAugmentedModelConfig config = {},
      std::optional<ProgramBranch> program_branch = std::nullopt);

  [[nodiscard]] const ProgramAugmentedModelConfig &config() const noexcept;
  [[nodiscard]] ExecutionBackend backend() const noexcept;
  [[nodiscard]] bool has_program() const noexcept;

  // flattened_inputs has shape [batch_size, config().context_length].
  [[nodiscard]] ProgramAugmentedForwardResult
  forward(std::span<const TokenId> flattened_inputs, std::size_t batch_size,
          const ProgramAugmentedForwardOptions &options = {}) const;

  // Explicitly transfers frozen lowered-program tensors in addition to the
  // registered Parameter tree.
  void to(ExecutionBackend backend_value) override;

  [[nodiscard]] ProgrammedSequenceCore *program_core() noexcept;
  [[nodiscard]] const ProgrammedSequenceCore *program_core() const noexcept;
  [[nodiscard]] ParameterList parameters();

private:
  ProgramAugmentedModelConfig config_;
  std::mt19937 initialization_random_;
  Embedding token_embedding_;
  Embedding position_embedding_;
  FeedForward feed_forward_;
  std::vector<std::shared_ptr<CausalSelfAttention>> attention_branches_;
  ModuleList attention_branch_modules_;
  std::unique_ptr<Linear> attention_merge_;
  std::unique_ptr<Linear> output_;
  std::size_t program_source_offset_ = 0;
  std::size_t program_target_offset_ = 0;
  std::unique_ptr<ProgrammedSequenceCore> program_core_;
  std::unique_ptr<Linear> program_merge_;
};

} // namespace riftco_transformer::programmed
