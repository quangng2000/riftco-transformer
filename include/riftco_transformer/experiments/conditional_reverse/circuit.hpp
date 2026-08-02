#pragma once

#include "riftco_transformer/analysis/representation.hpp"
#include "riftco_transformer/experiments/conditional_reverse/program.hpp"
#include "riftco_transformer/experiments/conditional_reverse/task.hpp"
#include "riftco_transformer/lowering/config.hpp"
#include "riftco_transformer/nn/module.hpp"
#include "riftco_transformer/programmed/sequence_placement.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace riftco_transformer::experiments::conditional_reverse {

struct CircuitConfig {
  TaskConfig task;
  lowering::NeuralLoweringConfig lowering;
};

struct EvaluationMetrics {
  std::size_t example_count = 0;
  std::size_t token_count = 0;
  std::size_t correct_token_count = 0;
  std::size_t correct_sequence_count = 0;
  double token_accuracy = 0.0;
  double exact_sequence_accuracy = 0.0;
};

struct EvaluationResult {
  EvaluationMetrics metrics;
  std::vector<SymbolId> predictions;
  std::vector<double> per_example_token_accuracy;
  std::vector<double> per_example_exact_accuracy;
  analysis::RepresentationTrace representations;
};

// Exact task circuit with learned, Adam-compatible projections surrounding a
// frozen (or explicitly configured trainable) compiled attention head. The
// source is represented in the first L residual positions and the program is
// placed in the second L positions.
class Circuit final : public Module {
public:
  explicit Circuit(CircuitConfig config = {});

  [[nodiscard]] const CircuitConfig &config() const noexcept;
  [[nodiscard]] const Task &task() const noexcept;
  [[nodiscard]] const ResourceMetadata &resources() const noexcept;
  [[nodiscard]] ExecutionBackend backend() const noexcept;

  // flattened_sources contains batch_size consecutive source sequences.
  [[nodiscard]] programmed::SequencePlacementResult
  forward(std::span<const SymbolId> flattened_sources, std::size_t batch_size,
          const programmed::SequenceForwardOptions &options = {}) const;

  [[nodiscard]] std::vector<SymbolId>
  decode_target(const Tensor &sequence_states) const;
  [[nodiscard]] EvaluationResult
  evaluate(std::span<const Example> examples,
           const programmed::SequenceForwardOptions &options = {}) const;

  // Explicitly reaches the adapter's frozen compiled coefficient storage.
  void to(ExecutionBackend backend) override;

  [[nodiscard]] programmed::ProgrammedSequenceAdapter &adapter() noexcept;
  [[nodiscard]] const programmed::ProgrammedSequenceAdapter &
  adapter() const noexcept;
  [[nodiscard]] ParameterList parameters();

private:
  [[nodiscard]] Tensor
  encode_source_states(std::span<const SymbolId> flattened_sources,
                       std::size_t batch_size) const;

  CircuitConfig config_;
  Task task_;
  ResourceMetadata resources_;
  std::unique_ptr<programmed::ProgrammedSequenceAdapter> adapter_;
};

} // namespace riftco_transformer::experiments::conditional_reverse
