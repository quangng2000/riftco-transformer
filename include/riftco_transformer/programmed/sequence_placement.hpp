#pragma once

#include "riftco_transformer/analysis/representation.hpp"
#include "riftco_transformer/lowering/module.hpp"
#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace riftco_transformer::programmed {

enum class ProgramInputSource : std::uint8_t {
  WholeSource = 0,
  SourcePosition = 1,
};

struct ProgramInputLayout {
  ProgramInputSource source = ProgramInputSource::WholeSource;
  // Used only by SourcePosition and relative to the selected source span.
  std::size_t position = 0;
};

struct SequencePlacementConfig {
  std::size_t source_offset = 0;
  // May be zero only for a zero-arity program with no input layouts.
  std::size_t source_length = 1;
  std::size_t output_length = 1;
  std::size_t target_offset = 1;
  std::vector<ProgramInputLayout> inputs;
};

// An affine intervention in a projected program-input basis. Empty positions
// selects every source position. Empty scales and offsets mean ones and zeros,
// respectively; otherwise each vector must match the selected input width.
struct ProgramInputSteering {
  std::size_t input_index = 0;
  std::vector<std::size_t> positions;
  std::vector<float> scales;
  std::vector<float> offsets;
};

// Resample-style activation ablation. Destination batch row i receives row
// (i + shift) mod batch_size. Program-input indices refer to representations
// after projection and steering; program_output applies after the compiled
// circuit and before the learned output projection.
struct BatchRollAblation {
  std::vector<std::size_t> program_input_indices;
  bool program_output = false;
  std::size_t shift = 1;
};

struct SequenceForwardOptions {
  bool capture_representations = false;
  std::vector<ProgramInputSteering> steering;
  std::optional<BatchRollAblation> ablation;
};

struct SequencePlacementResult {
  Variable output;
  analysis::RepresentationTrace representations;
};

struct ProjectionState {
  Tensor weight;
  Tensor bias;
};

struct SequenceProjectionState {
  std::vector<ProjectionState> inputs;
  ProjectionState output;
};

// Configuration for the reusable program executor before output projection or
// residual placement. Each logical input maps to one projection group. Empty
// input_projection_groups means one unique group per input; repeated group IDs
// share the same Linear module and Parameter objects.
struct SequenceCoreConfig {
  std::size_t source_length = 1;
  std::size_t output_length = 1;
  std::vector<ProgramInputLayout> inputs;
  std::vector<std::size_t> input_projection_groups;
  bool input_projection_bias = true;
};

// One state per unique projection group, in dense group-index order.
struct CoreProjectionState {
  Tensor weight;
  std::optional<Tensor> bias;
};

struct SequenceCoreProjectionState {
  std::vector<CoreProjectionState> inputs;
};

struct SequenceCoreResult {
  // [batch, output_length, output_basis_width], before any model-width output
  // projection or residual placement.
  Variable output;
  analysis::RepresentationTrace representations;
};

// Runs a lowered multilinear program over a source sequence and returns its
// raw program-basis output. Projection groups make parameter sharing explicit
// and register every shared child exactly once.
class ProgrammedSequenceCore final : public Module {
public:
  ProgrammedSequenceCore(
      std::size_t model_width, SequenceCoreConfig config,
      std::unique_ptr<lowering::LoweredMultilinearModule> program,
      std::mt19937 &random);
  ProgrammedSequenceCore(
      std::size_t model_width, SequenceCoreConfig config,
      std::unique_ptr<lowering::LoweredMultilinearModule> program,
      SequenceCoreProjectionState projections);

  [[nodiscard]] std::size_t model_width() const noexcept;
  [[nodiscard]] std::size_t output_basis_width() const noexcept;
  [[nodiscard]] const SequenceCoreConfig &config() const noexcept;

  [[nodiscard]] SequenceCoreResult
  forward(const Variable &source,
          const SequenceForwardOptions &options = {}) const;

  void to(ExecutionBackend backend) override;

  [[nodiscard]] std::size_t logical_input_count() const noexcept;
  [[nodiscard]] std::size_t input_projection_count() const noexcept;
  [[nodiscard]] std::size_t
  input_projection_group(std::size_t logical_input_index) const;
  [[nodiscard]] Linear &
  input_projection_for_input(std::size_t logical_input_index);
  [[nodiscard]] const Linear &
  input_projection_for_input(std::size_t logical_input_index) const;
  [[nodiscard]] Linear &input_projection(std::size_t projection_index);
  [[nodiscard]] const Linear &
  input_projection(std::size_t projection_index) const;
  [[nodiscard]] lowering::LoweredMultilinearModule &program() noexcept;
  [[nodiscard]] const lowering::LoweredMultilinearModule &
  program() const noexcept;

  [[nodiscard]] ParameterList parameters();

private:
  void initialize(std::mt19937 *random,
                  SequenceCoreProjectionState *projections);

  std::size_t model_width_;
  std::size_t output_basis_width_;
  SequenceCoreConfig config_;
  std::vector<std::size_t> logical_projection_groups_;
  std::vector<std::shared_ptr<Linear>> input_projections_;
  ModuleList input_projection_modules_;
  std::unique_ptr<lowering::LoweredMultilinearModule> program_;
};

// Places an arbitrary lowered multilinear program into a fixed source/target
// sequence layout. Learned projections translate model residual features into
// each program input basis and back. The compiled module remains a separately
// registered child, so frozen coefficients never appear in Adam parameters.
class ProgrammedSequenceAdapter final : public Module {
public:
  ProgrammedSequenceAdapter(
      std::size_t model_width, SequencePlacementConfig config,
      std::unique_ptr<lowering::LoweredMultilinearModule> program,
      std::mt19937 &random);
  ProgrammedSequenceAdapter(
      std::size_t model_width, SequencePlacementConfig config,
      std::unique_ptr<lowering::LoweredMultilinearModule> program,
      SequenceProjectionState projections);

  [[nodiscard]] std::size_t model_width() const noexcept;
  [[nodiscard]] std::size_t output_basis_width() const noexcept;
  [[nodiscard]] const SequencePlacementConfig &config() const noexcept;

  [[nodiscard]] SequencePlacementResult
  forward(const Variable &sequence,
          const SequenceForwardOptions &options = {}) const;

  // Explicitly transfers the frozen compiled coefficients as well as the
  // normal registered Parameter tree.
  void to(ExecutionBackend backend) override;

  [[nodiscard]] std::size_t input_projection_count() const noexcept;
  [[nodiscard]] Linear &input_projection(std::size_t index);
  [[nodiscard]] const Linear &input_projection(std::size_t index) const;
  [[nodiscard]] Linear &output_projection() noexcept;
  [[nodiscard]] const Linear &output_projection() const noexcept;
  [[nodiscard]] lowering::LoweredMultilinearModule &program() noexcept;
  [[nodiscard]] const lowering::LoweredMultilinearModule &
  program() const noexcept;

  [[nodiscard]] ParameterList parameters();

private:
  void initialize(std::mt19937 *random, SequenceProjectionState *projections);

  std::size_t model_width_;
  std::size_t output_basis_width_;
  SequencePlacementConfig config_;
  std::vector<std::shared_ptr<Linear>> input_projections_;
  ModuleList input_projection_modules_;
  std::unique_ptr<lowering::LoweredMultilinearModule> program_;
  std::unique_ptr<Linear> output_projection_;
};

} // namespace riftco_transformer::programmed
