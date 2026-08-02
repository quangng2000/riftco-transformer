#include "riftco_transformer/programmed/sequence_placement.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace riftco_transformer::programmed {
namespace {

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char *description) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(description);
  }
  return left + right;
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char *description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(description);
  }
  return left * right;
}

bool spans_overlap(std::size_t first_start, std::size_t first_length,
                   std::size_t second_start, std::size_t second_length) {
  const std::size_t first_end = checked_add(first_start, first_length,
                                            "programmed source span overflows");
  const std::size_t second_end = checked_add(
      second_start, second_length, "programmed target span overflows");
  return first_start < second_end && second_start < first_end;
}

void validate_config(std::size_t model_width,
                     const SequencePlacementConfig &config,
                     const lowering::LoweringMetadata &metadata) {
  if (model_width == 0 || config.output_length == 0) {
    throw std::invalid_argument(
        "programmed sequence dimensions must be positive");
  }
  if (config.inputs.size() != metadata.input_dimensions.size()) {
    throw std::invalid_argument(
        "programmed input layouts must match the compiled arity");
  }
  if (config.source_length == 0 &&
      (!config.inputs.empty() || !metadata.input_dimensions.empty())) {
    throw std::invalid_argument(
        "only zero-arity programs may omit the source span");
  }
  static_cast<void>(checked_add(config.target_offset, config.output_length,
                                "programmed target span overflows"));
  if (config.source_length != 0) {
    static_cast<void>(checked_add(config.source_offset, config.source_length,
                                  "programmed source span overflows"));
    if (spans_overlap(config.source_offset, config.source_length,
                      config.target_offset, config.output_length)) {
      throw std::invalid_argument(
          "programmed source and target spans must not overlap");
    }
  }
  for (std::size_t index = 0; index < config.inputs.size(); ++index) {
    const auto dimension = metadata.input_dimensions[index];
    if (dimension == 0) {
      throw std::invalid_argument(
          "programmed input dimensions must be positive");
    }
    const ProgramInputLayout &layout = config.inputs[index];
    switch (layout.source) {
    case ProgramInputSource::WholeSource:
      if (dimension % config.source_length != 0) {
        throw std::invalid_argument(
            "whole-source program input dimension must be divisible "
            "by source length");
      }
      break;
    case ProgramInputSource::SourcePosition:
      if (layout.position >= config.source_length) {
        throw std::out_of_range(
            "program input source position is out of range");
      }
      break;
    default:
      throw std::invalid_argument("program input source selection is unknown");
    }
  }
  if (metadata.output_dimension == 0 ||
      metadata.output_dimension % config.output_length != 0) {
    throw std::invalid_argument(
        "program output dimension must be divisible by output length");
  }
}

std::size_t projection_width(const ProgramInputLayout &layout,
                             std::size_t input_dimension,
                             std::size_t source_length) {
  return layout.source == ProgramInputSource::WholeSource
             ? input_dimension / source_length
             : input_dimension;
}

struct ValidatedCoreLayout {
  std::vector<std::size_t> logical_projection_groups;
  std::vector<std::size_t> projection_widths;
};

ValidatedCoreLayout
validate_core_config(std::size_t model_width, const SequenceCoreConfig &config,
                     const lowering::LoweringMetadata &metadata) {
  if (model_width == 0 || config.output_length == 0) {
    throw std::invalid_argument("programmed core dimensions must be positive");
  }
  if (config.inputs.size() != metadata.input_dimensions.size()) {
    throw std::invalid_argument(
        "programmed core input layouts must match the compiled arity");
  }
  if (config.source_length == 0 && !config.inputs.empty()) {
    throw std::invalid_argument(
        "a programmed core with inputs requires a source span");
  }
  if (metadata.output_dimension == 0 ||
      metadata.output_dimension % config.output_length != 0) {
    throw std::invalid_argument(
        "programmed core output dimension must be divisible by output length");
  }

  ValidatedCoreLayout result;
  result.logical_projection_groups.resize(config.inputs.size());
  if (config.input_projection_groups.empty()) {
    for (std::size_t index = 0; index < config.inputs.size(); ++index) {
      result.logical_projection_groups[index] = index;
    }
  } else {
    if (config.input_projection_groups.size() != config.inputs.size()) {
      throw std::invalid_argument(
          "programmed core projection groups must match the compiled arity");
    }
    result.logical_projection_groups = config.input_projection_groups;
  }

  std::size_t group_count = 0;
  for (const std::size_t group : result.logical_projection_groups) {
    group_count = std::max(
        group_count,
        checked_add(group, 1U, "programmed projection group count overflows"));
  }
  if (group_count > result.logical_projection_groups.size()) {
    throw std::invalid_argument(
        "programmed core projection groups must be dense and zero-based");
  }
  result.projection_widths.assign(group_count, 0U);
  std::vector<bool> seen_groups(group_count, false);

  for (std::size_t index = 0; index < config.inputs.size(); ++index) {
    const std::size_t dimension = metadata.input_dimensions[index];
    if (dimension == 0) {
      throw std::invalid_argument(
          "programmed core input dimensions must be positive");
    }
    const ProgramInputLayout &layout = config.inputs[index];
    switch (layout.source) {
    case ProgramInputSource::WholeSource:
      if (dimension % config.source_length != 0) {
        throw std::invalid_argument(
            "whole-source core input dimension must be divisible by source "
            "length");
      }
      break;
    case ProgramInputSource::SourcePosition:
      if (layout.position >= config.source_length) {
        throw std::out_of_range("core input source position is out of range");
      }
      break;
    default:
      throw std::invalid_argument("core input source selection is unknown");
    }

    const std::size_t width =
        projection_width(layout, dimension, config.source_length);
    const std::size_t group = result.logical_projection_groups[index];
    if (!seen_groups[group]) {
      seen_groups[group] = true;
      result.projection_widths[group] = width;
    } else if (result.projection_widths[group] != width) {
      throw std::invalid_argument(
          "shared programmed input projections must have equal widths");
    }
  }
  if (std::any_of(seen_groups.begin(), seen_groups.end(),
                  [](bool seen) { return !seen; })) {
    throw std::invalid_argument(
        "programmed core projection groups must be dense and zero-based");
  }
  return result;
}

Variable gather_time_range(const Variable &input, std::size_t start,
                           std::size_t length) {
  const auto &shape = input.value().shape();
  const std::size_t batch = shape[0];
  const std::size_t time = shape[1];
  const std::size_t width = shape[2];
  std::vector<std::size_t> indices;
  indices.reserve(checked_multiply(batch, length,
                                   "programmed source index count overflows"));
  for (std::size_t row = 0; row < batch; ++row) {
    for (std::size_t position = 0; position < length; ++position) {
      indices.push_back(row * time + start + position);
    }
  }
  return gather_rows(reshape(input, {batch * time, width}), indices,
                     {batch, length});
}

Variable select_source_position(const Variable &projected,
                                std::size_t position) {
  const auto &shape = projected.value().shape();
  const std::size_t batch = shape[0];
  const std::size_t time = shape[1];
  const std::size_t width = shape[2];
  std::vector<std::size_t> indices;
  indices.reserve(batch);
  for (std::size_t row = 0; row < batch; ++row) {
    indices.push_back(row * time + position);
  }
  return gather_rows(reshape(projected, {batch * time, width}), indices,
                     {batch});
}

Variable roll_batch(const Variable &input, std::size_t shift) {
  if (input.value().rank() == 0) {
    throw std::invalid_argument("batch-roll ablation requires a batch axis");
  }
  const auto shape = input.value().shape();
  const std::size_t batch = shape[0];
  const std::size_t row_width = input.value().numel() / batch;
  const std::size_t normalized_shift = shift % batch;
  std::vector<std::size_t> indices;
  indices.reserve(batch);
  for (std::size_t row = 0; row < batch; ++row) {
    indices.push_back(row >= batch - normalized_shift
                          ? row - (batch - normalized_shift)
                          : row + normalized_shift);
  }
  return reshape(
      gather_rows(reshape(input, {batch, row_width}), indices, {batch}), shape);
}

Variable apply_steering(const Variable &input,
                        const ProgramInputSteering &steering) {
  const auto &shape = input.value().shape();
  if (shape.size() != 3) {
    throw std::invalid_argument(
        "program-input steering requires [batch, position, feature]");
  }
  const std::size_t position_count = shape[1];
  const std::size_t feature_count = shape[2];
  if (!steering.scales.empty() && steering.scales.size() != feature_count) {
    throw std::invalid_argument(
        "program-input steering scale width is incorrect");
  }
  if (!steering.offsets.empty() && steering.offsets.size() != feature_count) {
    throw std::invalid_argument(
        "program-input steering offset width is incorrect");
  }
  for (const float value : steering.scales) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "program-input steering scales must be finite");
    }
  }
  for (const float value : steering.offsets) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "program-input steering offsets must be finite");
    }
  }

  std::vector<bool> selected(position_count, steering.positions.empty());
  std::unordered_set<std::size_t> unique_positions;
  unique_positions.reserve(steering.positions.size());
  for (const std::size_t position : steering.positions) {
    if (position >= position_count) {
      throw std::out_of_range(
          "program-input steering position is out of range");
    }
    if (!unique_positions.insert(position).second) {
      throw std::invalid_argument(
          "program-input steering positions must be unique");
    }
    selected[position] = true;
  }

  std::vector<float> scales(input.value().numel(), 1.0F);
  std::vector<float> offsets(input.value().numel(), 0.0F);
  for (std::size_t flat = 0; flat < scales.size(); ++flat) {
    const std::size_t position = (flat / feature_count) % position_count;
    if (!selected[position]) {
      continue;
    }
    const std::size_t feature = flat % feature_count;
    if (!steering.scales.empty()) {
      scales[flat] = steering.scales[feature];
    }
    if (!steering.offsets.empty()) {
      offsets[flat] = steering.offsets[feature];
    }
  }
  const ExecutionBackend backend = input.value().backend();
  return input * Variable(Tensor(shape, std::move(scales), backend), false) +
         Variable(Tensor(shape, std::move(offsets), backend), false);
}

Variable place_program_output(const Variable &output, std::size_t total_time,
                              std::size_t target_offset) {
  const auto &shape = output.value().shape();
  if (shape.size() != 3) {
    throw std::invalid_argument(
        "programmed output placement requires [batch, position, feature]");
  }
  const std::size_t batch = shape[0];
  const std::size_t output_length = shape[1];
  const std::size_t width = shape[2];
  const std::size_t target_end =
      checked_add(target_offset, output_length,
                  "programmed placement target span overflows");
  if (target_end > total_time) {
    throw std::invalid_argument(
        "programmed output placement exceeds the sequence length");
  }

  const std::size_t output_row_elements = checked_multiply(
      output_length, width, "programmed output row size overflows");
  const std::size_t placed_row_elements = checked_multiply(
      total_time, width, "programmed placed row size overflows");
  const std::size_t output_elements = checked_multiply(
      batch, output_row_elements, "programmed output size overflows");
  const std::size_t placed_elements = checked_multiply(
      batch, placed_row_elements, "programmed placed output size overflows");
  if (output.value().numel() != output_elements) {
    throw std::logic_error(
        "programmed output shape has an inconsistent element count");
  }

  const std::size_t target_row_offset = checked_multiply(
      target_offset, width, "programmed placement target offset overflows");
  std::vector<float> placed_values(placed_elements, 0.0F);
  const std::span<const float> output_values = output.value().data();
  for (std::size_t row = 0; row < batch; ++row) {
    const std::size_t source_base = row * output_row_elements;
    const std::size_t destination_base =
        row * placed_row_elements + target_row_offset;
    for (std::size_t element = 0; element < output_row_elements; ++element) {
      placed_values[destination_base + element] =
          output_values[source_base + element];
    }
  }

  const ExecutionBackend backend = output.value().backend();
  Tensor placed({batch, total_time, width}, std::move(placed_values), backend);
  return custom_gradient(
      std::move(placed), std::span<const Variable>(&output, 1),
      [batch, total_time, output_length, width, target_row_offset,
       output_row_elements, placed_row_elements, output_elements,
       backend](const Tensor &upstream) {
        if (upstream.shape() != Tensor::Shape{batch, total_time, width} ||
            upstream.backend() != backend) {
          throw std::invalid_argument(
              "programmed placement gradient has the wrong layout");
        }

        std::vector<float> gradient_values(output_elements, 0.0F);
        const std::span<const float> upstream_values = upstream.data();
        for (std::size_t row = 0; row < batch; ++row) {
          const std::size_t source_base =
              row * placed_row_elements + target_row_offset;
          const std::size_t destination_base = row * output_row_elements;
          for (std::size_t element = 0; element < output_row_elements;
               ++element) {
            gradient_values[destination_base + element] =
                upstream_values[source_base + element];
          }
        }

        std::vector<Tensor> gradients;
        gradients.emplace_back(Tensor::Shape{batch, output_length, width},
                               std::move(gradient_values), backend);
        return gradients;
      });
}

void capture(analysis::RepresentationTrace &trace, std::string name,
             const Tensor &value) {
  if (value.backend() == ExecutionBackend::Cpu) {
    trace.capture(std::move(name), value.shape(), value.data());
    return;
  }
  const Tensor host = value.to(ExecutionBackend::Cpu);
  trace.capture(std::move(name), host.shape(), host.data());
}

std::unordered_set<std::size_t>
validate_ablation(const BatchRollAblation &ablation, std::size_t input_count) {
  if (ablation.shift == 0) {
    throw std::invalid_argument("batch-roll ablation shift must be positive");
  }
  if (ablation.program_input_indices.empty() && !ablation.program_output) {
    throw std::invalid_argument(
        "batch-roll ablation must select a representation");
  }
  std::unordered_set<std::size_t> selected;
  selected.reserve(ablation.program_input_indices.size());
  for (const std::size_t index : ablation.program_input_indices) {
    if (index >= input_count) {
      throw std::out_of_range("batch-roll program input index is out of range");
    }
    if (!selected.insert(index).second) {
      throw std::invalid_argument(
          "batch-roll program input indices must be unique");
    }
  }
  return selected;
}

} // namespace

ProgrammedSequenceCore::ProgrammedSequenceCore(
    std::size_t model_width, SequenceCoreConfig config,
    std::unique_ptr<lowering::LoweredMultilinearModule> program,
    std::mt19937 &random)
    : model_width_(model_width), output_basis_width_(0),
      config_(std::move(config)), program_(std::move(program)) {
  initialize(&random, nullptr);
}

ProgrammedSequenceCore::ProgrammedSequenceCore(
    std::size_t model_width, SequenceCoreConfig config,
    std::unique_ptr<lowering::LoweredMultilinearModule> program,
    SequenceCoreProjectionState projections)
    : model_width_(model_width), output_basis_width_(0),
      config_(std::move(config)), program_(std::move(program)) {
  initialize(nullptr, &projections);
}

void ProgrammedSequenceCore::initialize(
    std::mt19937 *random, SequenceCoreProjectionState *projections) {
  if ((random == nullptr) == (projections == nullptr)) {
    throw std::invalid_argument(
        "programmed core needs exactly one projection initializer");
  }
  if (program_ == nullptr) {
    throw std::invalid_argument("programmed core requires a program");
  }

  ValidatedCoreLayout layout =
      validate_core_config(model_width_, config_, program_->metadata());
  logical_projection_groups_ = std::move(layout.logical_projection_groups);
  output_basis_width_ =
      program_->metadata().output_dimension / config_.output_length;
  if (projections != nullptr &&
      projections->inputs.size() != layout.projection_widths.size()) {
    throw std::invalid_argument(
        "programmed core projection state must match the unique projection "
        "count");
  }

  input_projections_.reserve(layout.projection_widths.size());
  for (std::size_t index = 0; index < layout.projection_widths.size();
       ++index) {
    const std::size_t width = layout.projection_widths[index];
    std::shared_ptr<Linear> projection;
    if (projections == nullptr) {
      projection = std::make_shared<Linear>(model_width_, width, *random,
                                            config_.input_projection_bias);
    } else {
      CoreProjectionState &state = projections->inputs[index];
      if (state.weight.shape() != Tensor::Shape{width, model_width_}) {
        throw std::invalid_argument(
            "programmed core input projection weight has the wrong shape");
      }
      if (config_.input_projection_bias) {
        if (!state.bias.has_value() ||
            state.bias->shape() != Tensor::Shape{width}) {
          throw std::invalid_argument(
              "programmed core input projection bias has the wrong shape");
        }
        projection = std::make_shared<Linear>(std::move(state.weight),
                                              std::move(*state.bias));
      } else {
        if (state.bias.has_value()) {
          throw std::invalid_argument(
              "biasless programmed core projection state must omit bias");
        }
        projection = std::make_shared<Linear>(std::move(state.weight));
      }
    }
    input_projection_modules_.append(projection);
    input_projections_.push_back(std::move(projection));
  }

  register_module("input_projections", input_projection_modules_);
  register_module("program", *program_);
  to(program_->backend());
}

std::size_t ProgrammedSequenceCore::model_width() const noexcept {
  return model_width_;
}

std::size_t ProgrammedSequenceCore::output_basis_width() const noexcept {
  return output_basis_width_;
}

const SequenceCoreConfig &ProgrammedSequenceCore::config() const noexcept {
  return config_;
}

SequenceCoreResult
ProgrammedSequenceCore::forward(const Variable &source,
                                const SequenceForwardOptions &options) const {
  const auto &shape = source.value().shape();
  if (shape.size() != 3 || shape[2] != model_width_ ||
      (config_.source_length != 0 && shape[1] != config_.source_length)) {
    throw std::invalid_argument("programmed core source must have shape "
                                "[batch, source_length, model_width]");
  }
  if (source.value().backend() != program_->backend()) {
    throw std::invalid_argument(
        "programmed core source and circuit must share a backend");
  }

  std::unordered_set<std::size_t> ablated_inputs;
  if (options.ablation.has_value()) {
    ablated_inputs =
        validate_ablation(*options.ablation, logical_input_count());
  }
  for (const auto &steering : options.steering) {
    if (steering.input_index >= logical_input_count()) {
      throw std::out_of_range("program-input steering index is out of range");
    }
  }

  analysis::RepresentationTrace trace;
  if (options.capture_representations) {
    capture(trace, "source", source.value());
  }

  std::vector<std::optional<Variable>> shared_projected(
      input_projections_.size());
  std::vector<Variable> program_inputs;
  program_inputs.reserve(logical_input_count());
  for (std::size_t index = 0; index < logical_input_count(); ++index) {
    const std::size_t group = logical_projection_groups_[index];
    if (!shared_projected[group].has_value()) {
      shared_projected[group].emplace(
          input_projections_[group]->forward(source));
    }
    Variable projected = *shared_projected[group];
    for (const auto &steering : options.steering) {
      if (steering.input_index == index) {
        projected = apply_steering(projected, steering);
      }
    }
    if (ablated_inputs.contains(index)) {
      projected = roll_batch(projected, options.ablation->shift);
    }
    if (options.capture_representations) {
      capture(trace, "program_input." + std::to_string(index) + ".projected",
              projected.value());
    }

    Variable arranged = [&] {
      if (config_.inputs[index].source == ProgramInputSource::WholeSource) {
        return reshape(
            projected,
            {shape[0], program_->metadata().input_dimensions[index]});
      }
      return select_source_position(projected, config_.inputs[index].position);
    }();
    if (options.capture_representations) {
      capture(trace, "program_input." + std::to_string(index),
              arranged.value());
    }
    program_inputs.push_back(std::move(arranged));
  }

  Variable program_output = program_inputs.empty()
                                ? program_->forward_constant({shape[0]})
                                : program_->forward(program_inputs);
  program_output = reshape(
      program_output, {shape[0], config_.output_length, output_basis_width_});
  if (options.ablation.has_value() && options.ablation->program_output) {
    program_output = roll_batch(program_output, options.ablation->shift);
  }
  if (options.capture_representations) {
    capture(trace, "program_output", program_output.value());
  }
  return {std::move(program_output), std::move(trace)};
}

void ProgrammedSequenceCore::to(ExecutionBackend backend) {
  Module::to(backend);
}

std::size_t ProgrammedSequenceCore::logical_input_count() const noexcept {
  return logical_projection_groups_.size();
}

std::size_t ProgrammedSequenceCore::input_projection_count() const noexcept {
  return input_projections_.size();
}

std::size_t ProgrammedSequenceCore::input_projection_group(
    std::size_t logical_input_index) const {
  if (logical_input_index >= logical_input_count()) {
    throw std::out_of_range("programmed logical input index is out of range");
  }
  return logical_projection_groups_[logical_input_index];
}

Linear &ProgrammedSequenceCore::input_projection_for_input(
    std::size_t logical_input_index) {
  return input_projection(input_projection_group(logical_input_index));
}

const Linear &ProgrammedSequenceCore::input_projection_for_input(
    std::size_t logical_input_index) const {
  return input_projection(input_projection_group(logical_input_index));
}

Linear &ProgrammedSequenceCore::input_projection(std::size_t projection_index) {
  if (projection_index >= input_projections_.size()) {
    throw std::out_of_range(
        "programmed input projection index is out of range");
  }
  return *input_projections_[projection_index];
}

const Linear &
ProgrammedSequenceCore::input_projection(std::size_t projection_index) const {
  if (projection_index >= input_projections_.size()) {
    throw std::out_of_range(
        "programmed input projection index is out of range");
  }
  return *input_projections_[projection_index];
}

lowering::LoweredMultilinearModule &ProgrammedSequenceCore::program() noexcept {
  return *program_;
}

const lowering::LoweredMultilinearModule &
ProgrammedSequenceCore::program() const noexcept {
  return *program_;
}

ParameterList ProgrammedSequenceCore::parameters() {
  return Module::parameters();
}

ProgrammedSequenceAdapter::ProgrammedSequenceAdapter(
    std::size_t model_width, SequencePlacementConfig config,
    std::unique_ptr<lowering::LoweredMultilinearModule> program,
    std::mt19937 &random)
    : model_width_(model_width), output_basis_width_(0),
      config_(std::move(config)), program_(std::move(program)) {
  initialize(&random, nullptr);
}

ProgrammedSequenceAdapter::ProgrammedSequenceAdapter(
    std::size_t model_width, SequencePlacementConfig config,
    std::unique_ptr<lowering::LoweredMultilinearModule> program,
    SequenceProjectionState projections)
    : model_width_(model_width), output_basis_width_(0),
      config_(std::move(config)), program_(std::move(program)) {
  initialize(nullptr, &projections);
}

void ProgrammedSequenceAdapter::initialize(
    std::mt19937 *random, SequenceProjectionState *projections) {
  if ((random == nullptr) == (projections == nullptr)) {
    throw std::invalid_argument(
        "programmed adapter needs exactly one projection initializer");
  }
  if (program_ == nullptr) {
    throw std::invalid_argument(
        "programmed sequence adapter requires a program");
  }
  validate_config(model_width_, config_, program_->metadata());
  output_basis_width_ =
      program_->metadata().output_dimension / config_.output_length;

  const auto &dimensions = program_->metadata().input_dimensions;
  if (projections != nullptr &&
      projections->inputs.size() != dimensions.size()) {
    throw std::invalid_argument(
        "programmed projection state must match the compiled arity");
  }
  input_projections_.reserve(dimensions.size());
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    const std::size_t width = projection_width(
        config_.inputs[index], dimensions[index], config_.source_length);
    std::shared_ptr<Linear> projection;
    if (projections == nullptr) {
      projection = std::make_shared<Linear>(model_width_, width, *random);
    } else {
      ProjectionState &state = projections->inputs[index];
      if (state.weight.shape() != Tensor::Shape{width, model_width_} ||
          state.bias.shape() != Tensor::Shape{width}) {
        throw std::invalid_argument(
            "programmed input projection state has the wrong shape");
      }
      projection = std::make_shared<Linear>(std::move(state.weight),
                                            std::move(state.bias));
    }
    input_projection_modules_.append(projection);
    input_projections_.push_back(std::move(projection));
  }
  if (projections == nullptr) {
    output_projection_ =
        std::make_unique<Linear>(output_basis_width_, model_width_, *random);
  } else {
    if (projections->output.weight.shape() !=
            Tensor::Shape{model_width_, output_basis_width_} ||
        projections->output.bias.shape() != Tensor::Shape{model_width_}) {
      throw std::invalid_argument(
          "programmed output projection state has the wrong shape");
    }
    output_projection_ =
        std::make_unique<Linear>(std::move(projections->output.weight),
                                 std::move(projections->output.bias));
  }

  register_module("input_projections", input_projection_modules_);
  register_module("program", *program_);
  register_module("output_projection", *output_projection_);
  to(program_->backend());
}

std::size_t ProgrammedSequenceAdapter::model_width() const noexcept {
  return model_width_;
}

std::size_t ProgrammedSequenceAdapter::output_basis_width() const noexcept {
  return output_basis_width_;
}

const SequencePlacementConfig &
ProgrammedSequenceAdapter::config() const noexcept {
  return config_;
}

SequencePlacementResult ProgrammedSequenceAdapter::forward(
    const Variable &sequence, const SequenceForwardOptions &options) const {
  const auto &shape = sequence.value().shape();
  if (shape.size() != 3 || shape[2] != model_width_) {
    throw std::invalid_argument("programmed sequence input must have shape "
                                "[batch, time, model_width]");
  }
  std::size_t required_time =
      checked_add(config_.target_offset, config_.output_length,
                  "programmed target span overflows");
  if (config_.source_length != 0) {
    required_time = std::max(
        required_time, checked_add(config_.source_offset, config_.source_length,
                                   "programmed source span overflows"));
  }
  if (shape[1] < required_time) {
    throw std::invalid_argument(
        "programmed sequence input is too short for its layout");
  }
  if (sequence.value().backend() != program_->backend()) {
    throw std::invalid_argument(
        "programmed sequence input and circuit must share a backend");
  }

  std::unordered_set<std::size_t> ablated_inputs;
  if (options.ablation.has_value()) {
    ablated_inputs =
        validate_ablation(*options.ablation, input_projections_.size());
  }
  for (const auto &steering : options.steering) {
    if (steering.input_index >= input_projections_.size()) {
      throw std::out_of_range("program-input steering index is out of range");
    }
  }

  analysis::RepresentationTrace trace;
  std::optional<Variable> source;
  if (config_.source_length != 0) {
    source.emplace(gather_time_range(sequence, config_.source_offset,
                                     config_.source_length));
    if (options.capture_representations) {
      capture(trace, "source", source->value());
    }
  }

  std::vector<Variable> program_inputs;
  program_inputs.reserve(input_projections_.size());
  for (std::size_t index = 0; index < input_projections_.size(); ++index) {
    if (!source.has_value()) {
      throw std::logic_error(
          "programmed input projection requires a source span");
    }
    Variable projected = input_projections_[index]->forward(*source);
    for (const auto &steering : options.steering) {
      if (steering.input_index == index) {
        projected = apply_steering(projected, steering);
      }
    }
    if (ablated_inputs.contains(index)) {
      projected = roll_batch(projected, options.ablation->shift);
    }
    if (options.capture_representations) {
      capture(trace, "program_input." + std::to_string(index) + ".projected",
              projected.value());
    }

    Variable arranged = [&] {
      if (config_.inputs[index].source == ProgramInputSource::WholeSource) {
        return reshape(
            projected,
            {shape[0], program_->metadata().input_dimensions[index]});
      }
      return select_source_position(projected, config_.inputs[index].position);
    }();
    if (options.capture_representations) {
      capture(trace, "program_input." + std::to_string(index),
              arranged.value());
    }
    program_inputs.push_back(std::move(arranged));
  }

  Variable program_output = program_inputs.empty()
                                ? program_->forward_constant({shape[0]})
                                : program_->forward(program_inputs);
  program_output = reshape(
      program_output, {shape[0], config_.output_length, output_basis_width_});
  if (options.ablation.has_value() && options.ablation->program_output) {
    program_output = roll_batch(program_output, options.ablation->shift);
  }
  if (options.capture_representations) {
    capture(trace, "program_output", program_output.value());
  }

  const Variable projected_output = output_projection_->forward(program_output);
  const Variable programmed_branch =
      place_program_output(projected_output, shape[1], config_.target_offset);
  if (options.capture_representations) {
    capture(trace, "programmed_branch", programmed_branch.value());
  }
  Variable output = sequence + programmed_branch;
  if (options.capture_representations) {
    capture(trace, "output", output.value());
  }
  return {std::move(output), std::move(trace)};
}

void ProgrammedSequenceAdapter::to(ExecutionBackend backend) {
  Module::to(backend);
}

std::size_t ProgrammedSequenceAdapter::input_projection_count() const noexcept {
  return input_projections_.size();
}

Linear &ProgrammedSequenceAdapter::input_projection(std::size_t index) {
  if (index >= input_projections_.size()) {
    throw std::out_of_range(
        "programmed input projection index is out of range");
  }
  return *input_projections_[index];
}

const Linear &
ProgrammedSequenceAdapter::input_projection(std::size_t index) const {
  if (index >= input_projections_.size()) {
    throw std::out_of_range(
        "programmed input projection index is out of range");
  }
  return *input_projections_[index];
}

Linear &ProgrammedSequenceAdapter::output_projection() noexcept {
  return *output_projection_;
}

const Linear &ProgrammedSequenceAdapter::output_projection() const noexcept {
  return *output_projection_;
}

lowering::LoweredMultilinearModule &
ProgrammedSequenceAdapter::program() noexcept {
  return *program_;
}

const lowering::LoweredMultilinearModule &
ProgrammedSequenceAdapter::program() const noexcept {
  return *program_;
}

ParameterList ProgrammedSequenceAdapter::parameters() {
  return Module::parameters();
}

} // namespace riftco_transformer::programmed
