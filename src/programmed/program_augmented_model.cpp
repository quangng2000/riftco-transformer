#include "riftco_transformer/programmed/program_augmented_model.hpp"

#include "riftco_transformer/core/autograd.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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

ProgramAugmentedModelConfig checked_config(ProgramAugmentedModelConfig config) {
  config.validate();
  return config;
}

void validate_program_span(const ProgramBranch &branch,
                           std::size_t context_length) {
  if (branch.program == nullptr) {
    throw std::invalid_argument(
        "program-augmented model requires a nonnull lowered program");
  }

  if (branch.core_config.source_length == 0) {
    if (branch.source_offset > context_length) {
      throw std::invalid_argument(
          "zero-arity program source offset exceeds the context length");
    }
  } else {
    const std::size_t source_end =
        checked_add(branch.source_offset, branch.core_config.source_length,
                    "program source span overflows");
    if (source_end > context_length) {
      throw std::invalid_argument(
          "program source span exceeds the context length");
    }
  }

  const std::size_t target_end =
      checked_add(branch.target_offset, branch.core_config.output_length,
                  "program target span overflows");
  if (target_end > context_length) {
    throw std::invalid_argument(
        "program target span exceeds the context length");
  }
}

Variable gather_time_range(const Variable &input, std::size_t start,
                           std::size_t length) {
  const auto &shape = input.value().shape();
  if (shape.size() != 3) {
    throw std::invalid_argument(
        "program source selection requires [batch, time, feature]");
  }
  const std::size_t batch = shape[0];
  const std::size_t time = shape[1];
  const std::size_t width = shape[2];
  const std::size_t end =
      checked_add(start, length, "program source selection span overflows");
  if (length == 0 || end > time) {
    throw std::invalid_argument(
        "program source selection must be a nonempty in-range span");
  }

  std::vector<std::size_t> indices;
  indices.reserve(
      checked_multiply(batch, length, "program source index count overflows"));
  for (std::size_t row = 0; row < batch; ++row) {
    for (std::size_t position = 0; position < length; ++position) {
      indices.push_back(row * time + start + position);
    }
  }
  return gather_rows(reshape(input, {batch * time, width}), indices,
                     {batch, length});
}

Variable roll_batch(const Variable &input, std::size_t shift) {
  if (input.value().rank() == 0) {
    throw std::invalid_argument(
        "learned-attention batch roll requires a batch axis");
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

Variable place_program_output(const Variable &output, std::size_t total_time,
                              std::size_t target_offset) {
  const auto &shape = output.value().shape();
  if (shape.size() != 3) {
    throw std::invalid_argument(
        "program output placement requires [batch, position, feature]");
  }
  const std::size_t batch = shape[0];
  const std::size_t output_length = shape[1];
  const std::size_t width = shape[2];
  const std::size_t target_end = checked_add(target_offset, output_length,
                                             "program target span overflows");
  if (target_end > total_time) {
    throw std::invalid_argument(
        "program output placement exceeds the context length");
  }

  const ExecutionBackend backend = output.value().backend();
  const std::size_t suffix_length = total_time - target_end;
  const Variable time_last = permute(output, {0, 2, 1});
  std::vector<Variable> pieces;
  pieces.reserve(3);
  if (target_offset != 0) {
    pieces.emplace_back(Tensor::zeros({batch, width, target_offset}, backend),
                        false);
  }
  pieces.push_back(time_last);
  if (suffix_length != 0) {
    pieces.emplace_back(Tensor::zeros({batch, width, suffix_length}, backend),
                        false);
  }
  if (pieces.size() == 1) {
    return output;
  }
  return permute(concatenate_last_axis(std::span<const Variable>(pieces)),
                 {0, 2, 1});
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

void copy_capture(analysis::RepresentationTrace &destination, std::string name,
                  const analysis::NamedRepresentation &source) {
  destination.capture({
      std::move(name),
      source.leading_shape,
      source.observations,
  });
}

void append_program_trace(analysis::RepresentationTrace &destination,
                          const analysis::RepresentationTrace &source) {
  constexpr std::string_view input_prefix = "program_input.";
  for (const analysis::NamedRepresentation &entry : source.entries()) {
    if (entry.name == "source" || entry.name == "program_output") {
      continue;
    }
    const std::string_view name = entry.name;
    if (name.starts_with(input_prefix)) {
      copy_capture(destination,
                   "program.input." +
                       std::string(name.substr(input_prefix.size())),
                   entry);
      continue;
    }
    copy_capture(destination, "program." + entry.name, entry);
  }
}

} // namespace

void ProgramAugmentedModelConfig::validate() const {
  if (vocabulary_size == 0 || context_length == 0 || model_width == 0 ||
      head_count == 0 || attention_branch_count == 0 ||
      feed_forward_width == 0) {
    throw std::invalid_argument(
        "program-augmented model dimensions must be positive");
  }
  if (model_width % head_count != 0) {
    throw std::invalid_argument(
        "program-augmented model width must be divisible by head count");
  }
  static_cast<void>(
      checked_multiply(attention_branch_count, model_width,
                       "program-augmented attention merge width overflows"));
  if (context_length >
      static_cast<std::size_t>(std::numeric_limits<TokenId>::max())) {
    throw std::overflow_error(
        "program-augmented positions exceed the TokenId range");
  }
  switch (attention_kind) {
  case FullSequenceAttentionKind::Materialized:
  case FullSequenceAttentionKind::Flash:
    break;
  default:
    throw std::invalid_argument(
        "program-augmented attention implementation is unknown");
  }
}

ProgramAugmentedModel::ProgramAugmentedModel(
    ProgramAugmentedModelConfig config,
    std::optional<ProgramBranch> program_branch)
    : config_(checked_config(std::move(config))),
      initialization_random_(config_.seed),
      token_embedding_(config_.vocabulary_size, config_.model_width,
                       initialization_random_),
      position_embedding_(config_.context_length, config_.model_width,
                          initialization_random_),
      feed_forward_(config_.model_width, config_.feed_forward_width,
                    initialization_random_, FeedForwardActivation::Relu) {
  if (program_branch.has_value()) {
    validate_program_span(*program_branch, config_.context_length);
  }

  attention_branches_.reserve(config_.attention_branch_count);
  for (std::size_t index = 0; index < config_.attention_branch_count; ++index) {
    auto attention = std::make_shared<CausalSelfAttention>(
        config_.model_width, config_.head_count, initialization_random_,
        config_.attention_kind);
    attention_branch_modules_.append(attention);
    attention_branches_.push_back(std::move(attention));
  }
  attention_merge_ = std::make_unique<Linear>(
      checked_multiply(config_.attention_branch_count, config_.model_width,
                       "program-augmented attention merge width overflows"),
      config_.model_width, initialization_random_);
  output_ = std::make_unique<Linear>(
      config_.model_width, config_.vocabulary_size, initialization_random_);

  if (program_branch.has_value()) {
    ProgramBranch &branch = *program_branch;
    program_source_offset_ = branch.source_offset;
    program_target_offset_ = branch.target_offset;
    program_core_ = std::make_unique<ProgrammedSequenceCore>(
        config_.model_width, std::move(branch.core_config),
        std::move(branch.program), initialization_random_);
    program_merge_ = std::make_unique<Linear>(
        checked_add(config_.model_width, program_core_->output_basis_width(),
                    "program-augmented program merge width overflows"),
        config_.model_width, initialization_random_, branch.merge_bias);
  }

  register_module("token_embedding", token_embedding_);
  register_module("position_embedding", position_embedding_);
  register_module("feed_forward", feed_forward_);
  register_module("attention_branches", attention_branch_modules_);
  register_module("attention_merge", *attention_merge_);
  if (has_program()) {
    register_module("program", *program_core_);
    register_module("program_merge", *program_merge_);
  }
  register_module("output", *output_);

  if (has_program()) {
    to(program_core_->program().backend());
  }
}

const ProgramAugmentedModelConfig &
ProgramAugmentedModel::config() const noexcept {
  return config_;
}

ExecutionBackend ProgramAugmentedModel::backend() const noexcept {
  return token_embedding_.weight().value().backend();
}

bool ProgramAugmentedModel::has_program() const noexcept {
  return program_core_ != nullptr;
}

ProgramAugmentedForwardResult ProgramAugmentedModel::forward(
    std::span<const TokenId> flattened_inputs, std::size_t batch_size,
    const ProgramAugmentedForwardOptions &options) const {
  if (batch_size == 0) {
    throw std::invalid_argument(
        "program-augmented forward requires a nonempty batch");
  }
  const std::size_t expected =
      checked_multiply(batch_size, config_.context_length,
                       "program-augmented forward batch size overflows");
  if (flattened_inputs.size() != expected) {
    throw std::invalid_argument(
        "program-augmented input batch has the wrong size");
  }
  for (const TokenId token : flattened_inputs) {
    if (static_cast<std::size_t>(token) >= config_.vocabulary_size) {
      throw std::out_of_range(
          "program-augmented input token is outside the vocabulary");
    }
  }
  if (options.batch_roll_attention && options.batch_roll_shift == 0) {
    throw std::invalid_argument(
        "learned-attention batch-roll shift must be positive");
  }
  if (!has_program() && (!options.program.steering.empty() ||
                         options.program.ablation.has_value())) {
    throw std::invalid_argument(
        "program interventions require a programmed branch");
  }

  std::vector<TokenId> positions;
  positions.reserve(expected);
  for (std::size_t row = 0; row < batch_size; ++row) {
    for (std::size_t position = 0; position < config_.context_length;
         ++position) {
      positions.push_back(static_cast<TokenId>(position));
    }
  }

  const Variable token_state = token_embedding_.forward(
      flattened_inputs, {batch_size, config_.context_length});
  const Variable position_state = position_embedding_.forward(
      positions, {batch_size, config_.context_length});
  const Variable embedding_sum = token_state + position_state;
  const Variable residual_pre_attention =
      feed_forward_.forward(embedding_sum) + embedding_sum;

  std::vector<Variable> attention_outputs;
  attention_outputs.reserve(attention_branches_.size());
  for (const auto &attention : attention_branches_) {
    attention_outputs.push_back(attention->forward(residual_pre_attention));
  }
  Variable learned_attention = attention_merge_->forward(
      concatenate_last_axis(std::span<const Variable>(attention_outputs)));
  if (options.batch_roll_attention) {
    learned_attention = roll_batch(learned_attention, options.batch_roll_shift);
  }

  const bool capture_representations = options.capture_representations ||
                                       options.program.capture_representations;
  analysis::RepresentationTrace trace;
  if (capture_representations) {
    capture(trace, "embedding.sum", embedding_sum.value());
    capture(trace, "residual.pre_attention", residual_pre_attention.value());
    capture(trace, "learned_attention.merged", learned_attention.value());
  }

  Variable residual_post_merge = learned_attention + residual_pre_attention;
  if (has_program()) {
    const std::size_t source_length = program_core_->config().source_length;
    // A zero-arity program needs only the batch/backend carried by the source;
    // the core deliberately ignores its time extent in that case.
    const Variable program_source =
        source_length == 0
            ? residual_pre_attention
            : gather_time_range(residual_pre_attention, program_source_offset_,
                                source_length);
    SequenceForwardOptions program_options = options.program;
    program_options.capture_representations = capture_representations;
    SequenceCoreResult program_result =
        program_core_->forward(program_source, program_options);
    const Variable placed_program = place_program_output(
        program_result.output, config_.context_length, program_target_offset_);
    residual_post_merge = program_merge_->forward(concatenate_last_axis(
                              learned_attention, placed_program)) +
                          residual_pre_attention;

    if (capture_representations) {
      capture(trace, "program.source", program_source.value());
      append_program_trace(trace, program_result.representations);
      capture(trace, "program.output.raw", program_result.output.value());
      capture(trace, "program.output.placed", placed_program.value());
    }
  }

  if (capture_representations) {
    capture(trace, "residual.post_merge", residual_post_merge.value());
  }
  const Variable logits = output_->forward(residual_post_merge);
  if (capture_representations) {
    capture(trace, "logits", logits.value());
  }
  return {std::move(logits), std::move(trace)};
}

void ProgramAugmentedModel::to(ExecutionBackend backend_value) {
  Module::to(backend_value);
}

ProgrammedSequenceCore *ProgramAugmentedModel::program_core() noexcept {
  return program_core_.get();
}

const ProgrammedSequenceCore *
ProgramAugmentedModel::program_core() const noexcept {
  return program_core_.get();
}

ParameterList ProgramAugmentedModel::parameters() {
  return Module::parameters();
}

} // namespace riftco_transformer::programmed
