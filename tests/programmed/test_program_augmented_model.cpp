#include "riftco_transformer/compiler/cajal/multilinear_map.hpp"
#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/lowering/config.hpp"
#include "riftco_transformer/lowering/strategy.hpp"
#include "riftco_transformer/programmed/program_augmented_model.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using riftco_transformer::ExecutionBackend;
using riftco_transformer::FullSequenceAttentionKind;
using riftco_transformer::ParameterList;
using riftco_transformer::Tensor;
using riftco_transformer::TokenId;
using riftco_transformer::analysis::NamedRepresentation;
using riftco_transformer::analysis::RepresentationTrace;
using riftco_transformer::compiler::cajal::MultilinearMap;
using riftco_transformer::lowering::NeuralLoweringConfig;
using riftco_transformer::programmed::BatchRollAblation;
using riftco_transformer::programmed::ProgramAugmentedForwardOptions;
using riftco_transformer::programmed::ProgramAugmentedModel;
using riftco_transformer::programmed::ProgramAugmentedModelConfig;
using riftco_transformer::programmed::ProgramBranch;
using riftco_transformer::programmed::ProgramInputLayout;
using riftco_transformer::programmed::ProgramInputSource;
using riftco_transformer::programmed::ProgramInputSteering;

class TrackingFrozenProgram final
    : public riftco_transformer::lowering::LoweredMultilinearModule {
public:
  TrackingFrozenProgram() : coefficients_(Tensor({2}, {1.0F, -1.0F})) {
    metadata_.requested_strategy = "tracking_test";
    metadata_.selected_strategy = "tracking_test";
    metadata_.preserves_compiled_map = true;
    metadata_.output_dimension = 2;
    metadata_.logical_coefficient_shape = {2};
    metadata_.tensors.push_back({
        .name = "coefficients",
        .semantic_role = "test_constant",
        .shape = {2},
        .trainable = false,
    });
  }

  [[nodiscard]] riftco_transformer::Variable
  forward(std::span<const riftco_transformer::Variable> inputs) const override {
    if (!inputs.empty()) {
      throw std::invalid_argument("tracking program is zero-arity");
    }
    return forward_constant();
  }

  [[nodiscard]] riftco_transformer::Variable
  forward_constant(Tensor::Shape leading_shape = {}) const override {
    if (!leading_shape.empty()) {
      throw std::invalid_argument(
          "tracking program test does not execute batched forward");
    }
    return riftco_transformer::Variable(coefficients_, false);
  }

  [[nodiscard]] const riftco_transformer::lowering::LoweringMetadata &
  metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] const Tensor &
  named_tensor(std::string_view name) const override {
    if (name != "coefficients") {
      throw std::invalid_argument("unknown tracking-program tensor");
    }
    return coefficients_;
  }

  [[nodiscard]] ExecutionBackend backend() const noexcept override {
    return coefficients_.backend();
  }

  void to(ExecutionBackend) override {
    ++direct_to_calls_;
    throw std::runtime_error(
        "a parent transfer must not call a child transfer directly");
  }

  [[nodiscard]] std::size_t direct_to_calls() const noexcept {
    return direct_to_calls_;
  }

  [[nodiscard]] std::size_t prepare_calls() const noexcept {
    return prepare_calls_;
  }

  [[nodiscard]] std::size_t commit_calls() const noexcept {
    return commit_calls_;
  }

private:
  class PreparedCounterTransfer final : public PreparedBackendTransfer {
  public:
    explicit PreparedCounterTransfer(std::size_t &counter)
        : counter_(&counter) {}

    void commit() noexcept override { ++*counter_; }

  private:
    std::size_t *counter_;
  };

  [[nodiscard]] PreparedBackendTransferList
  prepare_extra_backend_transfers(ExecutionBackend) override {
    ++prepare_calls_;
    PreparedBackendTransferList result;
    result.push_back(
        std::make_unique<PreparedCounterTransfer>(commit_calls_));
    return result;
  }

  Tensor coefficients_;
  riftco_transformer::lowering::LoweringMetadata metadata_;
  std::size_t direct_to_calls_ = 0;
  std::size_t prepare_calls_ = 0;
  std::size_t commit_calls_ = 0;
};

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_throws(Function &&function, const std::string &message) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception &) {
    threw = true;
  }
  require(threw, message);
}

void require_close(float actual, float expected, const std::string &message,
                   float tolerance = 1.0e-5F) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

ProgramAugmentedModelConfig model_config(std::size_t branch_count = 3) {
  ProgramAugmentedModelConfig config;
  config.vocabulary_size = 7;
  config.context_length = 5;
  config.model_width = 4;
  config.head_count = 2;
  config.attention_branch_count = branch_count;
  config.feed_forward_width = 8;
  config.attention_kind = FullSequenceAttentionKind::Materialized;
  config.seed = 17;
  return config;
}

ProgramBranch identity_branch(bool trainable = false,
                              std::size_t source_offset = 2,
                              std::size_t target_offset = 1) {
  NeuralLoweringConfig lowering;
  lowering.strategy = riftco_transformer::lowering::kLinearStrategy;
  lowering.trainable = trainable;

  ProgramBranch branch;
  branch.source_offset = source_offset;
  branch.target_offset = target_offset;
  branch.core_config.source_length = 2;
  branch.core_config.output_length = 2;
  branch.core_config.inputs = {
      ProgramInputLayout{ProgramInputSource::WholeSource, 0},
  };
  branch.core_config.input_projection_groups = {0};
  branch.core_config.input_projection_bias = false;
  branch.program = riftco_transformer::lowering::lower_to_neural(
      MultilinearMap::identity(4), lowering);
  branch.merge_bias = false;
  return branch;
}

ProgramBranch constant_branch() {
  NeuralLoweringConfig lowering;
  lowering.strategy = riftco_transformer::lowering::kDenseContractionStrategy;

  ProgramBranch branch;
  branch.source_offset = 5;
  branch.target_offset = 4;
  branch.core_config.source_length = 0;
  branch.core_config.output_length = 1;
  branch.core_config.input_projection_bias = false;
  branch.program = riftco_transformer::lowering::lower_to_neural(
      MultilinearMap(std::vector<std::size_t>{}, 2,
                     std::vector<double>{1.25, -0.75}),
      lowering);
  return branch;
}

std::vector<TokenId> tokens(std::size_t batch_size) {
  std::vector<TokenId> result;
  result.reserve(batch_size * 5);
  for (std::size_t row = 0; row < batch_size; ++row) {
    for (std::size_t position = 0; position < 5; ++position) {
      result.push_back(static_cast<TokenId>((row * 3 + position) % 7));
    }
  }
  return result;
}

void require_trace_names(const RepresentationTrace &trace,
                         const std::vector<std::string> &expected,
                         const std::string &message) {
  const auto entries = trace.entries();
  require(entries.size() == expected.size(), message + ": entry count");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    require(entries[index].name == expected[index],
            message + ": name at index " + std::to_string(index));
  }
}

void require_representation_shape(const NamedRepresentation &entry,
                                  std::vector<std::size_t> leading_shape,
                                  std::size_t width,
                                  const std::string &message) {
  require(entry.leading_shape == leading_shape,
          message + ": leading shape mismatch");
  require(entry.observations.columns == width,
          message + ": feature width mismatch");
  std::size_t rows = 1;
  for (const std::size_t dimension : leading_shape) {
    rows *= dimension;
  }
  require(entry.observations.rows == rows, message + ": row count mismatch");
  require(entry.observations.values.size() == rows * width,
          message + ": value count mismatch");
}

bool contains_parameter(const ParameterList &parameters,
                        std::string_view name) {
  return std::any_of(
      parameters.begin(), parameters.end(),
      [name](const auto &parameter) { return parameter.name == name; });
}

bool prefix_has_nonzero_gradient(const ParameterList &parameters,
                                 std::string_view prefix) {
  bool found = false;
  bool nonzero = false;
  for (const auto &named : parameters) {
    if (!std::string_view(named.name).starts_with(prefix)) {
      continue;
    }
    found = true;
    for (const float value : named.parameter->gradient().data()) {
      require(std::isfinite(value),
              "gradient must be finite for " + named.name);
      nonzero = nonzero || std::fabs(value) > 1.0e-9F;
    }
  }
  require(found, "missing parameter prefix " + std::string(prefix));
  return nonzero;
}

void require_batch_roll(const NamedRepresentation &actual,
                        const NamedRepresentation &baseline,
                        std::size_t batch_size, std::size_t shift,
                        const std::string &message) {
  require(actual.leading_shape == baseline.leading_shape &&
              actual.observations.rows == baseline.observations.rows &&
              actual.observations.columns == baseline.observations.columns,
          message + ": representation shape mismatch");
  require(!actual.leading_shape.empty() &&
              actual.leading_shape.front() == batch_size,
          message + ": batch shape mismatch");
  const std::size_t rows_per_batch = actual.observations.rows / batch_size;
  const std::size_t width = actual.observations.columns;
  const std::size_t normalized_shift = shift % batch_size;
  for (std::size_t row = 0; row < batch_size; ++row) {
    const std::size_t source_row = (row + normalized_shift) % batch_size;
    for (std::size_t observation = 0; observation < rows_per_batch;
         ++observation) {
      for (std::size_t feature = 0; feature < width; ++feature) {
        const std::size_t actual_index =
            (row * rows_per_batch + observation) * width + feature;
        const std::size_t expected_index =
            (source_row * rows_per_batch + observation) * width + feature;
        require_close(actual.observations.values[actual_index],
                      baseline.observations.values[expected_index], message);
      }
    }
  }
}

void test_config_and_span_validation() {
  ProgramAugmentedModelConfig config = model_config();
  config.validate();

  config.attention_branch_count = 0;
  require_throws([&] { config.validate(); },
                 "zero attention branches must be rejected");

  config = model_config();
  config.model_width = 5;
  require_throws([&] { config.validate(); },
                 "indivisible head width must be rejected");

  config = model_config();
  config.attention_kind = static_cast<FullSequenceAttentionKind>(255);
  require_throws([&] { config.validate(); },
                 "unknown attention implementation must be rejected");

  config = model_config();
  config.model_width = 2;
  config.head_count = 1;
  config.attention_branch_count = std::numeric_limits<std::size_t>::max();
  require_throws([&] { config.validate(); },
                 "attention concatenation overflow must be rejected");

  if constexpr (std::numeric_limits<std::size_t>::max() >
                std::numeric_limits<TokenId>::max()) {
    config = model_config();
    config.context_length =
        static_cast<std::size_t>(std::numeric_limits<TokenId>::max()) + 1;
    require_throws([&] { config.validate(); },
                   "unrepresentable position IDs must be rejected");
  }

  config = model_config();
  ProgramBranch null_branch = identity_branch();
  null_branch.program.reset();
  require_throws(
      [&] {
        ProgramAugmentedModel model(
            config, std::optional<ProgramBranch>{std::move(null_branch)});
      },
      "null lowered programs must be rejected");

  ProgramBranch source_out_of_range = identity_branch(false, 4, 1);
  require_throws(
      [&] {
        ProgramAugmentedModel model(
            config,
            std::optional<ProgramBranch>{std::move(source_out_of_range)});
      },
      "out-of-range source spans must be rejected");

  ProgramBranch target_out_of_range = identity_branch(false, 2, 4);
  require_throws(
      [&] {
        ProgramAugmentedModel model(
            config,
            std::optional<ProgramBranch>{std::move(target_out_of_range)});
      },
      "out-of-range target spans must be rejected");

  ProgramBranch source_overflow =
      identity_branch(false, std::numeric_limits<std::size_t>::max(), 1);
  require_throws(
      [&] {
        ProgramAugmentedModel model(
            config, std::optional<ProgramBranch>{std::move(source_overflow)});
      },
      "overflowing source spans must be rejected");

  ProgramBranch mismatched_arity = identity_branch();
  mismatched_arity.core_config.inputs.clear();
  mismatched_arity.core_config.input_projection_groups.clear();
  require_throws(
      [&] {
        ProgramAugmentedModel model(
            config, std::optional<ProgramBranch>{std::move(mismatched_arity)});
      },
      "core layouts must match lowered program arity");
}

void test_learned_only_forward_and_schema() {
  ProgramAugmentedModel model(model_config());
  require(!model.has_program() && model.program_core() == nullptr,
          "learned-only model must omit the program");
  require(model.backend() == ExecutionBackend::Cpu,
          "new learned-only model must use CPU storage");
  require(model.config().attention_branch_count == 3,
          "model must retain the configured attention count");

  const std::vector<TokenId> input = tokens(2);
  const auto uncaptured = model.forward(input, 2);
  require(uncaptured.logits.value().shape() == Tensor::Shape{2, 5, 7},
          "learned-only logits shape");
  require(uncaptured.representations.entries().empty(),
          "capture-disabled forward must return an empty trace");

  ProgramAugmentedForwardOptions options;
  options.capture_representations = true;
  const auto captured = model.forward(input, 2, options);
  require_trace_names(captured.representations,
                      {
                          "embedding.sum",
                          "residual.pre_attention",
                          "learned_attention.merged",
                          "residual.post_merge",
                          "logits",
                      },
                      "learned-only trace schema");
  require_representation_shape(captured.representations.at("embedding.sum"),
                               {2, 5}, 4, "embedding trace");
  require_representation_shape(
      captured.representations.at("residual.pre_attention"), {2, 5}, 4,
      "pre-attention residual trace");
  require_representation_shape(
      captured.representations.at("learned_attention.merged"), {2, 5}, 4,
      "learned-attention trace");
  require_representation_shape(captured.representations.at("logits"), {2, 5}, 7,
                               "logit trace");

  const ParameterList parameters = model.parameters();
  require(
      contains_parameter(parameters, "attention_branches.0.query.weight") &&
          contains_parameter(parameters, "attention_branches.1.query.weight") &&
          contains_parameter(parameters, "attention_branches.2.query.weight"),
      "every configured attention branch must be registered");
  require(!contains_parameter(parameters, "program_merge.weight"),
          "learned-only parameters must omit program merge");
  std::set<riftco_transformer::Parameter *> unique_parameters;
  for (const auto &parameter : parameters) {
    require(parameter.parameter != nullptr,
            "parameter schema must not contain null parameters");
    require(unique_parameters.insert(parameter.parameter).second,
            "parameter schema must not contain duplicate identities");
  }

  require_throws([&] { static_cast<void>(model.forward(input, 0)); },
                 "empty batches must be rejected");
  require_throws(
      [&] {
        static_cast<void>(model.forward(
            std::span<const TokenId>(input.data(), input.size() - 1), 2));
      },
      "wrong fixed-context token counts must be rejected");
  std::vector<TokenId> invalid_tokens = input;
  invalid_tokens.front() = 7;
  require_throws([&] { static_cast<void>(model.forward(invalid_tokens, 2)); },
                 "out-of-vocabulary tokens must be rejected");

  options = {};
  options.batch_roll_attention = true;
  options.batch_roll_shift = 0;
  require_throws([&] { static_cast<void>(model.forward(input, 2, options)); },
                 "zero learned-attention roll shifts must be rejected");

  options = {};
  options.program.ablation = BatchRollAblation{
      .program_input_indices = {},
      .program_output = true,
      .shift = 1,
  };
  require_throws([&] { static_cast<void>(model.forward(input, 2, options)); },
                 "program ablation without a program must be rejected");
  options = {};
  options.program.steering.push_back(ProgramInputSteering{});
  require_throws([&] { static_cast<void>(model.forward(input, 2, options)); },
                 "program steering without a program must be rejected");
}

void test_learned_attention_batch_roll() {
  ProgramAugmentedModel model(model_config(1));
  const std::vector<TokenId> input = tokens(3);

  ProgramAugmentedForwardOptions baseline_options;
  baseline_options.capture_representations = true;
  const auto baseline = model.forward(input, 3, baseline_options);

  ProgramAugmentedForwardOptions rolled_options = baseline_options;
  rolled_options.batch_roll_attention = true;
  rolled_options.batch_roll_shift = 4;
  const auto rolled = model.forward(input, 3, rolled_options);
  require_batch_roll(rolled.representations.at("learned_attention.merged"),
                     baseline.representations.at("learned_attention.merged"), 3,
                     4, "learned-attention roll");

  ProgramAugmentedForwardOptions identity_options = baseline_options;
  identity_options.batch_roll_attention = true;
  identity_options.batch_roll_shift = 3;
  const auto identity = model.forward(input, 3, identity_options);
  require_batch_roll(identity.representations.at("learned_attention.merged"),
                     baseline.representations.at("learned_attention.merged"), 3,
                     3, "full-batch learned-attention roll");
}

void test_program_trace_placement_and_interventions() {
  ProgramAugmentedModel model(model_config(),
                              std::optional<ProgramBranch>{identity_branch()});
  require(model.has_program() && model.program_core() != nullptr,
          "programmed model must expose its core");
  require(model.program_core()->logical_input_count() == 1 &&
              model.program_core()->input_projection_count() == 1,
          "programmed model must preserve core projection sharing");

  const std::vector<TokenId> input = tokens(2);
  ProgramAugmentedForwardOptions options;
  options.capture_representations = true;
  const auto baseline = model.forward(input, 2, options);
  require(baseline.logits.value().shape() == Tensor::Shape{2, 5, 7},
          "programmed logits shape");
  require_trace_names(baseline.representations,
                      {
                          "embedding.sum",
                          "residual.pre_attention",
                          "learned_attention.merged",
                          "program.source",
                          "program.input.0.projected",
                          "program.input.0",
                          "program.output.raw",
                          "program.output.placed",
                          "residual.post_merge",
                          "logits",
                      },
                      "programmed trace schema");
  require_representation_shape(baseline.representations.at("program.source"),
                               {2, 2}, 4, "program source trace");
  require_representation_shape(
      baseline.representations.at("program.input.0.projected"), {2, 2}, 2,
      "projected program input trace");
  require_representation_shape(baseline.representations.at("program.input.0"),
                               {2}, 4, "arranged program input trace");
  require_representation_shape(
      baseline.representations.at("program.output.raw"), {2, 2}, 2,
      "raw program output trace");
  require_representation_shape(
      baseline.representations.at("program.output.placed"), {2, 5}, 2,
      "placed program output trace");

  const NamedRepresentation &raw =
      baseline.representations.at("program.output.raw");
  const NamedRepresentation &placed =
      baseline.representations.at("program.output.placed");
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t position = 0; position < 5; ++position) {
      for (std::size_t feature = 0; feature < 2; ++feature) {
        const float value =
            placed.observations.values[(row * 5 + position) * 2 + feature];
        if (position >= 1 && position < 3) {
          const float expected =
              raw.observations.values[(row * 2 + (position - 1)) * 2 + feature];
          require_close(value, expected,
                        "placed program target must equal raw output");
        } else {
          require_close(value, 0.0F,
                        "placed program output must be zero outside target");
        }
      }
    }
  }

  const ParameterList parameters = model.parameters();
  require(contains_parameter(parameters, "program.input_projections.0.weight"),
          "program core projection must be trainable");
  require(!contains_parameter(parameters, "program.input_projections.0.bias"),
          "biasless core projection must omit bias");
  require(contains_parameter(parameters, "program_merge.weight") &&
              !contains_parameter(parameters, "program_merge.bias"),
          "biasless program merge must expose only its weight");
  require(!contains_parameter(parameters, "program.program.coefficients"),
          "frozen program coefficients must stay out of optimizer schema");

  ProgramAugmentedForwardOptions rolled_options = options;
  rolled_options.program.ablation = BatchRollAblation{
      .program_input_indices = {},
      .program_output = true,
      .shift = 1,
  };
  const auto rolled = model.forward(input, 2, rolled_options);
  require_batch_roll(rolled.representations.at("program.output.raw"), raw, 2, 1,
                     "program-output roll");

  ProgramAugmentedForwardOptions steered_options = options;
  steered_options.program.steering.push_back(ProgramInputSteering{
      .input_index = 0,
      .positions = {},
      .scales = {0.0F, 0.0F},
      .offsets = {},
  });
  const auto steered = model.forward(input, 2, steered_options);
  for (const float value :
       steered.representations.at("program.output.raw").observations.values) {
    require_close(value, 0.0F,
                  "zero input steering must zero an identity program");
  }

  ProgramAugmentedForwardOptions invalid_options = options;
  invalid_options.program.steering.push_back(ProgramInputSteering{
      .input_index = 1,
  });
  require_throws(
      [&] { static_cast<void>(model.forward(input, 2, invalid_options)); },
      "out-of-range program steering must be rejected");

  invalid_options = options;
  invalid_options.program.ablation = BatchRollAblation{
      .program_input_indices = {},
      .program_output = true,
      .shift = 0,
  };
  require_throws(
      [&] { static_cast<void>(model.forward(input, 2, invalid_options)); },
      "zero program roll shifts must be rejected");
}

void test_trainable_program_backward_reaches_every_branch() {
  ProgramAugmentedModel model(
      model_config(), std::optional<ProgramBranch>{identity_branch(true)});
  ParameterList parameters = model.parameters();
  require(contains_parameter(parameters, "program.program.coefficients"),
          "trainable program coefficients must enter optimizer schema");
  for (const auto &parameter : parameters) {
    parameter.parameter->zero_gradient();
  }

  auto result = model.forward(tokens(2), 2);
  std::vector<float> seed_values(result.logits.value().numel(), 0.0F);
  for (std::size_t index = 0; index < seed_values.size(); ++index) {
    const int centered = static_cast<int>(index % 13) - 6;
    seed_values[index] = 0.03F * static_cast<float>(centered) + 0.01F;
  }
  result.logits.backward(
      Tensor(result.logits.value().shape(), std::move(seed_values)));

  for (const std::string_view prefix : {
           "token_embedding.",
           "position_embedding.",
           "feed_forward.",
           "attention_branches.0.",
           "attention_branches.1.",
           "attention_branches.2.",
           "attention_merge.",
           "program.input_projections.0.",
           "program.program.",
           "program_merge.",
           "output.",
       }) {
    require(prefix_has_nonzero_gradient(parameters, prefix),
            "backward must reach " + std::string(prefix));
  }
}

void test_zero_arity_program_and_backend_transfer() {
  ProgramAugmentedModel model(model_config(1),
                              std::optional<ProgramBranch>{constant_branch()});
  ProgramAugmentedForwardOptions options;
  // The embedded SequenceForwardOptions capture flag is also honored as a
  // request for the complete composite trace.
  options.program.capture_representations = true;
  const auto result = model.forward(tokens(2), 2, options);
  const auto &source = result.representations.at("program.source");
  require_representation_shape(source, {2, 5}, 4,
                               "zero-arity carrier source trace");
  const auto &raw = result.representations.at("program.output.raw");
  require_representation_shape(raw, {2, 1}, 2, "zero-arity raw output trace");
  for (std::size_t row = 0; row < 2; ++row) {
    require_close(raw.observations.values[row * 2], 1.25F,
                  "constant program first coordinate");
    require_close(raw.observations.values[row * 2 + 1], -0.75F,
                  "constant program second coordinate");
  }
  const auto &placed = result.representations.at("program.output.placed");
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t position = 0; position < 5; ++position) {
      for (std::size_t feature = 0; feature < 2; ++feature) {
        const float expected =
            position == 4 ? raw.observations.values[row * 2 + feature] : 0.0F;
        require_close(
            placed.observations.values[(row * 5 + position) * 2 + feature],
            expected, "constant program placement");
      }
    }
  }

  model.to(ExecutionBackend::Cpu);
  require(model.backend() == ExecutionBackend::Cpu &&
              model.program_core()->program().backend() ==
                  ExecutionBackend::Cpu,
          "CPU transfer must include frozen program storage");

  for (const ExecutionBackend backend : {
           ExecutionBackend::Metal,
           ExecutionBackend::Cuda,
           ExecutionBackend::Tpu,
       }) {
    if (!riftco_transformer::execution_backend_available(backend)) {
      continue;
    }
    model.to(backend);
    require(model.backend() == backend &&
                model.program_core()->program().backend() == backend,
            "backend transfer must include model and frozen program");
    {
      const ParameterList parameters = model.parameters();
      for (const auto &parameter : parameters) {
        require(parameter.parameter->value().backend() == backend &&
                    parameter.parameter->gradient().backend() == backend,
                "backend transfer must include every parameter");
      }
    }
    const auto &program = model.program_core()->program();
    for (const auto &tensor : program.metadata().tensors) {
      require(program.named_tensor(tensor.name).backend() == backend,
              "backend transfer must include every lowered tensor");
    }
    model.to(ExecutionBackend::Cpu);
  }
}

void test_composite_transfer_uses_one_tree_transaction() {
  auto program = std::make_unique<TrackingFrozenProgram>();
  TrackingFrozenProgram *const retained_program = program.get();
  ProgramBranch branch;
  branch.source_offset = 0;
  branch.target_offset = 0;
  branch.core_config.source_length = 0;
  branch.core_config.output_length = 1;
  branch.core_config.input_projection_bias = false;
  branch.program = std::move(program);

  ProgramAugmentedModel model(
      model_config(1), std::optional<ProgramBranch>{std::move(branch)});
  require(retained_program->direct_to_calls() == 0,
          "composite construction must not call child to() directly");
  require(retained_program->prepare_calls() == 2 &&
              retained_program->commit_calls() == 2,
          "core and model construction must each complete one tree transfer");

  model.to(ExecutionBackend::Cpu);
  require(retained_program->direct_to_calls() == 0,
          "model transfer must stay inside the parent transaction");
  require(retained_program->prepare_calls() == 3 &&
              retained_program->commit_calls() == 3,
          "model transfer must prepare and commit child transfer-only state");
  require(!contains_parameter(model.parameters(),
                              "program.program.coefficients"),
          "transfer-only frozen state must not become an Adam parameter");
}

} // namespace

int main() {
  try {
    test_config_and_span_validation();
    test_learned_only_forward_and_schema();
    test_learned_attention_batch_roll();
    test_program_trace_placement_and_interventions();
    test_trainable_program_backward_reaches_every_branch();
    test_zero_arity_program_and_backend_transfer();
    test_composite_transfer_uses_one_tree_transaction();
    std::cout << "program-augmented model tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "program-augmented model test failure: " << error.what()
              << '\n';
    return 1;
  }
}
