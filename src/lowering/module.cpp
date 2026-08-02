#include "lowering/detail/module_factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace riftco_transformer::lowering {
namespace {

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right,
                                           const char *quantity) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw LoweringError(std::string(quantity) + " exceeds size_t");
  }
  return left * right;
}

[[nodiscard]] std::size_t
checked_product(std::span<const std::size_t> dimensions, const char *quantity) {
  std::size_t result = 1;
  for (const std::size_t dimension : dimensions) {
    if (dimension == 0) {
      throw LoweringError(std::string(quantity) + " contains a zero dimension");
    }
    result = checked_multiply(result, dimension, quantity);
  }
  return result;
}

[[nodiscard]] Tensor::Shape
logical_coefficient_shape(const compiler::cajal::MultilinearMap &map) {
  Tensor::Shape result;
  result.reserve(map.arity() + 1);
  result.push_back(map.output_dimension());
  result.insert(result.end(), map.input_dimensions().begin(),
                map.input_dimensions().end());
  return result;
}

struct ConvertedCoefficients {
  std::vector<float> values;
  double maximum_error = 0.0;
};

[[nodiscard]] ConvertedCoefficients
convert_coefficients(const compiler::cajal::MultilinearMap &map,
                     const NeuralLoweringConfig &config) {
  if (map.coefficient_count() > config.max_coefficient_elements) {
    throw LoweringError(
        "multilinear map exceeds the neural lowering coefficient limit");
  }

  ConvertedCoefficients result;
  result.values.reserve(map.coefficient_count());
  if (config.initialization == CoefficientInitialization::RandomUniform) {
    std::mt19937 random(config.seed);
    std::uniform_real_distribution<float> unit_distribution(-1.0F, 1.0F);
    for (std::size_t index = 0; index < map.coefficient_count(); ++index) {
      const float coefficient = unit_distribution(random) * config.random_scale;
      if (!std::isfinite(coefficient)) {
        throw LoweringError(
            "random coefficient initialization produced a non-finite value");
      }
      result.values.push_back(coefficient);
    }
    return result;
  }

  for (const double coefficient : map.coefficients()) {
    if (!std::isfinite(coefficient)) {
      throw LoweringError("multilinear map contains a non-finite coefficient");
    }
    const float converted = static_cast<float>(coefficient);
    if (!std::isfinite(converted)) {
      throw LoweringError(
          "multilinear map coefficient is outside the float32 range");
    }
    const double restored = static_cast<double>(converted);
    const double error = std::abs(coefficient - restored);
    result.maximum_error = std::max(result.maximum_error, error);
    if (config.precision == CoefficientPrecision::RequireExactFloat32 &&
        restored != coefficient) {
      throw LoweringError(
          "multilinear map coefficient is not exactly representable as "
          "float32");
    }
    result.values.push_back(converted);
  }

  return result;
}

[[nodiscard]] std::string semantic_role(std::string_view strategy) {
  if (strategy == kDenseContractionStrategy) {
    return "dense_contraction";
  }
  if (strategy == kLinearStrategy) {
    return "linear_weight";
  }
  if (strategy == kLinearAttentionStrategy) {
    return "identity_kernel_linear_attention";
  }
  throw LoweringError("cannot materialize unknown built-in strategy: " +
                      std::string(strategy));
}

struct InputLayout {
  Tensor::Shape leading_shape;
  std::size_t row_count = 1;
};

class BuiltinMultilinearModule final : public LoweredMultilinearModule {
public:
  BuiltinMultilinearModule(LoweringMetadata metadata, Tensor coefficient_tensor)
      : metadata_(std::move(metadata)),
        input_combination_count_(checked_product(metadata_.input_dimensions,
                                                 "multilinear input dimension "
                                                 "product")) {
    if (coefficient_tensor.shape() != metadata_.logical_coefficient_shape) {
      throw LoweringError(
          "materialized coefficient tensor has an unexpected shape");
    }
    if (metadata_.trainable) {
      trainable_coefficients_.emplace(std::move(coefficient_tensor));
      register_parameter("coefficients", *trainable_coefficients_);
    } else {
      frozen_coefficients_.emplace(std::move(coefficient_tensor), false);
    }
  }

  [[nodiscard]] Variable
  forward(std::span<const Variable> inputs) const override {
    if (metadata_.input_dimensions.empty()) {
      if (!inputs.empty()) {
        throw LoweringError("constant lowering requires zero inputs");
      }
      return forward_constant({});
    }

    const InputLayout layout = checked_input_layout(inputs);
    if (metadata_.selected_strategy == kDenseContractionStrategy) {
      return forward_dense(inputs, layout);
    }
    if (metadata_.selected_strategy == kLinearStrategy) {
      if (inputs.size() != 1) {
        throw LoweringError("linear lowering requires exactly one input");
      }
      return forward_linear(inputs.front(), layout);
    }
    if (metadata_.selected_strategy == kLinearAttentionStrategy) {
      if (inputs.size() != 2) {
        throw LoweringError(
            "linear-attention lowering requires exactly two inputs");
      }
      return forward_linear_attention(inputs, layout);
    }
    throw LoweringError("lowered module has an unknown strategy");
  }

  [[nodiscard]] Variable
  forward_constant(Tensor::Shape leading_shape) const override {
    if (!metadata_.input_dimensions.empty()) {
      throw LoweringError(
          "forward_constant requires a zero-arity multilinear map");
    }
    static_cast<void>(checked_product(leading_shape, "constant leading shape"));

    Variable constant = coefficient_variable();
    if (leading_shape.empty()) {
      return constant;
    }
    leading_shape.push_back(metadata_.output_dimension);
    return broadcast_to(constant, std::move(leading_shape));
  }

  [[nodiscard]] const LoweringMetadata &metadata() const noexcept override {
    return metadata_;
  }

  [[nodiscard]] const Tensor &
  named_tensor(std::string_view name) const override {
    if (name != "coefficients") {
      throw LoweringError("lowered module has no tensor named '" +
                          std::string(name) + "'");
    }
    return coefficient_tensor();
  }

  [[nodiscard]] ExecutionBackend backend() const noexcept override {
    return coefficient_tensor().backend();
  }

  void to(ExecutionBackend target_backend) override {
    Module::to(target_backend);
  }

private:
  class PreparedFrozenCoefficientTransfer final
      : public PreparedBackendTransfer {
  public:
    PreparedFrozenCoefficientTransfer(Variable &destination,
                                      Variable transferred)
        : destination_(&destination), transferred_(std::move(transferred)) {}

    void commit() noexcept override {
      *destination_ = std::move(transferred_);
    }

  private:
    Variable *destination_;
    Variable transferred_;
  };

  [[nodiscard]] PreparedBackendTransferList
  prepare_extra_backend_transfers(ExecutionBackend target_backend) override {
    PreparedBackendTransferList result;
    if (!metadata_.trainable && target_backend != backend()) {
      result.push_back(
          std::make_unique<PreparedFrozenCoefficientTransfer>(
              *frozen_coefficients_,
              Variable(frozen_coefficients_->value().to(target_backend),
                       false)));
    }
    return result;
  }

  [[nodiscard]] const Tensor &coefficient_tensor() const noexcept {
    return metadata_.trainable ? trainable_coefficients_->value()
                               : frozen_coefficients_->value();
  }

  [[nodiscard]] InputLayout
  checked_input_layout(std::span<const Variable> inputs) const {
    if (inputs.size() != metadata_.input_dimensions.size()) {
      throw LoweringError("lowered multilinear module received " +
                          std::to_string(inputs.size()) + " inputs, expected " +
                          std::to_string(metadata_.input_dimensions.size()));
    }

    InputLayout result;
    for (std::size_t input_index = 0; input_index < inputs.size();
         ++input_index) {
      const Tensor &input = inputs[input_index].value();
      if (input.rank() == 0) {
        throw LoweringError("lowered multilinear input " +
                            std::to_string(input_index) +
                            " must have a final coordinate axis");
      }
      if (input.shape().back() != metadata_.input_dimensions[input_index]) {
        throw LoweringError(
            "lowered multilinear input " + std::to_string(input_index) +
            " has coordinate dimension " +
            std::to_string(input.shape().back()) + ", expected " +
            std::to_string(metadata_.input_dimensions[input_index]));
      }
      if (input.backend() != backend()) {
        throw LoweringError("lowered multilinear input " +
                            std::to_string(input_index) +
                            " and module tensors must share a backend");
      }

      Tensor::Shape leading(input.shape().begin(), input.shape().end() - 1);
      if (input_index == 0) {
        result.leading_shape = std::move(leading);
      } else if (leading != result.leading_shape) {
        throw LoweringError("lowered multilinear input " +
                            std::to_string(input_index) +
                            " has a different leading shape");
      }
    }
    result.row_count =
        checked_product(result.leading_shape, "multilinear leading shape");
    return result;
  }

  [[nodiscard]] Tensor::Shape output_shape(const InputLayout &layout) const {
    Tensor::Shape result = layout.leading_shape;
    result.push_back(metadata_.output_dimension);
    return result;
  }

  [[nodiscard]] Variable coefficient_variable() const {
    if (metadata_.trainable) {
      return trainable_coefficients_->variable();
    }
    return *frozen_coefficients_;
  }

  [[nodiscard]] Variable project_features(const Variable &features,
                                          std::size_t feature_width,
                                          const InputLayout &layout) const {
    if (feature_width != input_combination_count_) {
      throw LoweringError(
          "lowered feature width does not match the multilinear map");
    }
    const Variable coefficient_matrix =
        reshape(coefficient_variable(),
                {metadata_.output_dimension, input_combination_count_});
    const Variable projected =
        matmul(features, transpose_2d(coefficient_matrix));
    return reshape(projected, output_shape(layout));
  }

  [[nodiscard]] Variable forward_dense(std::span<const Variable> inputs,
                                       const InputLayout &layout) const {
    std::size_t feature_width = metadata_.input_dimensions.front();
    Variable features =
        reshape(inputs.front(), {layout.row_count, feature_width});

    for (std::size_t input_index = 1; input_index < inputs.size();
         ++input_index) {
      const std::size_t next_width = metadata_.input_dimensions[input_index];
      const std::size_t combined_width = checked_multiply(
          feature_width, next_width, "dense Kronecker feature width");
      const Tensor::Shape broadcast_shape{layout.row_count, feature_width,
                                          next_width};
      const Variable left =
          reshape(features, {layout.row_count, feature_width, std::size_t{1}});
      const Variable right = reshape(
          inputs[input_index], {layout.row_count, std::size_t{1}, next_width});
      features = reshape(broadcast_to(left, broadcast_shape) *
                             broadcast_to(right, broadcast_shape),
                         {layout.row_count, combined_width});
      feature_width = combined_width;
    }
    return project_features(features, feature_width, layout);
  }

  [[nodiscard]] Variable forward_linear(const Variable &input,
                                        const InputLayout &layout) const {
    const std::size_t input_width = metadata_.input_dimensions.front();
    const Variable flattened = reshape(input, {layout.row_count, input_width});
    return project_features(flattened, input_width, layout);
  }

  [[nodiscard]] Variable
  forward_linear_attention(std::span<const Variable> inputs,
                           const InputLayout &layout) const {
    const std::size_t query_axis =
        metadata_.attention_query_axis.value_or(std::size_t{1});
    if (query_axis >= 2) {
      throw LoweringError("linear-attention query axis is out of range");
    }
    const std::size_t generator_axis = 1 - query_axis;
    const std::size_t query_width = metadata_.input_dimensions[query_axis];
    const std::size_t generator_width =
        metadata_.input_dimensions[generator_axis];
    const std::size_t matrix_width =
        checked_multiply(metadata_.output_dimension, query_width,
                         "linear-attention dynamic matrix width");

    // Coefficients are [output, input_0, input_1]. Moving the generator
    // coordinate first exposes an [generator, output * query] projection.
    const Variable coefficient_projection =
        reshape(permute(coefficient_variable(),
                        {generator_axis + 1, std::size_t{0}, query_axis + 1}),
                {generator_width, matrix_width});
    const Variable generator =
        reshape(inputs[generator_axis], {layout.row_count, generator_width});
    const Variable dynamic_matrix =
        reshape(matmul(generator, coefficient_projection),
                {layout.row_count, metadata_.output_dimension, query_width});
    const Variable query = reshape(
        inputs[query_axis], {layout.row_count, query_width, std::size_t{1}});
    const Variable projected = matmul(dynamic_matrix, query);
    return reshape(projected, output_shape(layout));
  }

  LoweringMetadata metadata_;
  std::size_t input_combination_count_;
  std::optional<Variable> frozen_coefficients_;
  std::optional<Parameter> trainable_coefficients_;
};

static_assert(std::is_nothrow_move_assignable_v<Variable>,
              "frozen backend-transfer commit must not throw");

void validate_analysis(const compiler::cajal::MultilinearMap &map,
                       const NeuralLoweringConfig &config,
                       const LoweringAnalysis &analysis) {
  if (!analysis.supported) {
    throw LoweringError(
        "cannot materialize an unsupported neural lowering analysis");
  }
  if (analysis.requested_strategy != config.strategy) {
    throw LoweringError(
        "neural lowering analysis does not match the requested strategy");
  }
  if (analysis.logical_coefficient_elements != map.coefficient_count()) {
    throw LoweringError(
        "neural lowering analysis coefficient count is inconsistent");
  }
  if (analysis.selected_strategy == kDenseContractionStrategy) {
    return;
  }
  if (analysis.selected_strategy == kLinearStrategy) {
    if (map.arity() != 1) {
      throw LoweringError("linear lowering requires a unary map");
    }
    return;
  }
  if (analysis.selected_strategy == kLinearAttentionStrategy) {
    if (map.arity() != 2) {
      throw LoweringError("linear-attention lowering requires a bilinear map");
    }
    return;
  }
  throw LoweringError("analysis selected an unknown built-in strategy: " +
                      analysis.selected_strategy);
}

} // namespace

namespace detail {

std::unique_ptr<LoweredMultilinearModule>
make_builtin_module(const compiler::cajal::MultilinearMap &map,
                    const NeuralLoweringConfig &config,
                    const LoweringAnalysis &analysis) {
  config.validate();
  validate_analysis(map, config, analysis);

  const Tensor::Shape shape = logical_coefficient_shape(map);
  const std::size_t input_elements = checked_product(
      map.input_dimensions(), "multilinear input dimension product");
  const std::size_t expected_elements =
      checked_multiply(map.output_dimension(), input_elements,
                       "multilinear coefficient element count");
  if (expected_elements != map.coefficient_count()) {
    throw LoweringError(
        "multilinear coefficient count does not match its dimensions");
  }

  ConvertedCoefficients converted = convert_coefficients(map, config);
  const std::optional<std::size_t> query_axis =
      analysis.selected_strategy == kLinearAttentionStrategy
          ? std::optional<std::size_t>(
                config.attention_query_axis.value_or(std::size_t{1}))
          : std::nullopt;
  LoweringMetadata metadata;
  metadata.requested_strategy = analysis.requested_strategy;
  metadata.selected_strategy = analysis.selected_strategy;
  metadata.used_fallback = analysis.used_fallback;
  metadata.preserves_compiled_map =
      analysis.exact &&
      config.initialization == CoefficientInitialization::Compiled &&
      converted.maximum_error == 0.0;
  metadata.reason = analysis.reason;
  metadata.input_dimensions.assign(map.input_dimensions().begin(),
                                   map.input_dimensions().end());
  metadata.output_dimension = map.output_dimension();
  metadata.logical_coefficient_shape = shape;
  metadata.tensors = {{"coefficients",
                       semantic_role(analysis.selected_strategy), shape,
                       config.trainable}};
  metadata.precision = config.precision;
  metadata.initialization = config.initialization;
  metadata.trainable = config.trainable;
  metadata.maximum_float_conversion_error = converted.maximum_error;
  metadata.attention_query_axis = query_axis;

  Tensor coefficients(shape, std::move(converted.values), config.backend);
  return std::make_unique<BuiltinMultilinearModule>(std::move(metadata),
                                                    std::move(coefficients));
}

} // namespace detail
} // namespace riftco_transformer::lowering
