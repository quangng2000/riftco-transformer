#include "riftco_transformer/experiments/conditional_reverse/program.hpp"
#include "riftco_transformer/lowering/cajal.hpp"
#include "riftco_transformer/programmed/sequence_placement.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace cajal = riftco_transformer::compiler::cajal;
namespace conditional_reverse =
    riftco_transformer::experiments::conditional_reverse;
namespace lowering = riftco_transformer::lowering;
namespace programmed = riftco_transformer::programmed;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Function>
void require_throws_as(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  } catch (const std::exception &error) {
    throw std::runtime_error(message + ": wrong exception: " + error.what());
  }
  throw std::runtime_error(message + ": expected an exception");
}

[[nodiscard]] std::vector<std::size_t>
symbol_indices(std::size_t flat, std::size_t symbol_count,
               std::size_t sequence_length) {
  std::vector<std::size_t> result(sequence_length, 0U);
  for (std::size_t index = sequence_length; index > 0U; --index) {
    result[index - 1U] = flat % symbol_count;
    flat /= symbol_count;
  }
  return result;
}

[[nodiscard]] std::size_t checked_power(std::size_t base,
                                        std::size_t exponent) {
  std::size_t result = 1U;
  for (std::size_t index = 0; index < exponent; ++index) {
    require(base == 0U ||
                result <= std::numeric_limits<std::size_t>::max() / base,
            "test cardinality exceeds size_t");
    result *= base;
  }
  return result;
}

[[nodiscard]] cajal::Value
expected_value(conditional_reverse::Condition condition,
               std::size_t symbol_count,
               const std::vector<std::size_t> &symbols) {
  if (condition == conditional_reverse::Condition::Copy) {
    return conditional_reverse::make_sequence_value(symbol_count, symbols);
  }
  std::vector<std::size_t> reversed(symbols.rbegin(), symbols.rend());
  return conditional_reverse::make_sequence_value(symbol_count, reversed);
}

void test_config_and_resource_preflight() {
  const conditional_reverse::ProgramConfig config{
      .sequence_length = 4,
      .symbol_count = 3,
      .max_coefficient_elements = 1U << 20U,
  };
  const conditional_reverse::ResourceMetadata resources =
      conditional_reverse::analyze_resources(config);
  require(
      resources.condition_dimension == 2U && resources.symbol_dimension == 3U &&
          resources.sequence_dimension == 12U &&
          resources.output_dimension == 12U &&
          resources.input_dimensions == std::array<std::size_t, 2>{2U, 12U} &&
          resources.coefficient_elements == 288U &&
          resources.nonzero_coefficient_elements == 24U &&
          resources.dense_coefficient_bytes == 288U * sizeof(double),
      "conditional-reverse resource metadata");

  auto invalid = config;
  invalid.sequence_length = 0U;
  require_throws_as<conditional_reverse::ProgramError>(
      [&] {
        static_cast<void>(conditional_reverse::analyze_resources(invalid));
      },
      "zero sequence length must be rejected");
  invalid = config;
  invalid.symbol_count = 0U;
  require_throws_as<conditional_reverse::ProgramError>(
      [&] { static_cast<void>(conditional_reverse::compile_program(invalid)); },
      "zero symbol count must be rejected");
  invalid = config;
  invalid.max_coefficient_elements = 287U;
  require_throws_as<conditional_reverse::ProgramError>(
      [&] { static_cast<void>(conditional_reverse::compile_program(invalid)); },
      "coefficient limit must reject before compilation");
  invalid = config;
  invalid.sequence_length = std::numeric_limits<std::size_t>::max();
  invalid.max_coefficient_elements = std::numeric_limits<std::size_t>::max();
  require_throws_as<conditional_reverse::ProgramError>(
      [&] {
        static_cast<void>(conditional_reverse::analyze_resources(invalid));
      },
      "resource dimension overflow must be rejected");
}

void test_finite_types_and_values() {
  const cajal::Type condition_type = conditional_reverse::make_condition_type();
  require(condition_type.kind() == cajal::Type::Kind::Sum &&
              condition_type.dimension() == 2U,
          "condition is a two-way unit sum");
  require(cajal::encode(conditional_reverse::make_condition_value(
              conditional_reverse::Condition::Reverse)) ==
                  cajal::EncodedValue({1.0, 0.0}) &&
              cajal::encode(conditional_reverse::make_condition_value(
                  conditional_reverse::Condition::Copy)) ==
                  cajal::EncodedValue({0.0, 1.0}),
          "condition injection coordinates");

  const cajal::Type symbol_type = conditional_reverse::make_symbol_type(3U);
  require(symbol_type.dimension() == 3U &&
              symbol_type.kind() == cajal::Type::Kind::Sum &&
              symbol_type.right().kind() == cajal::Type::Kind::Sum,
          "three-way symbol is a right-associated unit sum");
  for (std::size_t symbol = 0; symbol < 3U; ++symbol) {
    std::vector<double> expected(3U, 0.0);
    expected[symbol] = 1.0;
    const cajal::Value value =
        conditional_reverse::make_symbol_value(3U, symbol);
    require(value.type() == symbol_type &&
                cajal::encode(value) == cajal::EncodedValue(expected),
            "finite symbol value at index " + std::to_string(symbol));
  }

  const std::array<std::size_t, 3> symbols{0U, 2U, 1U};
  const cajal::Value sequence =
      conditional_reverse::make_sequence_value(3U, symbols);
  const cajal::Type sequence_type =
      conditional_reverse::make_sequence_type(3U, symbols.size());
  require(sequence.type() == sequence_type && sequence_type.dimension() == 9U,
          "finite sequence value and type agree");
  require(sequence_type.kind() == cajal::Type::Kind::Product &&
              sequence_type.first() == symbol_type &&
              sequence_type.second().kind() == cajal::Type::Kind::Product &&
              sequence_type.second().first() == symbol_type &&
              sequence_type.second().second() == symbol_type,
          "finite sequence type is a right-associated product");
  require(
      cajal::encode(sequence) ==
          cajal::EncodedValue({1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0}),
      "finite sequence coordinates concatenate symbol blocks");

  require_throws_as<conditional_reverse::ProgramError>(
      [] { static_cast<void>(conditional_reverse::make_symbol_value(3U, 3U)); },
      "out-of-range symbol value must be rejected");
  require_throws_as<conditional_reverse::ProgramError>(
      [] {
        static_cast<void>(conditional_reverse::make_sequence_value(
            2U, std::span<const std::size_t>{}));
      },
      "empty sequence value must be rejected");
  require_throws_as<conditional_reverse::ProgramError>(
      [] {
        static_cast<void>(conditional_reverse::make_condition_value(
            static_cast<conditional_reverse::Condition>(UINT8_C(255))));
      },
      "unknown condition value must be rejected");
}

void test_compiled_program_structure_and_coefficients() {
  const conditional_reverse::ProgramConfig config{
      .sequence_length = 3,
      .symbol_count = 2,
      .max_coefficient_elements = 1024U,
  };
  const conditional_reverse::Program program =
      conditional_reverse::compile_program(config);
  require(program.input_context.size() == 2U &&
              program.input_context[0].name ==
                  conditional_reverse::kConditionInputName &&
              program.input_context[1].name ==
                  conditional_reverse::kSequenceInputName &&
              program.input_context[0].type == program.condition_type &&
              program.input_context[1].type == program.sequence_type,
          "compiled program preserves condition/sequence context order");
  require(program.compiled.output_type() == program.sequence_type &&
              program.compiled.map().arity() == 2U &&
              program.compiled.map().input_dimensions()[0] == 2U &&
              program.compiled.map().input_dimensions()[1] == 6U &&
              program.compiled.map().output_dimension() == 6U &&
              program.compiled.map().coefficient_count() == 72U,
          "compiled conditional-reverse map shape");

  std::size_t nonzero = 0U;
  for (const double coefficient : program.compiled.map().coefficients()) {
    require(coefficient == 0.0 || coefficient == 1.0,
            "compiled coefficients remain exactly representable");
    nonzero += coefficient != 0.0 ? 1U : 0U;
  }
  require(nonzero == 12U &&
              nonzero == program.resources.nonzero_coefficient_elements,
          "compiled conditional-reverse coefficient sparsity");
}

void test_exhaustive_discrete_semantics() {
  const conditional_reverse::ProgramConfig config{
      .sequence_length = 3,
      .symbol_count = 3,
      .max_coefficient_elements = 4096U,
  };
  const conditional_reverse::Program program =
      conditional_reverse::compile_program(config);
  const std::size_t sequence_count =
      checked_power(config.symbol_count, config.sequence_length);

  for (const conditional_reverse::Condition condition : {
           conditional_reverse::Condition::Reverse,
           conditional_reverse::Condition::Copy,
       }) {
    for (std::size_t flat = 0; flat < sequence_count; ++flat) {
      const std::vector<std::size_t> symbols =
          symbol_indices(flat, config.symbol_count, config.sequence_length);
      const cajal::Value sequence = conditional_reverse::make_sequence_value(
          config.symbol_count, symbols);
      const cajal::Environment environment{
          {conditional_reverse::kConditionInputName,
           conditional_reverse::make_condition_value(condition)},
          {conditional_reverse::kSequenceInputName, sequence},
      };
      const cajal::Value expected =
          expected_value(condition, config.symbol_count, symbols);
      const cajal::Value interpreted =
          cajal::evaluate(program.expression, environment);
      require(interpreted == expected,
              "conditional-reverse interpreter result on discrete input");

      const cajal::EncodedValue compiled = program.compiled.apply(environment);
      require(compiled == cajal::encode(expected) &&
                  cajal::decode(program.sequence_type, compiled) == expected,
              "conditional-reverse compiler result on discrete input");
    }
  }
}

void test_exhaustive_binary_diagonal_and_off_diagonal_semantics() {
  const conditional_reverse::ProgramConfig config{
      .sequence_length = 4,
      .symbol_count = 2,
      .max_coefficient_elements = 4096U,
  };
  const conditional_reverse::Program program =
      conditional_reverse::compile_program(config);
  const std::size_t sequence_count =
      checked_power(config.symbol_count, config.sequence_length);

  for (std::size_t flat = 0; flat < sequence_count; ++flat) {
    const std::vector<std::size_t> symbols =
        symbol_indices(flat, config.symbol_count, config.sequence_length);
    const cajal::Value sequence =
        conditional_reverse::make_sequence_value(config.symbol_count, symbols);
    const conditional_reverse::Condition diagonal_condition =
        symbols.front() == 0U ? conditional_reverse::Condition::Reverse
                              : conditional_reverse::Condition::Copy;
    const conditional_reverse::Condition off_diagonal_condition =
        diagonal_condition == conditional_reverse::Condition::Reverse
            ? conditional_reverse::Condition::Copy
            : conditional_reverse::Condition::Reverse;

    for (const conditional_reverse::Condition condition : {
             diagonal_condition,
             off_diagonal_condition,
         }) {
      const cajal::Environment environment{
          {conditional_reverse::kConditionInputName,
           conditional_reverse::make_condition_value(condition)},
          {conditional_reverse::kSequenceInputName, sequence},
      };
      const cajal::Value expected =
          expected_value(condition, config.symbol_count, symbols);
      require(cajal::evaluate(program.expression, environment) == expected &&
                  program.compiled.apply(environment) ==
                      cajal::encode(expected),
              condition == diagonal_condition
                  ? "diagonal condition/first-symbol semantics"
                  : "off-diagonal condition independently controls branch");
    }
  }
}

void test_off_basis_bilinear_semantics() {
  const conditional_reverse::Program program =
      conditional_reverse::compile_program({
          .sequence_length = 3,
          .symbol_count = 2,
          .max_coefficient_elements = 1024U,
      });
  const cajal::EncodedValue condition({2.0, 3.0});
  const cajal::EncodedValue sequence({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  const cajal::EncodedValue actual = program.compiled.map().apply(
      std::array<cajal::EncodedValue, 2>{condition, sequence});

  // The algebraic extension is 2 * reverse(sequence) + 3 * sequence.
  const cajal::EncodedValue expected({13.0, 18.0, 15.0, 20.0, 17.0, 22.0});
  require(cajal::approximately_equal(actual, expected),
          "off-basis condition linearly combines reverse and copy branches");
}

void test_reverse_program_resources_lowering_and_semantics() {
  const conditional_reverse::ProgramConfig config{
      .sequence_length = 3,
      .symbol_count = 2,
      .max_coefficient_elements = 1024U,
  };
  const conditional_reverse::ReverseResourceMetadata resources =
      conditional_reverse::analyze_reverse_resources(config);
  require(resources.symbol_dimension == 2U &&
              resources.sequence_dimension == 6U &&
              resources.output_dimension == 6U &&
              resources.input_dimensions == std::array<std::size_t, 1>{6U} &&
              resources.coefficient_elements == 36U &&
              resources.nonzero_coefficient_elements == 6U &&
              resources.dense_coefficient_bytes == 36U * sizeof(double),
          "unary reverse resource metadata");

  auto limited = config;
  limited.max_coefficient_elements = resources.coefficient_elements - 1U;
  require_throws_as<conditional_reverse::ProgramError>(
      [&] {
        static_cast<void>(
            conditional_reverse::analyze_reverse_resources(limited));
      },
      "unary reverse resource limit");

  const conditional_reverse::ReverseProgram program =
      conditional_reverse::compile_reverse_program(config);
  require(program.input_context.size() == 1U &&
              program.input_context.front().name ==
                  conditional_reverse::kSequenceInputName &&
              program.compiled.map().input_dimensions().size() == 1U &&
              program.compiled.map().input_dimensions()[0] == 6U,
          "unary reverse program structure");
  const lowering::LoweringAnalysis lowering_analysis =
      lowering::analyze_neural_lowering(program.compiled);
  require(lowering_analysis.supported &&
              lowering_analysis.selected_strategy ==
                  lowering::kLinearStrategy &&
              lowering_analysis.exact,
          "unary reverse selects exact linear lowering");

  const std::size_t sequence_count =
      checked_power(config.symbol_count, config.sequence_length);
  for (std::size_t flat = 0; flat < sequence_count; ++flat) {
    const std::vector<std::size_t> symbols =
        symbol_indices(flat, config.symbol_count, config.sequence_length);
    const cajal::Value sequence =
        conditional_reverse::make_sequence_value(config.symbol_count, symbols);
    const std::vector<std::size_t> reversed(symbols.rbegin(), symbols.rend());
    const cajal::Value expected =
        conditional_reverse::make_sequence_value(config.symbol_count, reversed);
    const cajal::Environment environment{
        {conditional_reverse::kSequenceInputName, sequence},
    };
    require(cajal::evaluate(program.expression, environment) == expected &&
                program.compiled.apply(environment) == cajal::encode(expected),
            "unary reverse exhaustive discrete semantics");
  }

  const cajal::EncodedValue off_basis({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  require(program.compiled.map().apply(
              std::array<cajal::EncodedValue, 1>{off_basis}) ==
              cajal::EncodedValue({5.0, 6.0, 3.0, 4.0, 1.0, 2.0}),
          "unary reverse off-basis linear semantics");
}

void test_two_sequence_program_resources_and_exhaustive_semantics() {
  const conditional_reverse::ProgramConfig config{
      .sequence_length = 2,
      .symbol_count = 3,
      .max_coefficient_elements = 1024U,
  };
  const conditional_reverse::TwoSequenceResourceMetadata resources =
      conditional_reverse::analyze_two_sequence_resources(config);
  require(
      resources.symbol_dimension == 3U && resources.sequence_dimension == 6U &&
          resources.output_dimension == 6U &&
          resources.input_dimensions == std::array<std::size_t, 2>{6U, 6U} &&
          resources.coefficient_elements == 216U &&
          resources.nonzero_coefficient_elements == 18U &&
          resources.dense_coefficient_bytes == 216U * sizeof(double),
      "two-sequence conditional resource metadata");

  auto limited = config;
  limited.max_coefficient_elements = resources.coefficient_elements - 1U;
  require_throws_as<conditional_reverse::ProgramError>(
      [&] {
        static_cast<void>(
            conditional_reverse::analyze_two_sequence_resources(limited));
      },
      "two-sequence conditional resource limit");

  const conditional_reverse::TwoSequenceProgram program =
      conditional_reverse::compile_two_sequence_program(config);
  require(program.input_context.size() == 2U &&
              program.input_context[0].name ==
                  conditional_reverse::kFirstSequenceInputName &&
              program.input_context[1].name ==
                  conditional_reverse::kSecondSequenceInputName &&
              program.compiled.map().input_dimensions().size() == 2U &&
              program.compiled.map().input_dimensions()[0] == 6U &&
              program.compiled.map().input_dimensions()[1] == 6U,
          "two-sequence conditional context and map structure");
  const lowering::LoweringAnalysis lowering_analysis =
      lowering::analyze_neural_lowering(program.compiled);
  require(lowering_analysis.supported &&
              lowering_analysis.selected_strategy ==
                  lowering::kLinearAttentionStrategy &&
              lowering_analysis.exact,
          "two-sequence conditional selects exact attention lowering");

  const std::size_t sequence_count =
      checked_power(config.symbol_count, config.sequence_length);
  for (std::size_t first_flat = 0; first_flat < sequence_count; ++first_flat) {
    const std::vector<std::size_t> first_symbols =
        symbol_indices(first_flat, config.symbol_count, config.sequence_length);
    const cajal::Value first = conditional_reverse::make_sequence_value(
        config.symbol_count, first_symbols);
    for (std::size_t second_flat = 0; second_flat < sequence_count;
         ++second_flat) {
      const std::vector<std::size_t> second_symbols = symbol_indices(
          second_flat, config.symbol_count, config.sequence_length);
      const cajal::Value second = conditional_reverse::make_sequence_value(
          config.symbol_count, second_symbols);
      std::vector<std::size_t> expected_symbols = second_symbols;
      if (first_symbols.front() == 0U) {
        std::reverse(expected_symbols.begin(), expected_symbols.end());
      }
      const cajal::Value expected = conditional_reverse::make_sequence_value(
          config.symbol_count, expected_symbols);
      const cajal::Environment environment{
          {conditional_reverse::kFirstSequenceInputName, first},
          {conditional_reverse::kSecondSequenceInputName, second},
      };
      require(cajal::evaluate(program.expression, environment) == expected &&
                  program.compiled.apply(environment) ==
                      cajal::encode(expected),
              "two-sequence conditional exhaustive discrete semantics");
    }
  }

  const cajal::EncodedValue selector({2.0, 3.0, 5.0, 7.0, 11.0, 13.0});
  const cajal::EncodedValue sequence({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  const cajal::EncodedValue off_basis = program.compiled.map().apply(
      std::array<cajal::EncodedValue, 2>{selector, sequence});
  require(
      cajal::approximately_equal(
          off_basis, cajal::EncodedValue({16.0, 26.0, 36.0, 34.0, 44.0, 54.0})),
      "two-sequence off-basis selector uses only first-position "
      "coordinates");

  const conditional_reverse::TwoSequenceProgram single_symbol =
      conditional_reverse::compile_two_sequence_program({
          .sequence_length = 3U,
          .symbol_count = 1U,
          .max_coefficient_elements = 64U,
      });
  const cajal::EncodedValue single_symbol_output =
      single_symbol.compiled.map().apply(std::array<cajal::EncodedValue, 2>{
          cajal::EncodedValue({2.0, 7.0, 11.0}),
          cajal::EncodedValue({1.0, 2.0, 3.0}),
      });
  require(cajal::approximately_equal(single_symbol_output,
                                     cajal::EncodedValue({6.0, 4.0, 2.0})),
          "single-symbol selector always reverses and ignores later control "
          "positions");
}

[[nodiscard]] std::vector<float>
one_hot_sequence_batch(std::size_t symbol_count, std::size_t sequence_length) {
  const std::size_t sequence_count =
      checked_power(symbol_count, sequence_length);
  std::vector<float> values(sequence_count * sequence_length * symbol_count,
                            0.0F);
  for (std::size_t row = 0; row < sequence_count; ++row) {
    const std::vector<std::size_t> symbols =
        symbol_indices(row, symbol_count, sequence_length);
    for (std::size_t position = 0; position < sequence_length; ++position) {
      values[(row * sequence_length + position) * symbol_count +
             symbols[position]] = 1.0F;
    }
  }
  return values;
}

[[nodiscard]] programmed::ProgrammedSequenceCore make_shared_identity_core(
    const conditional_reverse::TwoSequenceProgram &program,
    std::size_t sequence_length, std::size_t symbol_count) {
  lowering::NeuralLoweringConfig lowering_config;
  lowering_config.strategy = lowering::kLinearAttentionStrategy;
  lowering_config.attention_query_axis = 1U;
  auto lowered = lowering::lower_to_neural(program.compiled, lowering_config);
  std::vector<float> identity(symbol_count * symbol_count, 0.0F);
  for (std::size_t index = 0; index < symbol_count; ++index) {
    identity[index * symbol_count + index] = 1.0F;
  }
  programmed::SequenceCoreProjectionState projections{
      {
          {
              riftco_transformer::Tensor({symbol_count, symbol_count},
                                         std::move(identity)),
              std::nullopt,
          },
      },
  };
  return programmed::ProgrammedSequenceCore(
      symbol_count,
      {
          .source_length = sequence_length,
          .output_length = sequence_length,
          .inputs =
              {
                  {.source = programmed::ProgramInputSource::WholeSource},
                  {.source = programmed::ProgramInputSource::WholeSource},
              },
          .input_projection_groups = {0U, 0U},
          .input_projection_bias = false,
      },
      std::move(lowered), std::move(projections));
}

void test_shared_programmed_core_raw_output_interventions_and_gradients() {
  constexpr std::size_t kLength = 3U;
  constexpr std::size_t kSymbols = 2U;
  const conditional_reverse::TwoSequenceProgram program =
      conditional_reverse::compile_two_sequence_program({
          .sequence_length = kLength,
          .symbol_count = kSymbols,
          .max_coefficient_elements = 4096U,
      });
  programmed::ProgrammedSequenceCore core =
      make_shared_identity_core(program, kLength, kSymbols);

  require(core.logical_input_count() == 2U &&
              core.input_projection_count() == 1U &&
              core.input_projection_group(0U) == 0U &&
              core.input_projection_group(1U) == 0U &&
              &core.input_projection_for_input(0U) ==
                  &core.input_projection_for_input(1U) &&
              !core.input_projection(0U).has_bias(),
          "programmed core uses one genuinely shared biasless projection");
  {
    const riftco_transformer::ParameterList parameters = core.parameters();
    require(parameters.size() == 1U &&
                parameters.front().name == "input_projections.0.weight",
            "shared biasless core registers one weight and no duplicate or "
            "bias parameter");
  }

  const std::size_t batch = checked_power(kSymbols, kLength);
  riftco_transformer::Variable source(
      riftco_transformer::Tensor({batch, kLength, kSymbols},
                                 one_hot_sequence_batch(kSymbols, kLength)),
      true);
  programmed::SequenceForwardOptions options;
  options.capture_representations = true;
  programmed::SequenceCoreResult result = core.forward(source, options);
  require(result.output.value().shape() ==
                  riftco_transformer::Tensor::Shape{batch, kLength, kSymbols} &&
              result.representations.contains("source") &&
              result.representations.contains("program_input.0.projected") &&
              result.representations.contains("program_input.1") &&
              result.representations.contains("program_output"),
          "programmed core returns and captures raw program-basis output");

  const std::span<const float> actual = result.output.value().data();
  for (std::size_t row = 0; row < batch; ++row) {
    const std::vector<std::size_t> symbols =
        symbol_indices(row, kSymbols, kLength);
    std::vector<std::size_t> expected = symbols;
    if (symbols.front() == 0U) {
      std::reverse(expected.begin(), expected.end());
    }
    for (std::size_t position = 0; position < kLength; ++position) {
      for (std::size_t symbol = 0; symbol < kSymbols; ++symbol) {
        const float expected_value = symbol == expected[position] ? 1.0F : 0.0F;
        const std::size_t flat = (row * kLength + position) * kSymbols + symbol;
        require(actual[flat] == expected_value,
                "programmed core exhaustive raw-output semantics");
      }
    }
  }

  const std::vector<float> copy_source{0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 0.0F};
  programmed::SequenceForwardOptions force_reverse;
  force_reverse.steering.push_back({
      .input_index = 0U,
      .positions = {0U},
      .scales = {0.0F, 0.0F},
      .offsets = {1.0F, 0.0F},
  });
  const programmed::SequenceCoreResult steered = core.forward(
      riftco_transformer::Variable(
          riftco_transformer::Tensor({1U, kLength, kSymbols}, copy_source),
          false),
      force_reverse);
  const std::vector<float> expected_reversed{1.0F, 0.0F, 1.0F,
                                             0.0F, 0.0F, 1.0F};
  require(std::equal(steered.output.value().data().begin(),
                     steered.output.value().data().end(),
                     expected_reversed.begin()),
          "logical-input steering remains independent above a shared "
          "projection");

  const std::vector<float> ablation_source{
      1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 0.0F,
  };
  const riftco_transformer::Variable ablation_input(
      riftco_transformer::Tensor({2U, kLength, kSymbols}, ablation_source),
      false);
  const programmed::SequenceCoreResult unablated = core.forward(ablation_input);
  programmed::SequenceForwardOptions roll_output;
  roll_output.ablation = programmed::BatchRollAblation{
      .program_input_indices = {},
      .program_output = true,
      .shift = 1U,
  };
  const programmed::SequenceCoreResult ablated =
      core.forward(ablation_input, roll_output);
  const std::size_t row_width = kLength * kSymbols;
  require(std::equal(ablated.output.value().data().begin(),
                     ablated.output.value().data().begin() +
                         static_cast<std::ptrdiff_t>(row_width),
                     unablated.output.value().data().begin() +
                         static_cast<std::ptrdiff_t>(row_width)) &&
              std::equal(ablated.output.value().data().begin() +
                             static_cast<std::ptrdiff_t>(row_width),
                         ablated.output.value().data().end(),
                         unablated.output.value().data().begin()),
          "programmed core applies raw-output batch-roll ablation");

  std::vector<float> seed(result.output.value().numel(), 0.0F);
  for (std::size_t index = 0; index < seed.size(); ++index) {
    seed[index] = static_cast<float>((index % 7U) + 1U) / 7.0F;
  }
  result.output.backward(riftco_transformer::Tensor(
      result.output.value().shape(), std::move(seed)));
  const auto has_nonzero_finite = [](std::span<const float> values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); }) &&
           std::any_of(values.begin(), values.end(),
                       [](float value) { return value != 0.0F; });
  };
  require(has_nonzero_finite(source.gradient().data()),
          "raw programmed core propagates gradients to its source");
  const riftco_transformer::ParameterList parameters = core.parameters();
  require(
      parameters.size() == 1U &&
          has_nonzero_finite(parameters.front().parameter->gradient().data()),
      "both logical program inputs accumulate into one shared projection "
      "gradient");
}

void test_programmed_core_validation_and_backend_resources() {
  const conditional_reverse::TwoSequenceProgram program =
      conditional_reverse::compile_two_sequence_program({
          .sequence_length = 3U,
          .symbol_count = 2U,
          .max_coefficient_elements = 4096U,
      });
  auto make_lowered = [&] {
    lowering::NeuralLoweringConfig config;
    config.strategy = lowering::kLinearAttentionStrategy;
    return lowering::lower_to_neural(program.compiled, config);
  };
  std::mt19937 random(123U);
  require_throws_as<std::invalid_argument>(
      [&] {
        static_cast<void>(programmed::ProgrammedSequenceCore(
            2U,
            {
                .source_length = 3U,
                .output_length = 3U,
                .inputs =
                    {
                        {.source = programmed::ProgramInputSource::WholeSource},
                        {.source = programmed::ProgramInputSource::WholeSource},
                    },
                .input_projection_groups = {0U, 2U},
                .input_projection_bias = false,
            },
            make_lowered(), random));
      },
      "programmed core rejects sparse projection group IDs");

  programmed::SequenceCoreProjectionState biased_state{
      {
          {
              riftco_transformer::Tensor(
                  {2U, 2U}, std::vector<float>{1.0F, 0.0F, 0.0F, 1.0F}),
              riftco_transformer::Tensor::zeros({2U}),
          },
      },
  };
  require_throws_as<std::invalid_argument>(
      [&] {
        static_cast<void>(programmed::ProgrammedSequenceCore(
            2U,
            {
                .source_length = 3U,
                .output_length = 3U,
                .inputs =
                    {
                        {.source = programmed::ProgramInputSource::WholeSource},
                        {.source = programmed::ProgramInputSource::WholeSource},
                    },
                .input_projection_groups = {0U, 0U},
                .input_projection_bias = false,
            },
            make_lowered(), std::move(biased_state)));
      },
      "biasless programmed core state rejects a bias tensor");

  programmed::ProgrammedSequenceCore random_biasless_core(
      2U,
      {
          .source_length = 3U,
          .output_length = 3U,
          .inputs =
              {
                  {.source = programmed::ProgramInputSource::WholeSource},
                  {.source = programmed::ProgramInputSource::WholeSource},
              },
          .input_projection_groups = {0U, 0U},
          .input_projection_bias = false,
      },
      make_lowered(), random);
  {
    const riftco_transformer::ParameterList parameters =
        random_biasless_core.parameters();
    require(parameters.size() == 1U &&
                parameters.front().name == "input_projections.0.weight" &&
                !random_biasless_core.input_projection(0U).has_bias(),
            "random shared biasless core registers exactly one weight");
  }

  if (!riftco_transformer::execution_backend_available(
          riftco_transformer::ExecutionBackend::Metal)) {
    return;
  }
  random_biasless_core.input_projection(0U).quantize_weight_nf4(32U);
  random_biasless_core.to(riftco_transformer::ExecutionBackend::Metal);
  require(random_biasless_core.program().backend() ==
                  riftco_transformer::ExecutionBackend::Metal &&
              random_biasless_core.input_projection(0U)
                      .quantized_weight()
                      .backend() ==
                  riftco_transformer::ExecutionBackend::Metal &&
              &random_biasless_core.input_projection_for_input(0U) ==
                  &random_biasless_core.input_projection_for_input(1U),
          "programmed core transfers each shared packed resource once");
}

} // namespace

int main() {
  try {
    test_config_and_resource_preflight();
    test_finite_types_and_values();
    test_compiled_program_structure_and_coefficients();
    test_exhaustive_discrete_semantics();
    test_exhaustive_binary_diagonal_and_off_diagonal_semantics();
    test_off_basis_bilinear_semantics();
    test_reverse_program_resources_lowering_and_semantics();
    test_two_sequence_program_resources_and_exhaustive_semantics();
    test_shared_programmed_core_raw_output_interventions_and_gradients();
    test_programmed_core_validation_and_backend_resources();
    std::cout << "Conditional-reverse symbolic program tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Conditional-reverse symbolic program test failure: "
              << error.what() << '\n';
    return 1;
  }
}
