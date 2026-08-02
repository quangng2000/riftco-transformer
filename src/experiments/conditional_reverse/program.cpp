#include "riftco_transformer/experiments/conditional_reverse/program.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer::experiments::conditional_reverse {
namespace {

namespace cajal = compiler::cajal;

inline constexpr char kReversePayloadName[] = "reverse_condition_payload";
inline constexpr char kCopyPayloadName[] = "copy_condition_payload";

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right,
                                           const char *quantity) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    throw ProgramError(std::string(quantity) + " exceeds size_t");
  }
  return left * right;
}

[[nodiscard]] cajal::Expression
reverse_sequence_expression(std::string input_name,
                            std::size_t sequence_length) {
  cajal::Expression remaining =
      cajal::Expression::variable(std::move(input_name));
  if (sequence_length == 1U) {
    return remaining;
  }

  std::vector<cajal::Expression> symbols;
  symbols.reserve(sequence_length);
  for (std::size_t index = 0; index + 1U < sequence_length; ++index) {
    symbols.push_back(cajal::Expression::projection(remaining, 0U));
    remaining = cajal::Expression::projection(std::move(remaining), 1U);
  }
  symbols.push_back(std::move(remaining));

  cajal::Expression reversed = symbols.front();
  for (std::size_t index = 1; index < symbols.size(); ++index) {
    reversed = cajal::Expression::tuple(symbols[index], std::move(reversed));
  }
  return reversed;
}

[[nodiscard]] cajal::Expression
make_program_expression(std::size_t sequence_length) {
  cajal::Expression reverse_branch = cajal::Expression::sequence(
      cajal::Expression::variable(kReversePayloadName),
      reverse_sequence_expression(kSequenceInputName, sequence_length));
  cajal::Expression copy_branch = cajal::Expression::sequence(
      cajal::Expression::variable(kCopyPayloadName),
      cajal::Expression::variable(kSequenceInputName));
  return cajal::Expression::case_of(
      cajal::Expression::variable(kConditionInputName), kReversePayloadName,
      std::move(reverse_branch), kCopyPayloadName, std::move(copy_branch));
}

[[nodiscard]] cajal::Expression
consume_enum_then(cajal::Expression value, const cajal::Type &type,
                  cajal::Expression continuation, std::size_t depth) {
  if (type.kind() == cajal::Type::Kind::Unit) {
    return cajal::Expression::sequence(std::move(value),
                                       std::move(continuation));
  }
  if (type.kind() != cajal::Type::Kind::Sum) {
    throw ProgramError(
        "conditional selector payload is not a finite enum type");
  }

  const std::string left_name =
      "discard_selector_left_" + std::to_string(depth);
  const std::string right_name =
      "discard_selector_right_" + std::to_string(depth);
  return cajal::Expression::case_of(
      std::move(value), left_name,
      consume_enum_then(cajal::Expression::variable(left_name), type.left(),
                        continuation, depth + 1U),
      right_name,
      consume_enum_then(cajal::Expression::variable(right_name), type.right(),
                        std::move(continuation), depth + 1U));
}

[[nodiscard]] cajal::Expression
make_two_sequence_program_expression(const cajal::Type &symbol_type,
                                     std::size_t sequence_length) {
  cajal::Expression selector = cajal::Expression::projection(
      cajal::Expression::variable(kFirstSequenceInputName), 0U);
  cajal::Expression reversed =
      reverse_sequence_expression(kSecondSequenceInputName, sequence_length);

  if (symbol_type.kind() == cajal::Type::Kind::Unit) {
    return cajal::Expression::sequence(std::move(selector),
                                       std::move(reversed));
  }
  if (symbol_type.kind() != cajal::Type::Kind::Sum) {
    throw ProgramError("conditional selector is not a finite enum type");
  }

  constexpr char kSelectReversePayloadName[] = "select_reverse_payload";
  constexpr char kSelectCopyPayloadName[] = "select_copy_payload";
  cajal::Expression reverse_branch = cajal::Expression::sequence(
      cajal::Expression::variable(kSelectReversePayloadName),
      std::move(reversed));
  cajal::Expression copy_branch = consume_enum_then(
      cajal::Expression::variable(kSelectCopyPayloadName), symbol_type.right(),
      cajal::Expression::variable(kSecondSequenceInputName), 0U);
  return cajal::Expression::case_of(
      std::move(selector), kSelectReversePayloadName, std::move(reverse_branch),
      kSelectCopyPayloadName, std::move(copy_branch));
}

void validate_compilation(const Program &program) {
  const cajal::MultilinearMap &map = program.compiled.map();
  if (program.compiled.output_type() != program.sequence_type ||
      map.arity() != program.resources.input_dimensions.size() ||
      !std::equal(map.input_dimensions().begin(), map.input_dimensions().end(),
                  program.resources.input_dimensions.begin()) ||
      map.output_dimension() != program.resources.output_dimension ||
      map.coefficient_count() != program.resources.coefficient_elements) {
    throw ProgramError(
        "compiled conditional-reverse map does not match its resource "
        "preflight");
  }

  std::size_t nonzero = 0;
  for (const double coefficient : map.coefficients()) {
    if (coefficient != 0.0 && coefficient != 1.0) {
      throw ProgramError(
          "conditional-reverse compilation produced a coefficient other "
          "than zero or one");
    }
    nonzero += coefficient != 0.0 ? 1U : 0U;
  }
  if (nonzero != program.resources.nonzero_coefficient_elements) {
    throw ProgramError(
        "compiled conditional-reverse map has an unexpected number of "
        "nonzero coefficients");
  }
}

void validate_reverse_compilation(const ReverseProgram &program) {
  const cajal::MultilinearMap &map = program.compiled.map();
  if (program.compiled.output_type() != program.sequence_type ||
      map.arity() != program.resources.input_dimensions.size() ||
      !std::equal(map.input_dimensions().begin(), map.input_dimensions().end(),
                  program.resources.input_dimensions.begin()) ||
      map.output_dimension() != program.resources.output_dimension ||
      map.coefficient_count() != program.resources.coefficient_elements) {
    throw ProgramError(
        "compiled reverse map does not match its resource preflight");
  }

  std::size_t nonzero = 0U;
  for (const double coefficient : map.coefficients()) {
    if (coefficient != 0.0 && coefficient != 1.0) {
      throw ProgramError(
          "reverse compilation produced a coefficient other than zero or one");
    }
    nonzero += coefficient != 0.0 ? 1U : 0U;
  }
  if (nonzero != program.resources.nonzero_coefficient_elements) {
    throw ProgramError("compiled reverse map has an unexpected number of "
                       "nonzero coefficients");
  }
}

void validate_two_sequence_compilation(const TwoSequenceProgram &program) {
  const cajal::MultilinearMap &map = program.compiled.map();
  if (program.compiled.output_type() != program.sequence_type ||
      map.arity() != program.resources.input_dimensions.size() ||
      !std::equal(map.input_dimensions().begin(), map.input_dimensions().end(),
                  program.resources.input_dimensions.begin()) ||
      map.output_dimension() != program.resources.output_dimension ||
      map.coefficient_count() != program.resources.coefficient_elements) {
    throw ProgramError(
        "compiled two-sequence conditional map does not match its resource "
        "preflight");
  }

  std::size_t nonzero = 0U;
  for (const double coefficient : map.coefficients()) {
    if (coefficient != 0.0 && coefficient != 1.0) {
      throw ProgramError(
          "two-sequence conditional compilation produced a coefficient other "
          "than zero or one");
    }
    nonzero += coefficient != 0.0 ? 1U : 0U;
  }
  if (nonzero != program.resources.nonzero_coefficient_elements) {
    throw ProgramError(
        "compiled two-sequence conditional map has an unexpected number of "
        "nonzero coefficients");
  }
}

} // namespace

void ProgramConfig::validate() const {
  if (sequence_length == 0U) {
    throw ProgramError(
        "conditional-reverse sequence length must be greater than zero");
  }
  if (symbol_count == 0U) {
    throw ProgramError(
        "conditional-reverse symbol count must be greater than zero");
  }
  if (max_coefficient_elements == 0U) {
    throw ProgramError(
        "conditional-reverse coefficient limit must be greater than zero");
  }
}

cajal::Type make_condition_type() {
  return cajal::Type::sum(cajal::Type::unit(), cajal::Type::unit());
}

cajal::Type make_symbol_type(std::size_t symbol_count) {
  if (symbol_count == 0U) {
    throw ProgramError("finite symbol count must be greater than zero");
  }

  cajal::Type result = cajal::Type::unit();
  for (std::size_t index = 1; index < symbol_count; ++index) {
    result = cajal::Type::sum(cajal::Type::unit(), std::move(result));
  }
  return result;
}

cajal::Type make_sequence_type(std::size_t symbol_count,
                               std::size_t sequence_length) {
  if (sequence_length == 0U) {
    throw ProgramError("finite sequence length must be greater than zero");
  }

  const cajal::Type symbol_type = make_symbol_type(symbol_count);
  cajal::Type result = symbol_type;
  for (std::size_t index = 1; index < sequence_length; ++index) {
    result = cajal::Type::product(symbol_type, std::move(result));
  }
  return result;
}

cajal::Value make_condition_value(Condition condition) {
  switch (condition) {
  case Condition::Reverse:
    return cajal::Value::inject_left(cajal::Value::unit(), cajal::Type::unit());
  case Condition::Copy:
    return cajal::Value::inject_right(cajal::Type::unit(),
                                      cajal::Value::unit());
  }
  throw ProgramError("conditional-reverse condition is not recognized");
}

cajal::Value make_symbol_value(std::size_t symbol_count,
                               std::size_t symbol_index) {
  if (symbol_count == 0U) {
    throw ProgramError("finite symbol count must be greater than zero");
  }
  if (symbol_index >= symbol_count) {
    throw ProgramError("finite symbol index is out of range");
  }

  cajal::Value result = cajal::Value::unit();
  if (symbol_index + 1U < symbol_count) {
    result = cajal::Value::inject_left(
        cajal::Value::unit(),
        make_symbol_type(symbol_count - symbol_index - 1U));
  }
  for (std::size_t depth = 0; depth < symbol_index; ++depth) {
    result = cajal::Value::inject_right(cajal::Type::unit(), std::move(result));
  }
  return result;
}

cajal::Value make_sequence_value(std::size_t symbol_count,
                                 std::span<const std::size_t> symbol_indices) {
  if (symbol_indices.empty()) {
    throw ProgramError("finite symbol sequence must not be empty");
  }

  cajal::Value result = make_symbol_value(symbol_count, symbol_indices.back());
  for (std::size_t index = symbol_indices.size() - 1U; index > 0U; --index) {
    result = cajal::Value::product(
        make_symbol_value(symbol_count, symbol_indices[index - 1U]),
        std::move(result));
  }
  return result;
}

ResourceMetadata analyze_resources(const ProgramConfig &config) {
  config.validate();

  ResourceMetadata resources;
  resources.symbol_dimension = config.symbol_count;
  resources.sequence_dimension =
      checked_multiply(config.symbol_count, config.sequence_length,
                       "conditional-reverse sequence coordinate dimension");
  resources.output_dimension = resources.sequence_dimension;
  resources.input_dimensions = {resources.condition_dimension,
                                resources.sequence_dimension};
  resources.coefficient_elements = checked_multiply(
      checked_multiply(resources.condition_dimension,
                       resources.sequence_dimension,
                       "conditional-reverse input combination count"),
      resources.output_dimension,
      "conditional-reverse dense coefficient count");
  resources.nonzero_coefficient_elements = checked_multiply(
      resources.condition_dimension, resources.sequence_dimension,
      "conditional-reverse nonzero coefficient count");
  resources.dense_coefficient_bytes =
      checked_multiply(resources.coefficient_elements, sizeof(double),
                       "conditional-reverse dense coefficient storage");

  if (resources.coefficient_elements > config.max_coefficient_elements) {
    throw ProgramError(
        "conditional-reverse program requires " +
        std::to_string(resources.coefficient_elements) +
        " coefficient elements, exceeding the configured limit of " +
        std::to_string(config.max_coefficient_elements));
  }
  return resources;
}

ReverseResourceMetadata analyze_reverse_resources(const ProgramConfig &config) {
  config.validate();

  ReverseResourceMetadata resources;
  resources.symbol_dimension = config.symbol_count;
  resources.sequence_dimension =
      checked_multiply(config.symbol_count, config.sequence_length,
                       "reverse sequence coordinate dimension");
  resources.output_dimension = resources.sequence_dimension;
  resources.input_dimensions = {resources.sequence_dimension};
  resources.coefficient_elements =
      checked_multiply(resources.sequence_dimension, resources.output_dimension,
                       "reverse dense coefficient count");
  resources.nonzero_coefficient_elements = resources.sequence_dimension;
  resources.dense_coefficient_bytes =
      checked_multiply(resources.coefficient_elements, sizeof(double),
                       "reverse dense coefficient storage");

  if (resources.coefficient_elements > config.max_coefficient_elements) {
    throw ProgramError("reverse program requires " +
                       std::to_string(resources.coefficient_elements) +
                       " coefficient elements, exceeding the configured "
                       "limit of " +
                       std::to_string(config.max_coefficient_elements));
  }
  return resources;
}

TwoSequenceResourceMetadata
analyze_two_sequence_resources(const ProgramConfig &config) {
  config.validate();

  TwoSequenceResourceMetadata resources;
  resources.symbol_dimension = config.symbol_count;
  resources.sequence_dimension =
      checked_multiply(config.symbol_count, config.sequence_length,
                       "two-sequence coordinate dimension");
  resources.output_dimension = resources.sequence_dimension;
  resources.input_dimensions = {resources.sequence_dimension,
                                resources.sequence_dimension};
  resources.coefficient_elements =
      checked_multiply(checked_multiply(resources.sequence_dimension,
                                        resources.sequence_dimension,
                                        "two-sequence input combination count"),
                       resources.output_dimension,
                       "two-sequence conditional dense coefficient count");
  resources.nonzero_coefficient_elements =
      checked_multiply(resources.symbol_dimension, resources.sequence_dimension,
                       "two-sequence conditional nonzero coefficient count");
  resources.dense_coefficient_bytes =
      checked_multiply(resources.coefficient_elements, sizeof(double),
                       "two-sequence conditional dense coefficient storage");

  if (resources.coefficient_elements > config.max_coefficient_elements) {
    throw ProgramError(
        "two-sequence conditional program requires " +
        std::to_string(resources.coefficient_elements) +
        " coefficient elements, exceeding the configured limit of " +
        std::to_string(config.max_coefficient_elements));
  }
  return resources;
}

Program compile_program(const ProgramConfig &config) {
  const ResourceMetadata resources = analyze_resources(config);
  const cajal::Type condition_type = make_condition_type();
  const cajal::Type symbol_type = make_symbol_type(config.symbol_count);
  const cajal::Type sequence_type =
      make_sequence_type(config.symbol_count, config.sequence_length);
  const cajal::Context input_context{
      {kConditionInputName, condition_type},
      {kSequenceInputName, sequence_type},
  };
  const cajal::Expression expression =
      make_program_expression(config.sequence_length);
  cajal::CompiledProgram compiled = cajal::compile(expression, input_context);

  Program program{
      config,        resources,     condition_type, symbol_type,
      sequence_type, input_context, expression,     std::move(compiled),
  };
  validate_compilation(program);
  return program;
}

ReverseProgram compile_reverse_program(const ProgramConfig &config) {
  const ReverseResourceMetadata resources = analyze_reverse_resources(config);
  const cajal::Type symbol_type = make_symbol_type(config.symbol_count);
  const cajal::Type sequence_type =
      make_sequence_type(config.symbol_count, config.sequence_length);
  const cajal::Context input_context{
      {kSequenceInputName, sequence_type},
  };
  const cajal::Expression expression =
      reverse_sequence_expression(kSequenceInputName, config.sequence_length);
  cajal::CompiledProgram compiled = cajal::compile(expression, input_context);

  ReverseProgram program{
      config,        resources,  symbol_type,         sequence_type,
      input_context, expression, std::move(compiled),
  };
  validate_reverse_compilation(program);
  return program;
}

TwoSequenceProgram compile_two_sequence_program(const ProgramConfig &config) {
  const TwoSequenceResourceMetadata resources =
      analyze_two_sequence_resources(config);
  const cajal::Type symbol_type = make_symbol_type(config.symbol_count);
  const cajal::Type sequence_type =
      make_sequence_type(config.symbol_count, config.sequence_length);
  const cajal::Context input_context{
      {kFirstSequenceInputName, sequence_type},
      {kSecondSequenceInputName, sequence_type},
  };
  const cajal::Expression expression =
      make_two_sequence_program_expression(symbol_type, config.sequence_length);
  cajal::CompiledProgram compiled = cajal::compile(expression, input_context);

  TwoSequenceProgram program{
      config,        resources,  symbol_type,         sequence_type,
      input_context, expression, std::move(compiled),
  };
  validate_two_sequence_compilation(program);
  return program;
}

} // namespace riftco_transformer::experiments::conditional_reverse
