#include "riftco_transformer/compiler/cajal/cajal.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace cajal = riftco_transformer::compiler::cajal;

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

[[nodiscard]] cajal::Type bit_type() {
  return cajal::Type::sum(cajal::Type::unit(), cajal::Type::unit());
}

[[nodiscard]] cajal::Expression bit_expression(bool one) {
  if (one) {
    return cajal::Expression::inject_right(cajal::Type::unit(),
                                           cajal::Expression::unit());
  }
  return cajal::Expression::inject_left(cajal::Expression::unit(),
                                        cajal::Type::unit());
}

[[nodiscard]] cajal::Value bit_value(bool one) {
  if (one) {
    return cajal::Value::inject_right(cajal::Type::unit(),
                                      cajal::Value::unit());
  }
  return cajal::Value::inject_left(cajal::Value::unit(), cajal::Type::unit());
}

void require_coordinates(const cajal::EncodedValue &actual,
                         std::vector<double> expected,
                         const std::string &message) {
  require(actual.size() == expected.size(), message + ": coordinate count");
  require(cajal::approximately_equal(actual,
                                     cajal::EncodedValue(std::move(expected))),
          message + ": coordinate values");
}

void test_value_encoding_decoding_and_enumeration() {
  require_coordinates(cajal::encode(cajal::Value::unit()), {1.0},
                      "unit encoding");
  require_coordinates(cajal::encode(bit_value(false)), {1.0, 0.0},
                      "left sum encoding");
  require_coordinates(cajal::encode(bit_value(true)), {0.0, 1.0},
                      "right sum encoding");

  const cajal::Type pair_type =
      cajal::Type::product(cajal::Type::unit(), bit_type());
  const cajal::Value pair =
      cajal::Value::product(cajal::Value::unit(), bit_value(true));
  require_coordinates(cajal::encode(pair), {1.0, 0.0, 1.0}, "product encoding");
  require(cajal::decode(pair_type, cajal::encode(pair)) == pair,
          "product decoding must invert a discrete encoding");

  const cajal::Value dictionary = cajal::Value::dictionary({
      {bit_value(false), bit_value(true)},
      {bit_value(true), bit_value(false)},
  });
  require_coordinates(cajal::encode(dictionary), {0.0, 1.0, 1.0, 0.0},
                      "dictionary outer-product encoding");

  const auto bits = cajal::enumerate_values(bit_type());
  require(bits.size() == 2 && bits[0] == bit_value(false) &&
              bits[1] == bit_value(true),
          "sum enumeration must be deterministic and left-first");
  const auto pairs =
      cajal::enumerate_values(cajal::Type::product(bit_type(), bit_type()));
  require(pairs.size() == 4, "a Bit product must enumerate four source values");

  require_throws_as<cajal::EncodingError>(
      [] { static_cast<void>(cajal::EncodedValue({})); },
      "empty coordinate vectors must be rejected");
  require_throws_as<cajal::EncodingError>(
      [] {
        static_cast<void>(
            cajal::EncodedValue({std::numeric_limits<double>::infinity()}));
      },
      "non-finite coordinates must be rejected");
  require_throws_as<cajal::EncodingError>(
      [&] {
        static_cast<void>(
            cajal::decode(bit_type(), cajal::EncodedValue({1.0, 1.0})));
      },
      "ambiguous sum coordinates must be rejected");
  require_throws_as<cajal::EncodingError>(
      [&] {
        static_cast<void>(
            cajal::decode(dictionary.type(), cajal::encode(dictionary)));
      },
      "dictionary coordinates have no unique source decomposition");
  require_throws_as<cajal::EncodingError>(
      [&] { static_cast<void>(cajal::enumerate_values(dictionary.type())); },
      "dictionary source lists are unbounded");
  require_throws_as<cajal::EncodingError>(
      [] { static_cast<void>(cajal::enumerate_values(bit_type(), 1)); },
      "enumeration must honor its cardinality guard");
}

void test_multilinear_map_primitive() {
  const cajal::MultilinearMap constant =
      cajal::MultilinearMap::constant(cajal::EncodedValue({2.0, -1.0}));
  require(constant.arity() == 0 && constant.output_dimension() == 2,
          "constant maps are zero-linear");
  require_coordinates(constant.apply(std::vector<cajal::EncodedValue>{}),
                      {2.0, -1.0}, "constant application");

  const cajal::MultilinearMap identity = cajal::MultilinearMap::identity(3);
  require_coordinates(identity.apply(std::vector<cajal::EncodedValue>{
                          cajal::EncodedValue({2.0, 3.0, 4.0})}),
                      {2.0, 3.0, 4.0}, "identity application");

  const cajal::MultilinearMap bilinear({2, 2}, 1, {1.0, 2.0, 3.0, 4.0});
  require_coordinates(
      bilinear.apply(std::vector<cajal::EncodedValue>{
          cajal::EncodedValue({1.0, 2.0}), cajal::EncodedValue({3.0, 4.0})}),
      {61.0}, "dense bilinear contraction");
  const std::vector<std::size_t> coefficient_index = {1, 0};
  require(bilinear.coefficient_at(0, coefficient_index) == 3.0,
          "coefficient layout must be output-major row-major");

  require_throws_as<cajal::MultilinearMapError>(
      [] { static_cast<void>(cajal::MultilinearMap({2}, 2, {1.0})); },
      "coefficient count must match shape");
  require_throws_as<cajal::MultilinearMapError>(
      [&] {
        static_cast<void>(bilinear.apply(
            std::vector<cajal::EncodedValue>{cajal::EncodedValue({1.0, 2.0})}));
      },
      "application arity must match");
  require_throws_as<cajal::MultilinearMapError>(
      [] {
        static_cast<void>(cajal::MultilinearMap(
            {std::numeric_limits<std::size_t>::max(), 2}, 1, {}));
      },
      "input dimension products must reject overflow");
}

void test_sparse_multilinear_map_construction() {
  const std::vector<std::size_t> flat_indices{0, 4, 6, 11};
  const std::vector<double> values{2.0, 3.0, -1.0, 4.0};
  const cajal::MultilinearMap sparse =
      cajal::MultilinearMap::from_sparse({2, 3}, 2, flat_indices, values);
  const std::vector<double> expected_coefficients{
      2.0, 0.0, 0.0, 0.0, 3.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 4.0,
  };
  require(std::vector<double>(sparse.coefficients().begin(),
                              sparse.coefficients().end()) ==
              expected_coefficients,
          "sparse entries must materialize in output-major flat order");
  require(sparse.coefficient_at(0, std::vector<std::size_t>{1, 1}) == 3.0 &&
              sparse.coefficient_at(1, std::vector<std::size_t>{1, 2}) == 4.0,
          "sparse coefficients must use the dense map coordinate layout");
  require_coordinates(sparse.apply(std::vector<cajal::EncodedValue>{
                          cajal::EncodedValue({1.0, 2.0}),
                          cajal::EncodedValue({3.0, 4.0, 5.0})}),
                      {30.0, 37.0}, "sparse map application");

  const std::vector<std::size_t> empty_indices;
  const std::vector<double> empty_values;
  const cajal::MultilinearMap zero_constant =
      cajal::MultilinearMap::from_sparse({}, 3, empty_indices, empty_values);
  require(zero_constant.arity() == 0 && zero_constant.coefficient_count() == 3,
          "empty sparse constants must retain their output shape");
  require_coordinates(zero_constant.apply(std::vector<cajal::EncodedValue>{}),
                      {0.0, 0.0, 0.0}, "empty sparse constant application");

  const cajal::MultilinearMap zero_bilinear =
      cajal::MultilinearMap::from_sparse({2, 3}, 2, empty_indices,
                                         empty_values);
  require_coordinates(zero_bilinear.apply(std::vector<cajal::EncodedValue>{
                          cajal::EncodedValue({1.0, 2.0}),
                          cajal::EncodedValue({3.0, 4.0, 5.0})}),
                      {0.0, 0.0}, "empty sparse nonconstant application");

  const std::vector<std::size_t> one_index{0};
  const std::vector<std::size_t> two_indices{0, 1};
  const std::vector<double> one_value{1.0};
  const std::vector<double> two_values{1.0, 2.0};
  require_throws_as<cajal::MultilinearMapError>(
      [&] {
        static_cast<void>(
            cajal::MultilinearMap::from_sparse({2}, 1, two_indices, one_value));
      },
      "sparse index and value counts must match");

  const std::vector<std::size_t> out_of_range_index{2};
  require_throws_as<cajal::MultilinearMapError>(
      [&] {
        static_cast<void>(cajal::MultilinearMap::from_sparse(
            {2}, 1, out_of_range_index, one_value));
      },
      "sparse indices must be in range");

  const std::vector<std::size_t> duplicate_indices{1, 1};
  require_throws_as<cajal::MultilinearMapError>(
      [&] {
        static_cast<void>(cajal::MultilinearMap::from_sparse(
            {2}, 1, duplicate_indices, two_values));
      },
      "sparse indices must be unique");

  const std::vector<double> infinite_value{
      std::numeric_limits<double>::infinity()};
  require_throws_as<cajal::MultilinearMapError>(
      [&] {
        static_cast<void>(cajal::MultilinearMap::from_sparse({2}, 1, one_index,
                                                             infinite_value));
      },
      "sparse values must be finite");

  const std::vector<double> zero_value{0.0};
  require_throws_as<cajal::MultilinearMapError>(
      [&] {
        static_cast<void>(
            cajal::MultilinearMap::from_sparse({2}, 1, one_index, zero_value));
      },
      "explicit sparse zero values must be rejected");

  require_throws_as<cajal::MultilinearMapError>(
      [&] {
        static_cast<void>(cajal::MultilinearMap::from_sparse(
            {std::numeric_limits<std::size_t>::max(), 2}, 1, empty_indices,
            empty_values));
      },
      "sparse map input dimension products must reject overflow");
  require_throws_as<cajal::MultilinearMapError>(
      [&] {
        static_cast<void>(cajal::MultilinearMap::from_sparse(
            {2}, std::numeric_limits<std::size_t>::max(), empty_indices,
            empty_values));
      },
      "sparse map coefficient counts must reject overflow");
}

void check_environment(const cajal::Expression &expression,
                       const cajal::CompiledProgram &compiled,
                       const cajal::Environment &environment,
                       const std::string &label) {
  const cajal::Value interpreted = cajal::evaluate(expression, environment);
  const cajal::EncodedValue expected = cajal::encode(interpreted);
  const cajal::EncodedValue actual = compiled.apply(environment);
  require(cajal::approximately_equal(actual, expected),
          label + ": compiled coordinates differ for interpreted value " +
              interpreted.to_string());
  if (compiled.output_type().kind() != cajal::Type::Kind::Dictionary) {
    require(cajal::decode(compiled.output_type(), actual) == interpreted,
            label + ": decoded compiled result differs from interpreter");
  }
}

void enumerate_environments(const cajal::Expression &expression,
                            const cajal::CompiledProgram &compiled,
                            std::size_t binding_index,
                            cajal::Environment &environment,
                            const std::string &label) {
  if (binding_index == compiled.input_context().size()) {
    check_environment(expression, compiled, environment, label);
    return;
  }

  const cajal::Binding &binding = compiled.input_context()[binding_index];
  for (const cajal::Value &value : cajal::enumerate_values(binding.type)) {
    environment.push_back({binding.name, value});
    enumerate_environments(expression, compiled, binding_index + 1, environment,
                           label);
    environment.pop_back();
  }
}

void require_exhaustive_equivalence(const cajal::Expression &expression,
                                    const cajal::Context &context,
                                    const std::string &label) {
  const cajal::CompiledProgram compiled = cajal::compile(expression, context);
  require(compiled.input_context().size() == context.size(),
          label + ": compiler changed context arity");
  require(compiled.output_type() == cajal::type_check(expression, context),
          label + ": compiler output type differs from checker");
  require(compiled.map().arity() == context.size(),
          label + ": map arity differs from context");
  for (std::size_t index = 0; index < context.size(); ++index) {
    require(compiled.input_context()[index].name == context[index].name &&
                compiled.map().input_dimensions()[index] ==
                    context[index].type.dimension(),
            label + ": compiler did not preserve ordered context axes");
  }

  cajal::Environment environment;
  enumerate_environments(expression, compiled, 0, environment, label);
}

void test_compiler_equivalence_for_core_expressions() {
  require_exhaustive_equivalence(cajal::Expression::unit(), {}, "unit");
  require_exhaustive_equivalence(cajal::Expression::variable("x"),
                                 {{"x", bit_type()}}, "variable");

  const cajal::Expression tuple = cajal::Expression::tuple(
      cajal::Expression::variable("x"), cajal::Expression::variable("x"));
  require_exhaustive_equivalence(tuple, {{"x", bit_type()}}, "additive tuple");
  require_exhaustive_equivalence(cajal::Expression::projection(tuple, 1),
                                 {{"x", bit_type()}}, "tuple projection");

  require_exhaustive_equivalence(
      cajal::Expression::inject_left(cajal::Expression::variable("x"),
                                     bit_type()),
      {{"x", cajal::Type::unit()}}, "left injection");
  require_exhaustive_equivalence(
      cajal::Expression::inject_right(bit_type(),
                                      cajal::Expression::variable("x")),
      {{"x", cajal::Type::unit()}}, "right injection");

  const cajal::Expression sequence =
      cajal::Expression::sequence(cajal::Expression::variable("tick"),
                                  cajal::Expression::variable("result"));
  require_exhaustive_equivalence(
      sequence, {{"result", bit_type()}, {"tick", cajal::Type::unit()}},
      "sequence with reversed context order");
  const cajal::CompiledProgram compiled_sequence = cajal::compile(
      sequence, {{"result", bit_type()}, {"tick", cajal::Type::unit()}});
  require_coordinates(
      compiled_sequence.map().apply(std::vector<cajal::EncodedValue>{
          cajal::EncodedValue({2.0, 3.0}), cajal::EncodedValue({4.0})}),
      {8.0, 12.0}, "sequence remains bilinear off the discrete basis");

  const cajal::Expression let_identity =
      cajal::Expression::let("bound", cajal::Expression::variable("input"),
                             cajal::Expression::variable("bound"));
  require_exhaustive_equivalence(let_identity, {{"input", bit_type()}},
                                 "let composition");
  const cajal::CompiledProgram compiled_let_identity =
      cajal::compile(let_identity, {{"input", bit_type()}});
  require_coordinates(
      compiled_let_identity.map().apply(
          std::vector<cajal::EncodedValue>{cajal::EncodedValue({2.0, 3.0})}),
      {2.0, 3.0}, "let contracts a multi-coordinate binder axis");

  const cajal::Expression scaled_let = cajal::Expression::let(
      "bound", cajal::Expression::variable("input"),
      cajal::Expression::sequence(cajal::Expression::variable("external"),
                                  cajal::Expression::variable("bound")));
  require_exhaustive_equivalence(
      scaled_let, {{"external", cajal::Type::unit()}, {"input", bit_type()}},
      "let with an external body context");
  const cajal::CompiledProgram compiled_scaled_let = cajal::compile(
      scaled_let, {{"external", cajal::Type::unit()}, {"input", bit_type()}});
  require_coordinates(
      compiled_scaled_let.map().apply(std::vector<cajal::EncodedValue>{
          cajal::EncodedValue({4.0}), cajal::EncodedValue({2.0, 3.0})}),
      {8.0, 12.0},
      "let composition preserves multilinearity across reordered axes");

  const cajal::Expression case_expression = cajal::Expression::case_of(
      cajal::Expression::variable("choice"), "left_payload",
      cajal::Expression::sequence(cajal::Expression::variable("left_payload"),
                                  cajal::Expression::variable("external")),
      "right_payload",
      cajal::Expression::sequence(cajal::Expression::variable("right_payload"),
                                  cajal::Expression::variable("external")));
  require_exhaustive_equivalence(
      case_expression, {{"external", bit_type()}, {"choice", bit_type()}},
      "case with reversed context order");
  const cajal::CompiledProgram compiled_case = cajal::compile(
      case_expression, {{"external", bit_type()}, {"choice", bit_type()}});
  require_coordinates(
      compiled_case.map().apply(std::vector<cajal::EncodedValue>{
          cajal::EncodedValue({2.0, 3.0}), cajal::EncodedValue({5.0, 7.0})}),
      {24.0, 36.0}, "case remains bilinear off the discrete basis");

  const cajal::Type wide_choice_type = cajal::Type::sum(bit_type(), bit_type());
  const cajal::Expression wide_case = cajal::Expression::case_of(
      cajal::Expression::variable("choice"), "left_payload",
      cajal::Expression::inject_left(
          cajal::Expression::variable("left_payload"), bit_type()),
      "right_payload",
      cajal::Expression::inject_right(
          bit_type(), cajal::Expression::variable("right_payload")));
  require_exhaustive_equivalence(wide_case, {{"choice", wide_choice_type}},
                                 "case with wide payloads");
  const cajal::CompiledProgram compiled_wide_case =
      cajal::compile(wide_case, {{"choice", wide_choice_type}});
  require_coordinates(
      compiled_wide_case.map().apply(std::vector<cajal::EncodedValue>{
          cajal::EncodedValue({2.0, 3.0, 5.0, 7.0})}),
      {2.0, 3.0, 5.0, 7.0},
      "case contracts both multi-coordinate payload blocks");

  const cajal::Type uneven_choice_type =
      cajal::Type::sum(cajal::Type::unit(), bit_type());
  const cajal::Expression uneven_case = cajal::Expression::case_of(
      cajal::Expression::variable("choice"), "unit_payload",
      cajal::Expression::inject_left(
          cajal::Expression::variable("unit_payload"), bit_type()),
      "bit_payload",
      cajal::Expression::inject_right(
          cajal::Type::unit(), cajal::Expression::variable("bit_payload")));
  require_exhaustive_equivalence(uneven_case, {{"choice", uneven_choice_type}},
                                 "case with unequal payload dimensions");
  const cajal::CompiledProgram compiled_uneven_case =
      cajal::compile(uneven_case, {{"choice", uneven_choice_type}});
  require_coordinates(
      compiled_uneven_case.map().apply(std::vector<cajal::EncodedValue>{
          cajal::EncodedValue({2.0, 3.0, 5.0})}),
      {2.0, 3.0, 5.0}, "case contracts unequal left and right payload blocks");

  const cajal::Type pair_type =
      cajal::Type::product(cajal::Type::unit(), bit_type());
  require_exhaustive_equivalence(
      cajal::Expression::projection(cajal::Expression::variable("pair"), 1),
      {{"pair", pair_type}}, "projection from product input");
}

void test_dictionary_and_lookup_compilation() {
  const cajal::Expression swap_dictionary = cajal::Expression::dictionary({
      {bit_expression(false), bit_expression(true)},
      {bit_expression(true), bit_expression(false)},
  });
  require_exhaustive_equivalence(swap_dictionary, {}, "closed dictionary");

  const cajal::Expression open_dictionary = cajal::Expression::dictionary({
      {bit_expression(false), cajal::Expression::variable("value")},
      {bit_expression(true), cajal::Expression::variable("value")},
  });
  require_exhaustive_equivalence(open_dictionary, {{"value", bit_type()}},
                                 "open dictionary");

  const cajal::Expression dynamic_dictionary =
      cajal::Expression::dictionary({{cajal::Expression::variable("key"),
                                      cajal::Expression::variable("value")}});
  require_exhaustive_equivalence(dynamic_dictionary,
                                 {{"value", bit_type()}, {"key", bit_type()}},
                                 "dictionary with dynamic key and value");
  const cajal::CompiledProgram compiled_dynamic_dictionary = cajal::compile(
      dynamic_dictionary, {{"value", bit_type()}, {"key", bit_type()}});
  require_coordinates(
      compiled_dynamic_dictionary.map().apply(std::vector<cajal::EncodedValue>{
          cajal::EncodedValue({2.0, 3.0}), cajal::EncodedValue({5.0, 7.0})}),
      {10.0, 14.0, 15.0, 21.0},
      "dictionary lowers to a value-major key/value outer product");
  require_coordinates(
      compiled_dynamic_dictionary.apply(
          {{"key", bit_value(true)}, {"value", bit_value(false)}}),
      {0.0, 1.0, 0.0, 0.0},
      "compiled programs reorder named environments to explicit map axes");

  const cajal::Expression closed_lookup = cajal::Expression::lookup(
      swap_dictionary, cajal::Expression::variable("query"));
  require_exhaustive_equivalence(closed_lookup, {{"query", bit_type()}},
                                 "closed dictionary lookup");

  const cajal::Expression open_lookup = cajal::Expression::lookup(
      open_dictionary, cajal::Expression::variable("query"));
  require_exhaustive_equivalence(open_lookup,
                                 {{"query", bit_type()}, {"value", bit_type()}},
                                 "open bilinear dictionary lookup");
  const cajal::CompiledProgram compiled_open_lookup = cajal::compile(
      open_lookup, {{"query", bit_type()}, {"value", bit_type()}});
  require_coordinates(
      compiled_open_lookup.map().apply(std::vector<cajal::EncodedValue>{
          cajal::EncodedValue({5.0, 7.0}), cajal::EncodedValue({2.0, 3.0})}),
      {24.0, 36.0},
      "dictionary lookup remains bilinear off the discrete basis");
}

void test_compiled_program_environment_validation() {
  const cajal::CompiledProgram compiled =
      cajal::compile(cajal::Expression::variable("x"), {{"x", bit_type()}});
  require_coordinates(compiled.apply({{"x", bit_value(true)}}), {0.0, 1.0},
                      "compiled program application");
  require_throws_as<cajal::CompilationError>(
      [&] { static_cast<void>(compiled.apply({})); },
      "missing compiled input must be rejected");
  require_throws_as<cajal::CompilationError>(
      [&] {
        static_cast<void>(compiled.apply(
            {{"x", bit_value(false)}, {"extra", cajal::Value::unit()}}));
      },
      "extra compiled input must be rejected");
  require_throws_as<cajal::CompilationError>(
      [&] {
        static_cast<void>(
            compiled.apply({{"x", bit_value(false)}, {"x", bit_value(true)}}));
      },
      "duplicate compiled input must be rejected");
  require_throws_as<cajal::CompilationError>(
      [&] { static_cast<void>(compiled.apply({{"x", cajal::Value::unit()}})); },
      "wrong compiled input type must be rejected");
}

} // namespace

int main() {
  try {
    test_value_encoding_decoding_and_enumeration();
    test_multilinear_map_primitive();
    test_sparse_multilinear_map_construction();
    test_compiler_equivalence_for_core_expressions();
    test_dictionary_and_lookup_compilation();
    test_compiled_program_environment_validation();
    std::cout << "Cajal multilinear compiler tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Cajal multilinear compiler test failure: " << error.what()
              << '\n';
    return 1;
  }
}
