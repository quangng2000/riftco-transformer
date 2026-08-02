#include "riftco_transformer/compiler/cajal/cajal.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

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
    throw std::runtime_error(message + ": threw the wrong exception: " +
                             error.what());
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
  return cajal::Value::inject_left(cajal::Value::unit(),
                                   cajal::Type::unit());
}

void require_program(const cajal::Expression &expression,
                     const cajal::Context &context,
                     const cajal::Environment &environment,
                     const cajal::Type &expected_type,
                     const cajal::Value &expected_value,
                     const std::string &message) {
  const cajal::Type checked_type = cajal::type_check(expression, context);
  require(checked_type == expected_type, message + ": static type mismatch");

  const cajal::Value result = cajal::evaluate(expression, environment);
  require(result == expected_value, message + ": evaluated value mismatch");
  require(result.type() == checked_type,
          message + ": evaluated value type differs from checked type");
}

void test_type_structure_strings_and_dimensions() {
  const cajal::Type unit = cajal::Type::unit();
  const cajal::Type bit = bit_type();
  const cajal::Type payload =
      cajal::Type::product(cajal::Type::unit(), bit_type());
  const cajal::Type dictionary =
      cajal::Type::dictionary(bit_type(), cajal::Type::product(
                                              cajal::Type::unit(), bit_type()));

  require(unit.dimension() == 1 && bit.dimension() == 2 &&
              payload.dimension() == 3 && dictionary.dimension() == 6,
          "finite type dimensions must follow unit, sum, product, and "
          "dictionary layout rules");
  require(unit.to_string() == "Unit", "unit type string");
  require(bit.to_string() == "Sum<Unit, Unit>", "sum type string");
  require(payload.to_string() == "Product<Unit, Sum<Unit, Unit>>",
          "product type string");
  require(dictionary.to_string() ==
              "Dictionary<Sum<Unit, Unit>, Product<Unit, Sum<Unit, Unit>>>",
          "dictionary type string");

  const cajal::Type rebuilt = cajal::Type::dictionary(
      cajal::Type::sum(cajal::Type::unit(), cajal::Type::unit()),
      cajal::Type::product(
          cajal::Type::unit(),
          cajal::Type::sum(cajal::Type::unit(), cajal::Type::unit())));
  require(dictionary == rebuilt,
          "type equality must be structural rather than pointer-based");
  require(cajal::Type::product(cajal::Type::unit(), bit_type()) !=
              cajal::Type::product(bit_type(), cajal::Type::unit()),
          "type equality must preserve child order");
}

void test_variables_tuples_and_injections() {
  const cajal::Expression variable = cajal::Expression::variable("x");
  require_program(variable, {{"x", cajal::Type::unit()}},
                  {{"x", cajal::Value::unit()}}, cajal::Type::unit(),
                  cajal::Value::unit(), "variable");

  const cajal::Expression tuple = cajal::Expression::tuple(
      cajal::Expression::variable("x"),
      cajal::Expression::variable("x"));
  require_program(
      tuple, {{"x", cajal::Type::unit()}},
      {{"x", cajal::Value::unit()}},
      cajal::Type::product(cajal::Type::unit(), cajal::Type::unit()),
      cajal::Value::product(cajal::Value::unit(), cajal::Value::unit()),
      "additive tuple");

  const cajal::Expression left = cajal::Expression::inject_left(
      cajal::Expression::variable("payload"), bit_type());
  require_program(
      left, {{"payload", cajal::Type::unit()}},
      {{"payload", cajal::Value::unit()}},
      cajal::Type::sum(cajal::Type::unit(), bit_type()),
      cajal::Value::inject_left(cajal::Value::unit(), bit_type()),
      "left injection");

  const cajal::Expression right = cajal::Expression::inject_right(
      bit_type(), cajal::Expression::variable("payload"));
  require_program(
      right, {{"payload", cajal::Type::unit()}},
      {{"payload", cajal::Value::unit()}},
      cajal::Type::sum(bit_type(), cajal::Type::unit()),
      cajal::Value::inject_right(bit_type(), cajal::Value::unit()),
      "right injection");
}

void test_projection_let_and_sequence() {
  const cajal::Type pair_type =
      cajal::Type::product(cajal::Type::unit(), bit_type());
  const cajal::Expression projection = cajal::Expression::projection(
      cajal::Expression::variable("pair"), 1);
  require_program(
      projection, {{"pair", pair_type}},
      {{"pair", cajal::Value::product(cajal::Value::unit(),
                                       bit_value(true))}},
      bit_type(), bit_value(true), "product projection");

  const cajal::Expression let = cajal::Expression::let(
      "bound", cajal::Expression::unit(),
      cajal::Expression::variable("bound"));
  require_program(let, {}, {}, cajal::Type::unit(), cajal::Value::unit(),
                  "let expression");

  const cajal::Expression sequence = cajal::Expression::sequence(
      cajal::Expression::variable("tick"),
      cajal::Expression::variable("result"));
  require_program(
      sequence,
      {{"tick", cajal::Type::unit()}, {"result", bit_type()}},
      {{"tick", cajal::Value::unit()}, {"result", bit_value(false)}},
      bit_type(), bit_value(false), "sequence");

  const cajal::Expression invalid_sequence = cajal::Expression::sequence(
      bit_expression(false), cajal::Expression::unit());
  require_throws_as<cajal::TypeError>(
      [&] { static_cast<void>(cajal::type_check(invalid_sequence)); },
      "a sequence must begin with a Unit expression");

  const cajal::Expression missing_lookup = cajal::Expression::lookup(
      cajal::Expression::dictionary(
          {{bit_expression(false), cajal::Expression::unit()}}),
      bit_expression(true));
  const cajal::Expression lazy_projection = cajal::Expression::projection(
      cajal::Expression::tuple(cajal::Expression::unit(), missing_lookup), 0);
  require_program(lazy_projection, {}, {}, cajal::Type::unit(),
                  cajal::Value::unit(),
                  "projection must not evaluate the unselected component");
}

void test_exactly_once_errors() {
  require_throws_as<cajal::TypeError>(
      [] {
        static_cast<void>(cajal::type_check(
            cajal::Expression::unit(), {{"unused", cajal::Type::unit()}}));
      },
      "an unused context binding must be rejected");

  const cajal::Expression duplicated = cajal::Expression::sequence(
      cajal::Expression::variable("x"),
      cajal::Expression::variable("x"));
  require_throws_as<cajal::TypeError>(
      [&] {
        static_cast<void>(
            cajal::type_check(duplicated, {{"x", cajal::Type::unit()}}));
      },
      "a linear variable must not be consumed twice");

  const cajal::Expression mismatched_tuple = cajal::Expression::tuple(
      cajal::Expression::variable("x"),
      cajal::Expression::variable("y"));
  require_throws_as<cajal::TypeError>(
      [&] {
        static_cast<void>(cajal::type_check(
            mismatched_tuple,
            {{"x", cajal::Type::unit()}, {"y", cajal::Type::unit()}}));
      },
      "additive tuple components must consume the same context");

  require_throws_as<cajal::TypeError>(
      [] {
        static_cast<void>(
            cajal::type_check(cajal::Expression::variable("missing")));
      },
      "an unknown variable must be rejected");

  const cajal::Expression unused_let = cajal::Expression::let(
      "bound", cajal::Expression::unit(), cajal::Expression::unit());
  require_throws_as<cajal::TypeError>(
      [&] { static_cast<void>(cajal::type_check(unused_let)); },
      "an unused let binder must be rejected");

  const cajal::Expression shadowing = cajal::Expression::let(
      "x", cajal::Expression::unit(), cajal::Expression::variable("x"));
  require_throws_as<cajal::TypeError>(
      [&] {
        static_cast<void>(
            cajal::type_check(shadowing, {{"x", cajal::Type::unit()}}));
      },
      "let bindings must not shadow context bindings");
}

void test_case_branch_linearity_and_evaluation() {
  const cajal::Expression choose_external = cajal::Expression::case_of(
      cajal::Expression::variable("choice"), "left_payload",
      cajal::Expression::sequence(
          cajal::Expression::variable("left_payload"),
          cajal::Expression::variable("external")),
      "right_payload",
      cajal::Expression::sequence(
          cajal::Expression::variable("right_payload"),
          cajal::Expression::variable("external")));
  const cajal::Context context = {
      {"choice", bit_type()}, {"external", bit_type()}};
  const cajal::Type result_type = bit_type();

  require_program(
      choose_external, context,
      {{"choice", bit_value(false)}, {"external", bit_value(true)}},
      result_type, bit_value(true),
      "left case branch");
  require_program(
      choose_external, context,
      {{"choice", bit_value(true)}, {"external", bit_value(false)}},
      result_type, bit_value(false),
      "right case branch");

  const cajal::Expression mismatched_usage = cajal::Expression::case_of(
      cajal::Expression::variable("choice"), "left_payload",
      cajal::Expression::sequence(
          cajal::Expression::variable("left_payload"),
          cajal::Expression::variable("external")),
      "right_payload", cajal::Expression::variable("right_payload"));
  require_throws_as<cajal::TypeError>(
      [&] {
        static_cast<void>(cajal::type_check(
            mismatched_usage,
            {{"choice", bit_type()}, {"external", cajal::Type::unit()}}));
      },
      "case branches must consume the same external bindings");

  const cajal::Expression mismatched_results = cajal::Expression::case_of(
      cajal::Expression::variable("choice"), "left_payload",
      cajal::Expression::variable("left_payload"), "right_payload",
      cajal::Expression::inject_left(
          cajal::Expression::variable("right_payload"),
          cajal::Type::unit()));
  require_throws_as<cajal::TypeError>(
      [&] {
        static_cast<void>(cajal::type_check(
            mismatched_results, {{"choice", bit_type()}}));
      },
      "case branches must return the same type");
}

void test_dictionary_validation_and_lookup() {
  const cajal::Expression empty =
      cajal::Expression::dictionary(cajal::Expression::DictionaryEntries{});
  require_throws_as<cajal::TypeError>(
      [&] { static_cast<void>(cajal::type_check(empty)); },
      "an empty dictionary expression must not infer a type");
  require_throws_as<std::invalid_argument>(
      [] {
        static_cast<void>(
            cajal::Value::dictionary(cajal::Value::DictionaryEntries{}));
      },
      "an empty dictionary value must be rejected");

  const cajal::Expression heterogeneous = cajal::Expression::dictionary({
      {cajal::Expression::unit(), cajal::Expression::unit()},
      {bit_expression(false), cajal::Expression::unit()},
  });
  require_throws_as<cajal::TypeError>(
      [&] { static_cast<void>(cajal::type_check(heterogeneous)); },
      "dictionary keys must have one structural type");
  require_throws_as<std::invalid_argument>(
      [] {
        static_cast<void>(cajal::Value::dictionary({
            {cajal::Value::unit(), cajal::Value::unit()},
            {bit_value(false), cajal::Value::unit()},
        }));
      },
      "runtime dictionary keys must have one structural type");

  const cajal::Expression linear_dictionary = cajal::Expression::dictionary({
      {cajal::Expression::variable("key"),
       cajal::Expression::variable("value")},
      {cajal::Expression::variable("key"),
       cajal::Expression::variable("value")},
  });
  require_program(
      linear_dictionary,
      {{"key", bit_type()}, {"value", bit_type()}},
      {{"key", bit_value(false)}, {"value", bit_value(true)}},
      cajal::Type::dictionary(bit_type(), bit_type()),
      cajal::Value::dictionary({
          {bit_value(false), bit_value(true)},
          {bit_value(false), bit_value(true)},
      }),
      "dictionary keys and values must each share one linear context");

  const cajal::Expression mismatched_key_contexts =
      cajal::Expression::dictionary({
          {cajal::Expression::variable("first_key"),
           cajal::Expression::unit()},
          {cajal::Expression::variable("second_key"),
           cajal::Expression::unit()},
      });
  require_throws_as<cajal::TypeError>(
      [&] {
        static_cast<void>(cajal::type_check(
            mismatched_key_contexts,
            {{"first_key", bit_type()}, {"second_key", bit_type()}}));
      },
      "dictionary key entries must consume the same linear context");

  const cajal::Expression dictionary = cajal::Expression::dictionary({
      {bit_expression(false), bit_expression(true)},
      {bit_expression(true), bit_expression(false)},
  });
  const cajal::Expression unique_lookup = cajal::Expression::lookup(
      dictionary, bit_expression(true));
  require_program(unique_lookup, {}, {}, bit_type(), bit_value(false),
                  "unique dictionary lookup");

  const cajal::Expression failing_value = cajal::Expression::lookup(
      cajal::Expression::dictionary(
          {{bit_expression(false), cajal::Expression::unit()}}),
      bit_expression(true));
  const cajal::Expression lazy_value_lookup = cajal::Expression::lookup(
      cajal::Expression::dictionary({
          {bit_expression(false), cajal::Expression::unit()},
          {bit_expression(true), failing_value},
      }),
      bit_expression(false));
  require_program(lazy_value_lookup, {}, {}, cajal::Type::unit(),
                  cajal::Value::unit(),
                  "lookup must not evaluate an unselected dictionary value");

  const cajal::Expression missing_lookup = cajal::Expression::lookup(
      cajal::Expression::dictionary(
          {{bit_expression(false), bit_expression(true)}}),
      bit_expression(true));
  require(cajal::type_check(missing_lookup) == bit_type(),
          "missing-key lookup must remain statically well typed");
  require_throws_as<cajal::EvaluationError>(
      [&] { static_cast<void>(cajal::evaluate(missing_lookup)); },
      "a dictionary lookup with no matching key must fail at runtime");

  const cajal::Expression duplicate_lookup = cajal::Expression::lookup(
      cajal::Expression::dictionary({
          {bit_expression(false), bit_expression(false)},
          {bit_expression(false), bit_expression(true)},
      }),
      bit_expression(false));
  require(cajal::type_check(duplicate_lookup) == bit_type(),
          "duplicate-key lookup must remain statically well typed");
  require_throws_as<cajal::EvaluationError>(
      [&] { static_cast<void>(cajal::evaluate(duplicate_lookup)); },
      "a dictionary lookup with multiple matching keys must fail at runtime");

  const cajal::Expression product_key = cajal::Expression::tuple(
      cajal::Expression::unit(), cajal::Expression::unit());
  const cajal::Expression non_enum_lookup = cajal::Expression::lookup(
      cajal::Expression::dictionary(
          {{product_key, cajal::Expression::unit()}}),
      product_key);
  require_throws_as<cajal::TypeError>(
      [&] { static_cast<void>(cajal::type_check(non_enum_lookup)); },
      "identity lookup must reject non-enum key encodings");
}

} // namespace

int main() {
  try {
    test_type_structure_strings_and_dimensions();
    test_variables_tuples_and_injections();
    test_projection_let_and_sequence();
    test_exactly_once_errors();
    test_case_branch_linearity_and_evaluation();
    test_dictionary_validation_and_lookup();
    std::cout << "Cajal-lite tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Cajal-lite test failure: " << error.what() << '\n';
    return 1;
  }
}
