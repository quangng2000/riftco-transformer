#include "riftco_transformer/compiler/cajal/interpreter.hpp"

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <utility>

namespace riftco_transformer::compiler::cajal {
namespace {

using Names = std::set<std::string>;

[[nodiscard]] const Value &find_value(const Environment &environment,
                                      const std::string &name) {
  const auto binding =
      std::find_if(environment.begin(), environment.end(),
                   [&name](const NamedValue &candidate) {
                     return candidate.name == name;
                   });
  if (binding == environment.end()) {
    throw EvaluationError("no runtime value is bound to '" + name + "'");
  }
  return binding->value;
}

[[nodiscard]] Environment extend_environment(const Environment &environment,
                                             const std::string &name,
                                             const Value &value) {
  if (name.empty()) {
    throw EvaluationError("runtime binding names must not be empty");
  }
  const auto existing =
      std::find_if(environment.begin(), environment.end(),
                   [&name](const NamedValue &candidate) {
                     return candidate.name == name;
                   });
  if (existing != environment.end()) {
    throw EvaluationError("runtime binding '" + name +
                          "' would shadow an existing value");
  }

  Environment extended = environment;
  extended.push_back(NamedValue{name, value});
  return extended;
}

void validate_environment(const Environment &environment) {
  for (std::size_t index = 0; index < environment.size(); ++index) {
    const auto &name = environment[index].name;
    if (name.empty()) {
      throw EvaluationError("runtime binding names must not be empty");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (environment[prior].name == name) {
        throw EvaluationError("duplicate runtime binding '" + name + "'");
      }
    }
  }
}

[[nodiscard]] Context context_from(const Environment &environment) {
  Context context;
  context.reserve(environment.size());
  for (const auto &binding : environment) {
    context.push_back(Binding{binding.name, binding.value.type()});
  }
  return context;
}

void collect_free_variables(const Expression &expression, const Names &bound,
                            Names &free) {
  switch (expression.kind()) {
  case Expression::Kind::Variable:
    if (!bound.contains(expression.variable_name())) {
      free.insert(expression.variable_name());
    }
    return;
  case Expression::Kind::Unit:
    return;
  case Expression::Kind::Tuple:
  case Expression::Kind::Sequence:
    collect_free_variables(expression.first(), bound, free);
    collect_free_variables(expression.second(), bound, free);
    return;
  case Expression::Kind::InjectLeft:
  case Expression::Kind::InjectRight:
    collect_free_variables(expression.payload(), bound, free);
    return;
  case Expression::Kind::Dictionary:
    for (std::size_t index = 0; index < expression.dictionary_size(); ++index) {
      collect_free_variables(expression.dictionary_key(index), bound, free);
      collect_free_variables(expression.dictionary_value(index), bound, free);
    }
    return;
  case Expression::Kind::Projection:
    collect_free_variables(expression.projected_value(), bound, free);
    return;
  case Expression::Kind::Case: {
    collect_free_variables(expression.scrutinee(), bound, free);
    Names left_bound = bound;
    left_bound.insert(expression.left_binding_name());
    collect_free_variables(expression.left_body(), left_bound, free);
    Names right_bound = bound;
    right_bound.insert(expression.right_binding_name());
    collect_free_variables(expression.right_body(), right_bound, free);
    return;
  }
  case Expression::Kind::Let: {
    collect_free_variables(expression.bound_expression(), bound, free);
    Names body_bound = bound;
    body_bound.insert(expression.binding_name());
    collect_free_variables(expression.body(), body_bound, free);
    return;
  }
  case Expression::Kind::Lookup:
    collect_free_variables(expression.lookup_dictionary(), bound, free);
    collect_free_variables(expression.lookup_query(), bound, free);
    return;
  }
  throw EvaluationError("unknown Cajal expression kind");
}

[[nodiscard]] Type type_in_environment(const Expression &expression,
                                       const Environment &environment) {
  Names free;
  collect_free_variables(expression, {}, free);

  Context context;
  context.reserve(free.size());
  for (const NamedValue &binding : environment) {
    if (free.contains(binding.name)) {
      context.push_back({binding.name, binding.value.type()});
    }
  }
  if (context.size() != free.size()) {
    throw EvaluationError(
        "cannot determine a suspended product component's type because its "
        "environment is incomplete");
  }
  return type_check(expression, context);
}

} // namespace

namespace detail {

class Interpreter {
public:
  [[nodiscard]] static Value
  evaluate_expression(const Expression &expression,
                      const Environment &environment) {
    switch (expression.kind()) {
    case Expression::Kind::Variable:
      return find_value(environment, expression.variable_name());

    case Expression::Kind::Unit:
      return Value::unit();

    case Expression::Kind::Tuple: {
      Expression first_expression = expression.first();
      Expression second_expression = expression.second();
      const Type first_type =
          type_in_environment(first_expression, environment);
      const Type second_type =
          type_in_environment(second_expression, environment);
      Environment first_environment = environment;
      Environment second_environment = environment;

      return Value::deferred_product(
          first_type, second_type,
          [expression = std::move(first_expression),
           environment = std::move(first_environment)] {
            return Interpreter::evaluate_expression(expression, environment);
          },
          [expression = std::move(second_expression),
           environment = std::move(second_environment)] {
            return Interpreter::evaluate_expression(expression, environment);
          });
    }

    case Expression::Kind::InjectLeft: {
      Value payload = evaluate_expression(expression.payload(), environment);
      return Value::inject_left(std::move(payload), expression.other_type());
    }

    case Expression::Kind::InjectRight: {
      Value payload = evaluate_expression(expression.payload(), environment);
      return Value::inject_right(expression.other_type(), std::move(payload));
    }

    case Expression::Kind::Dictionary: {
      const Type dictionary_type = type_in_environment(expression, environment);
      Value::DeferredDictionaryEntries entries;
      entries.reserve(expression.dictionary_size());
      for (std::size_t index = 0; index < expression.dictionary_size();
           ++index) {
        Expression key_expression = expression.dictionary_key(index);
        Expression value_expression = expression.dictionary_value(index);
        Environment key_environment = environment;
        Environment value_environment = environment;
        entries.emplace_back(
            [expression = std::move(key_expression),
             environment = std::move(key_environment)] {
              return Interpreter::evaluate_expression(expression, environment);
            },
            [expression = std::move(value_expression),
             environment = std::move(value_environment)] {
              return Interpreter::evaluate_expression(expression, environment);
            });
      }
      return Value::deferred_dictionary(dictionary_type.key(),
                                        dictionary_type.value(),
                                        std::move(entries));
    }

    case Expression::Kind::Sequence:
      static_cast<void>(evaluate_expression(expression.first(), environment));
      return evaluate_expression(expression.second(), environment);

    case Expression::Kind::Projection: {
      Value product =
          evaluate_expression(expression.projected_value(), environment);
      if (product.kind() != Value::Kind::Product) {
        throw EvaluationError("projection requires a product value");
      }
      if (expression.projection_index() == 0) {
        return product.first();
      }
      if (expression.projection_index() == 1) {
        return product.second();
      }
      throw EvaluationError("projection index must be 0 or 1");
    }

    case Expression::Kind::Case: {
      Value sum = evaluate_expression(expression.scrutinee(), environment);
      if (sum.kind() != Value::Kind::Sum) {
        throw EvaluationError("case analysis requires a sum value");
      }

      const bool choose_left = sum.is_left();
      const std::string &binding_name =
          choose_left ? expression.left_binding_name()
                      : expression.right_binding_name();
      Environment branch_environment =
          extend_environment(environment, binding_name, sum.payload());
      return evaluate_expression(choose_left ? expression.left_body()
                                             : expression.right_body(),
                                 branch_environment);
    }

    case Expression::Kind::Let: {
      Value bound =
          evaluate_expression(expression.bound_expression(), environment);
      Environment body_environment =
          extend_environment(environment, expression.binding_name(), bound);
      return evaluate_expression(expression.body(), body_environment);
    }

    case Expression::Kind::Lookup: {
      Value dictionary =
          evaluate_expression(expression.lookup_dictionary(), environment);
      Value query =
          evaluate_expression(expression.lookup_query(), environment);
      if (dictionary.kind() != Value::Kind::Dictionary) {
        throw EvaluationError("lookup requires a dictionary value");
      }

      std::size_t match_index = dictionary.dictionary_size();
      for (std::size_t index = 0; index < dictionary.dictionary_size();
           ++index) {
        if (dictionary.dictionary_key(index) != query) {
          continue;
        }
        if (match_index != dictionary.dictionary_size()) {
          throw EvaluationError("dictionary lookup matched more than one key");
        }
        match_index = index;
      }
      if (match_index == dictionary.dictionary_size()) {
        throw EvaluationError("dictionary lookup found no matching key");
      }
      return dictionary.dictionary_value(match_index);
    }
    }

    throw EvaluationError("unknown Cajal expression kind");
  }
};

} // namespace detail

Value evaluate(const Expression &expression, const Environment &environment) {
  validate_environment(environment);
  const Context context = context_from(environment);
  const Type expected_type = type_check(expression, context);
  Value result = detail::Interpreter::evaluate_expression(expression,
                                                          environment);
  if (result.type() != expected_type) {
    throw EvaluationError("interpreter result type does not match checked type");
  }
  return result;
}

} // namespace riftco_transformer::compiler::cajal
