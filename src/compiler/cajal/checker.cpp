#include "riftco_transformer/compiler/cajal/checker.hpp"

#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace riftco_transformer::compiler::cajal {
namespace {

using Usage = std::set<std::string>;

struct CheckedExpression {
  Type type;
  Usage used_bindings;
};

[[nodiscard]] const Type *find_binding(const Context &context,
                                       const std::string &name) noexcept {
  for (auto binding = context.rbegin(); binding != context.rend(); ++binding) {
    if (binding->name == name) {
      return &binding->type;
    }
  }
  return nullptr;
}

[[nodiscard]] bool has_binding(const Context &context,
                               const std::string &name) noexcept {
  return find_binding(context, name) != nullptr;
}

void validate_context(const Context &context) {
  Usage names;
  for (const Binding &binding : context) {
    if (binding.name.empty()) {
      throw TypeError("context binding names must not be empty");
    }
    if (!names.insert(binding.name).second) {
      throw TypeError("context contains duplicate binding '" + binding.name +
                      "'");
    }
  }
}

void validate_new_binding(const Context &context, const std::string &name,
                          std::string_view construct) {
  if (name.empty()) {
    throw TypeError(std::string(construct) + " binding name must not be empty");
  }
  if (has_binding(context, name)) {
    throw TypeError(std::string(construct) + " binding '" + name +
                    "' would shadow an existing binding");
  }
}

[[nodiscard]] Usage merge_disjoint(Usage left, const Usage &right,
                                   std::string_view construct) {
  for (const std::string &name : right) {
    if (left.contains(name)) {
      throw TypeError("linear binding '" + name +
                      "' is consumed more than once in " +
                      std::string(construct));
    }
  }
  left.insert(right.begin(), right.end());
  return left;
}

void require_same_usage(const Usage &left, const Usage &right,
                        std::string_view construct,
                        std::string_view left_role,
                        std::string_view right_role) {
  for (const std::string &name : left) {
    if (!right.contains(name)) {
      throw TypeError(std::string(construct) +
                      " consume different linear bindings: " +
                      std::string(left_role) + " consumes '" + name +
                      "' but " + std::string(right_role) + " does not");
    }
  }
  for (const std::string &name : right) {
    if (!left.contains(name)) {
      throw TypeError(std::string(construct) +
                      " consume different linear bindings: " +
                      std::string(right_role) + " consumes '" + name +
                      "' but " + std::string(left_role) + " does not");
    }
  }
}

[[nodiscard]] bool is_enum_type(const Type &type) {
  if (type.kind() == Type::Kind::Unit) {
    return true;
  }
  return type.kind() == Type::Kind::Sum && is_enum_type(type.left()) &&
         is_enum_type(type.right());
}

[[nodiscard]] CheckedExpression check_expression(const Expression &expression,
                                                 const Context &context);

[[nodiscard]] CheckedExpression check_variable(const Expression &expression,
                                               const Context &context) {
  const std::string &name = expression.variable_name();
  if (name.empty()) {
    throw TypeError("variable name must not be empty");
  }

  const Type *const type = find_binding(context, name);
  if (type == nullptr) {
    throw TypeError("unknown variable '" + name + "'");
  }
  return {*type, Usage{name}};
}

[[nodiscard]] CheckedExpression check_tuple(const Expression &expression,
                                            const Context &context) {
  CheckedExpression first = check_expression(expression.first(), context);
  CheckedExpression second = check_expression(expression.second(), context);
  require_same_usage(first.used_bindings, second.used_bindings,
                     "tuple components", "the first component",
                     "the second component");
  return {Type::product(std::move(first.type), std::move(second.type)),
          std::move(first.used_bindings)};
}

[[nodiscard]] CheckedExpression check_injection(const Expression &expression,
                                                const Context &context) {
  CheckedExpression payload = check_expression(expression.payload(), context);
  if (expression.kind() == Expression::Kind::InjectLeft) {
    return {Type::sum(std::move(payload.type), expression.other_type()),
            std::move(payload.used_bindings)};
  }
  return {Type::sum(expression.other_type(), std::move(payload.type)),
          std::move(payload.used_bindings)};
}

[[nodiscard]] CheckedExpression check_dictionary(const Expression &expression,
                                                 const Context &context) {
  if (expression.dictionary_size() == 0) {
    throw TypeError(
        "dictionary must contain at least one entry so its type is known");
  }

  CheckedExpression first_key =
      check_expression(expression.dictionary_key(0), context);
  CheckedExpression first_value =
      check_expression(expression.dictionary_value(0), context);
  const Type key_type = first_key.type;
  const Type value_type = first_value.type;
  Usage key_usage = std::move(first_key.used_bindings);
  Usage value_usage = std::move(first_value.used_bindings);

  for (std::size_t index = 1; index < expression.dictionary_size(); ++index) {
    CheckedExpression key =
        check_expression(expression.dictionary_key(index), context);
    CheckedExpression value =
        check_expression(expression.dictionary_value(index), context);

    if (key.type != key_type) {
      throw TypeError("dictionary key at entry " + std::to_string(index) +
                      " has type " + key.type.to_string() + ", expected " +
                      key_type.to_string());
    }
    if (value.type != value_type) {
      throw TypeError("dictionary value at entry " + std::to_string(index) +
                      " has type " + value.type.to_string() + ", expected " +
                      value_type.to_string());
    }

    require_same_usage(key_usage, key.used_bindings, "dictionary keys",
                       "the first key",
                       "key " + std::to_string(index));
    require_same_usage(value_usage, value.used_bindings, "dictionary values",
                       "the first value",
                       "value " + std::to_string(index));
  }

  Usage used = merge_disjoint(std::move(key_usage), value_usage,
                              "dictionary key and value contexts");
  return {Type::dictionary(key_type, value_type), std::move(used)};
}

[[nodiscard]] CheckedExpression check_sequence(const Expression &expression,
                                               const Context &context) {
  CheckedExpression first = check_expression(expression.first(), context);
  if (first.type.kind() != Type::Kind::Unit) {
    throw TypeError("sequence first expression must have type Unit, got " +
                    first.type.to_string());
  }

  CheckedExpression second = check_expression(expression.second(), context);
  Usage used = merge_disjoint(std::move(first.used_bindings),
                              second.used_bindings, "sequence");
  return {std::move(second.type), std::move(used)};
}

[[nodiscard]] CheckedExpression check_projection(const Expression &expression,
                                                 const Context &context) {
  const std::size_t index = expression.projection_index();
  if (index > 1) {
    throw TypeError("projection index must be 0 or 1, got " +
                    std::to_string(index));
  }

  CheckedExpression value =
      check_expression(expression.projected_value(), context);
  if (value.type.kind() != Type::Kind::Product) {
    throw TypeError("projection expects a Product, got " +
                    value.type.to_string());
  }

  Type result_type = index == 0 ? value.type.first() : value.type.second();
  return {std::move(result_type), std::move(value.used_bindings)};
}

[[nodiscard]] CheckedExpression check_case(const Expression &expression,
                                           const Context &context) {
  const std::string &left_name = expression.left_binding_name();
  const std::string &right_name = expression.right_binding_name();
  validate_new_binding(context, left_name, "case left");
  validate_new_binding(context, right_name, "case right");

  CheckedExpression scrutinee =
      check_expression(expression.scrutinee(), context);
  if (scrutinee.type.kind() != Type::Kind::Sum) {
    throw TypeError("case scrutinee must have a Sum type, got " +
                    scrutinee.type.to_string());
  }

  Context left_context = context;
  left_context.push_back({left_name, scrutinee.type.left()});
  CheckedExpression left =
      check_expression(expression.left_body(), left_context);
  if (!left.used_bindings.erase(left_name)) {
    throw TypeError("case left binding '" + left_name +
                    "' must be consumed exactly once");
  }

  Context right_context = context;
  right_context.push_back({right_name, scrutinee.type.right()});
  CheckedExpression right =
      check_expression(expression.right_body(), right_context);
  if (!right.used_bindings.erase(right_name)) {
    throw TypeError("case right binding '" + right_name +
                    "' must be consumed exactly once");
  }

  if (left.type != right.type) {
    throw TypeError("case branches must have the same result type: left is " +
                    left.type.to_string() + ", right is " +
                    right.type.to_string());
  }
  require_same_usage(left.used_bindings, right.used_bindings, "case branches",
                     "the left branch", "the right branch");

  Usage used = merge_disjoint(std::move(scrutinee.used_bindings),
                              left.used_bindings, "case expression");
  return {std::move(left.type), std::move(used)};
}

[[nodiscard]] CheckedExpression check_let(const Expression &expression,
                                          const Context &context) {
  const std::string &name = expression.binding_name();
  validate_new_binding(context, name, "let");

  CheckedExpression bound =
      check_expression(expression.bound_expression(), context);
  Context body_context = context;
  body_context.push_back({name, bound.type});
  CheckedExpression body = check_expression(expression.body(), body_context);
  if (!body.used_bindings.erase(name)) {
    throw TypeError("let binding '" + name + "' must be consumed exactly once");
  }

  Usage used = merge_disjoint(std::move(bound.used_bindings),
                              body.used_bindings, "let expression");
  return {std::move(body.type), std::move(used)};
}

[[nodiscard]] CheckedExpression check_lookup(const Expression &expression,
                                             const Context &context) {
  CheckedExpression dictionary =
      check_expression(expression.lookup_dictionary(), context);
  if (dictionary.type.kind() != Type::Kind::Dictionary) {
    throw TypeError("lookup expects a Dictionary, got " +
                    dictionary.type.to_string());
  }

  CheckedExpression query =
      check_expression(expression.lookup_query(), context);
  if (query.type != dictionary.type.key()) {
    throw TypeError("lookup query has type " + query.type.to_string() +
                    ", but the dictionary key type is " +
                    dictionary.type.key().to_string());
  }
  if (!is_enum_type(query.type)) {
    throw TypeError(
        "identity dictionary lookup requires an enum key type built only "
        "from Unit and Sum, got " +
        query.type.to_string());
  }

  Usage used = merge_disjoint(std::move(dictionary.used_bindings),
                              query.used_bindings, "dictionary lookup");
  return {dictionary.type.value(), std::move(used)};
}

CheckedExpression check_expression(const Expression &expression,
                                   const Context &context) {
  switch (expression.kind()) {
  case Expression::Kind::Variable:
    return check_variable(expression, context);
  case Expression::Kind::Unit:
    return {Type::unit(), {}};
  case Expression::Kind::Tuple:
    return check_tuple(expression, context);
  case Expression::Kind::InjectLeft:
  case Expression::Kind::InjectRight:
    return check_injection(expression, context);
  case Expression::Kind::Dictionary:
    return check_dictionary(expression, context);
  case Expression::Kind::Sequence:
    return check_sequence(expression, context);
  case Expression::Kind::Projection:
    return check_projection(expression, context);
  case Expression::Kind::Case:
    return check_case(expression, context);
  case Expression::Kind::Let:
    return check_let(expression, context);
  case Expression::Kind::Lookup:
    return check_lookup(expression, context);
  }
  throw TypeError("expression has an unknown kind");
}

} // namespace

Type type_check(const Expression &expression, const Context &context) {
  validate_context(context);
  CheckedExpression checked = check_expression(expression, context);
  for (const Binding &binding : context) {
    if (!checked.used_bindings.contains(binding.name)) {
      throw TypeError("linear context binding '" + binding.name +
                      "' is not consumed by the expression");
    }
  }
  return std::move(checked.type);
}

} // namespace riftco_transformer::compiler::cajal
