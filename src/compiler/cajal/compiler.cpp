#include "riftco_transformer/compiler/cajal/compiler.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace riftco_transformer::compiler::cajal {
namespace {

using Names = std::set<std::string>;

struct LoweredExpression {
  Type type;
  Context context;
  MultilinearMap map;
};

[[noreturn]] void fail(std::string message) {
  throw CompilationError(std::move(message));
}

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right,
                                      std::string_view role) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    fail(std::string(role) + " exceeds size_t");
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right,
                                           std::string_view role) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    fail(std::string(role) + " exceeds size_t");
  }
  return left * right;
}

[[nodiscard]] std::size_t
combination_count(std::span<const std::size_t> dimensions) {
  std::size_t count = 1;
  for (const std::size_t dimension : dimensions) {
    if (dimension == 0) {
      fail("a compiled input axis has zero dimension");
    }
    count =
        checked_multiply(count, dimension, "compiled input combination count");
  }
  return count;
}

[[nodiscard]] std::vector<std::size_t>
context_dimensions(const Context &context) {
  std::vector<std::size_t> dimensions;
  dimensions.reserve(context.size());
  for (const Binding &binding : context) {
    dimensions.push_back(binding.type.dimension());
  }
  return dimensions;
}

[[nodiscard]] std::vector<double>
zero_coefficients(std::size_t output_dimension, const Context &context) {
  const std::vector<std::size_t> dimensions = context_dimensions(context);
  const std::size_t inputs = combination_count(dimensions);
  const std::size_t count =
      checked_multiply(output_dimension, inputs, "compiled coefficient count");
  if (count > std::vector<double>{}.max_size()) {
    fail("compiled coefficient count exceeds vector capacity");
  }
  return std::vector<double>(count, 0.0);
}

[[nodiscard]] const Binding *find_binding(const Context &context,
                                          const std::string &name) noexcept {
  for (const Binding &binding : context) {
    if (binding.name == name) {
      return &binding;
    }
  }
  return nullptr;
}

[[nodiscard]] bool same_context(const Context &left,
                                const Context &right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (left[index].name != right[index].name ||
        left[index].type != right[index].type) {
      return false;
    }
  }
  return true;
}

void require_same_context(const Context &left, const Context &right,
                          std::string_view construct) {
  if (!same_context(left, right)) {
    fail(std::string(construct) +
         " did not lower to identical ordered input contexts");
  }
}

void add_names(Names &destination, const Names &source) {
  destination.insert(source.begin(), source.end());
}

[[nodiscard]] Names free_variables(const Expression &expression) {
  switch (expression.kind()) {
  case Expression::Kind::Variable:
    return {expression.variable_name()};
  case Expression::Kind::Unit:
    return {};
  case Expression::Kind::Tuple:
  case Expression::Kind::Sequence: {
    Names result = free_variables(expression.first());
    add_names(result, free_variables(expression.second()));
    return result;
  }
  case Expression::Kind::InjectLeft:
  case Expression::Kind::InjectRight:
    return free_variables(expression.payload());
  case Expression::Kind::Dictionary: {
    Names result;
    for (std::size_t index = 0; index < expression.dictionary_size(); ++index) {
      add_names(result, free_variables(expression.dictionary_key(index)));
      add_names(result, free_variables(expression.dictionary_value(index)));
    }
    return result;
  }
  case Expression::Kind::Projection:
    return free_variables(expression.projected_value());
  case Expression::Kind::Case: {
    Names result = free_variables(expression.scrutinee());
    Names left = free_variables(expression.left_body());
    Names right = free_variables(expression.right_body());
    left.erase(expression.left_binding_name());
    right.erase(expression.right_binding_name());
    add_names(result, left);
    add_names(result, right);
    return result;
  }
  case Expression::Kind::Let: {
    Names result = free_variables(expression.bound_expression());
    Names body = free_variables(expression.body());
    body.erase(expression.binding_name());
    add_names(result, body);
    return result;
  }
  case Expression::Kind::Lookup: {
    Names result = free_variables(expression.lookup_dictionary());
    add_names(result, free_variables(expression.lookup_query()));
    return result;
  }
  }
  fail("cannot collect free variables for an unknown expression kind");
}

[[nodiscard]] Context used_context(const Expression &expression,
                                   const Context &available) {
  const Names free = free_variables(expression);
  Names found;
  Context result;
  result.reserve(free.size());
  for (const Binding &binding : available) {
    if (free.contains(binding.name)) {
      if (!found.insert(binding.name).second) {
        fail("available context contains duplicate binding '" + binding.name +
             "'");
      }
      result.push_back(binding);
    }
  }
  for (const std::string &name : free) {
    if (!found.contains(name)) {
      fail("free variable '" + name + "' is absent from the compiler context");
    }
  }
  return result;
}

void require_disjoint_cover(const Context &left, const Context &right,
                            const Context &combined,
                            std::string_view construct) {
  std::unordered_set<std::string> names;
  names.reserve(left.size() + right.size());
  for (const Binding &binding : left) {
    if (!names.insert(binding.name).second) {
      fail(std::string(construct) + " has a duplicate left input '" +
           binding.name + "'");
    }
  }
  for (const Binding &binding : right) {
    if (!names.insert(binding.name).second) {
      fail(std::string(construct) + " input contexts overlap at '" +
           binding.name + "'");
    }
  }
  if (names.size() != combined.size()) {
    fail(std::string(construct) + " inputs do not cover the result context");
  }
  for (const Binding &binding : combined) {
    if (!names.contains(binding.name)) {
      fail(std::string(construct) + " is missing result input '" +
           binding.name + "'");
    }
    const Binding *source = find_binding(left, binding.name);
    if (source == nullptr) {
      source = find_binding(right, binding.name);
    }
    if (source == nullptr || source->type != binding.type) {
      fail(std::string(construct) + " has inconsistent type for input '" +
           binding.name + "'");
    }
  }
}

[[nodiscard]] std::vector<std::size_t>
axis_positions(const Context &part, const Context &whole,
               std::string_view construct) {
  std::vector<std::size_t> positions;
  positions.reserve(part.size());
  for (const Binding &binding : part) {
    bool found = false;
    for (std::size_t index = 0; index < whole.size(); ++index) {
      if (whole[index].name == binding.name) {
        if (whole[index].type != binding.type) {
          fail(std::string(construct) + " has inconsistent input-axis types");
        }
        positions.push_back(index);
        found = true;
        break;
      }
    }
    if (!found) {
      fail(std::string(construct) + " cannot locate input axis '" +
           binding.name + "'");
    }
  }
  return positions;
}

void decode_flat_index(std::size_t flat,
                       std::span<const std::size_t> dimensions,
                       std::span<std::size_t> indices) {
  if (indices.size() != dimensions.size()) {
    fail("compiler index rank does not match its input shape");
  }
  for (std::size_t axis = dimensions.size(); axis-- > 0;) {
    indices[axis] = flat % dimensions[axis];
    flat /= dimensions[axis];
  }
  if (flat != 0) {
    fail("compiler flat input index is out of range");
  }
}

[[nodiscard]] std::size_t
projected_flat_index(std::span<const std::size_t> whole_indices,
                     std::span<const std::size_t> positions,
                     std::span<const std::size_t> part_dimensions) {
  if (positions.size() != part_dimensions.size()) {
    fail("compiler axis projection rank mismatch");
  }
  std::size_t flat = 0;
  for (std::size_t axis = 0; axis < positions.size(); ++axis) {
    if (positions[axis] >= whole_indices.size() ||
        whole_indices[positions[axis]] >= part_dimensions[axis]) {
      fail("compiler axis projection is out of range");
    }
    flat = checked_multiply(flat, part_dimensions[axis],
                            "compiler projected input index");
    flat = checked_add(flat, whole_indices[positions[axis]],
                       "compiler projected input index");
  }
  return flat;
}

[[nodiscard]] std::size_t map_input_count(const MultilinearMap &map) {
  return combination_count(map.input_dimensions());
}

[[nodiscard]] double coefficient(const MultilinearMap &map, std::size_t output,
                                 std::size_t flat_input) {
  const std::size_t inputs = map_input_count(map);
  if (output >= map.output_dimension() || flat_input >= inputs) {
    fail("compiler coefficient access is out of range");
  }
  return map.coefficients()[output * inputs + flat_input];
}

[[nodiscard]] LoweredExpression make_lowered(Type type, Context context,
                                             std::vector<double> coefficients) {
  const std::size_t output_dimension = type.dimension();
  std::vector<std::size_t> dimensions = context_dimensions(context);
  MultilinearMap map(std::move(dimensions), output_dimension,
                     std::move(coefficients));
  return {std::move(type), std::move(context), std::move(map)};
}

void copy_output_block(std::span<const double> source,
                       std::size_t source_output_dimension,
                       std::size_t input_count, std::size_t output_offset,
                       std::span<double> destination) {
  const std::size_t source_count = checked_multiply(
      source_output_dimension, input_count, "source coefficient count");
  if (source.size() != source_count) {
    fail("source coefficient map has an inconsistent shape");
  }
  const std::size_t destination_offset =
      checked_multiply(output_offset, input_count, "coefficient output offset");
  if (destination_offset > destination.size() ||
      source.size() > destination.size() - destination_offset) {
    fail("coefficient output block does not fit its destination");
  }
  std::copy(source.begin(), source.end(),
            destination.begin() +
                static_cast<std::ptrdiff_t>(destination_offset));
}

[[nodiscard]] LoweredExpression lower_expression(const Expression &expression,
                                                 const Context &available);

[[nodiscard]] LoweredExpression lower_variable(const Expression &expression,
                                               Context context) {
  const Binding *binding = find_binding(context, expression.variable_name());
  if (binding == nullptr || context.size() != 1) {
    fail("variable did not lower to exactly one input axis");
  }
  Type type = binding->type;
  return {type, std::move(context), MultilinearMap::identity(type.dimension())};
}

[[nodiscard]] LoweredExpression lower_unit(Context context) {
  if (!context.empty()) {
    fail("Unit unexpectedly acquired an input context");
  }
  Type type = Type::unit();
  return make_lowered(std::move(type), std::move(context), {1.0});
}

[[nodiscard]] LoweredExpression lower_injection(const Expression &expression,
                                                Context context) {
  LoweredExpression payload = lower_expression(expression.payload(), context);
  require_same_context(payload.context, context, "sum injection");

  const bool inject_left = expression.kind() == Expression::Kind::InjectLeft;
  Type type = inject_left ? Type::sum(payload.type, expression.other_type())
                          : Type::sum(expression.other_type(), payload.type);
  const std::size_t output_offset =
      inject_left ? 0 : expression.other_type().dimension();
  const std::size_t inputs = map_input_count(payload.map);
  std::vector<double> coefficients =
      zero_coefficients(type.dimension(), context);
  copy_output_block(payload.map.coefficients(), payload.map.output_dimension(),
                    inputs, output_offset, coefficients);
  return make_lowered(std::move(type), std::move(context),
                      std::move(coefficients));
}

[[nodiscard]] LoweredExpression lower_tuple(const Expression &expression,
                                            Context context) {
  LoweredExpression first = lower_expression(expression.first(), context);
  LoweredExpression second = lower_expression(expression.second(), context);
  require_same_context(first.context, second.context, "tuple components");
  require_same_context(first.context, context, "tuple");

  Type type = Type::product(first.type, second.type);
  const std::size_t inputs = map_input_count(first.map);
  if (inputs != map_input_count(second.map)) {
    fail("tuple components have inconsistent input shapes");
  }
  std::vector<double> coefficients =
      zero_coefficients(type.dimension(), context);
  copy_output_block(first.map.coefficients(), first.map.output_dimension(),
                    inputs, 0, coefficients);
  copy_output_block(second.map.coefficients(), second.map.output_dimension(),
                    inputs, first.map.output_dimension(), coefficients);
  return make_lowered(std::move(type), std::move(context),
                      std::move(coefficients));
}

[[nodiscard]] LoweredExpression lower_projection(const Expression &expression,
                                                 Context context) {
  LoweredExpression value =
      lower_expression(expression.projected_value(), context);
  require_same_context(value.context, context, "projection");
  if (value.type.kind() != Type::Kind::Product ||
      expression.projection_index() > 1) {
    fail("type checker admitted an invalid product projection");
  }

  const bool first = expression.projection_index() == 0;
  Type type = first ? value.type.first() : value.type.second();
  const std::size_t output_offset = first ? 0 : value.type.first().dimension();
  const std::size_t inputs = map_input_count(value.map);
  std::vector<double> coefficients =
      zero_coefficients(type.dimension(), context);
  const std::size_t source_offset =
      checked_multiply(output_offset, inputs, "projection source offset");
  const std::size_t count = checked_multiply(type.dimension(), inputs,
                                             "projection coefficient count");
  if (source_offset > value.map.coefficients().size() ||
      count > value.map.coefficients().size() - source_offset) {
    fail("projected coefficient block is out of range");
  }
  std::copy_n(value.map.coefficients().begin() +
                  static_cast<std::ptrdiff_t>(source_offset),
              count, coefficients.begin());
  return make_lowered(std::move(type), std::move(context),
                      std::move(coefficients));
}

[[nodiscard]] LoweredExpression lower_sequence(const Expression &expression,
                                               Context context) {
  LoweredExpression first = lower_expression(expression.first(), context);
  LoweredExpression second = lower_expression(expression.second(), context);
  if (first.type.kind() != Type::Kind::Unit ||
      first.map.output_dimension() != 1) {
    fail("type checker admitted a non-Unit sequence prefix");
  }
  require_disjoint_cover(first.context, second.context, context, "sequence");

  Type type = second.type;
  const std::vector<std::size_t> dimensions = context_dimensions(context);
  const std::size_t inputs = combination_count(dimensions);
  const std::vector<std::size_t> first_positions =
      axis_positions(first.context, context, "sequence prefix");
  const std::vector<std::size_t> second_positions =
      axis_positions(second.context, context, "sequence result");
  std::vector<std::size_t> indices(context.size(), 0);
  std::vector<double> coefficients =
      zero_coefficients(type.dimension(), context);

  for (std::size_t flat = 0; flat < inputs; ++flat) {
    decode_flat_index(flat, dimensions, indices);
    const std::size_t first_flat = projected_flat_index(
        indices, first_positions, first.map.input_dimensions());
    const std::size_t second_flat = projected_flat_index(
        indices, second_positions, second.map.input_dimensions());
    const double prefix = coefficient(first.map, 0, first_flat);
    for (std::size_t output = 0; output < type.dimension(); ++output) {
      coefficients[output * inputs + flat] =
          prefix * coefficient(second.map, output, second_flat);
    }
  }
  return make_lowered(std::move(type), std::move(context),
                      std::move(coefficients));
}

[[nodiscard]] LoweredExpression lower_dictionary(const Expression &expression,
                                                 Context context) {
  if (expression.dictionary_size() == 0) {
    fail("type checker admitted an empty dictionary");
  }

  std::vector<LoweredExpression> keys;
  std::vector<LoweredExpression> values;
  keys.reserve(expression.dictionary_size());
  values.reserve(expression.dictionary_size());
  for (std::size_t entry = 0; entry < expression.dictionary_size(); ++entry) {
    keys.push_back(lower_expression(expression.dictionary_key(entry), context));
    values.push_back(
        lower_expression(expression.dictionary_value(entry), context));
  }

  const Type key_type = keys.front().type;
  const Type value_type = values.front().type;
  for (std::size_t entry = 0; entry < keys.size(); ++entry) {
    if (keys[entry].type != key_type || values[entry].type != value_type) {
      fail("dictionary entries changed type after type checking");
    }
    require_same_context(keys.front().context, keys[entry].context,
                         "dictionary keys");
    require_same_context(values.front().context, values[entry].context,
                         "dictionary values");
  }
  require_disjoint_cover(keys.front().context, values.front().context, context,
                         "dictionary key/value contexts");

  Type type = Type::dictionary(key_type, value_type);
  const std::size_t key_dimension = key_type.dimension();
  const std::size_t value_dimension = value_type.dimension();
  const std::vector<std::size_t> dimensions = context_dimensions(context);
  const std::size_t inputs = combination_count(dimensions);
  const std::vector<std::size_t> key_positions =
      axis_positions(keys.front().context, context, "dictionary keys");
  const std::vector<std::size_t> value_positions =
      axis_positions(values.front().context, context, "dictionary values");
  std::vector<std::size_t> indices(context.size(), 0);
  std::vector<double> coefficients =
      zero_coefficients(type.dimension(), context);

  for (std::size_t flat = 0; flat < inputs; ++flat) {
    decode_flat_index(flat, dimensions, indices);
    const std::size_t key_flat = projected_flat_index(
        indices, key_positions, keys.front().map.input_dimensions());
    const std::size_t value_flat = projected_flat_index(
        indices, value_positions, values.front().map.input_dimensions());
    for (std::size_t entry = 0; entry < keys.size(); ++entry) {
      for (std::size_t value = 0; value < value_dimension; ++value) {
        const double value_coefficient =
            coefficient(values[entry].map, value, value_flat);
        for (std::size_t key = 0; key < key_dimension; ++key) {
          const std::size_t output = value * key_dimension + key;
          coefficients[output * inputs + flat] +=
              value_coefficient * coefficient(keys[entry].map, key, key_flat);
        }
      }
    }
  }
  return make_lowered(std::move(type), std::move(context),
                      std::move(coefficients));
}

[[nodiscard]] Context binder_external_context(const LoweredExpression &body,
                                              const std::string &name,
                                              const Type &type,
                                              std::string_view construct) {
  if (body.context.empty() || body.context.back().name != name ||
      body.context.back().type != type) {
    fail(std::string(construct) +
         " binder is not the final compiled input axis");
  }
  return Context(body.context.begin(), body.context.end() - 1);
}

[[nodiscard]] LoweredExpression lower_let(const Expression &expression,
                                          Context context) {
  LoweredExpression bound =
      lower_expression(expression.bound_expression(), context);
  Context body_available = context;
  body_available.push_back({expression.binding_name(), bound.type});
  LoweredExpression body = lower_expression(expression.body(), body_available);
  const Context external = binder_external_context(
      body, expression.binding_name(), bound.type, "let");
  require_disjoint_cover(bound.context, external, context, "let");

  Type type = body.type;
  const std::size_t bound_dimension = bound.type.dimension();
  const std::vector<std::size_t> dimensions = context_dimensions(context);
  const std::size_t inputs = combination_count(dimensions);
  const std::vector<std::size_t> bound_positions =
      axis_positions(bound.context, context, "let bound expression");
  const std::vector<std::size_t> external_positions =
      axis_positions(external, context, "let body");
  const std::vector<std::size_t> external_dimensions =
      context_dimensions(external);
  const std::size_t external_inputs = combination_count(external_dimensions);
  if (map_input_count(body.map) != checked_multiply(external_inputs,
                                                    bound_dimension,
                                                    "let body input count")) {
    fail("let body map does not have a final binder axis");
  }

  std::vector<std::size_t> indices(context.size(), 0);
  std::vector<double> coefficients =
      zero_coefficients(type.dimension(), context);
  for (std::size_t flat = 0; flat < inputs; ++flat) {
    decode_flat_index(flat, dimensions, indices);
    const std::size_t bound_flat = projected_flat_index(
        indices, bound_positions, bound.map.input_dimensions());
    const std::size_t external_flat =
        projected_flat_index(indices, external_positions, external_dimensions);
    for (std::size_t output = 0; output < type.dimension(); ++output) {
      double sum = 0.0;
      for (std::size_t binder = 0; binder < bound_dimension; ++binder) {
        const std::size_t body_flat = external_flat * bound_dimension + binder;
        sum += coefficient(body.map, output, body_flat) *
               coefficient(bound.map, binder, bound_flat);
      }
      coefficients[output * inputs + flat] = sum;
    }
  }
  return make_lowered(std::move(type), std::move(context),
                      std::move(coefficients));
}

[[nodiscard]] LoweredExpression lower_case(const Expression &expression,
                                           Context context) {
  LoweredExpression scrutinee =
      lower_expression(expression.scrutinee(), context);
  if (scrutinee.type.kind() != Type::Kind::Sum) {
    fail("type checker admitted a non-sum case scrutinee");
  }

  const Type left_type = scrutinee.type.left();
  const Type right_type = scrutinee.type.right();
  Context left_available = context;
  left_available.push_back({expression.left_binding_name(), left_type});
  Context right_available = context;
  right_available.push_back({expression.right_binding_name(), right_type});
  LoweredExpression left =
      lower_expression(expression.left_body(), left_available);
  LoweredExpression right =
      lower_expression(expression.right_body(), right_available);
  if (left.type != right.type) {
    fail("case branch result types changed after type checking");
  }
  const Context left_external = binder_external_context(
      left, expression.left_binding_name(), left_type, "case left branch");
  const Context right_external = binder_external_context(
      right, expression.right_binding_name(), right_type, "case right branch");
  require_same_context(left_external, right_external, "case branches");
  require_disjoint_cover(scrutinee.context, left_external, context,
                         "case expression");

  Type type = left.type;
  const std::size_t left_dimension = left_type.dimension();
  const std::size_t right_dimension = right_type.dimension();
  const std::vector<std::size_t> dimensions = context_dimensions(context);
  const std::size_t inputs = combination_count(dimensions);
  const std::vector<std::size_t> scrutinee_positions =
      axis_positions(scrutinee.context, context, "case scrutinee");
  const std::vector<std::size_t> external_positions =
      axis_positions(left_external, context, "case branches");
  const std::vector<std::size_t> external_dimensions =
      context_dimensions(left_external);
  const std::size_t external_inputs = combination_count(external_dimensions);
  if (map_input_count(left.map) !=
          checked_multiply(external_inputs, left_dimension,
                           "left case body input count") ||
      map_input_count(right.map) !=
          checked_multiply(external_inputs, right_dimension,
                           "right case body input count")) {
    fail("case branch map does not have a final binder axis");
  }

  std::vector<std::size_t> indices(context.size(), 0);
  std::vector<double> coefficients =
      zero_coefficients(type.dimension(), context);
  for (std::size_t flat = 0; flat < inputs; ++flat) {
    decode_flat_index(flat, dimensions, indices);
    const std::size_t scrutinee_flat = projected_flat_index(
        indices, scrutinee_positions, scrutinee.map.input_dimensions());
    const std::size_t external_flat =
        projected_flat_index(indices, external_positions, external_dimensions);
    for (std::size_t output = 0; output < type.dimension(); ++output) {
      double sum = 0.0;
      for (std::size_t binder = 0; binder < left_dimension; ++binder) {
        sum += coefficient(left.map, output,
                           external_flat * left_dimension + binder) *
               coefficient(scrutinee.map, binder, scrutinee_flat);
      }
      for (std::size_t binder = 0; binder < right_dimension; ++binder) {
        sum +=
            coefficient(right.map, output,
                        external_flat * right_dimension + binder) *
            coefficient(scrutinee.map, left_dimension + binder, scrutinee_flat);
      }
      coefficients[output * inputs + flat] = sum;
    }
  }
  return make_lowered(std::move(type), std::move(context),
                      std::move(coefficients));
}

[[nodiscard]] LoweredExpression lower_lookup(const Expression &expression,
                                             Context context) {
  LoweredExpression dictionary =
      lower_expression(expression.lookup_dictionary(), context);
  LoweredExpression query =
      lower_expression(expression.lookup_query(), context);
  if (dictionary.type.kind() != Type::Kind::Dictionary ||
      query.type != dictionary.type.key()) {
    fail("type checker admitted an invalid dictionary lookup");
  }
  require_disjoint_cover(dictionary.context, query.context, context,
                         "dictionary lookup");

  Type type = dictionary.type.value();
  const std::size_t key_dimension = query.type.dimension();
  const std::vector<std::size_t> dimensions = context_dimensions(context);
  const std::size_t inputs = combination_count(dimensions);
  const std::vector<std::size_t> dictionary_positions =
      axis_positions(dictionary.context, context, "lookup dictionary");
  const std::vector<std::size_t> query_positions =
      axis_positions(query.context, context, "lookup query");
  std::vector<std::size_t> indices(context.size(), 0);
  std::vector<double> coefficients =
      zero_coefficients(type.dimension(), context);

  for (std::size_t flat = 0; flat < inputs; ++flat) {
    decode_flat_index(flat, dimensions, indices);
    const std::size_t dictionary_flat = projected_flat_index(
        indices, dictionary_positions, dictionary.map.input_dimensions());
    const std::size_t query_flat = projected_flat_index(
        indices, query_positions, query.map.input_dimensions());
    for (std::size_t value = 0; value < type.dimension(); ++value) {
      double sum = 0.0;
      for (std::size_t key = 0; key < key_dimension; ++key) {
        sum += coefficient(dictionary.map, value * key_dimension + key,
                           dictionary_flat) *
               coefficient(query.map, key, query_flat);
      }
      coefficients[value * inputs + flat] = sum;
    }
  }
  return make_lowered(std::move(type), std::move(context),
                      std::move(coefficients));
}

LoweredExpression lower_expression(const Expression &expression,
                                   const Context &available) {
  Context context = used_context(expression, available);
  switch (expression.kind()) {
  case Expression::Kind::Variable:
    return lower_variable(expression, std::move(context));
  case Expression::Kind::Unit:
    return lower_unit(std::move(context));
  case Expression::Kind::Tuple:
    return lower_tuple(expression, std::move(context));
  case Expression::Kind::InjectLeft:
  case Expression::Kind::InjectRight:
    return lower_injection(expression, std::move(context));
  case Expression::Kind::Dictionary:
    return lower_dictionary(expression, std::move(context));
  case Expression::Kind::Sequence:
    return lower_sequence(expression, std::move(context));
  case Expression::Kind::Projection:
    return lower_projection(expression, std::move(context));
  case Expression::Kind::Case:
    return lower_case(expression, std::move(context));
  case Expression::Kind::Let:
    return lower_let(expression, std::move(context));
  case Expression::Kind::Lookup:
    return lower_lookup(expression, std::move(context));
  }
  fail("cannot lower an unknown expression kind");
}

} // namespace

CompiledProgram::CompiledProgram(Type output_type, Context input_context,
                                 MultilinearMap map)
    : output_type_(std::move(output_type)),
      input_context_(std::move(input_context)), map_(std::move(map)) {}

const Type &CompiledProgram::output_type() const noexcept {
  return output_type_;
}

const Context &CompiledProgram::input_context() const noexcept {
  return input_context_;
}

const MultilinearMap &CompiledProgram::map() const noexcept { return map_; }

EncodedValue CompiledProgram::apply(const Environment &environment) const {
  std::unordered_set<std::string> names;
  names.reserve(environment.size());
  for (const NamedValue &named : environment) {
    if (named.name.empty()) {
      throw CompilationError("compiled environment names must not be empty");
    }
    if (!names.insert(named.name).second) {
      throw CompilationError("compiled environment contains duplicate input '" +
                             named.name + "'");
    }
    const Binding *binding = find_binding(input_context_, named.name);
    if (binding == nullptr) {
      throw CompilationError("compiled environment contains extra input '" +
                             named.name + "'");
    }
    const Type actual_type = named.value.type();
    if (actual_type != binding->type) {
      throw CompilationError("compiled input '" + named.name + "' has type " +
                             actual_type.to_string() + ", expected " +
                             binding->type.to_string());
    }
  }

  std::vector<EncodedValue> inputs;
  inputs.reserve(input_context_.size());
  for (const Binding &binding : input_context_) {
    const NamedValue *value = nullptr;
    for (const NamedValue &candidate : environment) {
      if (candidate.name == binding.name) {
        value = &candidate;
        break;
      }
    }
    if (value == nullptr) {
      throw CompilationError("compiled environment is missing input '" +
                             binding.name + "'");
    }
    inputs.push_back(encode(value->value));
  }
  return map_.apply(inputs);
}

CompiledProgram compile(const Expression &expression,
                        const Context &ordered_context) {
  const Type checked_type = type_check(expression, ordered_context);
  LoweredExpression lowered = lower_expression(expression, ordered_context);
  if (lowered.type != checked_type) {
    fail("recursive compiler result type differs from the checked type");
  }
  require_same_context(lowered.context, ordered_context,
                       "top-level compilation");
  return CompiledProgram(std::move(lowered.type), std::move(lowered.context),
                         std::move(lowered.map));
}

} // namespace riftco_transformer::compiler::cajal
