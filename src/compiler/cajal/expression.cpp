#include "riftco_transformer/compiler/cajal/expression.hpp"

#include <optional>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::compiler::cajal {

struct Expression::Node {
  explicit Node(Kind node_kind) : kind(node_kind) {}

  Kind kind;
  std::string name;
  std::string alternate_name;
  std::optional<Expression> first;
  std::optional<Expression> second;
  std::optional<Expression> third;
  std::optional<Type> type;
  std::size_t index = 0;
  DictionaryEntries entries;
};

namespace {

void require_name(const std::string &name, const char *role) {
  if (name.empty()) {
    throw std::invalid_argument(std::string("Cajal ") + role +
                                " name must not be empty");
  }
}

} // namespace

Expression::Expression(std::shared_ptr<const Node> node)
    : node_(std::move(node)) {
  if (!node_) {
    throw std::invalid_argument("Cajal Expression requires a node");
  }
}

Expression Expression::variable(std::string name) {
  require_name(name, "variable");
  auto node = std::make_shared<Node>(Kind::Variable);
  node->name = std::move(name);
  return Expression(std::move(node));
}

Expression Expression::unit() {
  return Expression(std::make_shared<Node>(Kind::Unit));
}

Expression Expression::tuple(Expression first, Expression second) {
  auto node = std::make_shared<Node>(Kind::Tuple);
  node->first = std::move(first);
  node->second = std::move(second);
  return Expression(std::move(node));
}

Expression Expression::inject_left(Expression payload, Type right_type) {
  auto node = std::make_shared<Node>(Kind::InjectLeft);
  node->first = std::move(payload);
  node->type = std::move(right_type);
  return Expression(std::move(node));
}

Expression Expression::inject_right(Type left_type, Expression payload) {
  auto node = std::make_shared<Node>(Kind::InjectRight);
  node->first = std::move(payload);
  node->type = std::move(left_type);
  return Expression(std::move(node));
}

Expression Expression::dictionary(DictionaryEntries entries) {
  auto node = std::make_shared<Node>(Kind::Dictionary);
  node->entries = std::move(entries);
  return Expression(std::move(node));
}

Expression Expression::sequence(Expression first, Expression second) {
  auto node = std::make_shared<Node>(Kind::Sequence);
  node->first = std::move(first);
  node->second = std::move(second);
  return Expression(std::move(node));
}

Expression Expression::projection(Expression value, std::size_t index) {
  auto node = std::make_shared<Node>(Kind::Projection);
  node->first = std::move(value);
  node->index = index;
  return Expression(std::move(node));
}

Expression Expression::case_of(Expression scrutinee, std::string left_name,
                               Expression left_body,
                               std::string right_name,
                               Expression right_body) {
  require_name(left_name, "left case binding");
  require_name(right_name, "right case binding");
  auto node = std::make_shared<Node>(Kind::Case);
  node->first = std::move(scrutinee);
  node->second = std::move(left_body);
  node->third = std::move(right_body);
  node->name = std::move(left_name);
  node->alternate_name = std::move(right_name);
  return Expression(std::move(node));
}

Expression Expression::let(std::string name, Expression bound,
                           Expression body) {
  require_name(name, "let binding");
  auto node = std::make_shared<Node>(Kind::Let);
  node->name = std::move(name);
  node->first = std::move(bound);
  node->second = std::move(body);
  return Expression(std::move(node));
}

Expression Expression::lookup(Expression dictionary, Expression query) {
  auto node = std::make_shared<Node>(Kind::Lookup);
  node->first = std::move(dictionary);
  node->second = std::move(query);
  return Expression(std::move(node));
}

Expression::Expression(Expression &&other) noexcept : node_(other.node_) {}

Expression &Expression::operator=(Expression &&other) noexcept {
  if (this != &other) {
    node_ = other.node_;
  }
  return *this;
}

Expression::~Expression() = default;

Expression::Kind Expression::kind() const noexcept { return node_->kind; }

const std::string &Expression::variable_name() const {
  if (kind() != Kind::Variable) {
    throw std::logic_error("variable_name() requires a variable expression");
  }
  return node_->name;
}

const Expression &Expression::first() const {
  if (kind() != Kind::Tuple && kind() != Kind::Sequence) {
    throw std::logic_error("first() requires a tuple or sequence expression");
  }
  return *node_->first;
}

const Expression &Expression::second() const {
  if (kind() != Kind::Tuple && kind() != Kind::Sequence) {
    throw std::logic_error(
        "second() requires a tuple or sequence expression");
  }
  return *node_->second;
}

const Expression &Expression::payload() const {
  if (kind() != Kind::InjectLeft && kind() != Kind::InjectRight) {
    throw std::logic_error("payload() requires an injection expression");
  }
  return *node_->first;
}

const Type &Expression::other_type() const {
  if (kind() != Kind::InjectLeft && kind() != Kind::InjectRight) {
    throw std::logic_error("other_type() requires an injection expression");
  }
  return *node_->type;
}

std::size_t Expression::dictionary_size() const {
  if (kind() != Kind::Dictionary) {
    throw std::logic_error(
        "dictionary_size() requires a dictionary expression");
  }
  return node_->entries.size();
}

const Expression &Expression::dictionary_key(std::size_t index) const {
  if (kind() != Kind::Dictionary) {
    throw std::logic_error(
        "dictionary_key() requires a dictionary expression");
  }
  if (index >= node_->entries.size()) {
    throw std::out_of_range("Cajal dictionary key index is out of range");
  }
  return node_->entries[index].first;
}

const Expression &Expression::dictionary_value(std::size_t index) const {
  if (kind() != Kind::Dictionary) {
    throw std::logic_error(
        "dictionary_value() requires a dictionary expression");
  }
  if (index >= node_->entries.size()) {
    throw std::out_of_range("Cajal dictionary value index is out of range");
  }
  return node_->entries[index].second;
}

const Expression &Expression::projected_value() const {
  if (kind() != Kind::Projection) {
    throw std::logic_error(
        "projected_value() requires a projection expression");
  }
  return *node_->first;
}

std::size_t Expression::projection_index() const {
  if (kind() != Kind::Projection) {
    throw std::logic_error(
        "projection_index() requires a projection expression");
  }
  return node_->index;
}

const Expression &Expression::scrutinee() const {
  if (kind() != Kind::Case) {
    throw std::logic_error("scrutinee() requires a case expression");
  }
  return *node_->first;
}

const std::string &Expression::left_binding_name() const {
  if (kind() != Kind::Case) {
    throw std::logic_error(
        "left_binding_name() requires a case expression");
  }
  return node_->name;
}

const Expression &Expression::left_body() const {
  if (kind() != Kind::Case) {
    throw std::logic_error("left_body() requires a case expression");
  }
  return *node_->second;
}

const std::string &Expression::right_binding_name() const {
  if (kind() != Kind::Case) {
    throw std::logic_error(
        "right_binding_name() requires a case expression");
  }
  return node_->alternate_name;
}

const Expression &Expression::right_body() const {
  if (kind() != Kind::Case) {
    throw std::logic_error("right_body() requires a case expression");
  }
  return *node_->third;
}

const std::string &Expression::binding_name() const {
  if (kind() != Kind::Let) {
    throw std::logic_error("binding_name() requires a let expression");
  }
  return node_->name;
}

const Expression &Expression::bound_expression() const {
  if (kind() != Kind::Let) {
    throw std::logic_error(
        "bound_expression() requires a let expression");
  }
  return *node_->first;
}

const Expression &Expression::body() const {
  if (kind() != Kind::Let) {
    throw std::logic_error("body() requires a let expression");
  }
  return *node_->second;
}

const Expression &Expression::lookup_dictionary() const {
  if (kind() != Kind::Lookup) {
    throw std::logic_error(
        "lookup_dictionary() requires a lookup expression");
  }
  return *node_->first;
}

const Expression &Expression::lookup_query() const {
  if (kind() != Kind::Lookup) {
    throw std::logic_error("lookup_query() requires a lookup expression");
  }
  return *node_->second;
}

} // namespace riftco_transformer::compiler::cajal
