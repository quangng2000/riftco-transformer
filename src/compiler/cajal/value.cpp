#include "riftco_transformer/compiler/cajal/value.hpp"

#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::compiler::cajal {

struct Value::Node {
  Node(Kind node_kind, Type node_type)
      : kind(node_kind), value_type(std::move(node_type)) {}

  Kind kind;
  Type value_type;
  bool left_injection = false;
  std::optional<Value> first;
  std::optional<Value> second;
  std::function<Value()> deferred_first;
  std::function<Value()> deferred_second;
  DictionaryEntries entries;
  DeferredDictionaryEntries deferred_entries;
};

Value::Value(std::shared_ptr<const Node> node) : node_(std::move(node)) {
  if (!node_) {
    throw std::invalid_argument("Cajal Value requires a node");
  }
}

Value Value::unit() {
  return Value(std::make_shared<Node>(Kind::Unit, Type::unit()));
}

Value Value::inject_left(Value payload, Type right_type) {
  const Type sum_type = Type::sum(payload.type(), std::move(right_type));
  auto node = std::make_shared<Node>(Kind::Sum, sum_type);
  node->left_injection = true;
  node->first = std::move(payload);
  return Value(std::move(node));
}

Value Value::inject_right(Type left_type, Value payload) {
  const Type sum_type = Type::sum(std::move(left_type), payload.type());
  auto node = std::make_shared<Node>(Kind::Sum, sum_type);
  node->left_injection = false;
  node->first = std::move(payload);
  return Value(std::move(node));
}

Value Value::product(Value first, Value second) {
  const Type product_type = Type::product(first.type(), second.type());
  auto node = std::make_shared<Node>(Kind::Product, product_type);
  node->first = std::move(first);
  node->second = std::move(second);
  return Value(std::move(node));
}

Value Value::deferred_product(Type first_type, Type second_type,
                              std::function<Value()> first,
                              std::function<Value()> second) {
  if (!first || !second) {
    throw std::invalid_argument(
        "a deferred Cajal product requires both component computations");
  }
  auto node = std::make_shared<Node>(
      Kind::Product,
      Type::product(std::move(first_type), std::move(second_type)));
  node->deferred_first = std::move(first);
  node->deferred_second = std::move(second);
  return Value(std::move(node));
}

Value Value::dictionary(DictionaryEntries entries) {
  if (entries.empty()) {
    throw std::invalid_argument(
        "a Cajal-lite dictionary value must contain at least one entry");
  }

  const Type key_type = entries.front().first.type();
  const Type value_type = entries.front().second.type();
  for (const auto &[key, value] : entries) {
    if (key.type() != key_type || value.type() != value_type) {
      throw std::invalid_argument(
          "all Cajal dictionary entries must have the same key and value "
          "types");
    }
  }

  auto node = std::make_shared<Node>(
      Kind::Dictionary, Type::dictionary(key_type, value_type));
  node->entries = std::move(entries);
  return Value(std::move(node));
}

Value Value::deferred_dictionary(Type key_type, Type value_type,
                                 DeferredDictionaryEntries entries) {
  if (entries.empty()) {
    throw std::invalid_argument(
        "a deferred Cajal dictionary requires at least one entry");
  }
  for (const auto &[key, value] : entries) {
    if (!key || !value) {
      throw std::invalid_argument(
          "a deferred Cajal dictionary requires every key and value "
          "computation");
    }
  }

  auto node = std::make_shared<Node>(
      Kind::Dictionary,
      Type::dictionary(std::move(key_type), std::move(value_type)));
  node->deferred_entries = std::move(entries);
  return Value(std::move(node));
}

Value::Value(Value &&other) noexcept : node_(other.node_) {}

Value &Value::operator=(Value &&other) noexcept {
  if (this != &other) {
    node_ = other.node_;
  }
  return *this;
}

Value::~Value() = default;

Value::Kind Value::kind() const noexcept { return node_->kind; }

Type Value::type() const { return node_->value_type; }

bool Value::is_left() const {
  if (kind() != Kind::Sum) {
    throw std::logic_error("is_left() requires a Cajal sum value");
  }
  return node_->left_injection;
}

const Value &Value::payload() const {
  if (kind() != Kind::Sum) {
    throw std::logic_error("payload() requires a Cajal sum value");
  }
  return *node_->first;
}

Value Value::first() const {
  if (kind() != Kind::Product) {
    throw std::logic_error("first() requires a Cajal product value");
  }
  if (node_->first.has_value()) {
    return *node_->first;
  }
  if (!node_->deferred_first) {
    throw std::logic_error("Cajal product has no first component");
  }
  Value result = node_->deferred_first();
  if (result.type() != node_->value_type.first()) {
    throw std::logic_error(
        "deferred Cajal product produced the wrong first-component type");
  }
  return result;
}

Value Value::second() const {
  if (kind() != Kind::Product) {
    throw std::logic_error("second() requires a Cajal product value");
  }
  if (node_->second.has_value()) {
    return *node_->second;
  }
  if (!node_->deferred_second) {
    throw std::logic_error("Cajal product has no second component");
  }
  Value result = node_->deferred_second();
  if (result.type() != node_->value_type.second()) {
    throw std::logic_error(
        "deferred Cajal product produced the wrong second-component type");
  }
  return result;
}

std::size_t Value::dictionary_size() const {
  if (kind() != Kind::Dictionary) {
    throw std::logic_error(
        "dictionary_size() requires a Cajal dictionary value");
  }
  return node_->entries.empty() ? node_->deferred_entries.size()
                                : node_->entries.size();
}

Value Value::dictionary_key(std::size_t index) const {
  if (kind() != Kind::Dictionary) {
    throw std::logic_error(
        "dictionary_key() requires a Cajal dictionary value");
  }
  if (index >= dictionary_size()) {
    throw std::out_of_range("Cajal dictionary key index is out of range");
  }
  if (!node_->entries.empty()) {
    return node_->entries[index].first;
  }
  Value result = node_->deferred_entries[index].first();
  if (result.type() != node_->value_type.key()) {
    throw std::logic_error(
        "deferred Cajal dictionary produced the wrong key type");
  }
  return result;
}

Value Value::dictionary_value(std::size_t index) const {
  if (kind() != Kind::Dictionary) {
    throw std::logic_error(
        "dictionary_value() requires a Cajal dictionary value");
  }
  if (index >= dictionary_size()) {
    throw std::out_of_range("Cajal dictionary value index is out of range");
  }
  if (!node_->entries.empty()) {
    return node_->entries[index].second;
  }
  Value result = node_->deferred_entries[index].second();
  if (result.type() != node_->value_type.value()) {
    throw std::logic_error(
        "deferred Cajal dictionary produced the wrong value type");
  }
  return result;
}

std::string Value::to_string() const {
  switch (kind()) {
  case Kind::Unit:
    return "()";
  case Kind::Sum:
    return std::string(is_left() ? "left(" : "right(") +
           payload().to_string() + ")";
  case Kind::Product:
    return "(" + first().to_string() + ", " + second().to_string() + ")";
  case Kind::Dictionary: {
    std::ostringstream output;
    output << '{';
    for (std::size_t index = 0; index < dictionary_size(); ++index) {
      if (index != 0) {
        output << ", ";
      }
      output << dictionary_key(index).to_string() << ": "
             << dictionary_value(index).to_string();
    }
    output << '}';
    return output.str();
  }
  }
  throw std::logic_error("unknown Cajal value kind");
}

bool operator==(const Value &left, const Value &right) {
  if (left.node_ == right.node_) {
    return true;
  }
  if (left.kind() != right.kind() || left.type() != right.type()) {
    return false;
  }
  switch (left.kind()) {
  case Value::Kind::Unit:
    return true;
  case Value::Kind::Sum:
    return left.is_left() == right.is_left() &&
           left.payload() == right.payload();
  case Value::Kind::Product:
    return left.first() == right.first() && left.second() == right.second();
  case Value::Kind::Dictionary:
    if (left.dictionary_size() != right.dictionary_size()) {
      return false;
    }
    for (std::size_t index = 0; index < left.dictionary_size(); ++index) {
      if (left.dictionary_key(index) != right.dictionary_key(index) ||
          left.dictionary_value(index) != right.dictionary_value(index)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool operator!=(const Value &left, const Value &right) {
  return !(left == right);
}

} // namespace riftco_transformer::compiler::cajal
