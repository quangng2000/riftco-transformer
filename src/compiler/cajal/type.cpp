#include "riftco_transformer/compiler/cajal/type.hpp"

#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::compiler::cajal {

struct Type::Node {
  explicit Node(Kind node_kind) : kind(node_kind) {}

  Kind kind;
  std::optional<Type> first;
  std::optional<Type> second;
};

namespace {

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("Cajal type dimension exceeds size_t");
  }
  return left + right;
}

[[nodiscard]] std::size_t checked_multiply(std::size_t left,
                                           std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Cajal type dimension exceeds size_t");
  }
  return left * right;
}

} // namespace

Type::Type(std::shared_ptr<const Node> node) : node_(std::move(node)) {
  if (!node_) {
    throw std::invalid_argument("Cajal Type requires a node");
  }
}

Type Type::unit() { return Type(std::make_shared<Node>(Kind::Unit)); }

Type Type::sum(Type left, Type right) {
  auto node = std::make_shared<Node>(Kind::Sum);
  node->first = std::move(left);
  node->second = std::move(right);
  return Type(std::move(node));
}

Type Type::product(Type first, Type second) {
  auto node = std::make_shared<Node>(Kind::Product);
  node->first = std::move(first);
  node->second = std::move(second);
  return Type(std::move(node));
}

Type Type::dictionary(Type key, Type value) {
  auto node = std::make_shared<Node>(Kind::Dictionary);
  node->first = std::move(key);
  node->second = std::move(value);
  return Type(std::move(node));
}

Type::Type(Type &&other) noexcept : node_(other.node_) {}

Type &Type::operator=(Type &&other) noexcept {
  if (this != &other) {
    node_ = other.node_;
  }
  return *this;
}

Type::~Type() = default;

Type::Kind Type::kind() const noexcept { return node_->kind; }

const Type &Type::left() const {
  if (kind() != Kind::Sum) {
    throw std::logic_error("left() requires a Cajal sum type");
  }
  return *node_->first;
}

const Type &Type::right() const {
  if (kind() != Kind::Sum) {
    throw std::logic_error("right() requires a Cajal sum type");
  }
  return *node_->second;
}

const Type &Type::first() const {
  if (kind() != Kind::Product) {
    throw std::logic_error("first() requires a Cajal product type");
  }
  return *node_->first;
}

const Type &Type::second() const {
  if (kind() != Kind::Product) {
    throw std::logic_error("second() requires a Cajal product type");
  }
  return *node_->second;
}

const Type &Type::key() const {
  if (kind() != Kind::Dictionary) {
    throw std::logic_error("key() requires a Cajal dictionary type");
  }
  return *node_->first;
}

const Type &Type::value() const {
  if (kind() != Kind::Dictionary) {
    throw std::logic_error("value() requires a Cajal dictionary type");
  }
  return *node_->second;
}

std::size_t Type::dimension() const {
  switch (kind()) {
  case Kind::Unit:
    return 1;
  case Kind::Sum:
    return checked_add(left().dimension(), right().dimension());
  case Kind::Product:
    return checked_add(first().dimension(), second().dimension());
  case Kind::Dictionary:
    return checked_multiply(key().dimension(), value().dimension());
  }
  throw std::logic_error("unknown Cajal type kind");
}

std::string Type::to_string() const {
  switch (kind()) {
  case Kind::Unit:
    return "Unit";
  case Kind::Sum:
    return "Sum<" + left().to_string() + ", " + right().to_string() + ">";
  case Kind::Product:
    return "Product<" + first().to_string() + ", " +
           second().to_string() + ">";
  case Kind::Dictionary:
    return "Dictionary<" + key().to_string() + ", " + value().to_string() +
           ">";
  }
  throw std::logic_error("unknown Cajal type kind");
}

bool operator==(const Type &left, const Type &right) noexcept {
  if (left.node_ == right.node_) {
    return true;
  }
  if (left.kind() != right.kind()) {
    return false;
  }
  switch (left.kind()) {
  case Type::Kind::Unit:
    return true;
  case Type::Kind::Sum:
    return left.left() == right.left() && left.right() == right.right();
  case Type::Kind::Product:
    return left.first() == right.first() && left.second() == right.second();
  case Type::Kind::Dictionary:
    return left.key() == right.key() && left.value() == right.value();
  }
  return false;
}

bool operator!=(const Type &left, const Type &right) noexcept {
  return !(left == right);
}

} // namespace riftco_transformer::compiler::cajal
