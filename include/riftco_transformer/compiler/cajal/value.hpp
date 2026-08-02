#pragma once

#include "riftco_transformer/compiler/cajal/type.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer::compiler::cajal {

namespace detail {
class Interpreter;
}

// A finite runtime value for the Cajal-lite reference interpreter.
class Value {
public:
  enum class Kind : std::uint8_t {
    Unit,
    Sum,
    Product,
    Dictionary,
  };

  using DictionaryEntries = std::vector<std::pair<Value, Value>>;

  [[nodiscard]] static Value unit();
  [[nodiscard]] static Value inject_left(Value payload, Type right_type);
  [[nodiscard]] static Value inject_right(Type left_type, Value payload);
  [[nodiscard]] static Value product(Value first, Value second);
  [[nodiscard]] static Value dictionary(DictionaryEntries entries);

  Value(const Value &) noexcept = default;
  Value &operator=(const Value &) noexcept = default;
  Value(Value &&other) noexcept;
  Value &operator=(Value &&other) noexcept;
  ~Value();

  [[nodiscard]] Kind kind() const noexcept;
  [[nodiscard]] Type type() const;

  [[nodiscard]] bool is_left() const;
  [[nodiscard]] const Value &payload() const;
  // Product components may be suspended interpreter computations. Accessing a
  // component evaluates only that component and returns its value.
  [[nodiscard]] Value first() const;
  [[nodiscard]] Value second() const;

  [[nodiscard]] std::size_t dictionary_size() const;
  // Interpreter-produced dictionary entries may be suspended. Keys are
  // evaluated for matching; a value is evaluated only when selected.
  [[nodiscard]] Value dictionary_key(std::size_t index) const;
  [[nodiscard]] Value dictionary_value(std::size_t index) const;

  [[nodiscard]] std::string to_string() const;

  friend bool operator==(const Value &left, const Value &right);

private:
  struct Node;
  using DeferredDictionaryEntries = std::vector<
      std::pair<std::function<Value()>, std::function<Value()>>>;
  std::shared_ptr<const Node> node_;

  explicit Value(std::shared_ptr<const Node> node);
  [[nodiscard]] static Value deferred_product(
      Type first_type, Type second_type, std::function<Value()> first,
      std::function<Value()> second);
  [[nodiscard]] static Value deferred_dictionary(
      Type key_type, Type value_type, DeferredDictionaryEntries entries);

  friend class detail::Interpreter;
};

[[nodiscard]] bool operator!=(const Value &left, const Value &right);

} // namespace riftco_transformer::compiler::cajal
