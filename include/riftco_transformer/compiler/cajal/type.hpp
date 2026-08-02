#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace riftco_transformer::compiler::cajal {

// A finite type from the deterministic Cajal-lite language. Product values
// use concatenated coordinates, sum values use tagged coordinate blocks, and
// dictionaries use one coordinate for every key/value pair.
class Type {
public:
  enum class Kind : std::uint8_t {
    Unit,
    Sum,
    Product,
    Dictionary,
  };

  [[nodiscard]] static Type unit();
  [[nodiscard]] static Type sum(Type left, Type right);
  [[nodiscard]] static Type product(Type first, Type second);
  [[nodiscard]] static Type dictionary(Type key, Type value);

  Type(const Type &) noexcept = default;
  Type &operator=(const Type &) noexcept = default;
  Type(Type &&other) noexcept;
  Type &operator=(Type &&other) noexcept;
  ~Type();

  [[nodiscard]] Kind kind() const noexcept;
  [[nodiscard]] const Type &left() const;
  [[nodiscard]] const Type &right() const;
  [[nodiscard]] const Type &first() const;
  [[nodiscard]] const Type &second() const;
  [[nodiscard]] const Type &key() const;
  [[nodiscard]] const Type &value() const;

  [[nodiscard]] std::size_t dimension() const;
  [[nodiscard]] std::string to_string() const;

  friend bool operator==(const Type &left, const Type &right) noexcept;

private:
  struct Node;
  std::shared_ptr<const Node> node_;

  explicit Type(std::shared_ptr<const Node> node);
};

[[nodiscard]] bool operator!=(const Type &left, const Type &right) noexcept;

} // namespace riftco_transformer::compiler::cajal
