#pragma once

#include "riftco_transformer/compiler/cajal/value.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace riftco_transformer::compiler::cajal {

class EncodingError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Backend-neutral coordinates in the finite vector space denoted by a Cajal
// type. This intentionally uses double precision and has no Tensor dependency.
class EncodedValue {
public:
  explicit EncodedValue(std::vector<double> coordinates);

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::span<const double> coordinates() const noexcept;
  [[nodiscard]] double operator[](std::size_t index) const;

  friend bool operator==(const EncodedValue &left,
                         const EncodedValue &right) noexcept;

private:
  std::vector<double> coordinates_;
};

[[nodiscard]] bool operator!=(const EncodedValue &left,
                              const EncodedValue &right) noexcept;

[[nodiscard]] bool approximately_equal(const EncodedValue &left,
                                       const EncodedValue &right,
                                       double tolerance = 1.0e-9) noexcept;

// Unit maps to [1], sums use zero-padded blocks, products concatenate, and a
// dictionary is flattened value-major from the sum of value/key outer products.
[[nodiscard]] EncodedValue encode(const Value &value);

// Decodes discrete Unit, Sum, and Product encodings. Dictionary coordinates
// do not have a unique source-level decomposition and are rejected.
[[nodiscard]] Value decode(const Type &type, const EncodedValue &encoded,
                           double tolerance = 1.0e-9);

// Exhaustively enumerates Unit, Sum, and Product values. Dictionary values are
// unbounded source-level lists and are rejected. max_values prevents an
// accidental combinatorial explosion.
[[nodiscard]] std::vector<Value>
enumerate_values(const Type &type, std::size_t max_values = 4096);

} // namespace riftco_transformer::compiler::cajal
