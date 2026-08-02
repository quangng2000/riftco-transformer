#pragma once

#include "riftco_transformer/compiler/cajal/encoding.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace riftco_transformer::compiler::cajal {

class MultilinearMapError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Dense coefficients for a k-linear map. Coefficients are output-major with
// shape [output, input_0, ..., input_{k-1}]. An empty input shape is a constant
// (0-linear) map.
class MultilinearMap {
public:
  MultilinearMap(std::vector<std::size_t> input_dimensions,
                 std::size_t output_dimension,
                 std::vector<double> coefficients);

  // Materializes a dense map from output-major flat nonzero entries. Sparse
  // indices must be unique and in range, and values must be finite and
  // nonzero. Empty sparse arrays construct an all-zero map.
  [[nodiscard]] static MultilinearMap
  from_sparse(std::vector<std::size_t> input_dimensions,
              std::size_t output_dimension,
              std::span<const std::size_t> flat_indices,
              std::span<const double> values);

  [[nodiscard]] static MultilinearMap constant(const EncodedValue &value);
  [[nodiscard]] static MultilinearMap identity(std::size_t dimension);

  [[nodiscard]] std::size_t arity() const noexcept;
  [[nodiscard]] std::span<const std::size_t> input_dimensions() const noexcept;
  [[nodiscard]] std::size_t output_dimension() const noexcept;
  [[nodiscard]] std::size_t coefficient_count() const noexcept;
  [[nodiscard]] std::span<const double> coefficients() const noexcept;

  [[nodiscard]] double
  coefficient_at(std::size_t output_index,
                 std::span<const std::size_t> input_indices) const;
  [[nodiscard]] EncodedValue apply(std::span<const EncodedValue> inputs) const;

private:
  std::vector<std::size_t> input_dimensions_;
  std::size_t output_dimension_;
  std::size_t input_combination_count_;
  std::vector<double> coefficients_;
};

} // namespace riftco_transformer::compiler::cajal
