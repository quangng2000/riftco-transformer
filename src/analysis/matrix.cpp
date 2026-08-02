#include "riftco_transformer/analysis/matrix.hpp"

#include <limits>
#include <stdexcept>

namespace riftco_transformer::analysis {

MatrixView Matrix::view() const noexcept { return {rows, columns, values}; }

std::size_t checked_matrix_size(std::size_t rows, std::size_t columns) {
  if (rows == 0 || columns == 0) {
    throw std::invalid_argument("analysis matrix dimensions must be positive");
  }
  if (rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::overflow_error("analysis matrix shape is too large");
  }
  return rows * columns;
}

void validate_matrix_view(MatrixView matrix) {
  const std::size_t expected = checked_matrix_size(matrix.rows, matrix.columns);
  if (matrix.values.size() != expected) {
    throw std::invalid_argument(
        "analysis matrix value count does not match its shape");
  }
}

} // namespace riftco_transformer::analysis
