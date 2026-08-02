#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace riftco_transformer::analysis {

// Non-owning row-major matrix view. Both dimensions must be positive and the
// value span must contain exactly rows * columns elements.
struct MatrixView {
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::span<const float> values;
};

// Owning row-major matrix with the same shape contract as MatrixView.
struct Matrix {
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::vector<float> values;

  [[nodiscard]] MatrixView view() const noexcept;
};

// Validates positive dimensions and returns rows * columns with overflow
// checking.
[[nodiscard]] std::size_t checked_matrix_size(std::size_t rows,
                                              std::size_t columns);

// Validates shape arithmetic and the size of the backing value span.
void validate_matrix_view(MatrixView matrix);

} // namespace riftco_transformer::analysis
