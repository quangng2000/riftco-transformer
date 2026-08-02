#pragma once

#include "riftco_transformer/analysis/matrix.hpp"

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace riftco_transformer::analysis::detail {

inline void validate_finite_matrix(MatrixView matrix,
                                   std::string_view description) {
  validate_matrix_view(matrix);
  for (const float value : matrix.values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(std::string(description) +
                                  " must contain only finite values");
    }
  }
}

inline void validate_finite_vector(std::span<const float> values,
                                   std::string_view description) {
  for (const float value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(std::string(description) +
                                  " must contain only finite values");
    }
  }
}

inline float checked_float(double value, std::string_view description) {
  constexpr double maximum =
      static_cast<double>(std::numeric_limits<float>::max());
  if (!std::isfinite(value) || value < -maximum || value > maximum) {
    throw std::overflow_error(std::string(description) +
                              " is outside finite float range");
  }
  return static_cast<float>(value);
}

inline double nonzero_l2_norm(std::span<const float> values,
                              std::string_view description) {
  validate_finite_vector(values, description);
  double norm = 0.0;
  for (const float value : values) {
    norm = std::hypot(norm, static_cast<double>(value));
  }
  if (!std::isfinite(norm) || norm == 0.0) {
    throw std::invalid_argument(std::string(description) +
                                " must have finite nonzero L2 length");
  }
  return norm;
}

} // namespace riftco_transformer::analysis::detail
