#include "riftco_transformer/analysis/intervention.hpp"

#include "detail/validation.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace riftco_transformer::analysis {
namespace {

void validate_direction_size(std::span<const float> direction,
                             std::size_t feature_count) {
  if (direction.size() != feature_count) {
    throw std::invalid_argument(
        "intervention direction must contain one value per feature");
  }
}

Matrix apply_coordinate_replacement(MatrixView observations,
                                    const CoordinateReplacement &replacement) {
  if (replacement.feature_indices.empty() ||
      replacement.feature_indices.size() !=
          replacement.replacement_values.size()) {
    throw std::invalid_argument(
        "coordinate replacement indices and values must be nonempty "
        "and equally sized");
  }
  detail::validate_finite_vector(replacement.replacement_values,
                                 "coordinate replacement values");

  std::vector<bool> seen(observations.columns, false);
  for (const std::size_t feature : replacement.feature_indices) {
    if (feature >= observations.columns) {
      throw std::out_of_range("coordinate replacement feature is out of range");
    }
    if (seen[feature]) {
      throw std::invalid_argument(
          "coordinate replacement features must be unique");
    }
    seen[feature] = true;
  }

  Matrix result{
      observations.rows,
      observations.columns,
      std::vector<float>(observations.values.begin(),
                         observations.values.end()),
  };
  for (std::size_t row = 0; row < observations.rows; ++row) {
    const std::size_t offset = row * observations.columns;
    for (std::size_t index = 0; index < replacement.feature_indices.size();
         ++index) {
      result.values[offset + replacement.feature_indices[index]] =
          replacement.replacement_values[index];
    }
  }
  return result;
}

Matrix apply_direction_steering(MatrixView observations,
                                const DirectionSteering &steering) {
  validate_direction_size(steering.direction, observations.columns);
  if (!std::isfinite(steering.strength)) {
    throw std::invalid_argument("direction steering strength must be finite");
  }
  const double norm =
      detail::nonzero_l2_norm(steering.direction, "direction steering vector");
  const double factor = static_cast<double>(steering.strength) /
                        (steering.normalize ? norm : 1.0);
  if (!std::isfinite(factor)) {
    throw std::overflow_error(
        "direction steering scale is outside finite range");
  }

  Matrix result{
      observations.rows,
      observations.columns,
      std::vector<float>(observations.values.size()),
  };
  for (std::size_t row = 0; row < observations.rows; ++row) {
    const std::size_t offset = row * observations.columns;
    for (std::size_t feature = 0; feature < observations.columns; ++feature) {
      const double value =
          static_cast<double>(observations.values[offset + feature]) +
          factor * static_cast<double>(steering.direction[feature]);
      result.values[offset + feature] =
          detail::checked_float(value, "steered representation value");
    }
  }
  return result;
}

Matrix apply_direction_projection(MatrixView observations,
                                  const DirectionProjection &projection) {
  validate_direction_size(projection.direction, observations.columns);
  const double norm =
      detail::nonzero_l2_norm(projection.direction, "projection direction");
  if (!projection.origin.empty() &&
      projection.origin.size() != observations.columns) {
    throw std::invalid_argument(
        "projection origin must be empty or contain one value per feature");
  }
  detail::validate_finite_vector(projection.origin, "projection origin");

  std::vector<double> unit_direction(observations.columns);
  for (std::size_t feature = 0; feature < observations.columns; ++feature) {
    unit_direction[feature] =
        static_cast<double>(projection.direction[feature]) / norm;
  }

  Matrix result{
      observations.rows,
      observations.columns,
      std::vector<float>(observations.values.size()),
  };
  for (std::size_t row = 0; row < observations.rows; ++row) {
    const std::size_t offset = row * observations.columns;
    double component = 0.0;
    for (std::size_t feature = 0; feature < observations.columns; ++feature) {
      const double origin =
          projection.origin.empty()
              ? 0.0
              : static_cast<double>(projection.origin[feature]);
      component += (static_cast<double>(observations.values[offset + feature]) -
                    origin) *
                   unit_direction[feature];
    }
    if (!std::isfinite(component)) {
      throw std::overflow_error(
          "projected representation component is outside finite range");
    }
    for (std::size_t feature = 0; feature < observations.columns; ++feature) {
      const double value =
          static_cast<double>(observations.values[offset + feature]) -
          component * unit_direction[feature];
      result.values[offset + feature] =
          detail::checked_float(value, "projected representation value");
    }
  }
  return result;
}

} // namespace

Matrix apply_intervention(MatrixView observations,
                          const InterventionConfig &intervention) {
  detail::validate_finite_matrix(observations, "intervention observations");
  if (intervention.valueless_by_exception()) {
    throw std::invalid_argument("intervention configuration has no value");
  }
  return std::visit(
      [&](const auto &config) -> Matrix {
        using Config = std::decay_t<decltype(config)>;
        if constexpr (std::is_same_v<Config, CoordinateReplacement>) {
          return apply_coordinate_replacement(observations, config);
        } else if constexpr (std::is_same_v<Config, DirectionSteering>) {
          return apply_direction_steering(observations, config);
        } else {
          static_assert(std::is_same_v<Config, DirectionProjection>,
                        "unhandled intervention configuration");
          return apply_direction_projection(observations, config);
        }
      },
      intervention);
}

} // namespace riftco_transformer::analysis
