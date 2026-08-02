#include "riftco_transformer/analysis/pca.hpp"

#include "detail/validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer::analysis {
namespace {

struct MatrixNorms {
  double full;
  double off_diagonal;
};

struct SymmetricEigensystem {
  std::vector<double> eigenvalues;
  // Eigenvectors are columns of this row-major square matrix.
  std::vector<double> eigenvectors;
  std::size_t sweep_count;
  double off_diagonal_residual;
};

void compensated_add(double value, double &sum, double &correction) {
  const double adjusted = value - correction;
  const double updated = sum + adjusted;
  correction = (updated - sum) - adjusted;
  sum = updated;
}

MatrixNorms matrix_norms(std::span<const double> matrix,
                         std::size_t dimension) {
  double full = 0.0;
  double off_diagonal = 0.0;
  constexpr double square_root_two = 1.4142135623730950488;
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t column = 0; column < dimension; ++column) {
      const double value = matrix[row * dimension + column];
      full = std::hypot(full, value);
      if (column > row) {
        off_diagonal = std::hypot(off_diagonal, square_root_two * value);
      }
    }
  }
  return {full, off_diagonal};
}

bool converged(const MatrixNorms &norms, double tolerance) {
  return norms.full == 0.0 || norms.off_diagonal <= tolerance * norms.full;
}

SymmetricEigensystem diagonalize_symmetric(std::vector<double> matrix,
                                           std::size_t dimension,
                                           const PcaOptions &options) {
  std::vector<double> eigenvectors(checked_matrix_size(dimension, dimension),
                                   0.0);
  for (std::size_t index = 0; index < dimension; ++index) {
    eigenvectors[index * dimension + index] = 1.0;
  }

  MatrixNorms norms = matrix_norms(matrix, dimension);
  std::size_t sweep_count = 0;
  while (!converged(norms, options.tolerance) &&
         sweep_count < options.max_sweeps) {
    for (std::size_t first = 0; first < dimension; ++first) {
      for (std::size_t second = first + 1; second < dimension; ++second) {
        const double cross = matrix[first * dimension + second];
        if (cross == 0.0) {
          continue;
        }
        const double first_diagonal = matrix[first * dimension + first];
        const double second_diagonal = matrix[second * dimension + second];
        const double angle =
            0.5 * std::atan2(2.0 * cross, second_diagonal - first_diagonal);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);

        for (std::size_t index = 0; index < dimension; ++index) {
          if (index == first || index == second) {
            continue;
          }
          const double first_value = matrix[index * dimension + first];
          const double second_value = matrix[index * dimension + second];
          const double rotated_first =
              cosine * first_value - sine * second_value;
          const double rotated_second =
              sine * first_value + cosine * second_value;
          matrix[index * dimension + first] = rotated_first;
          matrix[first * dimension + index] = rotated_first;
          matrix[index * dimension + second] = rotated_second;
          matrix[second * dimension + index] = rotated_second;
        }

        const double cosine_squared = cosine * cosine;
        const double sine_squared = sine * sine;
        const double twice_sine_cosine = 2.0 * sine * cosine;
        matrix[first * dimension + first] = cosine_squared * first_diagonal -
                                            twice_sine_cosine * cross +
                                            sine_squared * second_diagonal;
        matrix[second * dimension + second] = sine_squared * first_diagonal +
                                              twice_sine_cosine * cross +
                                              cosine_squared * second_diagonal;
        matrix[first * dimension + second] = 0.0;
        matrix[second * dimension + first] = 0.0;

        for (std::size_t row = 0; row < dimension; ++row) {
          const double first_vector = eigenvectors[row * dimension + first];
          const double second_vector = eigenvectors[row * dimension + second];
          eigenvectors[row * dimension + first] =
              cosine * first_vector - sine * second_vector;
          eigenvectors[row * dimension + second] =
              sine * first_vector + cosine * second_vector;
        }
      }
    }
    ++sweep_count;
    norms = matrix_norms(matrix, dimension);
  }

  if (!converged(norms, options.tolerance)) {
    throw std::runtime_error(
        "PCA Jacobi eigensolver did not converge within max_sweeps");
  }

  std::vector<double> eigenvalues(dimension);
  for (std::size_t index = 0; index < dimension; ++index) {
    eigenvalues[index] = matrix[index * dimension + index];
    if (!std::isfinite(eigenvalues[index])) {
      throw std::overflow_error(
          "PCA eigensolver produced a nonfinite eigenvalue");
    }
  }
  return {
      std::move(eigenvalues),
      std::move(eigenvectors),
      sweep_count,
      norms.off_diagonal,
  };
}

std::size_t validated_component_count(MatrixView observations,
                                      const PcaOptions &options) {
  detail::validate_finite_matrix(observations, "PCA observations");
  if (observations.rows < 2) {
    throw std::invalid_argument("PCA requires at least two observations");
  }
  if (options.max_sweeps == 0) {
    throw std::invalid_argument("PCA max_sweeps must be positive");
  }
  if (!std::isfinite(options.tolerance) || options.tolerance <= 0.0) {
    throw std::invalid_argument("PCA tolerance must be finite and positive");
  }
  if (options.max_covariance_elements == 0) {
    throw std::invalid_argument(
        "PCA covariance element limit must be positive");
  }
  const std::size_t covariance_elements =
      checked_matrix_size(observations.columns, observations.columns);
  if (covariance_elements > options.max_covariance_elements) {
    throw std::length_error(
        "PCA covariance matrix exceeds max_covariance_elements");
  }

  const std::size_t maximum_components =
      std::min(observations.rows - 1, observations.columns);
  const std::size_t component_count = options.component_count == 0
                                          ? maximum_components
                                          : options.component_count;
  if (component_count == 0 || component_count > maximum_components) {
    throw std::invalid_argument(
        "PCA component count exceeds centered matrix rank bound");
  }
  return component_count;
}

std::vector<double> feature_means(MatrixView observations) {
  std::vector<double> sums(observations.columns, 0.0);
  std::vector<double> corrections(observations.columns, 0.0);
  for (std::size_t row = 0; row < observations.rows; ++row) {
    const std::size_t offset = row * observations.columns;
    for (std::size_t feature = 0; feature < observations.columns; ++feature) {
      compensated_add(
          static_cast<double>(observations.values[offset + feature]),
          sums[feature], corrections[feature]);
    }
  }
  const double denominator = static_cast<double>(observations.rows);
  for (double &sum : sums) {
    sum /= denominator;
    if (!std::isfinite(sum)) {
      throw std::overflow_error("PCA feature mean is nonfinite");
    }
  }
  return sums;
}

std::vector<double> sample_covariance(MatrixView observations,
                                      std::span<const double> means) {
  const std::size_t elements =
      checked_matrix_size(observations.columns, observations.columns);
  std::vector<double> covariance(elements, 0.0);
  std::vector<double> corrections(elements, 0.0);
  std::vector<double> centered(observations.columns);
  for (std::size_t row = 0; row < observations.rows; ++row) {
    const std::size_t offset = row * observations.columns;
    for (std::size_t feature = 0; feature < observations.columns; ++feature) {
      centered[feature] =
          static_cast<double>(observations.values[offset + feature]) -
          means[feature];
    }
    for (std::size_t first = 0; first < observations.columns; ++first) {
      for (std::size_t second = first; second < observations.columns;
           ++second) {
        const std::size_t index = first * observations.columns + second;
        const double contribution = centered[first] * centered[second];
        if (!std::isfinite(contribution)) {
          throw std::overflow_error("PCA covariance contribution is nonfinite");
        }
        compensated_add(contribution, covariance[index], corrections[index]);
      }
    }
  }

  const double denominator = static_cast<double>(observations.rows - 1);
  for (std::size_t first = 0; first < observations.columns; ++first) {
    for (std::size_t second = first; second < observations.columns; ++second) {
      const std::size_t upper = first * observations.columns + second;
      const std::size_t lower = second * observations.columns + first;
      covariance[upper] /= denominator;
      if (!std::isfinite(covariance[upper])) {
        throw std::overflow_error("PCA covariance value is nonfinite");
      }
      covariance[lower] = covariance[upper];
    }
  }
  return covariance;
}

void validate_pca_model(const PcaModel &model) {
  if (model.feature_count == 0 || model.component_count == 0) {
    throw std::invalid_argument("PCA model dimensions must be positive");
  }
  if (model.mean.size() != model.feature_count) {
    throw std::invalid_argument(
        "PCA model mean size does not match feature count");
  }
  detail::validate_finite_vector(model.mean, "PCA model mean");
  detail::validate_finite_matrix(model.components.view(), "PCA components");
  if (model.components.rows != model.component_count ||
      model.components.columns != model.feature_count) {
    throw std::invalid_argument(
        "PCA component matrix shape does not match model dimensions");
  }
  if (model.explained_variance.size() != model.component_count ||
      model.explained_variance_ratio.size() != model.component_count) {
    throw std::invalid_argument(
        "PCA variance metadata does not match component count");
  }
  for (const double variance : model.explained_variance) {
    if (!std::isfinite(variance) || variance < 0.0) {
      throw std::invalid_argument(
          "PCA explained variances must be finite and nonnegative");
    }
  }
  for (const double ratio : model.explained_variance_ratio) {
    if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
      throw std::invalid_argument(
          "PCA explained variance ratios must lie in [0, 1]");
    }
  }
}

std::vector<double>
nonnegative_sorted_eigenvalues(const SymmetricEigensystem &eigensystem,
                               std::span<const std::size_t> order,
                               double tolerance) {
  double maximum_absolute = 0.0;
  for (const double eigenvalue : eigensystem.eigenvalues) {
    maximum_absolute = std::max(maximum_absolute, std::fabs(eigenvalue));
  }
  const double negative_limit = tolerance * maximum_absolute;
  std::vector<double> result;
  result.reserve(order.size());
  for (const std::size_t index : order) {
    double eigenvalue = eigensystem.eigenvalues[index];
    if (eigenvalue < -negative_limit) {
      throw std::runtime_error(
          "PCA covariance eigensolver produced a negative eigenvalue");
    }
    if (eigenvalue < 0.0) {
      eigenvalue = 0.0;
    }
    result.push_back(eigenvalue);
  }
  return result;
}

} // namespace

PcaFit fit_pca(MatrixView observations, const PcaOptions &options) {
  const std::size_t component_count =
      validated_component_count(observations, options);
  const std::vector<double> means = feature_means(observations);
  SymmetricEigensystem eigensystem = diagonalize_symmetric(
      sample_covariance(observations, means), observations.columns, options);

  std::vector<std::size_t> order(observations.columns);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(
      order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return eigensystem.eigenvalues[left] > eigensystem.eigenvalues[right];
      });
  const std::vector<double> sorted_eigenvalues =
      nonnegative_sorted_eigenvalues(eigensystem, order, options.tolerance);

  double total_variance = 0.0;
  double total_variance_correction = 0.0;
  for (const double eigenvalue : sorted_eigenvalues) {
    compensated_add(eigenvalue, total_variance, total_variance_correction);
  }
  if (!std::isfinite(total_variance)) {
    throw std::overflow_error("PCA total variance is nonfinite");
  }

  PcaModel model;
  model.feature_count = observations.columns;
  model.component_count = component_count;
  model.mean.reserve(observations.columns);
  for (const double mean : means) {
    model.mean.push_back(detail::checked_float(mean, "PCA feature mean"));
  }
  model.components = {
      component_count,
      observations.columns,
      std::vector<float>(component_count * observations.columns),
  };
  model.explained_variance.reserve(component_count);
  model.explained_variance_ratio.reserve(component_count);

  for (std::size_t component = 0; component < component_count; ++component) {
    const std::size_t eigenvector_index = order[component];
    std::size_t sign_feature = 0;
    double largest_loading = -1.0;
    for (std::size_t feature = 0; feature < observations.columns; ++feature) {
      const double loading =
          eigensystem
              .eigenvectors[feature * observations.columns + eigenvector_index];
      const double magnitude = std::fabs(loading);
      if (magnitude > largest_loading) {
        largest_loading = magnitude;
        sign_feature = feature;
      }
    }
    const double sign_loading =
        eigensystem.eigenvectors[sign_feature * observations.columns +
                                 eigenvector_index];
    const double sign = sign_loading < 0.0 ? -1.0 : 1.0;
    for (std::size_t feature = 0; feature < observations.columns; ++feature) {
      const double loading =
          sign *
          eigensystem
              .eigenvectors[feature * observations.columns + eigenvector_index];
      model.components.values[component * observations.columns + feature] =
          detail::checked_float(loading, "PCA component loading");
    }

    const double variance = sorted_eigenvalues[component];
    model.explained_variance.push_back(variance);
    model.explained_variance_ratio.push_back(
        total_variance == 0.0 ? 0.0 : variance / total_variance);
  }

  Matrix scores = transform_pca(observations, model);
  return {
      std::move(model),
      std::move(scores),
      eigensystem.sweep_count,
      eigensystem.off_diagonal_residual,
  };
}

Matrix transform_pca(MatrixView observations, const PcaModel &model) {
  detail::validate_finite_matrix(observations, "PCA transform observations");
  validate_pca_model(model);
  if (observations.columns != model.feature_count) {
    throw std::invalid_argument(
        "PCA transform feature count does not match the model");
  }

  Matrix scores{
      observations.rows,
      model.component_count,
      std::vector<float>(
          checked_matrix_size(observations.rows, model.component_count)),
  };
  for (std::size_t row = 0; row < observations.rows; ++row) {
    for (std::size_t component = 0; component < model.component_count;
         ++component) {
      double score = 0.0;
      double correction = 0.0;
      for (std::size_t feature = 0; feature < model.feature_count; ++feature) {
        const double centered =
            static_cast<double>(
                observations.values[row * model.feature_count + feature]) -
            static_cast<double>(model.mean[feature]);
        const double contribution =
            centered *
            static_cast<double>(
                model.components
                    .values[component * model.feature_count + feature]);
        compensated_add(contribution, score, correction);
      }
      scores.values[row * model.component_count + component] =
          detail::checked_float(score, "PCA transformed score");
    }
  }
  return scores;
}

Matrix reconstruct_pca(MatrixView scores, const PcaModel &model) {
  detail::validate_finite_matrix(scores, "PCA reconstruction scores");
  validate_pca_model(model);
  if (scores.columns != model.component_count) {
    throw std::invalid_argument(
        "PCA reconstruction score count does not match the model");
  }

  Matrix reconstruction{
      scores.rows,
      model.feature_count,
      std::vector<float>(checked_matrix_size(scores.rows, model.feature_count)),
  };
  for (std::size_t row = 0; row < scores.rows; ++row) {
    for (std::size_t feature = 0; feature < model.feature_count; ++feature) {
      double value = static_cast<double>(model.mean[feature]);
      double correction = 0.0;
      for (std::size_t component = 0; component < model.component_count;
           ++component) {
        const double contribution =
            static_cast<double>(
                scores.values[row * model.component_count + component]) *
            static_cast<double>(
                model.components
                    .values[component * model.feature_count + feature]);
        compensated_add(contribution, value, correction);
      }
      reconstruction.values[row * model.feature_count + feature] =
          detail::checked_float(value, "PCA reconstructed value");
    }
  }
  return reconstruction;
}

} // namespace riftco_transformer::analysis
