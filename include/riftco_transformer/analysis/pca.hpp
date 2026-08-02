#pragma once

#include "riftco_transformer/analysis/matrix.hpp"

#include <cstddef>
#include <vector>

namespace riftco_transformer::analysis {

struct PcaOptions {
  // Zero retains every possible centered component:
  // min(observation_count - 1, feature_count).
  std::size_t component_count = 0;
  std::size_t max_sweeps = 100;
  double tolerance = 1.0e-12;
  std::size_t max_covariance_elements = std::size_t{1} << 24;
};

struct PcaModel {
  std::size_t feature_count = 0;
  std::size_t component_count = 0;
  std::vector<float> mean;
  // [component_count, feature_count], one principal direction per row.
  Matrix components;
  std::vector<double> explained_variance;
  std::vector<double> explained_variance_ratio;
};

struct PcaFit {
  PcaModel model;
  // [observation_count, component_count].
  Matrix scores;
  std::size_t sweep_count = 0;
  double off_diagonal_residual = 0.0;
};

// Fits centered sample-covariance PCA using a deterministic cyclic Jacobi
// eigensolver. Input values are never mutated.
[[nodiscard]] PcaFit fit_pca(MatrixView observations,
                             const PcaOptions &options = {});

// Applies a previously fitted model without refitting its mean or directions.
[[nodiscard]] Matrix transform_pca(MatrixView observations,
                                   const PcaModel &model);

// Maps [observation_count, component_count] scores back into feature space.
[[nodiscard]] Matrix reconstruct_pca(MatrixView scores, const PcaModel &model);

} // namespace riftco_transformer::analysis
