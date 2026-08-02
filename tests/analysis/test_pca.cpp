#include "riftco_transformer/analysis/pca.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace analysis = riftco_transformer::analysis;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(double actual, double expected, const std::string &message,
                   double tolerance = 1.0e-5) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

void require_values_close(std::span<const float> actual,
                          const std::vector<float> &expected,
                          const std::string &message,
                          double tolerance = 1.0e-5) {
  require(actual.size() == expected.size(), message + ": size");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require_close(static_cast<double>(actual[index]),
                  static_cast<double>(expected[index]),
                  message + " at index " + std::to_string(index), tolerance);
  }
}

template <typename Exception, typename Function>
void require_throws(Function &&function, const std::string &message) {
  bool threw_expected = false;
  try {
    function();
  } catch (const Exception &) {
    threw_expected = true;
  }
  require(threw_expected, message);
}

void test_axis_aligned_fit_transform_and_reconstruct() {
  const analysis::Matrix observations{
      4,
      2,
      {-2.0F, 0.0F, -1.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F},
  };
  analysis::PcaOptions options;
  options.component_count = 1;
  const analysis::PcaFit fit = analysis::fit_pca(observations.view(), options);

  require_values_close(observations.values,
                       {-2.0F, 0.0F, -1.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F},
                       "PCA input immutability");
  require(fit.model.feature_count == 2, "PCA feature count");
  require(fit.model.component_count == 1, "PCA component count");
  require_values_close(fit.model.mean, {0.0F, 0.0F}, "PCA mean");
  require_values_close(fit.model.components.values, {1.0F, 0.0F},
                       "axis-aligned component");
  require_values_close(fit.scores.values, {-2.0F, -1.0F, 1.0F, 2.0F},
                       "axis-aligned scores");
  require_close(fit.model.explained_variance.front(), 10.0 / 3.0,
                "axis-aligned variance");
  require_close(fit.model.explained_variance_ratio.front(), 1.0,
                "axis-aligned variance ratio");

  const analysis::Matrix transformed =
      analysis::transform_pca(observations.view(), fit.model);
  require_values_close(transformed.values, fit.scores.values,
                       "fit and transform scores");
  const analysis::Matrix reconstructed =
      analysis::reconstruct_pca(transformed.view(), fit.model);
  require_values_close(reconstructed.values, observations.values,
                       "rank-one reconstruction");
  const analysis::Matrix held_out = analysis::transform_pca(
      analysis::Matrix{1, 2, {3.0F, 0.0F}}.view(), fit.model);
  require_values_close(held_out.values, {3.0F}, "held-out transform");
}

void test_translation_rotation_and_determinism() {
  const analysis::Matrix diagonal{
      4,
      2,
      {-2.0F, -2.0F, -1.0F, -1.0F, 1.0F, 1.0F, 2.0F, 2.0F},
  };
  analysis::PcaOptions options;
  options.component_count = 1;
  const analysis::PcaFit first = analysis::fit_pca(diagonal.view(), options);
  const analysis::PcaFit replay = analysis::fit_pca(diagonal.view(), options);
  const double inverse_square_root_two = 1.0 / std::sqrt(2.0);
  require_close(first.model.components.values[0], inverse_square_root_two,
                "rotated first loading");
  require_close(first.model.components.values[1], inverse_square_root_two,
                "rotated second loading");
  require_close(first.model.explained_variance.front(), 20.0 / 3.0,
                "rotated variance");
  require(first.model.components.values == replay.model.components.values &&
              first.scores.values == replay.scores.values &&
              first.model.explained_variance == replay.model.explained_variance,
          "PCA must replay deterministically");

  analysis::Matrix translated = diagonal;
  for (std::size_t row = 0; row < translated.rows; ++row) {
    translated.values[row * 2] += 7.0F;
    translated.values[row * 2 + 1] -= 3.0F;
  }
  const analysis::PcaFit shifted =
      analysis::fit_pca(translated.view(), options);
  require_values_close(shifted.model.components.values,
                       first.model.components.values,
                       "translation-invariant components");
  require_values_close(shifted.scores.values, first.scores.values,
                       "translation-invariant scores");
  require_values_close(shifted.model.mean, {7.0F, -3.0F}, "translated mean");
}

void test_full_rank_and_zero_variance() {
  const analysis::Matrix observations{
      5,
      2,
      {-2.0F, 1.0F, -1.0F, -2.0F, 0.0F, 2.0F, 1.0F, -1.0F, 2.0F, 0.0F},
  };
  const analysis::PcaFit fit = analysis::fit_pca(observations.view());
  require(fit.model.component_count == 2, "default retains full rank bound");
  const analysis::Matrix reconstructed =
      analysis::reconstruct_pca(fit.scores.view(), fit.model);
  require_values_close(reconstructed.values, observations.values,
                       "full-rank reconstruction", 2.0e-5);
  require_close(fit.model.explained_variance_ratio[0] +
                    fit.model.explained_variance_ratio[1],
                1.0, "retained variance ratios");

  const analysis::Matrix constant{
      3,
      2,
      {4.0F, -2.0F, 4.0F, -2.0F, 4.0F, -2.0F},
  };
  const analysis::PcaFit constant_fit = analysis::fit_pca(constant.view());
  require_values_close(constant_fit.model.mean, {4.0F, -2.0F}, "constant mean");
  for (const float score : constant_fit.scores.values) {
    require_close(score, 0.0, "constant score");
  }
  for (const double ratio : constant_fit.model.explained_variance_ratio) {
    require_close(ratio, 0.0, "constant explained variance ratio");
  }
  const analysis::Matrix constant_reconstruction =
      analysis::reconstruct_pca(constant_fit.scores.view(), constant_fit.model);
  require_values_close(constant_reconstruction.values, constant.values,
                       "constant reconstruction");
}

void test_random_full_rank_invariants() {
  constexpr std::size_t row_count = 12;
  constexpr std::size_t feature_count = 5;
  std::mt19937 random(1729U);
  std::uniform_real_distribution<float> distribution(-3.0F, 3.0F);
  std::vector<float> values(row_count * feature_count);
  for (float &value : values) {
    value = distribution(random);
  }
  const analysis::Matrix observations{
      row_count,
      feature_count,
      values,
  };
  const analysis::PcaFit fit = analysis::fit_pca(observations.view());
  require(fit.model.component_count == feature_count,
          "random PCA retains every feature component");

  double ratio_sum = 0.0;
  for (std::size_t component = 0; component < feature_count; ++component) {
    ratio_sum += fit.model.explained_variance_ratio[component];
    if (component != 0) {
      require(fit.model.explained_variance[component - 1] >=
                  fit.model.explained_variance[component],
              "PCA variances are sorted descending");
    }

    std::size_t largest_feature = 0;
    float largest_magnitude = -1.0F;
    for (std::size_t feature = 0; feature < feature_count; ++feature) {
      const float loading =
          fit.model.components.values[component * feature_count + feature];
      if (std::fabs(loading) > largest_magnitude) {
        largest_magnitude = std::fabs(loading);
        largest_feature = feature;
      }
    }
    require(fit.model.components.values[component * feature_count +
                                        largest_feature] >= 0.0F,
            "PCA canonicalizes component sign");

    for (std::size_t other = 0; other < feature_count; ++other) {
      double dot = 0.0;
      for (std::size_t feature = 0; feature < feature_count; ++feature) {
        dot +=
            static_cast<double>(
                fit.model.components
                    .values[component * feature_count + feature]) *
            static_cast<double>(
                fit.model.components.values[other * feature_count + feature]);
      }
      require_close(dot, component == other ? 1.0 : 0.0,
                    "PCA components are orthonormal", 2.0e-5);
    }
  }
  require_close(ratio_sum, 1.0, "full PCA variance ratios sum to one", 1.0e-8);

  const analysis::Matrix reconstructed =
      analysis::reconstruct_pca(fit.scores.view(), fit.model);
  require_values_close(reconstructed.values, observations.values,
                       "random full-rank reconstruction", 3.0e-5);
}

void test_matrix_and_pca_validation() {
  require_throws<std::invalid_argument>(
      [] { static_cast<void>(analysis::checked_matrix_size(0, 1)); },
      "matrix rejects zero rows");
  require_throws<std::overflow_error>(
      [] {
        static_cast<void>(analysis::checked_matrix_size(
            std::numeric_limits<std::size_t>::max(), 2));
      },
      "matrix rejects size overflow");
  require_throws<std::invalid_argument>(
      [] { analysis::validate_matrix_view({2, 2, std::vector<float>{1.0F}}); },
      "matrix rejects mismatched storage");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(
            analysis::fit_pca(analysis::Matrix{1, 2, {1.0F, 2.0F}}.view()));
      },
      "PCA rejects one observation");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(analysis::fit_pca(analysis::Matrix{
            2,
            1,
            {1.0F, std::numeric_limits<float>::infinity()},
        }
                                                .view()));
      },
      "PCA rejects nonfinite observations");

  const analysis::Matrix observations{
      4,
      3,
      {
          1.0F,
          2.0F,
          4.0F,
          2.0F,
          -1.0F,
          3.0F,
          -2.0F,
          3.0F,
          1.0F,
          4.0F,
          0.5F,
          -3.0F,
      },
  };
  analysis::PcaOptions options;
  options.component_count = 4;
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::fit_pca(observations.view(), options));
      },
      "PCA rejects too many components");
  options = {};
  options.max_sweeps = 0;
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::fit_pca(observations.view(), options));
      },
      "PCA rejects zero sweeps");
  options = {};
  options.tolerance = std::numeric_limits<double>::quiet_NaN();
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::fit_pca(observations.view(), options));
      },
      "PCA rejects NaN tolerance");
  options = {};
  options.max_covariance_elements = 8;
  require_throws<std::length_error>(
      [&] {
        static_cast<void>(analysis::fit_pca(observations.view(), options));
      },
      "PCA enforces covariance resource limit");
  options = {};
  options.max_sweeps = 1;
  options.tolerance = std::numeric_limits<double>::min();
  require_throws<std::runtime_error>(
      [&] {
        static_cast<void>(analysis::fit_pca(observations.view(), options));
      },
      "PCA reports nonconvergence");

  const analysis::PcaFit fit = analysis::fit_pca(observations.view());
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::transform_pca(
            analysis::Matrix{2, 2, {1.0F, 2.0F, 3.0F, 4.0F}}.view(),
            fit.model));
      },
      "PCA transform rejects feature mismatch");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::reconstruct_pca(
            analysis::Matrix{2, 1, {1.0F, 2.0F}}.view(), fit.model));
      },
      "PCA reconstruction rejects score mismatch");
  analysis::PcaModel invalid_model = fit.model;
  invalid_model.mean.pop_back();
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(
            analysis::transform_pca(observations.view(), invalid_model));
      },
      "PCA transform validates its model");
}

} // namespace

int main() {
  try {
    test_axis_aligned_fit_transform_and_reconstruct();
    test_translation_rotation_and_determinism();
    test_full_rank_and_zero_variance();
    test_random_full_rank_invariants();
    test_matrix_and_pca_validation();
    std::cout << "Analysis PCA tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Analysis PCA test failure: " << error.what() << '\n';
    return 1;
  }
}
