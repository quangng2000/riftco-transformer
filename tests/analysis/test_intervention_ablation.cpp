#include "riftco_transformer/analysis/ablation.hpp"
#include "riftco_transformer/analysis/intervention.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
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
                   double tolerance = 1.0e-6) {
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
                          double tolerance = 1.0e-6) {
  require(actual.size() == expected.size(), message + ": size");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require_close(actual[index], expected[index], message, tolerance);
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

void test_coordinate_and_direction_interventions() {
  const analysis::Matrix source{
      2,
      3,
      {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
  };
  const analysis::Matrix replaced = analysis::apply_intervention(
      source.view(), analysis::CoordinateReplacement{{0, 2}, {-1.0F, 9.0F}});
  require_values_close(replaced.values, {-1.0F, 2.0F, 9.0F, -1.0F, 5.0F, 9.0F},
                       "coordinate replacement");
  require_values_close(source.values, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                       "intervention source immutability");

  const analysis::Matrix normalized = analysis::apply_intervention(
      source.view(),
      analysis::DirectionSteering{{3.0F, 4.0F, 0.0F}, 2.0F, true});
  require_values_close(normalized.values, {2.2F, 3.6F, 3.0F, 5.2F, 6.6F, 6.0F},
                       "normalized steering");
  const analysis::Matrix raw = analysis::apply_intervention(
      source.view(),
      analysis::DirectionSteering{{1.0F, -2.0F, 0.5F}, 0.5F, false});
  require_values_close(raw.values, {1.5F, 1.0F, 3.25F, 4.5F, 4.0F, 6.25F},
                       "raw steering");

  const analysis::Matrix projected = analysis::apply_intervention(
      analysis::Matrix{1, 2, {3.0F, 1.0F}}.view(),
      analysis::DirectionProjection{{1.0F, 1.0F}, {}});
  require_values_close(projected.values, {1.0F, -1.0F},
                       "origin-zero projection");
  const analysis::Matrix centered_projection = analysis::apply_intervention(
      analysis::Matrix{1, 2, {3.0F, 1.0F}}.view(),
      analysis::DirectionProjection{{1.0F, 1.0F}, {1.0F, 1.0F}});
  require_values_close(centered_projection.values, {2.0F, 0.0F},
                       "centered projection");
}

void test_intervention_validation() {
  const analysis::Matrix source{1, 2, {1.0F, 2.0F}};
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(), analysis::CoordinateReplacement{{}, {}}));
      },
      "replacement rejects empty coordinates");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(), analysis::CoordinateReplacement{{0, 1}, {1.0F}}));
      },
      "replacement rejects mismatched coordinates and values");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(),
            analysis::CoordinateReplacement{{0, 0}, {1.0F, 2.0F}}));
      },
      "replacement rejects duplicate coordinates");
  require_throws<std::out_of_range>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(), analysis::CoordinateReplacement{{2}, {0.0F}}));
      },
      "replacement rejects out-of-range coordinates");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(), analysis::CoordinateReplacement{
                               {0},
                               {std::numeric_limits<float>::quiet_NaN()},
                           }));
      },
      "replacement rejects nonfinite values");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(),
            analysis::DirectionSteering{{0.0F, 0.0F}, 1.0F, true}));
      },
      "steering rejects zero direction");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(), analysis::DirectionSteering{
                               {std::numeric_limits<float>::infinity(), 0.0F},
                               1.0F,
                               true,
                           }));
      },
      "steering rejects nonfinite direction");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(), analysis::DirectionSteering{
                               {1.0F, 0.0F},
                               std::numeric_limits<float>::quiet_NaN(),
                               true,
                           }));
      },
      "steering rejects nonfinite strength");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(), analysis::DirectionProjection{{1.0F}, {}}));
      },
      "projection rejects direction width mismatch");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(), analysis::DirectionProjection{{0.0F, 0.0F}, {}}));
      },
      "projection rejects zero direction");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(),
            analysis::DirectionProjection{{1.0F, 0.0F}, {0.0F}}));
      },
      "projection rejects origin width mismatch");
  require_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(analysis::apply_intervention(
            source.view(), analysis::DirectionProjection{
                               {1.0F, 0.0F},
                               {std::numeric_limits<float>::infinity(), 0.0F},
                           }));
      },
      "projection rejects nonfinite origin");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(analysis::apply_intervention(
            analysis::Matrix{
                1,
                2,
                {1.0F, std::numeric_limits<float>::infinity()},
            }
                .view(),
            analysis::DirectionSteering{{1.0F, 0.0F}, 1.0F, true}));
      },
      "intervention rejects nonfinite observations");
  require_throws<std::overflow_error>(
      [] {
        const float maximum = std::numeric_limits<float>::max();
        static_cast<void>(analysis::apply_intervention(
            analysis::Matrix{1, 1, {maximum}}.view(),
            analysis::DirectionSteering{{maximum}, maximum, false}));
      },
      "intervention rejects output overflow");
}

void test_paired_ablation_summaries() {
  const std::vector<double> baseline{1.0, 2.0, 3.0};
  const std::vector<double> intervened{0.5, 1.0, 4.0};
  const analysis::AblationSummary maximize = analysis::summarize_ablation(
      baseline, intervened, analysis::MetricGoal::Maximize);
  require(maximize.sample_count == 3, "ablation sample count");
  require_close(maximize.baseline_mean, 2.0, "baseline mean");
  require_close(maximize.intervened_mean, 11.0 / 6.0, "intervened mean");
  require_close(maximize.mean_delta, -1.0 / 6.0, "mean delta");
  require_close(maximize.mean_degradation, 1.0 / 6.0, "maximize degradation");
  require_close(maximize.paired_standard_error, std::sqrt(13.0 / 36.0),
                "paired standard error");
  require_close(maximize.root_mean_square_delta, std::sqrt(3.0 / 4.0),
                "root mean square delta");
  require_close(maximize.maximum_absolute_delta, 1.0, "maximum delta");
  require_close(maximize.fraction_degraded, 2.0 / 3.0,
                "maximize degraded fraction");

  const analysis::AblationSummary minimize = analysis::summarize_ablation(
      baseline, intervened, analysis::MetricGoal::Minimize);
  require_close(minimize.mean_degradation, -1.0 / 6.0, "minimize degradation");
  require_close(minimize.fraction_degraded, 1.0 / 3.0,
                "minimize degraded fraction");
  require_close(minimize.paired_standard_error, maximize.paired_standard_error,
                "goal-independent standard error");

  const analysis::AblationSummary singleton = analysis::summarize_ablation(
      std::vector<double>{2.0}, std::vector<double>{2.0},
      analysis::MetricGoal::Maximize);
  require_close(singleton.paired_standard_error, 0.0, "singleton error");
  require_close(singleton.root_mean_square_delta, 0.0, "singleton RMS");
  require_close(singleton.fraction_degraded, 0.0, "singleton fraction");
}

void test_ablation_validation() {
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(analysis::summarize_ablation(
            {}, {}, analysis::MetricGoal::Maximize));
      },
      "ablation rejects empty samples");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(analysis::summarize_ablation(
            std::vector<double>{1.0}, std::vector<double>{1.0, 2.0},
            analysis::MetricGoal::Maximize));
      },
      "ablation rejects mismatched samples");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(analysis::summarize_ablation(
            std::vector<double>{1.0},
            std::vector<double>{
                std::numeric_limits<double>::quiet_NaN(),
            },
            analysis::MetricGoal::Maximize));
      },
      "ablation rejects nonfinite samples");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(analysis::summarize_ablation(
            std::vector<double>{1.0}, std::vector<double>{1.0},
            static_cast<analysis::MetricGoal>(UINT8_C(255))));
      },
      "ablation rejects unknown metric goal");
}

} // namespace

int main() {
  try {
    test_coordinate_and_direction_interventions();
    test_intervention_validation();
    test_paired_ablation_summaries();
    test_ablation_validation();
    std::cout << "Analysis intervention and ablation tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Analysis intervention/ablation test failure: " << error.what()
              << '\n';
    return 1;
  }
}
