#include "riftco_transformer/analysis/ablation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace riftco_transformer::analysis {
namespace {

double checked_double(long double value, std::string_view description) {
  constexpr long double maximum =
      static_cast<long double>(std::numeric_limits<double>::max());
  if (!std::isfinite(value) || value < -maximum || value > maximum) {
    throw std::overflow_error(std::string(description) +
                              " is outside finite double range");
  }
  return static_cast<double>(value);
}

long double degradation_for(long double delta, MetricGoal goal) {
  switch (goal) {
  case MetricGoal::Maximize:
    return -delta;
  case MetricGoal::Minimize:
    return delta;
  }
  throw std::invalid_argument("ablation metric goal is not recognized");
}

} // namespace

AblationSummary summarize_ablation(std::span<const double> baseline,
                                   std::span<const double> intervened,
                                   MetricGoal goal) {
  static_cast<void>(degradation_for(0.0L, goal));
  if (baseline.empty()) {
    throw std::invalid_argument(
        "ablation summary requires at least one paired sample");
  }
  if (baseline.size() != intervened.size()) {
    throw std::invalid_argument(
        "ablation baseline and intervention sample counts must match");
  }

  long double baseline_mean = 0.0L;
  long double intervened_mean = 0.0L;
  long double delta_mean = 0.0L;
  long double degradation_mean = 0.0L;
  long double degradation_m2 = 0.0L;
  long double rms_scale = 0.0L;
  long double rms_scaled_sum = 1.0L;
  long double maximum_absolute_delta = 0.0L;
  std::size_t degraded_count = 0;

  for (std::size_t index = 0; index < baseline.size(); ++index) {
    if (!std::isfinite(baseline[index]) || !std::isfinite(intervened[index])) {
      throw std::invalid_argument(
          "ablation samples must contain only finite values");
    }
    const long double sample_count = static_cast<long double>(index + 1);
    const long double baseline_value =
        static_cast<long double>(baseline[index]);
    const long double intervened_value =
        static_cast<long double>(intervened[index]);
    const long double delta = intervened_value - baseline_value;
    const long double degradation = degradation_for(delta, goal);
    if (!std::isfinite(delta) || !std::isfinite(degradation)) {
      throw std::overflow_error(
          "ablation paired delta is outside finite range");
    }

    baseline_mean += (baseline_value - baseline_mean) / sample_count;
    intervened_mean += (intervened_value - intervened_mean) / sample_count;
    delta_mean += (delta - delta_mean) / sample_count;

    const long double previous_degradation_mean = degradation_mean;
    degradation_mean += (degradation - degradation_mean) / sample_count;
    degradation_m2 += (degradation - previous_degradation_mean) *
                      (degradation - degradation_mean);

    const long double absolute_delta = std::fabs(delta);
    maximum_absolute_delta = std::max(maximum_absolute_delta, absolute_delta);
    if (absolute_delta != 0.0L) {
      if (rms_scale < absolute_delta) {
        const long double ratio = rms_scale / absolute_delta;
        rms_scaled_sum = 1.0L + rms_scaled_sum * ratio * ratio;
        rms_scale = absolute_delta;
      } else {
        const long double ratio = absolute_delta / rms_scale;
        rms_scaled_sum += ratio * ratio;
      }
    }
    if (degradation > 0.0L) {
      ++degraded_count;
    }
  }

  const long double count = static_cast<long double>(baseline.size());
  const long double nonnegative_degradation_m2 = std::max(degradation_m2, 0.0L);
  const long double paired_standard_error =
      baseline.size() == 1
          ? 0.0L
          : std::sqrt(nonnegative_degradation_m2 / (count - 1.0L) / count);
  const long double root_mean_square_delta =
      rms_scale == 0.0L ? 0.0L : rms_scale * std::sqrt(rms_scaled_sum / count);

  return {
      baseline.size(),
      checked_double(baseline_mean, "ablation baseline mean"),
      checked_double(intervened_mean, "ablation intervention mean"),
      checked_double(delta_mean, "ablation mean delta"),
      checked_double(degradation_mean, "ablation mean degradation"),
      checked_double(paired_standard_error, "ablation paired standard error"),
      checked_double(root_mean_square_delta, "ablation root-mean-square delta"),
      checked_double(maximum_absolute_delta, "ablation maximum absolute delta"),
      static_cast<double>(degraded_count) /
          static_cast<double>(baseline.size()),
  };
}

} // namespace riftco_transformer::analysis
