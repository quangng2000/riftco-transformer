#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace riftco_transformer::analysis {

enum class MetricGoal : std::uint8_t {
  Maximize = 0,
  Minimize = 1,
};

// Every delta is paired by sample. mean_delta is intervened minus baseline;
// mean_degradation is positive when the intervention made the configured
// metric worse.
struct AblationSummary {
  std::size_t sample_count = 0;
  double baseline_mean = 0.0;
  double intervened_mean = 0.0;
  double mean_delta = 0.0;
  double mean_degradation = 0.0;
  double paired_standard_error = 0.0;
  double root_mean_square_delta = 0.0;
  double maximum_absolute_delta = 0.0;
  double fraction_degraded = 0.0;
};

[[nodiscard]] AblationSummary
summarize_ablation(std::span<const double> baseline,
                   std::span<const double> intervened, MetricGoal goal);

} // namespace riftco_transformer::analysis
