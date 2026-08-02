#pragma once

#include "riftco_transformer/analysis/matrix.hpp"

#include <cstddef>
#include <variant>
#include <vector>

namespace riftco_transformer::analysis {

// Replaces the listed feature in every row. The index and replacement arrays
// must be nonempty, equally sized, unique, and in range.
struct CoordinateReplacement {
  std::vector<std::size_t> feature_indices;
  std::vector<float> replacement_values;
};

// Adds strength * direction to every row. When normalize is true, direction
// is first normalized to unit L2 length.
struct DirectionSteering {
  std::vector<float> direction;
  float strength = 1.0F;
  bool normalize = true;
};

// Removes each row's component parallel to direction. An empty origin means
// the zero vector; otherwise origin must contain one value per feature.
struct DirectionProjection {
  std::vector<float> direction;
  std::vector<float> origin;
};

using InterventionConfig =
    std::variant<CoordinateReplacement, DirectionSteering, DirectionProjection>;

// Applies an intervention row-wise and returns independent owned storage.
[[nodiscard]] Matrix apply_intervention(MatrixView observations,
                                        const InterventionConfig &intervention);

} // namespace riftco_transformer::analysis
