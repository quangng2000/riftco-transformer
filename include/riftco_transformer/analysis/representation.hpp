#pragma once

#include "riftco_transformer/analysis/matrix.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer::analysis {

struct NamedRepresentation {
  std::string name;
  // Original shape excluding its final feature axis. An empty leading shape
  // represents one unbatched observation.
  std::vector<std::size_t> leading_shape;
  // [product(leading_shape), feature_count].
  Matrix observations;
};

// Ordered, owning collection of uniquely named representation captures.
class RepresentationTrace {
public:
  // Captures a full shape [..., features], flattening every leading axis
  // into matrix rows and copying the values into trace-owned storage.
  void capture(std::string name, std::span<const std::size_t> shape,
               std::span<const float> values);

  // Captures an already flattened representation after validating that its
  // leading shape agrees with observations.rows.
  void capture(NamedRepresentation representation);

  [[nodiscard]] bool contains(std::string_view name) const noexcept;
  [[nodiscard]] const NamedRepresentation &at(std::string_view name) const;
  [[nodiscard]] std::span<const NamedRepresentation> entries() const noexcept;

private:
  std::vector<NamedRepresentation> entries_;
};

} // namespace riftco_transformer::analysis
