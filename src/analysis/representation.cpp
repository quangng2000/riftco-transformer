#include "riftco_transformer/analysis/representation.hpp"

#include "detail/validation.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::analysis {
namespace {

std::size_t checked_leading_rows(std::span<const std::size_t> leading_shape) {
  std::size_t rows = 1;
  for (const std::size_t dimension : leading_shape) {
    if (dimension == 0) {
      throw std::invalid_argument(
          "representation leading dimensions must be positive");
    }
    if (rows > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error("representation leading shape is too large");
    }
    rows *= dimension;
  }
  return rows;
}

void validate_representation(const NamedRepresentation &representation) {
  if (representation.name.empty()) {
    throw std::invalid_argument("representation capture name must be nonempty");
  }
  detail::validate_finite_matrix(representation.observations.view(),
                                 "representation observations");
  if (checked_leading_rows(representation.leading_shape) !=
      representation.observations.rows) {
    throw std::invalid_argument(
        "representation leading shape does not match observation rows");
  }
}

} // namespace

void RepresentationTrace::capture(std::string name,
                                  std::span<const std::size_t> shape,
                                  std::span<const float> values) {
  if (shape.empty()) {
    throw std::invalid_argument(
        "representation shape must include a feature axis");
  }
  const auto leading_shape = shape.first(shape.size() - 1);
  const std::size_t rows = checked_leading_rows(leading_shape);
  const std::size_t columns = shape.back();
  const std::size_t expected = checked_matrix_size(rows, columns);
  if (values.size() != expected) {
    throw std::invalid_argument(
        "representation value count does not match its full shape");
  }

  capture({
      std::move(name),
      std::vector<std::size_t>(leading_shape.begin(), leading_shape.end()),
      {rows, columns, std::vector<float>(values.begin(), values.end())},
  });
}

void RepresentationTrace::capture(NamedRepresentation representation) {
  validate_representation(representation);
  if (contains(representation.name)) {
    throw std::invalid_argument("representation capture names must be unique");
  }
  entries_.push_back(std::move(representation));
}

bool RepresentationTrace::contains(std::string_view name) const noexcept {
  for (const auto &entry : entries_) {
    if (entry.name == name) {
      return true;
    }
  }
  return false;
}

const NamedRepresentation &
RepresentationTrace::at(std::string_view name) const {
  for (const auto &entry : entries_) {
    if (entry.name == name) {
      return entry;
    }
  }
  throw std::out_of_range("representation capture name was not found");
}

std::span<const NamedRepresentation>
RepresentationTrace::entries() const noexcept {
  return entries_;
}

} // namespace riftco_transformer::analysis
