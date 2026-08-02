#include "riftco_transformer/compiler/cajal/multilinear_map.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer::compiler::cajal {
namespace {

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right,
                                           const char *quantity) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw MultilinearMapError(std::string(quantity) + " exceeds size_t");
  }
  return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right,
                                      const char *quantity) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw MultilinearMapError(std::string(quantity) + " exceeds size_t");
  }
  return left + right;
}

[[nodiscard]] std::string dimension_error(std::size_t input_index,
                                          std::size_t actual,
                                          std::size_t expected) {
  return "multilinear map input " + std::to_string(input_index) +
         " has dimension " + std::to_string(actual) + ", expected " +
         std::to_string(expected);
}

} // namespace

MultilinearMap::MultilinearMap(std::vector<std::size_t> input_dimensions,
                               std::size_t output_dimension,
                               std::vector<double> coefficients)
    : input_dimensions_(std::move(input_dimensions)),
      output_dimension_(output_dimension), input_combination_count_(1),
      coefficients_(std::move(coefficients)) {
  if (output_dimension_ == 0) {
    throw MultilinearMapError(
        "multilinear map output dimension must be greater than zero");
  }

  for (std::size_t input_index = 0; input_index < input_dimensions_.size();
       ++input_index) {
    const std::size_t dimension = input_dimensions_[input_index];
    if (dimension == 0) {
      throw MultilinearMapError("multilinear map input dimension " +
                                std::to_string(input_index) +
                                " must be greater than zero");
    }
    input_combination_count_ =
        checked_multiply(input_combination_count_, dimension,
                         "multilinear map input dimension product");
  }

  const std::size_t expected_coefficient_count =
      checked_multiply(output_dimension_, input_combination_count_,
                       "multilinear map coefficient count");
  if (coefficients_.size() != expected_coefficient_count) {
    throw MultilinearMapError("multilinear map requires exactly " +
                              std::to_string(expected_coefficient_count) +
                              " coefficients, got " +
                              std::to_string(coefficients_.size()));
  }

  for (std::size_t index = 0; index < coefficients_.size(); ++index) {
    if (!std::isfinite(coefficients_[index])) {
      throw MultilinearMapError("multilinear map coefficient " +
                                std::to_string(index) + " must be finite");
    }
  }
}

MultilinearMap MultilinearMap::constant(const EncodedValue &value) {
  const std::span<const double> coordinates = value.coordinates();
  return MultilinearMap(
      {}, coordinates.size(),
      std::vector<double>(coordinates.begin(), coordinates.end()));
}

MultilinearMap MultilinearMap::identity(std::size_t dimension) {
  if (dimension == 0) {
    throw MultilinearMapError(
        "multilinear map identity dimension must be greater than zero");
  }

  const std::size_t coefficient_count = checked_multiply(
      dimension, dimension, "multilinear map identity coefficient count");
  std::vector<double> coefficients(coefficient_count, 0.0);
  for (std::size_t index = 0; index < dimension; ++index) {
    const std::size_t row_offset = checked_multiply(
        index, dimension, "multilinear map identity coefficient index");
    coefficients[checked_add(
        row_offset, index, "multilinear map identity coefficient index")] = 1.0;
  }
  return MultilinearMap({dimension}, dimension, std::move(coefficients));
}

std::size_t MultilinearMap::arity() const noexcept {
  return input_dimensions_.size();
}

std::span<const std::size_t> MultilinearMap::input_dimensions() const noexcept {
  return {input_dimensions_.data(), input_dimensions_.size()};
}

std::size_t MultilinearMap::output_dimension() const noexcept {
  return output_dimension_;
}

std::size_t MultilinearMap::coefficient_count() const noexcept {
  return coefficients_.size();
}

std::span<const double> MultilinearMap::coefficients() const noexcept {
  return {coefficients_.data(), coefficients_.size()};
}

double MultilinearMap::coefficient_at(
    std::size_t output_index,
    std::span<const std::size_t> input_indices) const {
  if (input_indices.size() != arity()) {
    throw MultilinearMapError("multilinear map coefficient lookup requires " +
                              std::to_string(arity()) + " input indices, got " +
                              std::to_string(input_indices.size()));
  }
  if (output_index >= output_dimension_) {
    throw MultilinearMapError(
        "multilinear map output index " + std::to_string(output_index) +
        " is out of range for dimension " + std::to_string(output_dimension_));
  }

  std::size_t input_offset = 0;
  for (std::size_t input_index = 0; input_index < arity(); ++input_index) {
    const std::size_t coordinate = input_indices[input_index];
    const std::size_t dimension = input_dimensions_[input_index];
    if (coordinate >= dimension) {
      throw MultilinearMapError(
          "multilinear map input index " + std::to_string(coordinate) +
          " on axis " + std::to_string(input_index) +
          " is out of range for dimension " + std::to_string(dimension));
    }
    input_offset =
        checked_add(checked_multiply(input_offset, dimension,
                                     "multilinear map coefficient index"),
                    coordinate, "multilinear map coefficient index");
  }

  const std::size_t output_offset =
      checked_multiply(output_index, input_combination_count_,
                       "multilinear map coefficient index");
  return coefficients_[checked_add(output_offset, input_offset,
                                   "multilinear map coefficient index")];
}

EncodedValue MultilinearMap::apply(std::span<const EncodedValue> inputs) const {
  if (inputs.size() != arity()) {
    throw MultilinearMapError("multilinear map application requires " +
                              std::to_string(arity()) + " inputs, got " +
                              std::to_string(inputs.size()));
  }

  std::vector<std::span<const double>> input_coordinates;
  input_coordinates.reserve(inputs.size());
  for (std::size_t input_index = 0; input_index < inputs.size();
       ++input_index) {
    const std::span<const double> coordinates =
        inputs[input_index].coordinates();
    const std::size_t expected_dimension = input_dimensions_[input_index];
    if (coordinates.size() != expected_dimension) {
      throw MultilinearMapError(
          dimension_error(input_index, coordinates.size(), expected_dimension));
    }
    for (std::size_t coordinate_index = 0;
         coordinate_index < coordinates.size(); ++coordinate_index) {
      if (!std::isfinite(coordinates[coordinate_index])) {
        throw MultilinearMapError("multilinear map input " +
                                  std::to_string(input_index) + " coordinate " +
                                  std::to_string(coordinate_index) +
                                  " must be finite");
      }
    }
    input_coordinates.push_back(coordinates);
  }

  std::vector<double> result(output_dimension_, 0.0);
  for (std::size_t output_index = 0; output_index < output_dimension_;
       ++output_index) {
    const std::size_t coefficient_offset =
        checked_multiply(output_index, input_combination_count_,
                         "multilinear map coefficient index");
    double sum = 0.0;
    for (std::size_t combination = 0; combination < input_combination_count_;
         ++combination) {
      double contribution =
          coefficients_[checked_add(coefficient_offset, combination,
                                    "multilinear map coefficient index")];
      std::size_t remaining = combination;
      for (std::size_t input_index = arity(); input_index-- > 0;) {
        const std::size_t dimension = input_dimensions_[input_index];
        const std::size_t coordinate_index = remaining % dimension;
        remaining /= dimension;
        contribution *= input_coordinates[input_index][coordinate_index];
        if (!std::isfinite(contribution)) {
          throw MultilinearMapError(
              "multilinear map application produced a non-finite "
              "contribution for output " +
              std::to_string(output_index) + " at input combination " +
              std::to_string(combination));
        }
      }
      sum += contribution;
      if (!std::isfinite(sum)) {
        throw MultilinearMapError(
            "multilinear map application produced a non-finite result for "
            "output " +
            std::to_string(output_index));
      }
    }
    result[output_index] = sum;
  }

  return EncodedValue(std::move(result));
}

} // namespace riftco_transformer::compiler::cajal
