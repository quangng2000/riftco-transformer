#include "riftco_transformer/compiler/cajal/encoding.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace riftco_transformer::compiler::cajal {
namespace {

[[nodiscard]] bool coordinates_close(double left, double right,
                                     double tolerance) noexcept {
  if (left == right) {
    return true;
  }

  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  return std::abs((left / scale) - (right / scale)) <= tolerance;
}

void validate_coordinate_count(const Type &type, std::size_t actual) {
  if (actual != type.dimension()) {
    throw EncodingError("encoded coordinate count does not match Cajal type "
                        "dimension");
  }
}

void validate_finite(std::span<const double> coordinates) {
  if (!std::all_of(
          coordinates.begin(), coordinates.end(),
          [](double coordinate) { return std::isfinite(coordinate); })) {
    throw EncodingError("encoded coordinates must all be finite");
  }
}

[[nodiscard]] std::vector<double> encode_coordinates(const Value &value) {
  const Type type = value.type();
  std::vector<double> result;

  switch (value.kind()) {
  case Value::Kind::Unit:
    result.push_back(1.0);
    break;

  case Value::Kind::Sum: {
    const std::size_t left_dimension = type.left().dimension();
    const std::size_t right_dimension = type.right().dimension();
    result.assign(type.dimension(), 0.0);

    const std::vector<double> payload = encode_coordinates(value.payload());
    if (value.is_left()) {
      if (payload.size() != left_dimension) {
        throw EncodingError(
            "left sum payload does not match its Cajal type dimension");
      }
      std::copy(payload.begin(), payload.end(), result.begin());
    } else {
      if (payload.size() != right_dimension) {
        throw EncodingError(
            "right sum payload does not match its Cajal type dimension");
      }
      std::copy(payload.begin(), payload.end(),
                result.begin() + static_cast<std::ptrdiff_t>(left_dimension));
    }
    break;
  }

  case Value::Kind::Product: {
    const Value first = value.first();
    const Value second = value.second();
    std::vector<double> first_coordinates = encode_coordinates(first);
    std::vector<double> second_coordinates = encode_coordinates(second);

    result.reserve(type.dimension());
    result.insert(result.end(), first_coordinates.begin(),
                  first_coordinates.end());
    result.insert(result.end(), second_coordinates.begin(),
                  second_coordinates.end());
    break;
  }

  case Value::Kind::Dictionary: {
    const std::size_t key_dimension = type.key().dimension();
    const std::size_t value_dimension = type.value().dimension();
    result.assign(type.dimension(), 0.0);

    for (std::size_t entry = 0; entry < value.dictionary_size(); ++entry) {
      const Value key = value.dictionary_key(entry);
      const Value mapped_value = value.dictionary_value(entry);
      const std::vector<double> key_coordinates = encode_coordinates(key);
      const std::vector<double> value_coordinates =
          encode_coordinates(mapped_value);

      if (key.type() != type.key() || key_coordinates.size() != key_dimension) {
        throw EncodingError(
            "dictionary key does not match its Cajal type dimension");
      }
      if (mapped_value.type() != type.value() ||
          value_coordinates.size() != value_dimension) {
        throw EncodingError(
            "dictionary value does not match its Cajal type dimension");
      }

      for (std::size_t value_index = 0; value_index < value_dimension;
           ++value_index) {
        for (std::size_t key_index = 0; key_index < key_dimension;
             ++key_index) {
          const std::size_t index = (value_index * key_dimension) + key_index;
          const double updated =
              result[index] +
              (value_coordinates[value_index] * key_coordinates[key_index]);
          if (!std::isfinite(updated)) {
            throw EncodingError(
                "dictionary outer-product encoding is not finite");
          }
          result[index] = updated;
        }
      }
    }
    break;
  }
  }

  validate_coordinate_count(type, result.size());
  validate_finite(result);
  return result;
}

[[nodiscard]] bool is_zero_block(std::span<const double> coordinates,
                                 double tolerance) noexcept {
  return std::all_of(coordinates.begin(), coordinates.end(),
                     [tolerance](double coordinate) {
                       return coordinates_close(coordinate, 0.0, tolerance);
                     });
}

[[nodiscard]] Value decode_coordinates(const Type &type,
                                       std::span<const double> coordinates,
                                       double tolerance);

[[nodiscard]] std::optional<Value>
try_decode_coordinates(const Type &type, std::span<const double> coordinates,
                       double tolerance) {
  try {
    return decode_coordinates(type, coordinates, tolerance);
  } catch (const EncodingError &) {
    return std::nullopt;
  }
}

[[nodiscard]] Value decode_coordinates(const Type &type,
                                       std::span<const double> coordinates,
                                       double tolerance) {
  validate_coordinate_count(type, coordinates.size());

  switch (type.kind()) {
  case Type::Kind::Unit:
    if (!coordinates_close(coordinates.front(), 1.0, tolerance)) {
      throw EncodingError("invalid Unit encoding: coordinate is not one");
    }
    return Value::unit();

  case Type::Kind::Sum: {
    const std::size_t left_dimension = type.left().dimension();
    const std::size_t right_dimension = type.right().dimension();
    const std::span<const double> left_coordinates =
        coordinates.first(left_dimension);
    const std::span<const double> right_coordinates =
        coordinates.subspan(left_dimension, right_dimension);

    const bool left_zero = is_zero_block(left_coordinates, tolerance);
    const bool right_zero = is_zero_block(right_coordinates, tolerance);
    const std::optional<Value> left =
        try_decode_coordinates(type.left(), left_coordinates, tolerance);
    const std::optional<Value> right =
        try_decode_coordinates(type.right(), right_coordinates, tolerance);
    const bool left_candidate = left.has_value() && right_zero;
    const bool right_candidate = right.has_value() && left_zero;

    if (left_candidate && !right_candidate) {
      return Value::inject_left(*left, type.right());
    }
    if (right_candidate && !left_candidate) {
      return Value::inject_right(type.left(), *right);
    }
    if (left_candidate && right_candidate) {
      throw EncodingError(
          "ambiguous Sum encoding: both branches satisfy decoding");
    }
    if ((left.has_value() && right.has_value()) ||
        (!left_zero && !right_zero)) {
      throw EncodingError(
          "ambiguous Sum encoding: more than one branch is active");
    }
    if (left_zero && right_zero) {
      throw EncodingError("invalid Sum encoding: no unique active branch");
    }
    throw EncodingError("invalid Sum encoding: active branch does not "
                        "recursively decode");
  }

  case Type::Kind::Product: {
    const std::size_t first_dimension = type.first().dimension();
    const std::size_t second_dimension = type.second().dimension();
    const Value first = decode_coordinates(
        type.first(), coordinates.first(first_dimension), tolerance);
    const Value second = decode_coordinates(
        type.second(), coordinates.subspan(first_dimension, second_dimension),
        tolerance);
    return Value::product(first, second);
  }

  case Type::Kind::Dictionary:
    throw EncodingError(
        "Dictionary coordinates have no unique source-level decoding");
  }

  throw EncodingError("unknown Cajal type kind while decoding");
}

[[nodiscard]] std::size_t checked_cardinality(const Type &type,
                                              std::size_t limit) {
  switch (type.kind()) {
  case Type::Kind::Unit:
    if (limit < 1) {
      throw EncodingError("Cajal value enumeration exceeds max_values");
    }
    return 1;

  case Type::Kind::Sum: {
    const std::size_t left = checked_cardinality(type.left(), limit);
    const std::size_t right = checked_cardinality(type.right(), limit);
    if (right > limit - left) {
      throw EncodingError("Cajal value enumeration exceeds max_values");
    }
    return left + right;
  }

  case Type::Kind::Product: {
    const std::size_t first = checked_cardinality(type.first(), limit);
    const std::size_t second = checked_cardinality(type.second(), limit);
    if (second > limit / first) {
      throw EncodingError("Cajal value enumeration exceeds max_values");
    }
    return first * second;
  }

  case Type::Kind::Dictionary:
    throw EncodingError(
        "Dictionary values are unbounded source lists and cannot be "
        "enumerated");
  }

  throw EncodingError("unknown Cajal type kind while enumerating values");
}

[[nodiscard]] std::vector<Value> enumerate_unchecked(const Type &type) {
  switch (type.kind()) {
  case Type::Kind::Unit:
    return {Value::unit()};

  case Type::Kind::Sum: {
    std::vector<Value> left = enumerate_unchecked(type.left());
    std::vector<Value> right = enumerate_unchecked(type.right());
    std::vector<Value> result;
    result.reserve(left.size() + right.size());

    for (Value &payload : left) {
      result.push_back(Value::inject_left(std::move(payload), type.right()));
    }
    for (Value &payload : right) {
      result.push_back(Value::inject_right(type.left(), std::move(payload)));
    }
    return result;
  }

  case Type::Kind::Product: {
    const std::vector<Value> first = enumerate_unchecked(type.first());
    const std::vector<Value> second = enumerate_unchecked(type.second());
    std::vector<Value> result;
    result.reserve(first.size() * second.size());

    for (const Value &first_value : first) {
      for (const Value &second_value : second) {
        result.push_back(Value::product(first_value, second_value));
      }
    }
    return result;
  }

  case Type::Kind::Dictionary:
    throw EncodingError(
        "Dictionary values are unbounded source lists and cannot be "
        "enumerated");
  }

  throw EncodingError("unknown Cajal type kind while enumerating values");
}

} // namespace

EncodedValue::EncodedValue(std::vector<double> coordinates)
    : coordinates_(std::move(coordinates)) {
  if (coordinates_.empty()) {
    throw EncodingError("an encoded value requires at least one coordinate");
  }
  validate_finite(coordinates_);
}

std::size_t EncodedValue::size() const noexcept { return coordinates_.size(); }

std::span<const double> EncodedValue::coordinates() const noexcept {
  return coordinates_;
}

double EncodedValue::operator[](std::size_t index) const {
  return coordinates_.at(index);
}

bool operator==(const EncodedValue &left, const EncodedValue &right) noexcept {
  return left.coordinates_ == right.coordinates_;
}

bool operator!=(const EncodedValue &left, const EncodedValue &right) noexcept {
  return !(left == right);
}

bool approximately_equal(const EncodedValue &left, const EncodedValue &right,
                         double tolerance) noexcept {
  if (!std::isfinite(tolerance) || tolerance < 0.0 ||
      left.size() != right.size()) {
    return false;
  }

  for (std::size_t index = 0; index < left.size(); ++index) {
    if (!coordinates_close(left[index], right[index], tolerance)) {
      return false;
    }
  }
  return true;
}

EncodedValue encode(const Value &value) {
  std::vector<double> coordinates = encode_coordinates(value);
  validate_coordinate_count(value.type(), coordinates.size());
  return EncodedValue(std::move(coordinates));
}

Value decode(const Type &type, const EncodedValue &encoded, double tolerance) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0) {
    throw EncodingError("decode tolerance must be finite and positive");
  }
  validate_coordinate_count(type, encoded.size());
  return decode_coordinates(type, encoded.coordinates(), tolerance);
}

std::vector<Value> enumerate_values(const Type &type, std::size_t max_values) {
  if (max_values == 0) {
    throw EncodingError("max_values must be greater than zero");
  }

  const std::size_t count = checked_cardinality(type, max_values);
  if (count > std::vector<Value>{}.max_size()) {
    throw EncodingError("Cajal value enumeration exceeds vector capacity");
  }

  std::vector<Value> result = enumerate_unchecked(type);
  if (result.size() != count) {
    throw EncodingError("Cajal value enumeration cardinality mismatch");
  }
  return result;
}

} // namespace riftco_transformer::compiler::cajal
