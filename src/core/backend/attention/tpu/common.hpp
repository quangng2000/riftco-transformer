#pragma once

#include "core/backend/adapters/tpu/runtime.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer::backend_detail::attention_tpu_detail {

inline std::int64_t checked_dimension(std::size_t value) {
    if (value == 0 || value > static_cast<std::size_t>(
                                  std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "TPU attention dimension exceeds the PJRT int64 range");
    }
    return static_cast<std::int64_t>(value);
}

inline std::string tensor_type(
    std::initializer_list<std::size_t> dimensions,
    std::string_view element_type = "f32") {
    std::string result = "tensor<";
    bool first = true;
    for (const std::size_t dimension : dimensions) {
        if (!first) {
            result += "x";
        }
        result += std::to_string(dimension);
        first = false;
    }
    result += "x";
    result += element_type;
    result += ">";
    return result;
}

inline std::string float_literal(float value) {
    std::ostringstream stream;
    stream << std::scientific
           << std::setprecision(std::numeric_limits<float>::max_digits10)
           << value;
    return stream.str();
}

inline float attention_scale(std::size_t head_width) {
    return 1.0F / std::sqrt(static_cast<float>(head_width));
}

inline void require_valid_softmax_status(const std::vector<float>& status) {
    for (const float value : status) {
        if (value != 0.0F || !std::isfinite(value)) {
            throw std::domain_error(
                "TPU attention softmax encountered an invalid row");
        }
    }
}

}  // namespace riftco_transformer::backend_detail::attention_tpu_detail
