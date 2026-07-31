#include "adam_reference.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace riftco_transformer::backend_detail {
namespace {

float checked_float(double value, const char* description) {
    const float result = static_cast<float>(value);
    if (!std::isfinite(value) || !std::isfinite(result)) {
        throw std::overflow_error(
            std::string("Adam produced a non-finite ") + description
        );
    }
    return result;
}

}  // namespace

void adam_reference_update(const AdamUpdateRequest& request) {
    for (const auto& tensor : request.tensors) {
        const auto value = tensor.value.data();
        const auto gradient = tensor.gradient.data();
        const auto first = tensor.first_moment.data();
        const auto second = tensor.second_moment.data();
        auto next_value = tensor.next_value.data();
        auto next_first = tensor.next_first_moment.data();
        auto next_second = tensor.next_second_moment.data();

        for (std::size_t index = 0; index < value.size(); ++index) {
            const double clipped_gradient =
                static_cast<double>(gradient[index]) *
                request.clip_scale;
            const double first_value =
                static_cast<double>(request.beta1) *
                    static_cast<double>(first[index]) +
                (1.0 - static_cast<double>(request.beta1)) *
                    clipped_gradient;
            const double second_value =
                static_cast<double>(request.beta2) *
                    static_cast<double>(second[index]) +
                (1.0 - static_cast<double>(request.beta2)) *
                    clipped_gradient * clipped_gradient;
            const double corrected_first =
                first_value / request.first_correction;
            const double corrected_second =
                second_value / request.second_correction;
            const double update =
                static_cast<double>(request.learning_rate) *
                corrected_first /
                (
                    std::sqrt(corrected_second) +
                    static_cast<double>(request.epsilon)
                );

            next_first[index] = checked_float(
                first_value,
                "first moment"
            );
            next_second[index] = checked_float(
                second_value,
                "second moment"
            );
            next_value[index] = checked_float(
                static_cast<double>(value[index]) - update,
                "parameter value"
            );
        }
    }
}

}  // namespace riftco_transformer::backend_detail
