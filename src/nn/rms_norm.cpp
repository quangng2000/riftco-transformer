#include "riftco_transformer/nn/rms_norm.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace riftco_transformer {
namespace {

float checked_epsilon(float epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument(
            "RMS normalization epsilon must be finite and positive"
        );
    }
    return epsilon;
}

Tensor initialized_scale(std::size_t width) {
    if (width == 0) {
        throw std::invalid_argument(
            "RMS normalization width must be greater than zero"
        );
    }
    return Tensor::full({width}, 1.0F);
}

Tensor checked_scale(Tensor scale) {
    if (scale.rank() != 1 || scale.shape()[0] == 0) {
        throw std::invalid_argument(
            "RMS normalization scale must have shape [width]"
        );
    }
    return scale;
}

}  // namespace

Variable rms_norm(
    const Variable& input,
    const Variable& scale,
    float epsilon
) {
    if (input.value().rank() == 0) {
        throw std::invalid_argument(
            "RMS normalization requires rank at least one"
        );
    }
    const std::size_t width = input.value().shape().back();
    if (width == 0 || scale.value().shape() != Tensor::Shape{width}) {
        throw std::invalid_argument(
            "RMS normalization scale must match the final input width"
        );
    }
    if (input.value().backend() != scale.value().backend()) {
        throw std::invalid_argument(
            "RMS normalization input and scale must use the same backend"
        );
    }
    epsilon = checked_epsilon(epsilon);

    const Variable mean_square = mean(
        input * input,
        input.value().rank() - 1,
        true
    );
    const Variable inverse_root_mean_square =
        1.0F / sqrt(mean_square + epsilon);
    return input * broadcast_to(
        inverse_root_mean_square,
        input.value().shape()
    ) * broadcast_to(scale, input.value().shape());
}

RMSNorm::RMSNorm(std::size_t width, float epsilon)
    : scale_(initialized_scale(width)),
      epsilon_(checked_epsilon(epsilon)) {
    register_parameter("scale", scale_);
}

RMSNorm::RMSNorm(Tensor scale, float epsilon)
    : scale_(checked_scale(std::move(scale))),
      epsilon_(checked_epsilon(epsilon)) {
    register_parameter("scale", scale_);
}

std::size_t RMSNorm::width() const noexcept {
    return scale_.value().shape()[0];
}

float RMSNorm::epsilon() const noexcept {
    return epsilon_;
}

Variable RMSNorm::forward(const Variable& input) const {
    return rms_norm(input, scale_.variable(), epsilon_);
}

void RMSNorm::to(ExecutionBackend backend) {
    Module::to(backend);
}

const Parameter& RMSNorm::scale() const noexcept {
    return scale_;
}

ParameterList RMSNorm::parameters() {
    return Module::parameters();
}

}  // namespace riftco_transformer
