#include "riftco_transformer/model/rotary_embedding.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer {
namespace {

struct RotaryShape {
    std::size_t batch;
    std::size_t heads;
    std::size_t time;
    std::size_t width;
};

RotaryShape checked_rotary_shape(
    const Tensor& input,
    std::size_t position_offset,
    float theta
) {
    if (input.rank() != 4) {
        throw std::invalid_argument(
            "rotary embedding requires [batch, head, time, head_width]"
        );
    }
    const auto& shape = input.shape();
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0 || shape[3] == 0) {
        throw std::invalid_argument(
            "rotary embedding dimensions must be greater than zero"
        );
    }
    if (shape[3] % 2 != 0) {
        throw std::invalid_argument(
            "rotary embedding head width must be even"
        );
    }
    if (!std::isfinite(theta) || theta <= 0.0F) {
        throw std::invalid_argument(
            "rotary embedding theta must be finite and positive"
        );
    }
    if (position_offset >
        std::numeric_limits<std::size_t>::max() - (shape[2] - 1)) {
        throw std::overflow_error("rotary embedding position overflows");
    }
    return {shape[0], shape[1], shape[2], shape[3]};
}

std::size_t offset(
    const RotaryShape& shape,
    std::size_t batch,
    std::size_t head,
    std::size_t time,
    std::size_t channel
) {
    return (((batch * shape.heads + head) * shape.time + time) *
            shape.width) + channel;
}

std::pair<float, float> rotary_cos_sin(
    std::size_t position,
    std::size_t channel,
    std::size_t width,
    float theta
) {
    const double exponent =
        -2.0 * static_cast<double>(channel) /
        static_cast<double>(width);
    const double angle =
        static_cast<double>(position) *
        std::pow(static_cast<double>(theta), exponent);
    return {
        static_cast<float>(std::cos(angle)),
        static_cast<float>(std::sin(angle)),
    };
}

Tensor rotate_forward(
    const Tensor& input,
    const RotaryShape& shape,
    std::size_t position_offset,
    float theta
) {
    Tensor output(input.shape(), input.backend());
    const std::size_t half = shape.width / 2;
    for (std::size_t batch = 0; batch < shape.batch; ++batch) {
        for (std::size_t head = 0; head < shape.heads; ++head) {
            for (std::size_t time = 0; time < shape.time; ++time) {
                for (std::size_t channel = 0; channel < half; ++channel) {
                    const auto [cosine, sine] = rotary_cos_sin(
                        position_offset + time,
                        channel,
                        shape.width,
                        theta
                    );
                    const std::size_t first = offset(
                        shape, batch, head, time, channel
                    );
                    const std::size_t second = offset(
                        shape, batch, head, time, channel + half
                    );
                    const float first_value = input.flat(first);
                    const float second_value = input.flat(second);
                    output.flat(first) =
                        first_value * cosine - second_value * sine;
                    output.flat(second) =
                        second_value * cosine + first_value * sine;
                }
            }
        }
    }
    return output;
}

Tensor rotate_backward(
    const Tensor& upstream,
    const RotaryShape& shape,
    std::size_t position_offset,
    float theta
) {
    Tensor gradient(upstream.shape(), upstream.backend());
    const std::size_t half = shape.width / 2;
    for (std::size_t batch = 0; batch < shape.batch; ++batch) {
        for (std::size_t head = 0; head < shape.heads; ++head) {
            for (std::size_t time = 0; time < shape.time; ++time) {
                for (std::size_t channel = 0; channel < half; ++channel) {
                    const auto [cosine, sine] = rotary_cos_sin(
                        position_offset + time,
                        channel,
                        shape.width,
                        theta
                    );
                    const std::size_t first = offset(
                        shape, batch, head, time, channel
                    );
                    const std::size_t second = offset(
                        shape, batch, head, time, channel + half
                    );
                    const float first_gradient = upstream.flat(first);
                    const float second_gradient = upstream.flat(second);
                    gradient.flat(first) =
                        first_gradient * cosine + second_gradient * sine;
                    gradient.flat(second) =
                        -first_gradient * sine + second_gradient * cosine;
                }
            }
        }
    }
    return gradient;
}

}  // namespace

Variable apply_rotary_position_embedding(
    const Variable& input,
    std::size_t position_offset,
    float theta
) {
    const RotaryShape shape = checked_rotary_shape(
        input.value(),
        position_offset,
        theta
    );
    const std::array inputs{input};
    return custom_gradient(
        rotate_forward(input.value(), shape, position_offset, theta),
        inputs,
        [shape, position_offset, theta](const Tensor& upstream) {
            if (upstream.shape() != Tensor::Shape{
                    shape.batch,
                    shape.heads,
                    shape.time,
                    shape.width,
                }) {
                throw std::invalid_argument(
                    "rotary embedding upstream gradient shape changed"
                );
            }
            return std::vector<Tensor>{rotate_backward(
                upstream,
                shape,
                position_offset,
                theta
            )};
        }
    );
}

}  // namespace riftco_transformer
