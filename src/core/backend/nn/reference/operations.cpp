#include "core/backend/nn/reference/operations.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

std::vector<std::size_t> make_strides(std::span<const std::size_t> shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index > 1; --index) {
        const auto current = index - 2;
        strides[current] = strides[current + 1] * shape[current + 1];
    }
    return strides;
}

} // namespace

void nn_reference_unary_elementwise(const UnaryElementwiseRequest& request) {
    const auto input = request.input.data();
    auto output = request.output.data();
    for (std::size_t index = 0; index < request.element_count; ++index) {
        const float value = input[index];
        switch (request.operation) {
        case UnaryOperation::Negate:
            output[index] = -value;
            break;
        case UnaryOperation::Exp:
            output[index] = std::exp(value);
            break;
        case UnaryOperation::Log:
            if (value <= 0.0F) {
                throw std::domain_error(
                    "log requires strictly positive values");
            }
            output[index] = std::log(value);
            break;
        case UnaryOperation::Sqrt:
            if (value < 0.0F) {
                throw std::domain_error("sqrt requires non-negative values");
            }
            output[index] = std::sqrt(value);
            break;
        case UnaryOperation::Erf:
            output[index] = std::erf(value);
            break;
        }
    }
}

void nn_reference_binary_elementwise(const BinaryElementwiseRequest& request) {
    const auto left = request.left.data();
    const auto right = request.right.data();
    auto output = request.output.data();
    for (std::size_t index = 0; index < request.element_count; ++index) {
        switch (request.operation) {
        case BinaryOperation::Add:
            output[index] = left[index] + right[index];
            break;
        case BinaryOperation::Subtract:
            output[index] = left[index] - right[index];
            break;
        case BinaryOperation::Multiply:
            output[index] = left[index] * right[index];
            break;
        case BinaryOperation::Divide:
            if (right[index] == 0.0F) {
                throw std::domain_error("division by zero in tensor operation");
            }
            output[index] = left[index] / right[index];
            break;
        }
    }
}

void nn_reference_scale(const ScaleRequest& request) {
    const auto input = request.input.data();
    auto output = request.output.data();
    for (std::size_t index = 0; index < request.element_count; ++index) {
        output[index] = input[index] * request.scale;
    }
}

void nn_reference_gelu_forward(const GeluForwardRequest& request) {
    constexpr float one_over_sqrt_two = 0.70710678118654752440F;
    const auto input = request.input.data();
    auto output = request.output.data();
    for (std::size_t index = 0; index < request.element_count; ++index) {
        const float value = input[index];
        output[index] =
            0.5F * value * (1.0F + std::erf(value * one_over_sqrt_two));
    }
}

void nn_reference_gelu_backward(const GeluBackwardRequest& request) {
    constexpr float one_over_sqrt_two = 0.70710678118654752440F;
    constexpr float one_over_sqrt_two_pi = 0.39894228040143267794F;
    const auto input = request.input.data();
    const auto upstream = request.upstream.data();
    auto input_gradient = request.input_gradient.data();
    for (std::size_t index = 0; index < request.element_count; ++index) {
        const float value = input[index];
        const float derivative =
            0.5F * (1.0F + std::erf(value * one_over_sqrt_two)) +
            value * one_over_sqrt_two_pi * std::exp(-0.5F * value * value);
        input_gradient[index] = upstream[index] * derivative;
    }
}

void nn_reference_reduce(const ReductionRequest& request) {
    const auto input = request.input.data();
    auto output = request.output.data();
    const auto& dimensions = request.dimensions;
    for (std::size_t outer = 0; outer < dimensions.outer; ++outer) {
        for (std::size_t inner = 0; inner < dimensions.inner; ++inner) {
            float total = 0.0F;
            const auto base =
                outer * dimensions.width * dimensions.inner + inner;
            for (std::size_t item = 0; item < dimensions.width; ++item) {
                total += input[base + item * dimensions.inner];
            }
            if (request.operation == ReductionOperation::Mean) {
                total /= static_cast<float>(dimensions.width);
            }
            output[outer * dimensions.inner + inner] = total;
        }
    }
}

void nn_reference_copy(const CopyRequest& request) {
    const auto input = request.input.data();
    auto output = request.output.data();
    std::copy_n(input.begin(), request.element_count, output.begin());
}

void nn_reference_permute(const PermuteRequest& request) {
    const auto input = request.input.data();
    auto output = request.output.data();
    const auto input_strides = make_strides(request.input_shape);

    std::vector<std::size_t> output_shape;
    output_shape.reserve(request.axes.size());
    for (const auto axis : request.axes) {
        output_shape.push_back(request.input_shape[axis]);
    }
    const auto output_strides = make_strides(output_shape);

    for (std::size_t output_index = 0; output_index < output.size();
         ++output_index) {
        std::size_t remainder = output_index;
        std::size_t input_index = 0;
        for (std::size_t output_dimension = 0;
             output_dimension < output_shape.size(); ++output_dimension) {
            const auto coordinate =
                remainder / output_strides[output_dimension];
            remainder %= output_strides[output_dimension];
            input_index +=
                coordinate * input_strides[request.axes[output_dimension]];
        }
        output[output_index] = input[input_index];
    }
}

void nn_reference_broadcast(const BroadcastRequest& request) {
    const auto input = request.input.data();
    auto output = request.output.data();
    const auto input_strides = make_strides(request.input_shape);
    const auto output_strides = make_strides(request.output_shape);
    const auto rank_offset =
        request.output_shape.size() - request.input_shape.size();

    for (std::size_t output_index = 0; output_index < output.size();
         ++output_index) {
        std::size_t remainder = output_index;
        std::size_t input_index = 0;
        for (std::size_t output_dimension = 0;
             output_dimension < request.output_shape.size();
             ++output_dimension) {
            const auto coordinate =
                remainder / output_strides[output_dimension];
            remainder %= output_strides[output_dimension];
            if (output_dimension < rank_offset) {
                continue;
            }
            const auto input_dimension = output_dimension - rank_offset;
            const auto input_coordinate =
                request.input_shape[input_dimension] == 1 ? 0 : coordinate;
            input_index += input_coordinate * input_strides[input_dimension];
        }
        output[output_index] = input[input_index];
    }
}

void nn_reference_sum_to_shape(const SumToShapeRequest& request) {
    const auto input = request.input.data();
    auto output = request.output.data();
    std::fill(output.begin(), output.end(), 0.0F);

    const auto input_strides = make_strides(request.input_shape);
    const auto output_strides = make_strides(request.output_shape);
    const auto rank_offset =
        request.input_shape.size() - request.output_shape.size();

    for (std::size_t input_index = 0; input_index < input.size();
         ++input_index) {
        std::size_t remainder = input_index;
        std::size_t output_index = 0;
        for (std::size_t input_dimension = 0;
             input_dimension < request.input_shape.size(); ++input_dimension) {
            const auto coordinate = remainder / input_strides[input_dimension];
            remainder %= input_strides[input_dimension];
            if (input_dimension < rank_offset) {
                continue;
            }
            const auto output_dimension = input_dimension - rank_offset;
            const auto output_coordinate =
                request.output_shape[output_dimension] == 1 ? 0 : coordinate;
            output_index +=
                output_coordinate * output_strides[output_dimension];
        }
        output[output_index] += input[input_index];
    }
}

void nn_reference_softmax_forward(const SoftmaxForwardRequest& request) {
    const auto input = request.input.data();
    auto probabilities = request.probabilities.data();
    const auto& dimensions = request.dimensions;

    for (std::size_t outer = 0; outer < dimensions.outer; ++outer) {
        for (std::size_t inner = 0; inner < dimensions.inner; ++inner) {
            const auto base =
                outer * dimensions.width * dimensions.inner + inner;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t item = 0; item < dimensions.width; ++item) {
                const float value = input[base + item * dimensions.inner];
                if (std::isnan(value) ||
                    value == std::numeric_limits<float>::infinity()) {
                    throw std::domain_error(
                        "softmax rejects NaN and positive infinity");
                }
                maximum = std::max(maximum, value);
            }
            if (maximum == -std::numeric_limits<float>::infinity()) {
                throw std::domain_error(
                    "softmax requires one finite value per slice");
            }

            double denominator = 0.0;
            for (std::size_t item = 0; item < dimensions.width; ++item) {
                const auto index = base + item * dimensions.inner;
                const double exponential =
                    std::exp(static_cast<double>(input[index] - maximum));
                probabilities[index] = static_cast<float>(exponential);
                denominator += exponential;
            }
            for (std::size_t item = 0; item < dimensions.width; ++item) {
                const auto index = base + item * dimensions.inner;
                probabilities[index] = static_cast<float>(
                    static_cast<double>(probabilities[index]) / denominator);
            }
        }
    }
}

void nn_reference_softmax_backward(const SoftmaxBackwardRequest& request) {
    const auto probabilities = request.probabilities.data();
    const auto upstream = request.upstream.data();
    auto input_gradient = request.input_gradient.data();
    const auto& dimensions = request.dimensions;

    for (std::size_t outer = 0; outer < dimensions.outer; ++outer) {
        for (std::size_t inner = 0; inner < dimensions.inner; ++inner) {
            const auto base =
                outer * dimensions.width * dimensions.inner + inner;
            double weighted_upstream = 0.0;
            for (std::size_t item = 0; item < dimensions.width; ++item) {
                const auto index = base + item * dimensions.inner;
                weighted_upstream += static_cast<double>(probabilities[index]) *
                                     static_cast<double>(upstream[index]);
            }
            for (std::size_t item = 0; item < dimensions.width; ++item) {
                const auto index = base + item * dimensions.inner;
                input_gradient[index] =
                    probabilities[index] *
                    (upstream[index] - static_cast<float>(weighted_upstream));
            }
        }
    }
}

void nn_reference_causal_softmax_forward(
    const CausalSoftmaxForwardRequest& request) {
    const auto scores = request.scores.data();
    auto probabilities = request.probabilities.data();
    std::fill(probabilities.begin(), probabilities.end(), 0.0F);

    for (std::size_t batch = 0; batch < request.batch; ++batch) {
        for (std::size_t head = 0; head < request.heads; ++head) {
            for (std::size_t query = 0; query < request.time; ++query) {
                const auto row =
                    (batch * request.heads + head) * request.time + query;
                const auto base = row * request.time;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t key = 0; key <= query; ++key) {
                    const float value =
                        scores[base + key] * request.score_scale;
                    if (std::isnan(value) ||
                        value == std::numeric_limits<float>::infinity()) {
                        throw std::domain_error(
                            "softmax rejects NaN and positive infinity");
                    }
                    maximum = std::max(maximum, value);
                }
                if (maximum == -std::numeric_limits<float>::infinity()) {
                    throw std::domain_error(
                        "softmax requires one finite value per slice");
                }

                double denominator = 0.0;
                for (std::size_t key = 0; key <= query; ++key) {
                    const double exponential = std::exp(static_cast<double>(
                        scores[base + key] * request.score_scale - maximum));
                    probabilities[base + key] = static_cast<float>(exponential);
                    denominator += exponential;
                }
                for (std::size_t key = 0; key <= query; ++key) {
                    probabilities[base + key] = static_cast<float>(
                        static_cast<double>(probabilities[base + key]) /
                        denominator);
                }
            }
        }
    }
}

void nn_reference_causal_softmax_backward(
    const CausalSoftmaxBackwardRequest& request) {
    const auto probabilities = request.probabilities.data();
    const auto upstream = request.upstream.data();
    auto score_gradient = request.score_gradient.data();
    std::fill(score_gradient.begin(), score_gradient.end(), 0.0F);

    for (std::size_t batch = 0; batch < request.batch; ++batch) {
        for (std::size_t head = 0; head < request.heads; ++head) {
            for (std::size_t query = 0; query < request.time; ++query) {
                const auto row =
                    (batch * request.heads + head) * request.time + query;
                const auto base = row * request.time;
                double weighted_upstream = 0.0;
                for (std::size_t key = 0; key <= query; ++key) {
                    weighted_upstream +=
                        static_cast<double>(probabilities[base + key]) *
                        static_cast<double>(upstream[base + key]);
                }
                for (std::size_t key = 0; key <= query; ++key) {
                    score_gradient[base + key] =
                        probabilities[base + key] *
                        (upstream[base + key] -
                         static_cast<float>(weighted_upstream)) *
                        request.score_scale;
                }
            }
        }
    }
}

void nn_reference_gather_rows(const GatherRowsRequest& request) {
    const auto table = request.table.data();
    auto output = request.output.data();
    for (std::size_t position = 0; position < request.row_indices.size();
         ++position) {
        const auto row =
            static_cast<std::size_t>(request.row_indices[position]);
        for (std::size_t column = 0; column < request.width; ++column) {
            output[position * request.width + column] =
                table[row * request.width + column];
        }
    }
}

void nn_reference_scatter_add_rows(const ScatterAddRowsRequest& request) {
    const auto upstream = request.upstream.data();
    auto table_gradient = request.table_gradient.data();
    std::fill(table_gradient.begin(), table_gradient.end(), 0.0F);
    for (std::size_t position = 0; position < request.row_indices.size();
         ++position) {
        const auto row =
            static_cast<std::size_t>(request.row_indices[position]);
        for (std::size_t column = 0; column < request.width; ++column) {
            table_gradient[row * request.width + column] +=
                upstream[position * request.width + column];
        }
    }
}

void nn_reference_layer_norm_forward(const LayerNormForwardRequest& request) {
    const auto input = request.input.data();
    const auto scale = request.scale.data();
    const auto bias = request.bias.data();
    auto output = request.output.data();
    auto mean = request.mean.data();
    auto inverse_standard_deviation = request.inverse_standard_deviation.data();

    for (std::size_t row = 0; row < request.rows; ++row) {
        const auto base = row * request.width;
        float row_mean = 0.0F;
        for (std::size_t column = 0; column < request.width; ++column) {
            row_mean += input[base + column];
        }
        row_mean /= static_cast<float>(request.width);

        float variance = 0.0F;
        for (std::size_t column = 0; column < request.width; ++column) {
            const float centered = input[base + column] - row_mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(request.width);
        const float inverse_std = 1.0F / std::sqrt(variance + request.epsilon);

        mean[row] = row_mean;
        inverse_standard_deviation[row] = inverse_std;
        for (std::size_t column = 0; column < request.width; ++column) {
            const float normalized =
                (input[base + column] - row_mean) * inverse_std;
            output[base + column] = normalized * scale[column] + bias[column];
        }
    }
}

void nn_reference_layer_norm_backward(const LayerNormBackwardRequest& request) {
    const auto input = request.input.data();
    const auto scale = request.scale.data();
    const auto mean = request.mean.data();
    const auto inverse_standard_deviation =
        request.inverse_standard_deviation.data();
    const auto upstream = request.upstream.data();
    auto input_gradient = request.input_gradient.data();
    auto scale_gradient = request.scale_gradient.data();
    auto bias_gradient = request.bias_gradient.data();
    std::fill(scale_gradient.begin(), scale_gradient.end(), 0.0F);
    std::fill(bias_gradient.begin(), bias_gradient.end(), 0.0F);

    for (std::size_t row = 0; row < request.rows; ++row) {
        const auto base = row * request.width;
        float sum_scaled_upstream = 0.0F;
        float sum_scaled_upstream_times_normalized = 0.0F;
        for (std::size_t column = 0; column < request.width; ++column) {
            const float normalized = (input[base + column] - mean[row]) *
                                     inverse_standard_deviation[row];
            const float scaled_upstream =
                upstream[base + column] * scale[column];
            sum_scaled_upstream += scaled_upstream;
            sum_scaled_upstream_times_normalized +=
                scaled_upstream * normalized;
            scale_gradient[column] += upstream[base + column] * normalized;
            bias_gradient[column] += upstream[base + column];
        }

        const float inverse_width = 1.0F / static_cast<float>(request.width);
        for (std::size_t column = 0; column < request.width; ++column) {
            const float normalized = (input[base + column] - mean[row]) *
                                     inverse_standard_deviation[row];
            const float scaled_upstream =
                upstream[base + column] * scale[column];
            input_gradient[base + column] =
                inverse_standard_deviation[row] *
                (scaled_upstream - sum_scaled_upstream * inverse_width -
                 normalized * sum_scaled_upstream_times_normalized *
                     inverse_width);
        }
    }
}

void nn_reference_cross_entropy_forward(
    const CrossEntropyForwardRequest& request) {
    nn_reference_softmax_forward({
        request.logits,
        request.base_gradient,
        {
            request.positions,
            request.classes,
            1,
        },
    });

    const auto logits = request.logits.data();
    auto base_gradient = request.base_gradient.data();
    auto loss = request.loss.data();
    double total_loss = 0.0;

    for (std::size_t position = 0; position < request.positions; ++position) {
        const auto row_offset = position * request.classes;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t item = 0; item < request.classes; ++item) {
            maximum = std::max(maximum, logits[row_offset + item]);
        }

        double exponential_sum = 0.0;
        for (std::size_t item = 0; item < request.classes; ++item) {
            exponential_sum += std::exp(
                static_cast<double>(logits[row_offset + item] - maximum));
        }

        const auto target = static_cast<std::size_t>(request.targets[position]);
        const float target_logit = logits[row_offset + target];
        if (!std::isfinite(target_logit)) {
            throw std::domain_error(
                "cross entropy target logit must be finite");
        }
        const double log_probability =
            static_cast<double>(target_logit - maximum) -
            std::log(exponential_sum);
        total_loss -= log_probability;
        base_gradient[row_offset + target] -= 1.0F;
    }

    const float inverse_positions =
        1.0F / static_cast<float>(request.positions);
    for (float& value : base_gradient) {
        value *= inverse_positions;
    }

    const double mean_loss =
        total_loss / static_cast<double>(request.positions);
    if (!std::isfinite(mean_loss) ||
        mean_loss > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::overflow_error(
            "cross entropy loss exceeds finite float range");
    }
    loss[0] = static_cast<float>(mean_loss);
}

} // namespace riftco_transformer::backend_detail
