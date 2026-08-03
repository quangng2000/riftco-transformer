#include "riftco_transformer/model/grouped_query_attention.hpp"

#include "riftco_transformer/model/causal_self_attention.hpp"
#include "riftco_transformer/model/rotary_embedding.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer {
namespace {

struct GroupedAttentionDimensions {
    std::size_t model_width;
    std::size_t query_heads;
    std::size_t key_value_heads;
    std::size_t head_width;
};

GroupedAttentionDimensions checked_dimensions(
    std::size_t model_width,
    std::size_t query_heads,
    std::size_t key_value_heads
) {
    if (model_width == 0 || query_heads == 0 || key_value_heads == 0) {
        throw std::invalid_argument(
            "grouped-query attention dimensions must be greater than zero"
        );
    }
    if (model_width % query_heads != 0) {
        throw std::invalid_argument(
            "model width must be divisible by query head count"
        );
    }
    if (query_heads % key_value_heads != 0) {
        throw std::invalid_argument(
            "query head count must be divisible by key/value head count"
        );
    }
    const std::size_t head_width = model_width / query_heads;
    if (head_width % 2 != 0) {
        throw std::invalid_argument(
            "grouped-query attention head width must be even for RoPE"
        );
    }
    return {model_width, query_heads, key_value_heads, head_width};
}

float checked_theta(float theta) {
    if (!std::isfinite(theta) || theta <= 0.0F) {
        throw std::invalid_argument(
            "grouped-query attention RoPE theta must be finite and positive"
        );
    }
    return theta;
}

Tensor repeated_heads_forward(
    const Tensor& input,
    std::size_t repetitions
) {
    const auto& shape = input.shape();
    if (shape[1] >
        std::numeric_limits<std::size_t>::max() / repetitions) {
        throw std::overflow_error("repeated key/value head count overflows");
    }
    Tensor output(
        {shape[0], shape[1] * repetitions, shape[2], shape[3]},
        input.backend()
    );
    const std::size_t head_elements = shape[2] * shape[3];
    for (std::size_t batch = 0; batch < shape[0]; ++batch) {
        for (std::size_t head = 0; head < shape[1]; ++head) {
            const std::size_t input_offset =
                (batch * shape[1] + head) * head_elements;
            for (std::size_t copy = 0; copy < repetitions; ++copy) {
                const std::size_t output_head = head * repetitions + copy;
                const std::size_t output_offset =
                    (batch * shape[1] * repetitions + output_head) *
                    head_elements;
                for (std::size_t index = 0;
                     index < head_elements;
                     ++index) {
                    output.flat(output_offset + index) =
                        input.flat(input_offset + index);
                }
            }
        }
    }
    return output;
}

Tensor repeated_heads_backward(
    const Tensor& upstream,
    const Tensor::Shape& input_shape,
    std::size_t repetitions
) {
    Tensor gradient(input_shape, 0.0F, upstream.backend());
    const std::size_t head_elements = input_shape[2] * input_shape[3];
    for (std::size_t batch = 0; batch < input_shape[0]; ++batch) {
        for (std::size_t head = 0; head < input_shape[1]; ++head) {
            const std::size_t input_offset =
                (batch * input_shape[1] + head) * head_elements;
            for (std::size_t copy = 0; copy < repetitions; ++copy) {
                const std::size_t output_head = head * repetitions + copy;
                const std::size_t output_offset =
                    (batch * input_shape[1] * repetitions + output_head) *
                    head_elements;
                for (std::size_t index = 0;
                     index < head_elements;
                     ++index) {
                    gradient.flat(input_offset + index) +=
                        upstream.flat(output_offset + index);
                }
            }
        }
    }
    return gradient;
}

}  // namespace

Variable repeat_key_value_heads(
    const Variable& input,
    std::size_t repetitions
) {
    if (input.value().rank() != 4) {
        throw std::invalid_argument(
            "key/value head repetition requires rank-four input"
        );
    }
    if (repetitions == 0) {
        throw std::invalid_argument(
            "key/value head repetitions must be greater than zero"
        );
    }
    if (repetitions == 1) {
        return input;
    }
    const Tensor::Shape input_shape = input.value().shape();
    const std::array inputs{input};
    return custom_gradient(
        repeated_heads_forward(input.value(), repetitions),
        inputs,
        [input_shape, repetitions](const Tensor& upstream) {
            const Tensor::Shape expected{
                input_shape[0],
                input_shape[1] * repetitions,
                input_shape[2],
                input_shape[3],
            };
            if (upstream.shape() != expected) {
                throw std::invalid_argument(
                    "repeated key/value upstream gradient shape changed"
                );
            }
            return std::vector<Tensor>{repeated_heads_backward(
                upstream,
                input_shape,
                repetitions
            )};
        }
    );
}

GroupedQueryAttention::GroupedQueryAttention(
    std::size_t model_width,
    std::size_t query_head_count,
    std::size_t key_value_head_count,
    float rope_theta,
    std::mt19937& random
) : query_head_count_(checked_dimensions(
        model_width,
        query_head_count,
        key_value_head_count
    ).query_heads),
    key_value_head_count_(key_value_head_count),
    rope_theta_(checked_theta(rope_theta)),
    query_(model_width, model_width, random, false),
    key_(
        model_width,
        key_value_head_count * (model_width / query_head_count),
        random,
        false
    ),
    value_(
        model_width,
        key_value_head_count * (model_width / query_head_count),
        random,
        false
    ),
    output_(model_width, model_width, random, false) {
    register_module("query", query_);
    register_module("key", key_);
    register_module("value", value_);
    register_module("output", output_);
}

std::size_t GroupedQueryAttention::model_width() const noexcept {
    return query_.input_width();
}

std::size_t GroupedQueryAttention::query_head_count() const noexcept {
    return query_head_count_;
}

std::size_t GroupedQueryAttention::key_value_head_count() const noexcept {
    return key_value_head_count_;
}

std::size_t GroupedQueryAttention::head_width() const noexcept {
    return model_width() / query_head_count();
}

float GroupedQueryAttention::rope_theta() const noexcept {
    return rope_theta_;
}

Variable GroupedQueryAttention::forward(
    const Variable& input,
    std::size_t position_offset
) const {
    if (input.value().rank() != 3 ||
        input.value().shape()[2] != model_width()) {
        throw std::invalid_argument(
            "grouped-query attention input must have shape "
            "[batch, time, model_width]"
        );
    }
    const Variable queries = apply_rotary_position_embedding(
        split_attention_heads(query_.forward(input), query_head_count()),
        position_offset,
        rope_theta()
    );
    const Variable keys = apply_rotary_position_embedding(
        split_attention_heads(key_.forward(input), key_value_head_count()),
        position_offset,
        rope_theta()
    );
    const Variable values = split_attention_heads(
        value_.forward(input),
        key_value_head_count()
    );
    const std::size_t repetitions =
        query_head_count() / key_value_head_count();
    CausalAttentionResult attention = causal_scaled_dot_product_attention(
        queries,
        repeat_key_value_heads(keys, repetitions),
        repeat_key_value_heads(values, repetitions)
    );
    return output_.forward(merge_attention_heads(attention.context));
}

void GroupedQueryAttention::to(ExecutionBackend backend) {
    Module::to(backend);
}

ParameterList GroupedQueryAttention::parameters() {
    return Module::parameters();
}

}  // namespace riftco_transformer
