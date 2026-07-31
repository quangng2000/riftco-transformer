#include "transformer_lab/model/transformer_block.hpp"

#include <stdexcept>

namespace transformer_lab {
namespace {

std::size_t checked_model_width(
    std::size_t model_width,
    std::size_t head_count,
    std::size_t feed_forward_width
) {
    if (model_width == 0 ||
        head_count == 0 ||
        feed_forward_width == 0) {
        throw std::invalid_argument(
            "transformer block dimensions must be greater than zero"
        );
    }
    if (model_width % head_count != 0) {
        throw std::invalid_argument(
            "transformer block model width must be divisible by head count"
        );
    }
    return model_width;
}

}  // namespace

TransformerBlock::TransformerBlock(
    std::size_t model_width,
    std::size_t head_count,
    std::size_t feed_forward_width,
    std::mt19937& random,
    float layer_norm_epsilon,
    FullSequenceAttentionKind attention_kind
)
    : model_width_(checked_model_width(
          model_width,
          head_count,
          feed_forward_width
      )),
      head_count_(head_count),
      feed_forward_width_(feed_forward_width),
      attention_norm_(model_width_, layer_norm_epsilon),
      attention_(
          model_width_,
          head_count_,
          random,
          attention_kind
      ),
      feed_forward_norm_(model_width_, layer_norm_epsilon),
      feed_forward_(
          model_width_,
          feed_forward_width_,
          random
      ) {
    register_module("attention_norm", attention_norm_);
    register_module("attention", attention_);
    register_module("feed_forward_norm", feed_forward_norm_);
    register_module("feed_forward", feed_forward_);
}

std::size_t TransformerBlock::model_width() const noexcept {
    return model_width_;
}

std::size_t TransformerBlock::head_count() const noexcept {
    return head_count_;
}

std::size_t TransformerBlock::feed_forward_width() const noexcept {
    return feed_forward_width_;
}

FullSequenceAttentionKind
TransformerBlock::full_sequence_attention_kind() const noexcept {
    return attention_.full_sequence_attention_kind();
}

void TransformerBlock::set_full_sequence_attention_kind(
    FullSequenceAttentionKind attention_kind
) {
    attention_.set_full_sequence_attention_kind(attention_kind);
}

Variable TransformerBlock::forward(const Variable& input) const {
    return forward(input, full_sequence_attention_kind());
}

Variable TransformerBlock::forward(
    const Variable& input,
    FullSequenceAttentionKind attention_kind
) const {
    if (input.value().rank() != 3 ||
        input.value().shape()[2] != model_width()) {
        throw std::invalid_argument(
            "transformer block input must have shape "
            "[batch, time, model_width]"
        );
    }

    const Variable attention_state =
        input + attention_.forward(
            attention_norm_.forward(input),
            attention_kind
        );
    return attention_state +
           feed_forward_.forward(
               feed_forward_norm_.forward(attention_state)
           );
}

void TransformerBlock::to(ExecutionBackend backend) {
    Module::to(backend);
}

ParameterList TransformerBlock::parameters() {
    return Module::parameters();
}

ParameterList TransformerBlock::lora_parameters() {
    ParameterList result;
    append_parameter_group(
        result,
        "attention",
        attention_.lora_parameters()
    );
    append_parameter_group(
        result,
        "feed_forward",
        feed_forward_.lora_parameters()
    );
    return result;
}

}  // namespace transformer_lab
