#include "riftco_transformer/nn/linear.hpp"

#include "riftco_transformer/core/tensor_ops.hpp"
#include "riftco_transformer/nn/initialization.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace riftco_transformer {
namespace {

Tensor checked_linear_weight(Tensor weight) {
    if (weight.rank() != 2) {
        throw std::invalid_argument(
            "linear weight must have shape [output, input]"
        );
    }
    return weight;
}

Tensor checked_linear_bias(
    Tensor bias,
    const Tensor& weight
) {
    if (bias.shape() != Tensor::Shape({weight.shape()[0]})) {
        throw std::invalid_argument(
            "linear bias must have shape [output]"
        );
    }
    if (bias.backend() != weight.backend()) {
        throw std::invalid_argument(
            "linear weight and bias must use the same backend"
        );
    }
    return bias;
}

Tensor initialized_linear_weight(
    std::size_t input_width,
    std::size_t output_width,
    std::mt19937& random
) {
    if (input_width == 0 || output_width == 0) {
        throw std::invalid_argument(
            "linear dimensions must be greater than zero"
        );
    }
    return xavier_uniform(
        {output_width, input_width},
        input_width,
        output_width,
        random
    );
}

}  // namespace

Linear::Linear(Tensor weight, Tensor bias)
    : weight_(checked_linear_weight(std::move(weight))),
      bias_(checked_linear_bias(
          std::move(bias),
          weight_.value()
      )) {
    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
}

Linear::Linear(
    std::size_t input_width,
    std::size_t output_width,
    std::mt19937& random
)
    : weight_(initialized_linear_weight(
          input_width,
          output_width,
          random
      )),
      bias_(Tensor::zeros({output_width})) {
    register_parameter("weight", weight_);
    register_parameter("bias", bias_);
}

std::size_t Linear::input_width() const noexcept {
    return weight_.value().shape()[1];
}

std::size_t Linear::output_width() const noexcept {
    return weight_.value().shape()[0];
}

Variable Linear::forward(const Variable& input) const {
    if (input.value().rank() == 0 ||
        input.value().shape().back() != input_width()) {
        throw std::invalid_argument(
            "linear input final dimension must equal input width"
        );
    }

    const auto rows = input.value().numel() / input_width();
    const Variable flattened = reshape(
        input,
        {rows, input_width()}
    );
    const Variable projected = matmul(
        flattened,
        transpose_2d(weight_.variable())
    );

    Tensor::Shape output_shape = input.value().shape();
    output_shape.back() = output_width();
    const Variable restored = reshape(projected, output_shape);
    const Variable base_output = restored + broadcast_to(
        bias_.variable(),
        std::move(output_shape)
    );
    if (!has_lora()) {
        return base_output;
    }
    return base_output + lora_->forward(input);
}

void Linear::to(ExecutionBackend backend) {
    Module::to(backend);
}

void Linear::attach_lora(
    std::size_t rank,
    float alpha,
    std::mt19937& random
) {
    if (lora_ != nullptr || lora_was_merged_) {
        throw std::logic_error(
            "LoRA can only be attached once to a Linear module"
        );
    }

    auto candidate = std::make_unique<LowRankAdapter>(
        input_width(),
        output_width(),
        rank,
        alpha,
        random,
        weight_.value().backend()
    );
    lora_ = std::move(candidate);
}

bool Linear::has_lora() const noexcept {
    return lora_ != nullptr && !lora_was_merged_;
}

ParameterList Linear::lora_parameters() {
    if (!has_lora()) {
        return {};
    }
    return lora_->parameters();
}

void Linear::merge_lora() {
    Tensor merged_weight = prepare_lora_merge();
    commit_prepared_lora_merge(std::move(merged_weight));
}

const Parameter& Linear::weight() const noexcept {
    return weight_;
}

const Parameter& Linear::bias() const noexcept {
    return bias_;
}

ParameterList Linear::parameters() {
    return Module::parameters();
}

ParameterList Linear::extra_parameters_for_transfer() {
    if (lora_ == nullptr) {
        return {};
    }
    return lora_->parameters();
}

Tensor Linear::prepare_lora_merge() const {
    if (!has_lora()) {
        if (lora_was_merged_) {
            throw std::logic_error(
                "LoRA has already been merged into this Linear module"
            );
        }
        throw std::logic_error(
            "cannot merge LoRA before attaching an adapter"
        );
    }

    Tensor candidate = tensor_ops::add(
        weight_.value(),
        lora_->weight_delta()
    );
    if (!std::all_of(
            candidate.data().begin(),
            candidate.data().end(),
            [](float value) {
                return std::isfinite(value);
            }
        )) {
        throw std::domain_error(
            "merged LoRA weight must contain only finite values"
        );
    }
    return candidate;
}

void Linear::commit_prepared_lora_merge(
    Tensor merged_weight
) noexcept {
    weight_.set_value(std::move(merged_weight));
    lora_was_merged_ = true;
}

void Linear::discard_unmerged_lora() noexcept {
    if (!lora_was_merged_) {
        lora_.reset();
    }
}

}  // namespace riftco_transformer
