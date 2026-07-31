#include "transformer_lab/nn/low_rank_adapter.hpp"

#include "transformer_lab/core/tensor_ops.hpp"
#include "transformer_lab/nn/initialization.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace transformer_lab {
namespace {

[[nodiscard]] std::size_t checked_width(
    std::size_t width,
    const char* name
) {
    if (width == 0) {
        throw std::invalid_argument(
            std::string("LoRA ") + name +
            " width must be greater than zero"
        );
    }
    return width;
}

[[nodiscard]] std::size_t checked_rank(
    std::size_t input_width,
    std::size_t output_width,
    std::size_t rank
) {
    if (rank == 0 ||
        rank > std::min(input_width, output_width)) {
        throw std::invalid_argument(
            "LoRA rank must be in [1, min(input_width, output_width)]"
        );
    }
    return rank;
}

[[nodiscard]] float checked_alpha(float alpha) {
    if (!std::isfinite(alpha) || alpha <= 0.0F) {
        throw std::invalid_argument(
            "LoRA alpha must be finite and positive"
        );
    }
    return alpha;
}

[[nodiscard]] float checked_scale(
    float alpha,
    std::size_t rank
) {
    const float scale =
        alpha / static_cast<float>(rank);
    if (!std::isfinite(scale) || scale <= 0.0F) {
        throw std::invalid_argument(
            "LoRA alpha divided by rank must be finite and positive"
        );
    }
    return scale;
}

[[nodiscard]] Tensor initialized_a(
    std::size_t input_width,
    std::size_t rank,
    std::mt19937& random,
    ExecutionBackend backend
) {
    return xavier_uniform(
        {rank, input_width},
        input_width,
        rank,
        random,
        backend
    );
}

}  // namespace

LowRankAdapter::LowRankAdapter(
    std::size_t input_width,
    std::size_t output_width,
    std::size_t rank,
    float alpha,
    std::mt19937& random,
    ExecutionBackend backend
)
    : input_width_(checked_width(input_width, "input")),
      output_width_(checked_width(output_width, "output")),
      rank_(checked_rank(input_width_, output_width_, rank)),
      alpha_(checked_alpha(alpha)),
      scale_(checked_scale(alpha_, rank_)),
      a_(initialized_a(input_width_, rank_, random, backend)),
      b_(Tensor::zeros(
          {output_width_, rank_},
          backend
      )),
      lora_a_(a_),
      lora_b_(b_) {
    register_module("lora_a", lora_a_);
    register_module("lora_b", lora_b_);
}

std::size_t LowRankAdapter::input_width() const noexcept {
    return input_width_;
}

std::size_t LowRankAdapter::output_width() const noexcept {
    return output_width_;
}

std::size_t LowRankAdapter::rank() const noexcept {
    return rank_;
}

float LowRankAdapter::alpha() const noexcept {
    return alpha_;
}

float LowRankAdapter::scale() const noexcept {
    return scale_;
}

Variable LowRankAdapter::forward(const Variable& input) const {
    if (input.value().rank() == 0 ||
        input.value().shape().back() != input_width_) {
        throw std::invalid_argument(
            "LoRA input final dimension must equal its input width"
        );
    }

    const std::size_t rows =
        input.value().numel() / input_width_;
    const Variable flattened = reshape(
        input,
        {rows, input_width_}
    );
    const Variable reduced = matmul(
        flattened,
        transpose_2d(a_.variable())
    );
    const Variable expanded = matmul(
        reduced,
        transpose_2d(b_.variable())
    );

    Tensor::Shape output_shape = input.value().shape();
    output_shape.back() = output_width_;
    return reshape(expanded, std::move(output_shape)) * scale_;
}

Tensor LowRankAdapter::weight_delta() const {
    return tensor_ops::scale(
        tensor_ops::matmul(b_.value(), a_.value()),
        scale_
    );
}

void LowRankAdapter::to(ExecutionBackend backend) {
    Module::to(backend);
}

const Parameter& LowRankAdapter::a() const noexcept {
    return a_;
}

const Parameter& LowRankAdapter::b() const noexcept {
    return b_;
}

ParameterList LowRankAdapter::parameters() {
    return Module::parameters();
}

}  // namespace transformer_lab
