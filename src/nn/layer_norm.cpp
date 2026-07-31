#include "transformer_lab/nn/layer_norm.hpp"

#include "core/backend/adapter.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace transformer_lab {
namespace {

float checked_epsilon(float epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        throw std::invalid_argument(
            "layer normalization epsilon must be finite and positive"
        );
    }
    return epsilon;
}

Tensor checked_layer_norm_scale(Tensor scale) {
    if (scale.rank() != 1) {
        throw std::invalid_argument(
            "layer normalization scale must have shape [width]"
        );
    }
    return scale;
}

Tensor checked_layer_norm_bias(Tensor bias, const Tensor& scale) {
    if (bias.shape() != scale.shape()) {
        throw std::invalid_argument(
            "layer normalization bias must match scale shape"
        );
    }
    if (bias.backend() != scale.backend()) {
        throw std::invalid_argument(
            "layer normalization scale and bias must use the same backend"
        );
    }
    return bias;
}

}  // namespace

Variable layer_norm(
    const Variable& input,
    const Variable& scale,
    const Variable& bias,
    float epsilon
) {
    if (input.value().rank() == 0) {
        throw std::invalid_argument(
            "layer normalization requires rank at least one"
        );
    }
    const auto width = input.value().shape().back();
    const Tensor::Shape parameter_shape{width};
    if (scale.value().shape() != parameter_shape ||
        bias.value().shape() != parameter_shape) {
        throw std::invalid_argument(
            "layer normalization parameters must match final input width"
        );
    }
    if (input.value().backend() != scale.value().backend() ||
        input.value().backend() != bias.value().backend()) {
        throw std::invalid_argument(
            "layer normalization tensors must use the same backend"
        );
    }
    epsilon = checked_epsilon(epsilon);

    const auto rows = input.value().numel() / width;
    Tensor output(input.value().shape(), input.value().backend());
    Tensor slice_mean({rows}, input.value().backend());
    Tensor inverse_standard_deviation({rows}, input.value().backend());
    backend_detail::dispatch_layer_norm_forward(
        input.value().backend(),
        {
            backend_detail::tensor_storage(input.value()),
            backend_detail::tensor_storage(scale.value()),
            backend_detail::tensor_storage(bias.value()),
            backend_detail::tensor_storage(output),
            backend_detail::tensor_storage(slice_mean),
            backend_detail::tensor_storage(inverse_standard_deviation),
            rows,
            width,
            epsilon,
        }
    );

    auto saved_mean = std::make_shared<Tensor>(std::move(slice_mean));
    auto saved_inverse_standard_deviation =
        std::make_shared<Tensor>(std::move(inverse_standard_deviation));
    const auto input_node = input.node_;
    const auto scale_node = scale.node_;
    const auto bias_node = bias.node_;
    return Variable::from_operation(
        std::move(output),
        {input_node, scale_node, bias_node},
        [input_node,
         scale_node,
         bias_node,
         input,
         scale,
         saved_mean,
         saved_inverse_standard_deviation,
         rows,
         width](const Tensor& upstream) {
            Tensor input_gradient(
                input.value().shape(),
                input.value().backend()
            );
            Tensor scale_gradient(
                scale.value().shape(),
                scale.value().backend()
            );
            Tensor bias_gradient(
                scale.value().shape(),
                scale.value().backend()
            );
            backend_detail::dispatch_layer_norm_backward(
                input.value().backend(),
                {
                    backend_detail::tensor_storage(input.value()),
                    backend_detail::tensor_storage(scale.value()),
                    backend_detail::tensor_storage(*saved_mean),
                    backend_detail::tensor_storage(
                        *saved_inverse_standard_deviation
                    ),
                    backend_detail::tensor_storage(upstream),
                    backend_detail::tensor_storage(input_gradient),
                    backend_detail::tensor_storage(scale_gradient),
                    backend_detail::tensor_storage(bias_gradient),
                    rows,
                    width,
                }
            );
            Variable::accumulate_gradient(input_node, input_gradient);
            Variable::accumulate_gradient(scale_node, scale_gradient);
            Variable::accumulate_gradient(bias_node, bias_gradient);
        }
    );
}

LayerNorm::LayerNorm(std::size_t width, float epsilon)
    : scale_(Tensor::full({width}, 1.0F)),
      bias_(Tensor::zeros({width})),
      epsilon_(checked_epsilon(epsilon)) {
    register_parameter("scale", scale_);
    register_parameter("bias", bias_);
}

LayerNorm::LayerNorm(Tensor scale, Tensor bias, float epsilon)
    : scale_(checked_layer_norm_scale(std::move(scale))),
      bias_(checked_layer_norm_bias(std::move(bias), scale_.value())),
      epsilon_(checked_epsilon(epsilon)) {
    register_parameter("scale", scale_);
    register_parameter("bias", bias_);
}

std::size_t LayerNorm::width() const noexcept {
    return scale_.value().shape()[0];
}

float LayerNorm::epsilon() const noexcept {
    return epsilon_;
}

Variable LayerNorm::forward(const Variable& input) const {
    return layer_norm(input, scale_.variable(), bias_.variable(), epsilon_);
}

void LayerNorm::to(ExecutionBackend backend) {
    Module::to(backend);
}

const Parameter& LayerNorm::scale() const noexcept {
    return scale_;
}

const Parameter& LayerNorm::bias() const noexcept {
    return bias_;
}

ParameterList LayerNorm::parameters() {
    return Module::parameters();
}

}  // namespace transformer_lab
