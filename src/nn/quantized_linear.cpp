#include "riftco_transformer/nn/quantized_linear.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/quantized_linear/dispatch.hpp"

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

namespace riftco_transformer {

Variable quantized_linear(
    const Variable& input,
    const QuantizedWeight& weight
) {
    if (weight.rank() != 2 ||
        weight.shape()[0] == 0 ||
        weight.shape()[1] == 0) {
        throw std::invalid_argument(
            "quantized linear weight must have shape [output,input]"
        );
    }
    const std::size_t input_width = weight.shape()[1];
    const std::size_t output_width = weight.shape()[0];
    if (input.value().rank() == 0 ||
        input.value().shape().back() != input_width) {
        throw std::invalid_argument(
            "quantized linear input final dimension must equal input width"
        );
    }
    if (input.value().backend() != weight.backend()) {
        throw std::invalid_argument(
            "quantized linear input and weight must use the same backend"
        );
    }

    const std::size_t rows = input.value().numel() / input_width;
    Tensor::Shape output_shape = input.value().shape();
    output_shape.back() = output_width;
    Tensor output(output_shape, input.value().backend());
    const backend_detail::QuantizedLinearDimensions dimensions{
        rows,
        input_width,
        output_width,
        weight.block_size(),
    };
    backend_detail::dispatch_quantized_linear_forward(
        input.value().backend(),
        {
            backend_detail::tensor_storage(input.value()),
            backend_detail::quantized_weight_storage(weight),
            backend_detail::tensor_storage(output),
            dimensions,
        }
    );

    const std::array dependencies{input};
    return custom_gradient(
        std::move(output),
        dependencies,
        [
            weight,
            input_shape = input.value().shape(),
            backend = input.value().backend(),
            dimensions
        ](const Tensor& upstream) {
            Tensor input_gradient(input_shape, backend);
            backend_detail::dispatch_quantized_linear_input_backward(
                backend,
                {
                    backend_detail::tensor_storage(upstream),
                    backend_detail::quantized_weight_storage(weight),
                    backend_detail::tensor_storage(input_gradient),
                    dimensions,
                }
            );
            return std::vector<Tensor>{std::move(input_gradient)};
        }
    );
}

}  // namespace riftco_transformer
