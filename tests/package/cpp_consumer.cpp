#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/core/autograd.hpp"
#include "transformer_lab/core/tensor.hpp"
#include "transformer_lab/core/tensor_ops.hpp"
#include "transformer_lab/model/activation_checkpointing.hpp"
#include "transformer_lab/model/lora.hpp"
#include "transformer_lab/nn/module.hpp"
#include "transformer_lab/stages/stages.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

class ConsumerModule final : public transformer_lab::Module {
public:
    ConsumerModule()
        : weight_(transformer_lab::Tensor({2}, {1.0F, 2.0F})) {
        register_parameter("weight", weight_);
    }

private:
    transformer_lab::Parameter weight_;
};

transformer_lab::ParameterList retained_parameters() {
    ConsumerModule module;
    return module.parameters();
}

}  // namespace

int main() {
    using transformer_lab::ExecutionBackend;
    using transformer_lab::Tensor;

    if (!transformer_lab::execution_backend_available(
            ExecutionBackend::Cpu
        )) {
        return EXIT_FAILURE;
    }

    const transformer_lab::LoraConfig lora;
    if (lora.rank != 4 ||
        lora.alpha != 8.0F ||
        lora.targets != transformer_lab::kLoraDefaultTargets) {
        return EXIT_FAILURE;
    }
    if (transformer_lab::ActivationCheckpointingKind::Disabled ==
        transformer_lab::ActivationCheckpointingKind::TransformerBlock) {
        return EXIT_FAILURE;
    }

    const Tensor left({1, 2}, {2.0F, 3.0F});
    const Tensor right({2, 1}, {4.0F, 5.0F});
    const Tensor product =
        transformer_lab::tensor_ops::matmul(
            left,
            right,
            ExecutionBackend::Cpu
        );

    const transformer_lab::ParameterList retained =
        retained_parameters();
    if (retained.size() != 1 ||
        retained.front().name != "weight" ||
        retained.front().parameter == nullptr ||
        retained.front().parameter->value().flat(1) != 2.0F) {
        return EXIT_FAILURE;
    }

    const transformer_lab::Variable input(
        Tensor({2}, {2.0F, 3.0F})
    );
    const Tensor saved_input = input.value();
    const std::array custom_inputs{input};
    const transformer_lab::Variable squared =
        transformer_lab::custom_gradient(
            transformer_lab::tensor_ops::multiply(
                saved_input,
                saved_input
            ),
            custom_inputs,
            [saved_input](const Tensor& upstream) {
                return std::vector<Tensor>{
                    transformer_lab::tensor_ops::scale(
                        transformer_lab::tensor_ops::multiply(
                            upstream,
                            saved_input
                        ),
                        2.0F
                    ),
                };
            }
        );
    transformer_lab::sum(squared).backward();
    if (input.gradient().shape() != Tensor::Shape({2}) ||
        std::fabs(input.gradient().flat(0) - 4.0F) >= 1.0e-6F ||
        std::fabs(input.gradient().flat(1) - 6.0F) >= 1.0e-6F) {
        return EXIT_FAILURE;
    }

    return product.shape() == Tensor::Shape({1, 1}) &&
                   std::fabs(product.flat(0) - 23.0F) < 1.0e-6F
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
