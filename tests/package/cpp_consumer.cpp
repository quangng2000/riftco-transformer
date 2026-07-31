#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/core/tensor.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"
#include "riftco_transformer/model/activation_checkpointing.hpp"
#include "riftco_transformer/model/lora.hpp"
#include "riftco_transformer/nn/module.hpp"
#include "riftco_transformer/stages/stages.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

class ConsumerModule final : public riftco_transformer::Module {
public:
    ConsumerModule()
        : weight_(riftco_transformer::Tensor({2}, {1.0F, 2.0F})) {
        register_parameter("weight", weight_);
    }

private:
    riftco_transformer::Parameter weight_;
};

riftco_transformer::ParameterList retained_parameters() {
    ConsumerModule module;
    return module.parameters();
}

}  // namespace

int main() {
    using riftco_transformer::ExecutionBackend;
    using riftco_transformer::Tensor;

    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Cpu
        )) {
        return EXIT_FAILURE;
    }

    const riftco_transformer::LoraConfig lora;
    if (lora.rank != 4 ||
        lora.alpha != 8.0F ||
        lora.targets != riftco_transformer::kLoraDefaultTargets) {
        return EXIT_FAILURE;
    }
    if (riftco_transformer::ActivationCheckpointingKind::Disabled ==
        riftco_transformer::ActivationCheckpointingKind::TransformerBlock) {
        return EXIT_FAILURE;
    }

    const Tensor left({1, 2}, {2.0F, 3.0F});
    const Tensor right({2, 1}, {4.0F, 5.0F});
    const Tensor product =
        riftco_transformer::tensor_ops::matmul(
            left,
            right,
            ExecutionBackend::Cpu
        );

    const riftco_transformer::ParameterList retained =
        retained_parameters();
    if (retained.size() != 1 ||
        retained.front().name != "weight" ||
        retained.front().parameter == nullptr ||
        retained.front().parameter->value().flat(1) != 2.0F) {
        return EXIT_FAILURE;
    }

    const riftco_transformer::Variable input(
        Tensor({2}, {2.0F, 3.0F})
    );
    const Tensor saved_input = input.value();
    const std::array custom_inputs{input};
    const riftco_transformer::Variable squared =
        riftco_transformer::custom_gradient(
            riftco_transformer::tensor_ops::multiply(
                saved_input,
                saved_input
            ),
            custom_inputs,
            [saved_input](const Tensor& upstream) {
                return std::vector<Tensor>{
                    riftco_transformer::tensor_ops::scale(
                        riftco_transformer::tensor_ops::multiply(
                            upstream,
                            saved_input
                        ),
                        2.0F
                    ),
                };
            }
        );
    riftco_transformer::sum(squared).backward();
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
