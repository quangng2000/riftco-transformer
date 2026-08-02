#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/core/quantized_weight.hpp"
#include "riftco_transformer/core/tensor.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"
#include "riftco_transformer/lowering/lowering.hpp"
#include "riftco_transformer/model/activation_checkpointing.hpp"
#include "riftco_transformer/model/lora.hpp"
#include "riftco_transformer/nn/module.hpp"
#include "riftco_transformer/nn/quantized_linear.hpp"
#include "riftco_transformer/stages/stages.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
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

    static_assert(
        static_cast<std::uint8_t>(ExecutionBackend::Cuda) == 2
    );
    static_assert(
        static_cast<std::uint8_t>(ExecutionBackend::Tpu) == 3
    );

    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Cpu
        ) ||
        !riftco_transformer::execution_backend_unavailability_reason(
            ExecutionBackend::Cpu
        ).empty() ||
        riftco_transformer::execution_backend_name(
            ExecutionBackend::Cuda
        ) != std::string_view("cuda") ||
        riftco_transformer::execution_backend_name(
            ExecutionBackend::Tpu
        ) != std::string_view("tpu")) {
        return EXIT_FAILURE;
    }
    riftco_transformer::set_execution_backend(
        ExecutionBackend::Cpu
    );
    if (riftco_transformer::execution_backend_available(
            ExecutionBackend::Cuda
        )) {
        {
            const riftco_transformer::ScopedExecutionBackend use_cuda(
                ExecutionBackend::Cuda
            );
            if (riftco_transformer::execution_backend() !=
                ExecutionBackend::Cuda) {
                return EXIT_FAILURE;
            }
        }
        if (riftco_transformer::execution_backend() !=
            ExecutionBackend::Cpu) {
            return EXIT_FAILURE;
        }
    } else {
        if (riftco_transformer::
                execution_backend_unavailability_reason(
                    ExecutionBackend::Cuda
                ).empty()) {
            return EXIT_FAILURE;
        }
        bool unavailable_threw = false;
        try {
            riftco_transformer::set_execution_backend(
                ExecutionBackend::Cuda
            );
        } catch (const std::runtime_error&) {
            unavailable_threw = true;
        } catch (...) {
            return EXIT_FAILURE;
        }
        if (!unavailable_threw ||
            riftco_transformer::execution_backend() !=
                ExecutionBackend::Cpu) {
            return EXIT_FAILURE;
        }
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
    const riftco_transformer::QuantizedWeight packed =
        riftco_transformer::QuantizedWeight::quantize_nf4(
            Tensor::zeros({32}, ExecutionBackend::Cpu),
            32
        );
    if (packed.packed_byte_count() != 16 ||
        packed.memory_usage().logical_payload_bytes != 20 ||
        packed.dequantize().numel() != 32) {
        return EXIT_FAILURE;
    }
    const riftco_transformer::QuantizedWeight double_packed =
        riftco_transformer::QuantizedWeight::
            quantize_nf4_double_quantized(
                Tensor::zeros({4096}, ExecutionBackend::Cpu),
                32,
                32
            );
    const auto double_memory = double_packed.memory_usage();
    if (!double_packed.uses_double_quantized_scales() ||
        double_memory.fp32_scale_bytes != 0 ||
        double_memory.scale_code_bytes != 128 ||
        double_memory.second_level_scale_bytes != 16 ||
        double_memory.scale_offset_bytes != 4) {
        return EXIT_FAILURE;
    }
    const riftco_transformer::QuantizedWeight packed_matrix =
        riftco_transformer::QuantizedWeight::quantize_nf4(
            Tensor::zeros({1, 32}, ExecutionBackend::Cpu),
            32
        );
    const riftco_transformer::Variable packed_output =
        riftco_transformer::quantized_linear(
            riftco_transformer::Variable(
                Tensor::zeros({1, 32}, ExecutionBackend::Cpu),
                false
            ),
            packed_matrix
        );
    if (packed_output.value().shape() != Tensor::Shape({1, 1}) ||
        packed_output.value().flat(0) != 0.0F) {
        return EXIT_FAILURE;
    }

    const auto lowering_analysis =
        riftco_transformer::lowering::analyze_neural_lowering(
            riftco_transformer::compiler::cajal::MultilinearMap::identity(2)
        );
    if (!lowering_analysis.supported ||
        lowering_analysis.selected_strategy !=
            riftco_transformer::lowering::kLinearStrategy ||
        !lowering_analysis.exact) {
        return EXIT_FAILURE;
    }

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
