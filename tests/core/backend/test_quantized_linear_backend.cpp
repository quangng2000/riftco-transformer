#include "core/backend/nn/quantized_linear/dispatch.hpp"

#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/quantized_weight.hpp"
#include "riftco_transformer/core/tensor.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using riftco_transformer::ExecutionBackend;
using riftco_transformer::QuantizedWeight;
using riftco_transformer::Tensor;
namespace backend = riftco_transformer::backend_detail;
namespace tensor_ops = riftco_transformer::tensor_ops;

#ifndef RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL
#define RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL 0
#endif
#ifndef RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA
#define RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA 0
#endif
#ifndef RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU
#define RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU 0
#endif

constexpr std::size_t kEdgeWidth = 33;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_tensor_close(
    const Tensor& actual,
    const Tensor& expected,
    const std::string& message,
    float tolerance = 3.0e-5F
) {
    require(actual.shape() == expected.shape(), message + ": shape mismatch");
    for (std::size_t index = 0; index < expected.numel(); ++index) {
        const float scale = std::max(
            {1.0F,
             std::fabs(actual.flat(index)),
             std::fabs(expected.flat(index))}
        );
        if (!std::isfinite(actual.flat(index)) ||
            !std::isfinite(expected.flat(index)) ||
            std::fabs(actual.flat(index) - expected.flat(index)) >
                tolerance * scale) {
            throw std::runtime_error(
                message + " at flat index " + std::to_string(index) +
                ": expected " + std::to_string(expected.flat(index)) +
                ", got " + std::to_string(actual.flat(index))
            );
        }
    }
}

template <typename Function>
void require_unavailable(Function&& function, const std::string& message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("unavailable") !=
            std::string::npos;
        if (!rejected) {
            throw std::runtime_error(
                message + ": unexpected error: " + error.what()
            );
        }
    }
    require(rejected, message);
}

[[nodiscard]] Tensor make_weight() {
    std::vector<float> values(3 * 32, 0.0F);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int centered = static_cast<int>(index % 19) - 9;
        values[index] = static_cast<float>(centered) * 0.1375F;
    }
    values[0] = -2.0F;
    values[31] = 1.75F;
    values[32] = 3.0F;
    values.back() = -4.0F;
    return Tensor({3, 32}, std::move(values), ExecutionBackend::Cpu);
}

[[nodiscard]] Tensor make_input() {
    std::vector<float> values(2 * 32, 0.0F);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int centered = static_cast<int>(index % 11) - 5;
        values[index] = static_cast<float>(centered) * 0.2F;
    }
    return Tensor({2, 32}, std::move(values), ExecutionBackend::Cpu);
}

[[nodiscard]] Tensor make_upstream() {
    return Tensor(
        {2, 3},
        {
            0.25F,
            -0.75F,
            1.5F,
            -1.0F,
            0.5F,
            0.125F,
        },
        ExecutionBackend::Cpu
    );
}

[[nodiscard]] Tensor make_edge_weight() {
    std::vector<float> values(kEdgeWidth * kEdgeWidth, 0.0F);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int centered = static_cast<int>(index % 37) - 18;
        values[index] = static_cast<float>(centered) * 0.0625F;
    }
    values[0] = -3.0F;
    values[31] = 2.5F;
    values[32] = -2.75F;
    values[1023] = 4.0F;
    values[1024] = -4.5F;
    values.back() = 5.0F;
    return Tensor(
        {kEdgeWidth, kEdgeWidth},
        std::move(values),
        ExecutionBackend::Cpu
    );
}

[[nodiscard]] Tensor make_edge_input() {
    std::vector<float> values(2 * kEdgeWidth, 0.0F);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int centered = static_cast<int>(index % 17) - 8;
        values[index] = static_cast<float>(centered) * 0.125F;
    }
    return Tensor(
        {2, kEdgeWidth},
        std::move(values),
        ExecutionBackend::Cpu
    );
}

[[nodiscard]] Tensor make_edge_upstream() {
    std::vector<float> values(2 * kEdgeWidth, 0.0F);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int centered = static_cast<int>(index % 13) - 6;
        values[index] = static_cast<float>(centered) * 0.075F;
    }
    return Tensor(
        {2, kEdgeWidth},
        std::move(values),
        ExecutionBackend::Cpu
    );
}

[[nodiscard]] Tensor quantized_forward(
    const Tensor& input,
    const QuantizedWeight& weight
) {
    const std::size_t rows = input.numel() / weight.shape()[1];
    Tensor output({rows, weight.shape()[0]}, input.backend());
    backend::dispatch_quantized_linear_forward(
        input.backend(),
        {
            backend::tensor_storage(input),
            backend::quantized_weight_storage(weight),
            backend::tensor_storage(output),
            {
                rows,
                weight.shape()[1],
                weight.shape()[0],
                weight.block_size(),
            },
        }
    );
    return output;
}

[[nodiscard]] Tensor quantized_input_backward(
    const Tensor& upstream,
    const QuantizedWeight& weight
) {
    const std::size_t rows = upstream.numel() / weight.shape()[0];
    Tensor input_gradient({rows, weight.shape()[1]}, upstream.backend());
    backend::dispatch_quantized_linear_input_backward(
        upstream.backend(),
        {
            backend::tensor_storage(upstream),
            backend::quantized_weight_storage(weight),
            backend::tensor_storage(input_gradient),
            {
                rows,
                weight.shape()[1],
                weight.shape()[0],
                weight.block_size(),
            },
        }
    );
    return input_gradient;
}

struct CpuFixture {
    Tensor input;
    Tensor upstream;
    QuantizedWeight weight;
    Tensor forward_oracle;
    Tensor backward_oracle;
};

[[nodiscard]] CpuFixture make_cpu_fixture_from(
    Tensor input,
    Tensor upstream,
    const Tensor& dense_weight,
    bool double_quantized
) {
    QuantizedWeight weight = double_quantized
        ? QuantizedWeight::quantize_nf4_double_quantized(
              dense_weight, 32, 32)
        : QuantizedWeight::quantize_nf4(dense_weight, 32);
    const Tensor dequantized = weight.dequantize();
    Tensor forward_oracle = tensor_ops::matmul(
        input,
        tensor_ops::transpose_2d(dequantized)
    );
    Tensor backward_oracle = tensor_ops::matmul(upstream, dequantized);
    return {
        std::move(input),
        std::move(upstream),
        std::move(weight),
        std::move(forward_oracle),
        std::move(backward_oracle),
    };
}

[[nodiscard]] CpuFixture make_cpu_fixture(bool double_quantized = false) {
    const Tensor dense_weight = make_weight();
    return make_cpu_fixture_from(
        make_input(),
        make_upstream(),
        dense_weight,
        double_quantized
    );
}

[[nodiscard]] CpuFixture make_edge_cpu_fixture(
    bool double_quantized = false
) {
    const Tensor dense_weight = make_edge_weight();
    return make_cpu_fixture_from(
        make_edge_input(),
        make_edge_upstream(),
        dense_weight,
        double_quantized
    );
}

void require_edge_layout(
    const QuantizedWeight& weight,
    const std::string& description
) {
    require(
        weight.numel() == kEdgeWidth * kEdgeWidth &&
            weight.numel() % 2 == 1,
        description + " must contain an odd number of NF4 values"
    );
    require(
        weight.block_size() == 32 && weight.block_count() == 35 &&
            weight.numel() % weight.block_size() == 1,
        description + " must end with a partial NF4 block"
    );

    const auto payload = weight.copy_payload_to_host();
    require(
        payload.packed_codes.size() == weight.packed_byte_count() &&
            (payload.packed_codes.back() & 0xF0U) == 0,
        description + " must retain a canonical odd packed-code tail"
    );
    if (!weight.uses_double_quantized_scales()) {
        return;
    }
    require(
        payload.double_quantized_scales.has_value() &&
            payload.double_quantized_scales->scale_block_size == 32 &&
            payload.double_quantized_scales->second_level_scales.size() == 2,
        description + " must cross two second-level scale groups"
    );
}

void test_cpu_double_quantized_forward_and_input_backward() {
    const CpuFixture fixture = make_cpu_fixture(true);
    require(
        fixture.weight.uses_double_quantized_scales(),
        "CPU fixture should retain nested scale quantization"
    );

    require_tensor_close(
        quantized_forward(fixture.input, fixture.weight),
        fixture.forward_oracle,
        "CPU double-quantized NF4 forward must match its decoded oracle"
    );
    require_tensor_close(
        quantized_input_backward(fixture.upstream, fixture.weight),
        fixture.backward_oracle,
        "CPU double-quantized NF4 backward must match its decoded oracle"
    );

    const auto usage = fixture.weight.memory_usage();
    require(
        usage.fp32_scale_bytes == 0 && usage.scale_code_bytes == 3 &&
            usage.second_level_scale_bytes == sizeof(float) &&
            usage.scale_offset_bytes == sizeof(float) &&
            usage.resident_payload_bytes == usage.logical_payload_bytes,
        "CPU double quantization must not retain first-level FP32 scales"
    );
}

void test_cpu_reference_forward_and_input_backward() {
    const CpuFixture fixture = make_cpu_fixture();
    const auto memory_before = fixture.weight.memory_usage();

    const Tensor output = quantized_forward(fixture.input, fixture.weight);
    const Tensor input_gradient =
        quantized_input_backward(fixture.upstream, fixture.weight);
    require_tensor_close(
        output,
        fixture.forward_oracle,
        "CPU NF4 forward must match a dequantized oracle"
    );
    require_tensor_close(
        input_gradient,
        fixture.backward_oracle,
        "CPU NF4 input backward must match a dequantized oracle"
    );

    const auto memory_after = fixture.weight.memory_usage();
    require(
        memory_before.resident_payload_bytes ==
            memory_after.resident_payload_bytes &&
            memory_after.resident_payload_bytes ==
                memory_after.logical_payload_bytes &&
            memory_after.resident_payload_bytes <
                memory_after.fp32_equivalent_bytes,
        "CPU execution must leave the frozen base weight in packed storage"
    );
}

void test_cpu_edge_layout_forward_and_input_backward() {
    for (const bool double_quantized : {false, true}) {
        const CpuFixture fixture = make_edge_cpu_fixture(double_quantized);
        const std::string description = double_quantized
            ? "CPU double-quantized edge fixture"
            : "CPU legacy edge fixture";
        require_edge_layout(fixture.weight, description);
        require_tensor_close(
            quantized_forward(fixture.input, fixture.weight),
            fixture.forward_oracle,
            description + " forward must match its decoded oracle"
        );
        require_tensor_close(
            quantized_input_backward(fixture.upstream, fixture.weight),
            fixture.backward_oracle,
            description + " input backward must match its decoded oracle"
        );
    }
}

void run_accelerator_oracle(
    ExecutionBackend execution_backend,
    const CpuFixture& fixture,
    bool edge_layout
) {
    const std::string backend_name(
        riftco_transformer::execution_backend_name(execution_backend)
    );
    const bool double_quantized =
        fixture.weight.uses_double_quantized_scales();
    const std::string scale_description = double_quantized
        ? " double-quantized NF4"
        : " NF4";
    const std::string fixture_description = edge_layout
        ? " edge fixture"
        : " aligned fixture";
    const QuantizedWeight accelerator_weight =
        fixture.weight.to(execution_backend);
    const Tensor accelerator_input = fixture.input.to(execution_backend);
    const Tensor accelerator_upstream = fixture.upstream.to(execution_backend);

    const Tensor output = quantized_forward(
        accelerator_input,
        accelerator_weight
    );
    const Tensor input_gradient = quantized_input_backward(
        accelerator_upstream,
        accelerator_weight
    );
    require_tensor_close(
        output.to(ExecutionBackend::Cpu),
        fixture.forward_oracle,
        backend_name + scale_description + fixture_description +
            " forward must match the CPU oracle"
    );
    require_tensor_close(
        input_gradient.to(ExecutionBackend::Cpu),
        fixture.backward_oracle,
        backend_name + scale_description + fixture_description +
            " input backward must match the CPU oracle"
    );

    const auto usage = accelerator_weight.memory_usage();
    require(
        usage.resident_payload_bytes == usage.logical_payload_bytes &&
            usage.resident_payload_bytes < usage.fp32_equivalent_bytes,
        backend_name + scale_description + fixture_description +
            " storage must remain smaller than its FP32 equivalent"
    );
    if (edge_layout) {
        require_edge_layout(
            accelerator_weight,
            backend_name + scale_description + fixture_description
        );
    }
    if (double_quantized) {
        require(
            usage.fp32_scale_bytes == 0 && usage.scale_code_bytes != 0 &&
                usage.second_level_scale_bytes != 0 &&
                usage.scale_offset_bytes == sizeof(float),
            backend_name +
                " double quantization must retain encoded first-level scales"
        );
    } else {
        require(
            usage.fp32_scale_bytes != 0 && usage.scale_code_bytes == 0 &&
                usage.second_level_scale_bytes == 0 &&
                usage.scale_offset_bytes == 0,
            backend_name + " legacy NF4 must retain only FP32 block scales"
        );
    }
}

[[nodiscard]] bool backend_is_required(
    ExecutionBackend execution_backend,
    bool require_tpu
) noexcept {
    switch (execution_backend) {
    case ExecutionBackend::Metal:
        return RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL != 0;
    case ExecutionBackend::Cuda:
        return RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA != 0;
    case ExecutionBackend::Tpu:
        return RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU != 0 || require_tpu;
    case ExecutionBackend::Cpu:
        return false;
    }
    return false;
}

void test_accelerator_when_available(
    ExecutionBackend execution_backend,
    bool require_tpu
) {
    const bool available =
        riftco_transformer::execution_backend_available(execution_backend);
    const std::string backend_name(
        riftco_transformer::execution_backend_name(execution_backend)
    );
    if (!available) {
        require(
            !backend_is_required(execution_backend, require_tpu),
            backend_name +
                " quantized-linear tests require an available backend"
        );
        for (const bool double_quantized : {false, true}) {
            const CpuFixture fixture = make_cpu_fixture(double_quantized);
            require_unavailable(
                [&] {
                    static_cast<void>(
                        fixture.weight.to(execution_backend)
                    );
                },
                backend_name +
                    " unavailable backend must reject packed weight transfer"
            );
        }
        return;
    }

    for (const bool double_quantized : {false, true}) {
        run_accelerator_oracle(
            execution_backend,
            make_cpu_fixture(double_quantized),
            false
        );
        run_accelerator_oracle(
            execution_backend,
            make_edge_cpu_fixture(double_quantized),
            true
        );
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool require_tpu = false;
        if (argc == 2 &&
            std::string_view(argv[1]) == "--require-tpu") {
            require_tpu = true;
        } else if (argc != 1) {
            throw std::invalid_argument(
                "usage: quantized_linear_backend_tests [--require-tpu]"
            );
        }
        test_cpu_reference_forward_and_input_backward();
        test_cpu_double_quantized_forward_and_input_backward();
        test_cpu_edge_layout_forward_and_input_backward();
        constexpr std::array<ExecutionBackend, 3> accelerators{
            ExecutionBackend::Metal,
            ExecutionBackend::Cuda,
            ExecutionBackend::Tpu,
        };
        for (const ExecutionBackend execution_backend : accelerators) {
            test_accelerator_when_available(
                execution_backend,
                require_tpu
            );
        }
        std::cout << "quantized-linear backend tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr <<
            "quantized-linear backend test failure: " << error.what() << '\n';
        return 1;
    }
}
