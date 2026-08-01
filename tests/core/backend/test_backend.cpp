#include "core/backend/adapters/tpu/compile_options.hpp"
#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using riftco_transformer::ExecutionBackend;
using riftco_transformer::Tensor;
using riftco_transformer::Variable;
namespace tensor_ops = riftco_transformer::tensor_ops;

#ifndef RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA
#define RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA 0
#endif
#ifndef RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU
#define RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU 0
#endif

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_pinned_tpu_compile_options() {
    constexpr std::array<std::uint8_t, 27> expected{
        0x1A, 0x19,
        0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0x01,
        0x20, 0x01,
        0x28, 0x01,
        0x62, 0x01, 0x00,
        0x92, 0x01, 0x01, 0x00,
        0xB8, 0x01, 0x01,
    };
    require(
        riftco_transformer::backend_detail::
            single_device_compile_options_proto == expected,
        "TPU compile options must match pinned OpenXLA defaults"
    );
}

void require_close(
    float actual,
    float expected,
    const std::string& message,
    float absolute_tolerance = 1.0e-6F,
    float relative_tolerance = 1.0e-6F
) {
    const float scale =
        std::max(std::fabs(actual), std::fabs(expected));
    const float tolerance =
        absolute_tolerance + relative_tolerance * scale;
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void require_tensor_close(
    const Tensor& actual,
    const Tensor& expected,
    const std::string& message,
    float absolute_tolerance = 1.0e-6F,
    float relative_tolerance = 1.0e-6F
) {
    require(
        actual.shape() == expected.shape(),
        message + ": shape mismatch"
    );
    require(
        actual.numel() == expected.numel(),
        message + ": value count mismatch"
    );
    for (std::size_t index = 0; index < expected.numel(); ++index) {
        require_close(
            actual.flat(index),
            expected.flat(index),
            message + " at flat index " + std::to_string(index),
            absolute_tolerance,
            relative_tolerance
        );
    }
}

template <typename Exception, typename Function>
void require_throws_as(
    Function&& function,
    const std::string& message
) {
    bool threw = false;
    try {
        function();
    } catch (const Exception&) {
        threw = true;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            message + ": wrong exception type: " + error.what()
        );
    } catch (...) {
        throw std::runtime_error(
            message + ": wrong non-standard exception type"
        );
    }
    require(threw, message);
}

Tensor rank_two_left() {
    return Tensor(
        {2, 3},
        {
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
        }
    );
}

Tensor rank_two_right() {
    return Tensor(
        {3, 4},
        {
            1.0F, 2.0F, 3.0F, 4.0F,
            5.0F, 6.0F, 7.0F, 8.0F,
            9.0F, 10.0F, 11.0F, 12.0F,
        }
    );
}

Tensor batched_left() {
    return Tensor(
        {2, 2, 3},
        {
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
            -1.0F, 2.0F, 0.0F,
            3.0F, 1.0F, 2.0F,
        }
    );
}

Tensor batched_right() {
    return Tensor(
        {2, 3, 2},
        {
            1.0F, 0.0F,
            0.0F, 1.0F,
            1.0F, 1.0F,
            2.0F, 1.0F,
            1.0F, -1.0F,
            0.0F, 2.0F,
        }
    );
}

void test_cpu_metadata_and_default_selection(
    ExecutionBackend initial_backend
) {
    require(
        initial_backend == ExecutionBackend::Cpu,
        "CPU should be the default execution backend"
    );
    require(
        riftco_transformer::execution_backend_available(
            ExecutionBackend::Cpu
        ),
        "CPU execution backend should always be available"
    );
    require(
        riftco_transformer::execution_backend_unavailability_reason(
            ExecutionBackend::Cpu
        ).empty(),
        "an available CPU backend should not report an unavailable reason"
    );
    require(
        riftco_transformer::execution_backend_name(
            ExecutionBackend::Cpu
        ) == std::string_view("cpu"),
        "CPU execution backend name"
    );
    require(
        riftco_transformer::execution_backend_name(
            ExecutionBackend::Metal
        ) == std::string_view("metal"),
        "Metal execution backend name"
    );
    require(
        static_cast<std::uint8_t>(ExecutionBackend::Cuda) == 2,
        "CUDA execution backend numeric identity"
    );
    require(
        riftco_transformer::execution_backend_name(
            ExecutionBackend::Cuda
        ) == std::string_view("cuda"),
        "CUDA execution backend name"
    );
    require(
        static_cast<std::uint8_t>(ExecutionBackend::Tpu) == 3,
        "TPU execution backend numeric identity"
    );
    require(
        riftco_transformer::execution_backend_name(
            ExecutionBackend::Tpu
        ) == std::string_view("tpu"),
        "TPU execution backend name"
    );
    require(
        riftco_transformer::execution_backend() ==
            ExecutionBackend::Cpu,
        "active backend should initially be CPU"
    );

    const Tensor left = rank_two_left();
    const Tensor right = rank_two_right();
    require_tensor_close(
        tensor_ops::matmul(left, right),
        tensor_ops::matmul(
            left,
            right,
            ExecutionBackend::Cpu
        ),
        "default matmul should dispatch to CPU"
    );
}

void test_explicit_cpu_matmul() {
    require_tensor_close(
        tensor_ops::matmul(
            rank_two_left(),
            rank_two_right(),
            ExecutionBackend::Cpu
        ),
        Tensor(
            {2, 4},
            {
                38.0F, 44.0F, 50.0F, 56.0F,
                83.0F, 98.0F, 113.0F, 128.0F,
            }
        ),
        "explicit CPU non-square matmul"
    );

    require_tensor_close(
        tensor_ops::matmul(
            batched_left(),
            batched_right(),
            ExecutionBackend::Cpu
        ),
        Tensor(
            {2, 2, 2},
            {
                4.0F, 5.0F,
                10.0F, 11.0F,
                0.0F, -3.0F,
                7.0F, 6.0F,
            }
        ),
        "explicit CPU batched matmul"
    );
}

void test_unknown_backend_is_rejected_transactionally() {
    constexpr auto unknown =
        static_cast<ExecutionBackend>(0xFF);
    riftco_transformer::set_execution_backend(
        ExecutionBackend::Cpu
    );

    require(
        !riftco_transformer::execution_backend_available(unknown),
        "an unknown backend must not report available"
    );
    require(
        riftco_transformer::execution_backend_unavailability_reason(
            unknown
        ).empty(),
        "an unknown backend should not imitate a recognized unavailable one"
    );
    require_throws_as<std::invalid_argument>(
        [] {
            static_cast<void>(
                riftco_transformer::execution_backend_name(unknown)
            );
        },
        "naming an unknown backend should throw invalid_argument"
    );
    require_throws_as<std::invalid_argument>(
        [] {
            riftco_transformer::set_execution_backend(unknown);
        },
        "selecting an unknown backend should throw invalid_argument"
    );
    require(
        riftco_transformer::execution_backend() ==
            ExecutionBackend::Cpu,
        "failed unknown selection should leave CPU active"
    );
    require_throws_as<std::invalid_argument>(
        [] {
            static_cast<void>(
                tensor_ops::matmul(
                    rank_two_left(),
                    rank_two_right(),
                    unknown
                )
            );
        },
        "dispatching an unknown backend should throw invalid_argument"
    );
    require(
        riftco_transformer::execution_backend() ==
            ExecutionBackend::Cpu,
        "failed explicit dispatch should not change selection"
    );
}

void test_unavailable_backends_have_diagnostics() {
    constexpr std::array optional_backends{
        ExecutionBackend::Metal,
        ExecutionBackend::Cuda,
        ExecutionBackend::Tpu,
    };
    for (const ExecutionBackend backend : optional_backends) {
        const bool available =
            riftco_transformer::execution_backend_available(backend);
        const std::string_view reason =
            riftco_transformer::execution_backend_unavailability_reason(
                backend
            );
        require(
            available ? reason.empty() : !reason.empty(),
            std::string(
                riftco_transformer::execution_backend_name(backend)
            ) + " backend availability diagnostic"
        );
    }
}

void test_selection_is_thread_local_if_available(
    ExecutionBackend accelerator,
    std::string_view name
) {
    if (!riftco_transformer::execution_backend_available(
            accelerator
        )) {
        return;
    }

    riftco_transformer::set_execution_backend(accelerator);
    std::exception_ptr worker_failure;
    std::thread worker([&worker_failure] {
        try {
            require(
                riftco_transformer::execution_backend() ==
                    ExecutionBackend::Cpu,
                "a new thread should begin with the CPU default"
            );
            riftco_transformer::set_execution_backend(
                ExecutionBackend::Cpu
            );
        } catch (...) {
            worker_failure = std::current_exception();
        }
    });
    worker.join();
    if (worker_failure) {
        std::rethrow_exception(worker_failure);
    }

    require(
        riftco_transformer::execution_backend() ==
            accelerator,
        "worker selection must not change the caller's " +
            std::string(name) + " backend"
    );
    riftco_transformer::set_execution_backend(
        ExecutionBackend::Cpu
    );
}

void test_scoped_selection_restores_previous_backend() {
    riftco_transformer::set_execution_backend(
        ExecutionBackend::Cpu
    );
    {
        const riftco_transformer::ScopedExecutionBackend scope(
            ExecutionBackend::Cpu
        );
        require(
            riftco_transformer::execution_backend() ==
                ExecutionBackend::Cpu,
            "scoped CPU selection"
        );
    }
    require(
        riftco_transformer::execution_backend() ==
            ExecutionBackend::Cpu,
        "scoped selection should restore CPU"
    );

    if (riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        {
            const riftco_transformer::ScopedExecutionBackend scope(
                ExecutionBackend::Metal
            );
            require(
                riftco_transformer::execution_backend() ==
                    ExecutionBackend::Metal,
                "scoped Metal selection"
            );
        }
        require(
            riftco_transformer::execution_backend() ==
                ExecutionBackend::Cpu,
            "scoped Metal selection should restore CPU"
        );
    }

    if (riftco_transformer::execution_backend_available(
            ExecutionBackend::Cuda
        )) {
        {
            const riftco_transformer::ScopedExecutionBackend scope(
                ExecutionBackend::Cuda
            );
            require(
                riftco_transformer::execution_backend() ==
                    ExecutionBackend::Cuda,
                "scoped CUDA selection"
            );
        }
        require(
            riftco_transformer::execution_backend() ==
                ExecutionBackend::Cpu,
            "scoped CUDA selection should restore CPU"
        );
    }

    if (riftco_transformer::execution_backend_available(
            ExecutionBackend::Tpu
        )) {
        {
            const riftco_transformer::ScopedExecutionBackend scope(
                ExecutionBackend::Tpu
            );
            require(
                riftco_transformer::execution_backend() ==
                    ExecutionBackend::Tpu,
                "scoped TPU selection"
            );
        }
        require(
            riftco_transformer::execution_backend() ==
                ExecutionBackend::Cpu,
            "scoped TPU selection should restore CPU"
        );
    }
}

struct AutogradSnapshot {
    Tensor output;
    Tensor left_gradient;
    Tensor right_gradient;
};

AutogradSnapshot run_batched_autograd(
    ExecutionBackend backend,
    bool switch_to_cpu_before_backward
) {
    riftco_transformer::set_execution_backend(backend);

    const Variable left(batched_left());
    const Variable right(batched_right());
    const Variable output = riftco_transformer::matmul(left, right);
    const Tensor seed(
        {2, 2, 2},
        {
            0.5F, -1.0F,
            1.5F, 0.25F,
            -0.75F, 0.6F,
            0.4F, -1.2F,
        }
    );

    if (switch_to_cpu_before_backward) {
        riftco_transformer::set_execution_backend(
            ExecutionBackend::Cpu
        );
    }
    output.backward(seed);

    return {
        output.value(),
        left.gradient(),
        right.gradient(),
    };
}

bool test_metal_parity_if_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        riftco_transformer::set_execution_backend(
            ExecutionBackend::Cpu
        );
        require_throws_as<std::runtime_error>(
            [] {
                riftco_transformer::set_execution_backend(
                    ExecutionBackend::Metal
                );
            },
            "selecting an unavailable Metal backend should throw"
        );
        require(
            riftco_transformer::execution_backend() ==
                ExecutionBackend::Cpu,
            "failed Metal selection should leave CPU active"
        );
        require_throws_as<std::runtime_error>(
            [] {
                static_cast<void>(
                    tensor_ops::matmul(
                        rank_two_left(),
                        rank_two_right(),
                        ExecutionBackend::Metal
                    )
                );
            },
            "dispatching unavailable Metal should throw runtime_error"
        );
        require(
            riftco_transformer::execution_backend() ==
                ExecutionBackend::Cpu,
            "failed explicit Metal dispatch should leave CPU active"
        );
        return false;
    }

    constexpr float metal_absolute_tolerance = 2.0e-4F;
    constexpr float metal_relative_tolerance = 2.0e-4F;

    const Tensor rank_two_cpu = tensor_ops::matmul(
        rank_two_left(),
        rank_two_right(),
        ExecutionBackend::Cpu
    );
    const Tensor rank_two_metal = tensor_ops::matmul(
        rank_two_left(),
        rank_two_right(),
        ExecutionBackend::Metal
    );
    require(
        rank_two_metal.backend() == ExecutionBackend::Cpu,
        "execution override must preserve CPU input storage"
    );
    require_tensor_close(
        rank_two_metal,
        rank_two_cpu,
        "Metal rank-two matmul parity",
        metal_absolute_tolerance,
        metal_relative_tolerance
    );

    const Tensor batched_cpu = tensor_ops::matmul(
        batched_left(),
        batched_right(),
        ExecutionBackend::Cpu
    );
    const Tensor batched_metal = tensor_ops::matmul(
        batched_left(),
        batched_right(),
        ExecutionBackend::Metal
    );
    require_tensor_close(
        batched_metal,
        batched_cpu,
        "Metal batched matmul parity",
        metal_absolute_tolerance,
        metal_relative_tolerance
    );

    riftco_transformer::set_execution_backend(
        ExecutionBackend::Metal
    );
    const Tensor cpu_executed_on_metal_storage =
        tensor_ops::matmul(
            rank_two_left(),
            rank_two_right(),
            ExecutionBackend::Cpu
        );
    require(
        cpu_executed_on_metal_storage.backend() ==
            ExecutionBackend::Metal,
        "execution override must preserve Metal input storage"
    );
    require_tensor_close(
        cpu_executed_on_metal_storage,
        rank_two_cpu,
        "explicit CPU matmul while Metal is selected"
    );
    require(
        riftco_transformer::execution_backend() ==
            ExecutionBackend::Metal,
        "explicit dispatch must not change the selected backend"
    );
    require_tensor_close(
        tensor_ops::matmul(rank_two_left(), rank_two_right()),
        rank_two_cpu,
        "default matmul should dispatch to intrinsic Metal backend",
        metal_absolute_tolerance,
        metal_relative_tolerance
    );

    const AutogradSnapshot cpu =
        run_batched_autograd(ExecutionBackend::Cpu, false);
    const AutogradSnapshot metal =
        run_batched_autograd(ExecutionBackend::Metal, false);
    require_tensor_close(
        metal.output,
        cpu.output,
        "Metal autograd matmul forward parity",
        metal_absolute_tolerance,
        metal_relative_tolerance
    );
    require_tensor_close(
        metal.left_gradient,
        cpu.left_gradient,
        "Metal autograd left-gradient parity",
        metal_absolute_tolerance,
        metal_relative_tolerance
    );
    require_tensor_close(
        metal.right_gradient,
        cpu.right_gradient,
        "Metal autograd right-gradient parity",
        metal_absolute_tolerance,
        metal_relative_tolerance
    );

    // Variable::matmul captures the input tensor backend when it creates the
    // graph. A later construction-default change cannot alter the backward
    // closure's implementation.
    const AutogradSnapshot switched_after_forward =
        run_batched_autograd(ExecutionBackend::Metal, true);
    require(
        riftco_transformer::execution_backend() ==
            ExecutionBackend::Cpu,
        "capture test should switch the active backend to CPU"
    );
    require_tensor_close(
        switched_after_forward.output,
        metal.output,
        "captured Metal backend forward value",
        metal_absolute_tolerance,
        metal_relative_tolerance
    );
    require_tensor_close(
        switched_after_forward.left_gradient,
        metal.left_gradient,
        "captured Metal backend left gradient",
        metal_absolute_tolerance,
        metal_relative_tolerance
    );
    require_tensor_close(
        switched_after_forward.right_gradient,
        metal.right_gradient,
        "captured Metal backend right gradient",
        metal_absolute_tolerance,
        metal_relative_tolerance
    );

    return true;
}

bool test_cuda_parity_if_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Cuda
        )) {
        riftco_transformer::set_execution_backend(
            ExecutionBackend::Cpu
        );
        require_throws_as<std::runtime_error>(
            [] {
                riftco_transformer::set_execution_backend(
                    ExecutionBackend::Cuda
                );
            },
            "selecting an unavailable CUDA backend should throw"
        );
        require(
            riftco_transformer::execution_backend() ==
                ExecutionBackend::Cpu,
            "failed CUDA selection should leave CPU active"
        );
        require_throws_as<std::runtime_error>(
            [] {
                static_cast<void>(
                    tensor_ops::matmul(
                        rank_two_left(),
                        rank_two_right(),
                        ExecutionBackend::Cuda
                    )
                );
            },
            "dispatching unavailable CUDA should throw runtime_error"
        );
        require(
            riftco_transformer::execution_backend() ==
                ExecutionBackend::Cpu,
            "failed explicit CUDA dispatch should leave CPU active"
        );
        return false;
    }

    constexpr float cuda_absolute_tolerance = 3.0e-4F;
    constexpr float cuda_relative_tolerance = 3.0e-4F;
    const Tensor rank_two_cpu = tensor_ops::matmul(
        rank_two_left(),
        rank_two_right(),
        ExecutionBackend::Cpu
    );
    const Tensor rank_two_cuda = tensor_ops::matmul(
        rank_two_left(),
        rank_two_right(),
        ExecutionBackend::Cuda
    );
    require(
        rank_two_cuda.backend() == ExecutionBackend::Cpu,
        "execution override must preserve CPU input storage"
    );
    require_tensor_close(
        rank_two_cuda,
        rank_two_cpu,
        "CUDA rank-two matmul parity",
        cuda_absolute_tolerance,
        cuda_relative_tolerance
    );

    riftco_transformer::set_execution_backend(
        ExecutionBackend::Cuda
    );
    const Tensor cuda_left = rank_two_left();
    const Tensor cuda_right = rank_two_right();
    require(
        cuda_left.backend() == ExecutionBackend::Cuda &&
            cuda_right.backend() == ExecutionBackend::Cuda,
        "CUDA selection must construct CUDA tensor storage"
    );
    require_tensor_close(
        tensor_ops::matmul(cuda_left, cuda_right),
        rank_two_cpu,
        "default matmul should dispatch to intrinsic CUDA backend",
        cuda_absolute_tolerance,
        cuda_relative_tolerance
    );

    const AutogradSnapshot cpu =
        run_batched_autograd(ExecutionBackend::Cpu, false);
    const AutogradSnapshot cuda =
        run_batched_autograd(ExecutionBackend::Cuda, false);
    require_tensor_close(
        cuda.output,
        cpu.output,
        "CUDA autograd matmul forward parity",
        cuda_absolute_tolerance,
        cuda_relative_tolerance
    );
    require_tensor_close(
        cuda.left_gradient,
        cpu.left_gradient,
        "CUDA autograd left-gradient parity",
        cuda_absolute_tolerance,
        cuda_relative_tolerance
    );
    require_tensor_close(
        cuda.right_gradient,
        cpu.right_gradient,
        "CUDA autograd right-gradient parity",
        cuda_absolute_tolerance,
        cuda_relative_tolerance
    );

    const AutogradSnapshot switched_after_forward =
        run_batched_autograd(ExecutionBackend::Cuda, true);
    require(
        riftco_transformer::execution_backend() ==
            ExecutionBackend::Cpu,
        "CUDA capture test should switch the active backend to CPU"
    );
    require_tensor_close(
        switched_after_forward.output,
        cuda.output,
        "captured CUDA backend forward value",
        cuda_absolute_tolerance,
        cuda_relative_tolerance
    );
    require_tensor_close(
        switched_after_forward.left_gradient,
        cuda.left_gradient,
        "captured CUDA backend left gradient",
        cuda_absolute_tolerance,
        cuda_relative_tolerance
    );
    require_tensor_close(
        switched_after_forward.right_gradient,
        cuda.right_gradient,
        "captured CUDA backend right gradient",
        cuda_absolute_tolerance,
        cuda_relative_tolerance
    );
    return true;
}

bool test_tpu_parity_if_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Tpu
        )) {
        riftco_transformer::set_execution_backend(
            ExecutionBackend::Cpu
        );
        require_throws_as<std::runtime_error>(
            [] {
                riftco_transformer::set_execution_backend(
                    ExecutionBackend::Tpu
                );
            },
            "selecting an unavailable TPU backend should throw"
        );
        require(
            riftco_transformer::execution_backend() ==
                ExecutionBackend::Cpu,
            "failed TPU selection should leave CPU active"
        );
        require_throws_as<std::runtime_error>(
            [] {
                static_cast<void>(
                    tensor_ops::matmul(
                        rank_two_left(),
                        rank_two_right(),
                        ExecutionBackend::Tpu
                    )
                );
            },
            "dispatching unavailable TPU should throw runtime_error"
        );
        return false;
    }

    constexpr float tpu_absolute_tolerance = 1.0e-3F;
    constexpr float tpu_relative_tolerance = 1.0e-3F;
    const Tensor rank_two_cpu = tensor_ops::matmul(
        rank_two_left(),
        rank_two_right(),
        ExecutionBackend::Cpu
    );
    const Tensor rank_two_tpu = tensor_ops::matmul(
        rank_two_left(),
        rank_two_right(),
        ExecutionBackend::Tpu
    );
    require(
        rank_two_tpu.backend() == ExecutionBackend::Cpu,
        "TPU execution override must preserve CPU input storage"
    );
    require_tensor_close(
        rank_two_tpu,
        rank_two_cpu,
        "TPU rank-two matmul parity",
        tpu_absolute_tolerance,
        tpu_relative_tolerance
    );

    riftco_transformer::set_execution_backend(ExecutionBackend::Tpu);
    const Tensor tpu_left = rank_two_left();
    const Tensor tpu_right = rank_two_right();
    require(
        tpu_left.backend() == ExecutionBackend::Tpu &&
            tpu_right.backend() == ExecutionBackend::Tpu,
        "TPU selection must construct TPU host-mirror storage"
    );
    require_tensor_close(
        tensor_ops::matmul(tpu_left, tpu_right),
        rank_two_cpu,
        "default matmul should dispatch through PJRT TPU",
        tpu_absolute_tolerance,
        tpu_relative_tolerance
    );

    const AutogradSnapshot cpu =
        run_batched_autograd(ExecutionBackend::Cpu, false);
    const AutogradSnapshot tpu =
        run_batched_autograd(ExecutionBackend::Tpu, false);
    require_tensor_close(
        tpu.output,
        cpu.output,
        "TPU autograd matmul forward parity",
        tpu_absolute_tolerance,
        tpu_relative_tolerance
    );
    require_tensor_close(
        tpu.left_gradient,
        cpu.left_gradient,
        "TPU autograd left-gradient parity",
        tpu_absolute_tolerance,
        tpu_relative_tolerance
    );
    require_tensor_close(
        tpu.right_gradient,
        cpu.right_gradient,
        "TPU autograd right-gradient parity",
        tpu_absolute_tolerance,
        tpu_relative_tolerance
    );

    const AutogradSnapshot switched_after_forward =
        run_batched_autograd(ExecutionBackend::Tpu, true);
    require(
        riftco_transformer::execution_backend() ==
            ExecutionBackend::Cpu,
        "TPU capture test should switch the active backend to CPU"
    );
    require_tensor_close(
        switched_after_forward.output,
        tpu.output,
        "captured TPU backend forward value",
        tpu_absolute_tolerance,
        tpu_relative_tolerance
    );
    require_tensor_close(
        switched_after_forward.left_gradient,
        tpu.left_gradient,
        "captured TPU backend left gradient",
        tpu_absolute_tolerance,
        tpu_relative_tolerance
    );
    require_tensor_close(
        switched_after_forward.right_gradient,
        tpu.right_gradient,
        "captured TPU backend right gradient",
        tpu_absolute_tolerance,
        tpu_relative_tolerance
    );
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool require_tpu = RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU != 0;
    if (argc == 2 &&
        std::string_view(argv[1]) == "--require-tpu") {
        require_tpu = true;
    } else if (argc != 1) {
        std::cerr << "usage: backend_tests [--require-tpu]\n";
        return 2;
    }

    const ExecutionBackend initial_backend =
        riftco_transformer::execution_backend();
    const riftco_transformer::ScopedExecutionBackend restore_backend(
        ExecutionBackend::Cpu
    );
    try {
#if RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA
        require(
            riftco_transformer::execution_backend_available(
                ExecutionBackend::Cuda
            ),
            "CUDA is required for this backend test, but no CUDA device "
            "is available"
        );
#endif
        if (require_tpu) {
            require(
                riftco_transformer::execution_backend_available(
                    ExecutionBackend::Tpu
                ),
                "TPU is required for this backend test, but no Cloud TPU "
                "device is available"
            );
        }
        test_cpu_metadata_and_default_selection(
            initial_backend
        );
        test_pinned_tpu_compile_options();
        test_explicit_cpu_matmul();
        test_unknown_backend_is_rejected_transactionally();
        test_unavailable_backends_have_diagnostics();
        test_selection_is_thread_local_if_available(
            ExecutionBackend::Metal,
            "Metal"
        );
        test_selection_is_thread_local_if_available(
            ExecutionBackend::Cuda,
            "CUDA"
        );
        test_selection_is_thread_local_if_available(
            ExecutionBackend::Tpu,
            "TPU"
        );
        test_scoped_selection_restores_previous_backend();
        const bool metal_tested =
            test_metal_parity_if_available();
        const bool cuda_tested =
            test_cuda_parity_if_available();
        const bool tpu_tested =
            test_tpu_parity_if_available();

        std::cout << "backend tests passed";
        if (!metal_tested) {
            std::cout << " (Metal unavailable; parity checks skipped)";
        }
        if (!cuda_tested) {
            std::cout << " (CUDA unavailable; parity checks skipped)";
        }
        if (!tpu_tested) {
            std::cout << " (TPU unavailable; parity checks skipped)";
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
