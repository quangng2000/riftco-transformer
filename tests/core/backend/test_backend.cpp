#include "transformer_lab/core/autograd.hpp"
#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/core/tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using transformer_lab::ExecutionBackend;
using transformer_lab::Tensor;
using transformer_lab::Variable;
namespace tensor_ops = transformer_lab::tensor_ops;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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
        transformer_lab::execution_backend_available(
            ExecutionBackend::Cpu
        ),
        "CPU execution backend should always be available"
    );
    require(
        transformer_lab::execution_backend_name(
            ExecutionBackend::Cpu
        ) == std::string_view("cpu"),
        "CPU execution backend name"
    );
    require(
        transformer_lab::execution_backend_name(
            ExecutionBackend::Metal
        ) == std::string_view("metal"),
        "Metal execution backend name"
    );
    require(
        transformer_lab::execution_backend() ==
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
    transformer_lab::set_execution_backend(
        ExecutionBackend::Cpu
    );

    require(
        !transformer_lab::execution_backend_available(unknown),
        "an unknown backend must not report available"
    );
    require_throws_as<std::invalid_argument>(
        [] {
            static_cast<void>(
                transformer_lab::execution_backend_name(unknown)
            );
        },
        "naming an unknown backend should throw invalid_argument"
    );
    require_throws_as<std::invalid_argument>(
        [] {
            transformer_lab::set_execution_backend(unknown);
        },
        "selecting an unknown backend should throw invalid_argument"
    );
    require(
        transformer_lab::execution_backend() ==
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
        transformer_lab::execution_backend() ==
            ExecutionBackend::Cpu,
        "failed explicit dispatch should not change selection"
    );
}

void test_selection_is_thread_local_if_metal_available() {
    if (!transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    transformer_lab::set_execution_backend(
        ExecutionBackend::Metal
    );
    std::exception_ptr worker_failure;
    std::thread worker([&worker_failure] {
        try {
            require(
                transformer_lab::execution_backend() ==
                    ExecutionBackend::Cpu,
                "a new thread should begin with the CPU default"
            );
            transformer_lab::set_execution_backend(
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
        transformer_lab::execution_backend() ==
            ExecutionBackend::Metal,
        "worker selection must not change the caller's backend"
    );
    transformer_lab::set_execution_backend(
        ExecutionBackend::Cpu
    );
}

void test_scoped_selection_restores_previous_backend() {
    transformer_lab::set_execution_backend(
        ExecutionBackend::Cpu
    );
    {
        const transformer_lab::ScopedExecutionBackend scope(
            ExecutionBackend::Cpu
        );
        require(
            transformer_lab::execution_backend() ==
                ExecutionBackend::Cpu,
            "scoped CPU selection"
        );
    }
    require(
        transformer_lab::execution_backend() ==
            ExecutionBackend::Cpu,
        "scoped selection should restore CPU"
    );

    if (transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        {
            const transformer_lab::ScopedExecutionBackend scope(
                ExecutionBackend::Metal
            );
            require(
                transformer_lab::execution_backend() ==
                    ExecutionBackend::Metal,
                "scoped Metal selection"
            );
        }
        require(
            transformer_lab::execution_backend() ==
                ExecutionBackend::Cpu,
            "scoped Metal selection should restore CPU"
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
    transformer_lab::set_execution_backend(backend);

    const Variable left(batched_left());
    const Variable right(batched_right());
    const Variable output = transformer_lab::matmul(left, right);
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
        transformer_lab::set_execution_backend(
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
    if (!transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        transformer_lab::set_execution_backend(
            ExecutionBackend::Cpu
        );
        require_throws_as<std::runtime_error>(
            [] {
                transformer_lab::set_execution_backend(
                    ExecutionBackend::Metal
                );
            },
            "selecting an unavailable Metal backend should throw"
        );
        require(
            transformer_lab::execution_backend() ==
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
            transformer_lab::execution_backend() ==
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

    transformer_lab::set_execution_backend(
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
        transformer_lab::execution_backend() ==
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
        transformer_lab::execution_backend() ==
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

}  // namespace

int main() {
    const ExecutionBackend initial_backend =
        transformer_lab::execution_backend();
    const transformer_lab::ScopedExecutionBackend restore_backend(
        ExecutionBackend::Cpu
    );
    try {
        test_cpu_metadata_and_default_selection(
            initial_backend
        );
        test_explicit_cpu_matmul();
        test_unknown_backend_is_rejected_transactionally();
        test_selection_is_thread_local_if_metal_available();
        test_scoped_selection_restores_previous_backend();
        const bool metal_tested =
            test_metal_parity_if_available();

        std::cout << "backend tests passed";
        if (!metal_tested) {
            std::cout << " (Metal unavailable; parity checks skipped)";
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
