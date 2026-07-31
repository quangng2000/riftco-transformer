#include "riftco_transformer/core/autograd.hpp"
#include "riftco_transformer/core/tensor.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using riftco_transformer::ExecutionBackend;
using riftco_transformer::Tensor;
using riftco_transformer::Variable;
namespace tensor_ops = riftco_transformer::tensor_ops;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(float actual, float expected, const std::string& message) {
    constexpr float tolerance = 1.0e-6F;
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

void test_shape_storage_and_strides() {
    Tensor tensor({2, 3, 4});
    const Tensor filled = Tensor::full({2}, 7.0F);
    const Tensor zeros = Tensor::zeros({2});

    require(tensor.rank() == 3, "rank should be three");
    require(tensor.numel() == 24, "2x3x4 should contain 24 values");
    require(tensor.shape() == Tensor::Shape({2, 3, 4}), "shape mismatch");
    require(tensor.strides() == Tensor::Shape({12, 4, 1}),
            "row-major strides mismatch");

    for (const float value : tensor.data()) {
        require_close(value, 0.0F, "default construction should zero data");
    }
    require_close(filled.flat(0), 7.0F, "full tensor value mismatch");
    require_close(filled.flat(1), 7.0F, "full tensor value mismatch");
    require_close(zeros.flat(0), 0.0F, "zero tensor value mismatch");
    require_close(zeros.flat(1), 0.0F, "zero tensor value mismatch");
}

void test_checked_indexing() {
    Tensor tensor({2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

    require_close(tensor.at({0, 0}), 1.0F, "first value mismatch");
    require_close(tensor.at({1, 2}), 6.0F, "last value mismatch");

    tensor.at({1, 1}) = 42.0F;
    require_close(tensor.flat(4), 42.0F, "indexed write used wrong offset");

    const Tensor& const_tensor = tensor;
    require_close(const_tensor.at({1, 1}), 42.0F, "const indexing mismatch");

    require_throws(
        [&] { static_cast<void>(tensor.at({0})); },
        "index rank mismatch should throw"
    );
    require_throws(
        [&] { static_cast<void>(tensor.at({2, 0})); },
        "out-of-range multidimensional index should throw"
    );
    require_throws(
        [&] { static_cast<void>(tensor.flat(6)); },
        "out-of-range flat index should throw"
    );
}

void test_scalar_and_invalid_construction() {
    Tensor scalar({}, 3.5F);
    require(scalar.rank() == 0, "scalar rank should be zero");
    require(scalar.numel() == 1, "scalar should contain one value");
    require_close(scalar.at({}), 3.5F, "scalar value mismatch");

    require_throws(
        [] { Tensor invalid({2, 0, 3}); },
        "zero-sized dimensions should be rejected"
    );
    require_throws(
        [] { Tensor invalid({2, 2}, {1.0F, 2.0F}); },
        "incorrect value count should be rejected"
    );
    require_throws(
        [] {
            Tensor invalid({
                std::numeric_limits<std::size_t>::max(),
                std::size_t{2},
            });
        },
        "shape size overflow should be rejected"
    );
}

void test_reshape() {
    const Tensor original(
        {2, 3},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}
    );
    const Tensor reshaped = original.reshape({3, 2});

    require(reshaped.shape() == Tensor::Shape({3, 2}), "reshape shape mismatch");
    require_close(reshaped.at({2, 1}), 6.0F, "reshape changed data order");
    require(original.shape() == Tensor::Shape({2, 3}),
            "reshape should not mutate its input");

    require_throws(
        [&] { static_cast<void>(original.reshape({4, 2})); },
        "reshape with a different element count should throw"
    );
}

void test_copy_and_same_backend_transfer_are_deep() {
    Tensor original(
        {2, 2},
        {1.0F, 2.0F, 3.0F, 4.0F},
        ExecutionBackend::Cpu
    );
    Tensor copied(original);
    copied.flat(0) = 100.0F;
    require_close(
        original.flat(0),
        1.0F,
        "copy construction must not alias source storage"
    );

    original.flat(1) = 200.0F;
    require_close(
        copied.flat(1),
        2.0F,
        "source mutation must not change copied storage"
    );

    Tensor assigned({1}, -1.0F, ExecutionBackend::Cpu);
    assigned = original;
    require(
        assigned.shape() == original.shape(),
        "copy assignment should replace the destination shape"
    );
    require(
        assigned.backend() == ExecutionBackend::Cpu,
        "copy assignment should preserve the source backend"
    );
    assigned.flat(2) = -3.0F;
    require_close(
        original.flat(2),
        3.0F,
        "copy assignment must not alias source storage"
    );

    Tensor same_backend = original.to(ExecutionBackend::Cpu);
    same_backend.flat(3) = -4.0F;
    require_close(
        original.flat(3),
        4.0F,
        "same-backend transfer must return independent storage"
    );
}

void test_moved_from_tensor_is_a_safe_singular_value() {
    Tensor source(
        {2},
        {1.0F, 2.0F},
        ExecutionBackend::Cpu
    );
    Tensor destination(std::move(source));
    require(destination.numel() == 2, "move retains destination values");
    require(source.numel() == 0, "moved-from tensor has zero elements");
    require(source.data().empty(), "moved-from tensor has empty data");
    static_cast<void>(source.backend());

    require_throws(
        [&] { static_cast<void>(source.flat(0)); },
        "moved-from flat access should throw"
    );
    require_throws(
        [&] { static_cast<void>(source.at({})); },
        "moved-from indexed access should throw"
    );
    require_throws(
        [&] { static_cast<void>(source.reshape({})); },
        "moved-from reshape should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(
                source.to(ExecutionBackend::Cpu)
            );
        },
        "moved-from transfer should throw"
    );

    const Tensor copied_singular(source);
    require(
        copied_singular.numel() == 0 &&
            copied_singular.data().empty(),
        "copying a singular tensor should remain safe"
    );
    source = Tensor({1}, 4.0F, ExecutionBackend::Cpu);
    require_close(
        source.flat(0),
        4.0F,
        "moved-from tensor should remain assignable"
    );
}

void test_operations_preserve_backend(ExecutionBackend backend) {
    const Tensor zeros = Tensor::zeros({2}, backend);
    const Tensor filled = Tensor::full({2}, 3.0F, backend);
    require(
        zeros.backend() == backend && filled.backend() == backend,
        "backend-specific factories should use the requested backend"
    );

    const Tensor left(
        {2, 2},
        {1.0F, 2.0F, 3.0F, 4.0F},
        backend
    );
    const Tensor right(
        {2, 2},
        {5.0F, 6.0F, 7.0F, 8.0F},
        backend
    );
    const Tensor added = tensor_ops::add(left, right);
    const Tensor product = tensor_ops::matmul(left, right, backend);
    const Tensor reduced = tensor_ops::sum(product);
    const Tensor reshaped = product.reshape({4});

    require(added.backend() == backend, "add output backend");
    require(product.backend() == backend, "matmul output backend");
    require(reduced.backend() == backend, "sum output backend");
    require(reshaped.backend() == backend, "reshape output backend");
    require_close(added.flat(0), 6.0F, "backend add value");
    require_close(product.flat(0), 19.0F, "backend matmul value");
    require_close(reduced.flat(0), 134.0F, "backend sum value");

    const Variable input(left);
    const Variable objective = riftco_transformer::sum(input * input);
    require(
        objective.value().backend() == backend,
        "autograd output should preserve the input backend"
    );
    objective.backward();
    require(
        input.gradient().backend() == backend,
        "autograd gradient should preserve the input backend"
    );
    for (std::size_t index = 0; index < input.gradient().numel(); ++index) {
        require_close(
            input.gradient().flat(index),
            2.0F * left.flat(index),
            "backend-preserving autograd gradient"
        );
    }
}

void test_metal_transfer_and_copy_if_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    Tensor host(
        {2, 2},
        {1.5F, -2.0F, 3.25F, 4.5F},
        ExecutionBackend::Cpu
    );
    Tensor device = host.to(ExecutionBackend::Metal);
    require(
        device.backend() == ExecutionBackend::Metal,
        "CPU-to-Metal transfer backend"
    );
    for (std::size_t index = 0; index < host.numel(); ++index) {
        require_close(
            device.flat(index),
            host.flat(index),
            "CPU-to-Metal transfer value"
        );
    }

    Tensor device_copy(device);
    device_copy.flat(0) = 99.0F;
    require_close(
        device.flat(0),
        1.5F,
        "Metal copy construction must clone device storage"
    );

    Tensor device_transfer = device.to(ExecutionBackend::Metal);
    device_transfer.flat(1) = 88.0F;
    require_close(
        device.flat(1),
        -2.0F,
        "same-Metal transfer must clone device storage"
    );

    Tensor round_trip = device.to(ExecutionBackend::Cpu);
    require(
        round_trip.backend() == ExecutionBackend::Cpu,
        "Metal-to-CPU transfer backend"
    );
    device.flat(2) = -7.0F;
    require_close(
        round_trip.flat(2),
        3.25F,
        "Metal-to-CPU transfer must not alias device storage"
    );
    require_close(
        host.flat(2),
        3.25F,
        "device mutation must not alias original host storage"
    );
}

void test_mixed_backend_inputs_are_rejected_if_metal_available() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    const Tensor cpu(
        {2, 2},
        {1.0F, 2.0F, 3.0F, 4.0F},
        ExecutionBackend::Cpu
    );
    const Tensor metal = cpu.to(ExecutionBackend::Metal);

    require_throws(
        [&] {
            static_cast<void>(tensor_ops::add(cpu, metal));
        },
        "elementwise mixed-backend inputs should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(tensor_ops::matmul(cpu, metal));
        },
        "matmul mixed-backend inputs should throw"
    );
    require_throws(
        [&] {
            const Variable cpu_variable(cpu);
            const Variable metal_variable(metal);
            static_cast<void>(cpu_variable + metal_variable);
        },
        "autograd mixed-backend inputs should throw"
    );
    require_throws(
        [&] {
            const Variable cpu_variable(cpu);
            cpu_variable.backward(metal);
        },
        "autograd mixed-backend seed should throw"
    );
}

}  // namespace

int main() {
    try {
        test_shape_storage_and_strides();
        test_checked_indexing();
        test_scalar_and_invalid_construction();
        test_reshape();
        test_copy_and_same_backend_transfer_are_deep();
        test_moved_from_tensor_is_a_safe_singular_value();
        test_operations_preserve_backend(ExecutionBackend::Cpu);
        test_metal_transfer_and_copy_if_available();
        test_mixed_backend_inputs_are_rejected_if_metal_available();
        const bool metal_available =
            riftco_transformer::execution_backend_available(
                ExecutionBackend::Metal
            );
        if (metal_available) {
            test_operations_preserve_backend(
                ExecutionBackend::Metal
            );
        }
        std::cout << "tensor tests passed";
        if (!metal_available) {
            std::cout << " (Metal unavailable; device checks skipped)";
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
