#include "core/backend/nn/cuda/launch.hpp"

#include "core/backend/nn/cuda/common.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace riftco_transformer::backend_detail {
namespace {

using nn_cuda_detail::DeviceBuffer;
using nn_cuda_detail::kThreadsPerBlock;

constexpr unsigned int kDomainError = 1U;
constexpr unsigned int kOverflowError = 2U;

__global__ void
cross_entropy_rows_kernel(const float* logits, const std::uint32_t* targets,
                          float* base_gradient, double* row_losses,
                          unsigned int* status, std::size_t positions,
                          std::size_t classes) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    std::size_t position =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (position < positions) {
        const std::size_t base = position * classes;
        float maximum = -CUDART_INF_F;
        bool invalid_value = false;
        for (std::size_t item = 0; item < classes; ++item) {
            const float value = logits[base + item];
            if (isnan(value) || value == CUDART_INF_F) {
                invalid_value = true;
            }
            if (maximum < value) {
                maximum = value;
            }
        }
        if (invalid_value || maximum == -CUDART_INF_F) {
            atomicOr(status, kDomainError);
            row_losses[position] = 0.0;
            for (std::size_t item = 0; item < classes; ++item) {
                base_gradient[base + item] = 0.0F;
            }
            position += stride;
            continue;
        }

        double exponential_sum = 0.0;
        for (std::size_t item = 0; item < classes; ++item) {
            const double exponential =
                exp(static_cast<double>(logits[base + item] - maximum));
            base_gradient[base + item] = static_cast<float>(exponential);
            exponential_sum += exponential;
        }
        if (!(exponential_sum > 0.0) || !isfinite(exponential_sum)) {
            atomicOr(status, kDomainError);
            row_losses[position] = 0.0;
            for (std::size_t item = 0; item < classes; ++item) {
                base_gradient[base + item] = 0.0F;
            }
            position += stride;
            continue;
        }

        const std::size_t target = static_cast<std::size_t>(targets[position]);
        const float target_logit = logits[base + target];
        if (!isfinite(target_logit)) {
            atomicOr(status, kDomainError);
            row_losses[position] = 0.0;
            for (std::size_t item = 0; item < classes; ++item) {
                base_gradient[base + item] = 0.0F;
            }
            position += stride;
            continue;
        }

        const float inverse_positions = 1.0F / static_cast<float>(positions);
        for (std::size_t item = 0; item < classes; ++item) {
            float gradient = static_cast<float>(
                static_cast<double>(base_gradient[base + item]) /
                exponential_sum);
            if (item == target) {
                gradient -= 1.0F;
            }
            base_gradient[base + item] = gradient * inverse_positions;
        }
        row_losses[position] =
            log(exponential_sum) - static_cast<double>(target_logit - maximum);
        position += stride;
    }
}

__global__ void cross_entropy_reduce_kernel(const double* row_losses,
                                            float* loss, unsigned int* status,
                                            std::size_t positions) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }
    if ((*status & kDomainError) != 0U) {
        return;
    }
    double total_loss = 0.0;
    for (std::size_t position = 0; position < positions; ++position) {
        total_loss += row_losses[position];
    }
    const double mean_loss = total_loss / static_cast<double>(positions);
    if (!isfinite(mean_loss) || mean_loss > static_cast<double>(FLT_MAX)) {
        atomicOr(status, kOverflowError);
        return;
    }
    loss[0] = static_cast<float>(mean_loss);
}

} // namespace

void cuda_nn_cross_entropy_forward(const CrossEntropyForwardRequest& request) {
    DeviceBuffer<std::uint32_t> targets(request.targets.size());
    targets.copy_from_host(request.targets, "cross-entropy target upload");
    DeviceBuffer<double> row_losses(request.positions);
    DeviceBuffer<unsigned int> status(1);
    status.zero("cross-entropy status initialization");

    const char* operation_name = "cross-entropy";
    cross_entropy_rows_kernel<<<
        nn_cuda_detail::block_count_for(request.positions), kThreadsPerBlock>>>(
        nn_cuda_detail::require_native_input(request.logits, operation_name),
        targets.data(),
        nn_cuda_detail::require_native_output(request.base_gradient,
                                              operation_name),
        row_losses.data(), status.data(), request.positions, request.classes);
    nn_cuda_detail::require_kernel_launch("cross-entropy row kernel launch");
    cross_entropy_reduce_kernel<<<1, 1>>>(
        row_losses.data(),
        nn_cuda_detail::require_native_output(request.loss, operation_name),
        status.data(), request.positions);
    nn_cuda_detail::require_kernel_launch(
        "cross-entropy reduction kernel launch");
    nn_cuda_detail::synchronize("cross-entropy synchronization");

    const unsigned int result =
        nn_cuda_detail::read_status(status, "cross-entropy status read");
    if ((result & kDomainError) != 0U) {
        throw std::domain_error("cross-entropy received invalid logits");
    }
    if ((result & kOverflowError) != 0U) {
        throw std::overflow_error(
            "cross entropy loss exceeds finite float range");
    }
}

} // namespace riftco_transformer::backend_detail
