#include "internal/bridge.hpp"

using namespace riftco_transformer::c_api::detail;

namespace {

void require_context(const rt_context* context) {
    if (context == nullptr) {
        throw std::invalid_argument(
            "context handle must not be null"
        );
    }
}

void require_tensor(const rt_tensor* tensor) {
    if (tensor == nullptr) {
        throw std::invalid_argument(
            "tensor handle must not be null"
        );
    }
}

}  // namespace


extern "C" {

rt_status RT_CALL rt_backend_is_available(
    rt_backend backend,
    int32_t* available
) {
    return guard([&] {
        if (available == nullptr) {
            throw std::invalid_argument(
                "availability output must not be null"
            );
        }
        riftco_transformer::ExecutionBackend native;
        switch (backend) {
            case RT_BACKEND_CPU:
                native =
                    riftco_transformer::ExecutionBackend::Cpu;
                break;
            case RT_BACKEND_METAL:
                native =
                    riftco_transformer::ExecutionBackend::Metal;
                break;
            case RT_BACKEND_CUDA:
                native =
                    riftco_transformer::ExecutionBackend::Cuda;
                break;
            case RT_BACKEND_TPU:
                native =
                    riftco_transformer::ExecutionBackend::Tpu;
                break;
            default:
                throw std::invalid_argument(
                    "unknown C API backend"
                );
        }
        *available =
            riftco_transformer::execution_backend_available(native)
                ? 1
                : 0;
    });
}

rt_status RT_CALL rt_context_create(
    rt_backend backend,
    rt_context** output
) {
    return guard([&] {
        require_output(output);
        const auto native = checked_backend(backend);
        auto result = std::make_unique<rt_context>(
            rt_context{native}
        );
        *output = result.release();
    });
}

void RT_CALL rt_context_release(rt_context* context) {
    delete context;
}

rt_status RT_CALL rt_context_backend(
    const rt_context* context,
    rt_backend* output
) {
    return guard([&] {
        require_context(context);
        if (output == nullptr) {
            throw std::invalid_argument(
                "backend output must not be null"
            );
        }
        *output = c_backend(context->backend);
    });
}

rt_status RT_CALL rt_tensor_create_f32(
    const rt_context* context,
    const uint64_t* shape,
    uint64_t rank,
    const float* values,
    uint64_t value_count,
    rt_tensor** output
) {
    return guard([&] {
        require_output(output);
        require_context(context);
        const std::size_t native_count =
            checked_size(value_count, "tensor value count");
        if (native_count != 0 && values == nullptr) {
            throw std::invalid_argument(
                "tensor values pointer must not be null"
            );
        }

        std::vector<float> owned_values;
        owned_values.reserve(native_count);
        if (native_count != 0) {
            owned_values.assign(
                values,
                values + native_count
            );
        }
        auto result = std::make_unique<rt_tensor>(
            rt_tensor{
                riftco_transformer::Tensor(
                    checked_shape(shape, rank),
                    std::move(owned_values),
                    context->backend
                ),
            }
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_tensor_zeros_f32(
    const rt_context* context,
    const uint64_t* shape,
    uint64_t rank,
    rt_tensor** output
) {
    return guard([&] {
        require_output(output);
        require_context(context);
        auto result = std::make_unique<rt_tensor>(
            rt_tensor{
                riftco_transformer::Tensor::zeros(
                    checked_shape(shape, rank),
                    context->backend
                ),
            }
        );
        *output = result.release();
    });
}

void RT_CALL rt_tensor_release(rt_tensor* tensor) {
    delete tensor;
}

rt_status RT_CALL rt_tensor_backend(
    const rt_tensor* tensor,
    rt_backend* output
) {
    return guard([&] {
        require_tensor(tensor);
        if (output == nullptr) {
            throw std::invalid_argument(
                "backend output must not be null"
            );
        }
        *output = c_backend(tensor->value.backend());
    });
}

rt_status RT_CALL rt_tensor_rank(
    const rt_tensor* tensor,
    uint64_t* output
) {
    return guard([&] {
        require_tensor(tensor);
        if (output == nullptr) {
            throw std::invalid_argument(
                "rank output must not be null"
            );
        }
        *output = checked_u64(
            tensor->value.rank(),
            "tensor rank"
        );
    });
}

rt_status RT_CALL rt_tensor_shape(
    const rt_tensor* tensor,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
) {
    return guard([&] {
        require_tensor(tensor);
        copy_tensor_shape(
            tensor->value,
            output_dimensions,
            dimension_capacity
        );
    });
}

rt_status RT_CALL rt_tensor_numel(
    const rt_tensor* tensor,
    uint64_t* output
) {
    return guard([&] {
        require_tensor(tensor);
        if (output == nullptr) {
            throw std::invalid_argument(
                "element-count output must not be null"
            );
        }
        *output = checked_u64(
            tensor->value.numel(),
            "tensor element count"
        );
    });
}

rt_status RT_CALL rt_tensor_copy_to_host_f32(
    const rt_tensor* tensor,
    float* output_values,
    uint64_t value_capacity
) {
    return guard([&] {
        require_tensor(tensor);
        copy_tensor_values(
            tensor->value,
            output_values,
            value_capacity
        );
    });
}

rt_status RT_CALL rt_tensor_matmul(
    const rt_tensor* left,
    const rt_tensor* right,
    rt_tensor** output
) {
    return guard([&] {
        require_output(output);
        require_tensor(left);
        require_tensor(right);
        if (left->value.backend() != right->value.backend()) {
            throw std::invalid_argument(
                "matmul tensors must use the same backend"
            );
        }
        auto result = std::make_unique<rt_tensor>(
            rt_tensor{
                riftco_transformer::tensor_ops::matmul(
                    left->value,
                    right->value
                ),
            }
        );
        *output = result.release();
    });
}

}  // extern "C"
