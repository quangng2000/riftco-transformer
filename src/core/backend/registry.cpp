#include "riftco_transformer/core/backend.hpp"

#include "adapter.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace riftco_transformer {
namespace {

thread_local ExecutionBackend active_backend =
    ExecutionBackend::Cpu;

const backend_detail::BackendAdapter& require_backend_adapter(
    ExecutionBackend backend
) {
    const auto* adapter =
        backend_detail::find_backend_adapter(backend);
    if (adapter == nullptr) {
        throw std::invalid_argument("unknown execution backend");
    }
    return *adapter;
}

void require_available(
    const backend_detail::BackendAdapter& adapter
) {
    if (!adapter.is_available()) {
        throw std::runtime_error(
            std::string(adapter.name()) +
            " execution backend is unavailable"
        );
    }
}

std::size_t checked_product(
    std::initializer_list<std::size_t> factors
) {
    std::size_t result = 1;
    for (const auto factor : factors) {
        if (factor != 0 &&
            result >
                std::numeric_limits<std::size_t>::max() / factor) {
            throw std::overflow_error(
                "backend operation size exceeds addressable storage"
            );
        }
        result *= factor;
    }
    return result;
}

std::size_t checked_shape_product(
    std::span<const std::size_t> dimensions
) {
    std::size_t result = 1;
    for (const auto dimension : dimensions) {
        if (dimension == 0) {
            throw std::invalid_argument(
                "backend operation dimensions must be positive"
            );
        }
        if (result >
            std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error(
                "backend operation size exceeds addressable storage"
            );
        }
        result *= dimension;
    }
    return result;
}

void require_storage_contract(
    const backend_detail::TensorStorage& storage,
    ExecutionBackend backend,
    std::size_t expected_size
) {
    if (storage.backend() != backend ||
        storage.size() != expected_size ||
        storage.data().size() != expected_size) {
        throw std::logic_error(
            "backend adapter returned invalid tensor storage"
        );
    }
}

void require_operation_storage(
    const backend_detail::TensorStorage& storage,
    ExecutionBackend backend,
    std::size_t expected_size
) {
    if (storage.backend() != backend) {
        throw std::invalid_argument(
            "backend operation storage does not match its backend"
        );
    }
    if (expected_size == 0 ||
        storage.size() != expected_size ||
        storage.data().size() != expected_size) {
        throw std::logic_error(
            "backend operation storage size does not match its dimensions"
        );
    }
}

bool storage_aliases(
    const backend_detail::TensorStorage& left,
    const backend_detail::TensorStorage& right
) noexcept {
    if (&left == &right) {
        return true;
    }
    const void* left_handle = left.native_handle();
    const void* right_handle = right.native_handle();
    return left_handle != nullptr && left_handle == right_handle;
}

void require_distinct(
    const backend_detail::TensorStorage& left,
    const backend_detail::TensorStorage& right
) {
    if (storage_aliases(left, right)) {
        throw std::invalid_argument(
            "backend operation inputs and outputs must not alias"
        );
    }
}

void require_output_separation(
    std::initializer_list<
        const backend_detail::TensorStorage*
    > inputs,
    std::initializer_list<
        const backend_detail::TensorStorage*
    > outputs
) {
    for (const auto* output : outputs) {
        for (const auto* input : inputs) {
            require_distinct(*input, *output);
        }
    }
    for (auto left = outputs.begin(); left != outputs.end(); ++left) {
        for (auto right = left + 1; right != outputs.end(); ++right) {
            require_distinct(**left, **right);
        }
    }
}

void require_axis_dimensions(
    const backend_detail::AxisDimensions& dimensions
) {
    if (dimensions.outer == 0 ||
        dimensions.width == 0 ||
        dimensions.inner == 0) {
        throw std::invalid_argument(
            "axis operation dimensions must be positive"
        );
    }
}

void require_valid_unary_operation(
    backend_detail::UnaryOperation operation
) {
    using backend_detail::UnaryOperation;
    switch (operation) {
        case UnaryOperation::Negate:
        case UnaryOperation::Exp:
        case UnaryOperation::Log:
        case UnaryOperation::Sqrt:
        case UnaryOperation::Erf:
            return;
    }
    throw std::invalid_argument(
        "unknown unary elementwise operation"
    );
}

void require_valid_binary_operation(
    backend_detail::BinaryOperation operation
) {
    using backend_detail::BinaryOperation;
    switch (operation) {
        case BinaryOperation::Add:
        case BinaryOperation::Subtract:
        case BinaryOperation::Multiply:
        case BinaryOperation::Divide:
            return;
    }
    throw std::invalid_argument(
        "unknown binary elementwise operation"
    );
}

void require_valid_reduction_operation(
    backend_detail::ReductionOperation operation
) {
    using backend_detail::ReductionOperation;
    switch (operation) {
        case ReductionOperation::Sum:
        case ReductionOperation::Mean:
            return;
    }
    throw std::invalid_argument("unknown reduction operation");
}

std::size_t checked_axis_input_size(
    const backend_detail::AxisDimensions& dimensions
) {
    require_axis_dimensions(dimensions);
    return checked_product({
        dimensions.outer,
        dimensions.width,
        dimensions.inner,
    });
}

std::size_t checked_axis_output_size(
    const backend_detail::AxisDimensions& dimensions
) {
    require_axis_dimensions(dimensions);
    return checked_product({
        dimensions.outer,
        dimensions.inner,
    });
}

}  // namespace

bool execution_backend_available(
    ExecutionBackend backend
) noexcept {
    const auto* adapter =
        backend_detail::find_backend_adapter(backend);
    return adapter != nullptr && adapter->is_available();
}

std::string_view execution_backend_name(
    ExecutionBackend backend
) {
    return require_backend_adapter(backend).name();
}

ExecutionBackend execution_backend() noexcept {
    return active_backend;
}

void set_execution_backend(ExecutionBackend backend) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    active_backend = backend;
}

ScopedExecutionBackend::ScopedExecutionBackend(
    ExecutionBackend backend
)
    : previous_(execution_backend()) {
    set_execution_backend(backend);
}

ScopedExecutionBackend::~ScopedExecutionBackend() noexcept {
    active_backend = previous_;
}

namespace backend_detail {

const BackendAdapter* find_backend_adapter(
    ExecutionBackend backend
) noexcept {
    switch (backend) {
        case ExecutionBackend::Cpu:
            return &cpu_backend_adapter();
        case ExecutionBackend::Metal:
            return &metal_backend_adapter();
    }
    return nullptr;
}

std::unique_ptr<TensorStorage> make_tensor_storage(
    ExecutionBackend backend,
    std::size_t element_count,
    float fill_value
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    auto storage = adapter.make_storage(element_count, fill_value);
    if (storage == nullptr) {
        throw std::logic_error(
            "backend adapter returned null tensor storage"
        );
    }
    require_storage_contract(*storage, backend, element_count);
    return storage;
}

std::unique_ptr<TensorStorage> make_tensor_storage(
    ExecutionBackend backend,
    std::vector<float> values
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    const auto element_count = values.size();
    auto storage = adapter.make_storage(std::move(values));
    if (storage == nullptr) {
        throw std::logic_error(
            "backend adapter returned null tensor storage"
        );
    }
    require_storage_contract(*storage, backend, element_count);
    return storage;
}

void dispatch_matmul(
    ExecutionBackend backend,
    const TensorStorage& left,
    const TensorStorage& right,
    TensorStorage& output,
    MatmulDimensions dimensions
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (left.backend() != right.backend() ||
        left.backend() != output.backend()) {
        throw std::invalid_argument(
            "matmul storage backends must match"
        );
    }
    const auto expected_left = checked_product({
        dimensions.batch_count,
        dimensions.rows,
        dimensions.shared,
    });
    const auto expected_right = checked_product({
        dimensions.batch_count,
        dimensions.shared,
        dimensions.columns,
    });
    const auto expected_output = checked_product({
        dimensions.batch_count,
        dimensions.rows,
        dimensions.columns,
    });
    if (left.size() != expected_left ||
        right.size() != expected_right ||
        output.size() != expected_output) {
        throw std::logic_error(
            "matmul storage size does not match its dimensions"
        );
    }
    adapter.matmul({left, right, output, dimensions});
}

void dispatch_unary_elementwise(
    ExecutionBackend backend,
    const UnaryElementwiseRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    require_valid_unary_operation(request.operation);
    require_operation_storage(
        request.input,
        backend,
        request.element_count
    );
    require_operation_storage(
        request.output,
        backend,
        request.element_count
    );
    require_distinct(request.input, request.output);
    adapter.unary_elementwise(request);
}

void dispatch_binary_elementwise(
    ExecutionBackend backend,
    const BinaryElementwiseRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    require_valid_binary_operation(request.operation);
    require_operation_storage(
        request.left,
        backend,
        request.element_count
    );
    require_operation_storage(
        request.right,
        backend,
        request.element_count
    );
    require_operation_storage(
        request.output,
        backend,
        request.element_count
    );
    require_distinct(request.left, request.output);
    require_distinct(request.right, request.output);
    adapter.binary_elementwise(request);
}

void dispatch_scale(
    ExecutionBackend backend,
    const ScaleRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    require_operation_storage(
        request.input,
        backend,
        request.element_count
    );
    require_operation_storage(
        request.output,
        backend,
        request.element_count
    );
    require_distinct(request.input, request.output);
    adapter.scale(request);
}

void dispatch_gelu_forward(
    ExecutionBackend backend,
    const GeluForwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    require_operation_storage(
        request.input,
        backend,
        request.element_count
    );
    require_operation_storage(
        request.output,
        backend,
        request.element_count
    );
    require_distinct(request.input, request.output);
    adapter.gelu_forward(request);
}

void dispatch_gelu_backward(
    ExecutionBackend backend,
    const GeluBackwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    require_operation_storage(
        request.input,
        backend,
        request.element_count
    );
    require_operation_storage(
        request.upstream,
        backend,
        request.element_count
    );
    require_operation_storage(
        request.input_gradient,
        backend,
        request.element_count
    );
    require_distinct(request.input, request.input_gradient);
    require_distinct(request.upstream, request.input_gradient);
    adapter.gelu_backward(request);
}

void dispatch_reduction(
    ExecutionBackend backend,
    const ReductionRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    require_valid_reduction_operation(request.operation);
    const auto input_size =
        checked_axis_input_size(request.dimensions);
    const auto output_size =
        checked_axis_output_size(request.dimensions);
    require_operation_storage(request.input, backend, input_size);
    require_operation_storage(request.output, backend, output_size);
    require_distinct(request.input, request.output);
    adapter.reduce(request);
}

void dispatch_copy(
    ExecutionBackend backend,
    const CopyRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    require_operation_storage(
        request.input,
        backend,
        request.element_count
    );
    require_operation_storage(
        request.output,
        backend,
        request.element_count
    );
    require_distinct(request.input, request.output);
    adapter.copy(request);
}

void dispatch_permute(
    ExecutionBackend backend,
    const PermuteRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.axes.size() != request.input_shape.size()) {
        throw std::invalid_argument(
            "permutation must provide one axis per dimension"
        );
    }
    std::vector<bool> seen(request.axes.size(), false);
    for (const auto axis : request.axes) {
        if (axis >= request.axes.size()) {
            throw std::out_of_range(
                "permutation axis is outside tensor rank"
            );
        }
        if (seen[axis]) {
            throw std::invalid_argument(
                "permutation axes must be unique"
            );
        }
        seen[axis] = true;
    }
    const auto element_count =
        checked_shape_product(request.input_shape);
    require_operation_storage(request.input, backend, element_count);
    require_operation_storage(request.output, backend, element_count);
    require_distinct(request.input, request.output);
    adapter.permute(request);
}

void dispatch_broadcast(
    ExecutionBackend backend,
    const BroadcastRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.input_shape.size() > request.output_shape.size()) {
        throw std::invalid_argument(
            "broadcast output rank cannot be smaller than input rank"
        );
    }
    const auto rank_offset =
        request.output_shape.size() - request.input_shape.size();
    for (std::size_t input_dimension = 0;
         input_dimension < request.input_shape.size();
         ++input_dimension) {
        const auto output_dimension =
            rank_offset + input_dimension;
        if (request.input_shape[input_dimension] != 1 &&
            request.input_shape[input_dimension] !=
                request.output_shape[output_dimension]) {
            throw std::invalid_argument(
                "tensor shape is not compatible with broadcast output"
            );
        }
    }
    const auto input_size =
        checked_shape_product(request.input_shape);
    const auto output_size =
        checked_shape_product(request.output_shape);
    require_operation_storage(request.input, backend, input_size);
    require_operation_storage(request.output, backend, output_size);
    require_distinct(request.input, request.output);
    adapter.broadcast(request);
}

void dispatch_sum_to_shape(
    ExecutionBackend backend,
    const SumToShapeRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.output_shape.size() > request.input_shape.size()) {
        throw std::invalid_argument(
            "sum-to-shape output rank cannot exceed input rank"
        );
    }
    const auto rank_offset =
        request.input_shape.size() - request.output_shape.size();
    for (std::size_t output_dimension = 0;
         output_dimension < request.output_shape.size();
         ++output_dimension) {
        const auto input_dimension =
            rank_offset + output_dimension;
        if (request.output_shape[output_dimension] != 1 &&
            request.output_shape[output_dimension] !=
                request.input_shape[input_dimension]) {
            throw std::invalid_argument(
                "sum-to-shape output is not broadcast-compatible"
            );
        }
    }
    const auto input_size =
        checked_shape_product(request.input_shape);
    const auto output_size =
        checked_shape_product(request.output_shape);
    require_operation_storage(request.input, backend, input_size);
    require_operation_storage(request.output, backend, output_size);
    require_distinct(request.input, request.output);
    adapter.sum_to_shape(request);
}

void dispatch_softmax_forward(
    ExecutionBackend backend,
    const SoftmaxForwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    const auto element_count =
        checked_axis_input_size(request.dimensions);
    require_operation_storage(request.input, backend, element_count);
    require_operation_storage(
        request.probabilities,
        backend,
        element_count
    );
    require_distinct(request.input, request.probabilities);
    adapter.softmax_forward(request);
}

void dispatch_softmax_backward(
    ExecutionBackend backend,
    const SoftmaxBackwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    const auto element_count =
        checked_axis_input_size(request.dimensions);
    require_operation_storage(
        request.probabilities,
        backend,
        element_count
    );
    require_operation_storage(
        request.upstream,
        backend,
        element_count
    );
    require_operation_storage(
        request.input_gradient,
        backend,
        element_count
    );
    require_distinct(
        request.probabilities,
        request.input_gradient
    );
    require_distinct(request.upstream, request.input_gradient);
    adapter.softmax_backward(request);
}

void dispatch_causal_softmax_forward(
    ExecutionBackend backend,
    const CausalSoftmaxForwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.batch == 0 ||
        request.heads == 0 ||
        request.time == 0) {
        throw std::invalid_argument(
            "causal softmax dimensions must be positive"
        );
    }
    if (!std::isfinite(request.score_scale) ||
        request.score_scale <= 0.0F) {
        throw std::invalid_argument(
            "causal softmax scale must be finite and positive"
        );
    }
    const auto element_count = checked_product({
        request.batch,
        request.heads,
        request.time,
        request.time,
    });
    require_operation_storage(request.scores, backend, element_count);
    require_operation_storage(
        request.probabilities,
        backend,
        element_count
    );
    require_distinct(request.scores, request.probabilities);
    adapter.causal_softmax_forward(request);
}

void dispatch_causal_softmax_backward(
    ExecutionBackend backend,
    const CausalSoftmaxBackwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.batch == 0 ||
        request.heads == 0 ||
        request.time == 0) {
        throw std::invalid_argument(
            "causal softmax dimensions must be positive"
        );
    }
    if (!std::isfinite(request.score_scale) ||
        request.score_scale <= 0.0F) {
        throw std::invalid_argument(
            "causal softmax scale must be finite and positive"
        );
    }
    const auto element_count = checked_product({
        request.batch,
        request.heads,
        request.time,
        request.time,
    });
    require_operation_storage(
        request.probabilities,
        backend,
        element_count
    );
    require_operation_storage(
        request.upstream,
        backend,
        element_count
    );
    require_operation_storage(
        request.score_gradient,
        backend,
        element_count
    );
    require_distinct(
        request.probabilities,
        request.score_gradient
    );
    require_distinct(request.upstream, request.score_gradient);
    adapter.causal_softmax_backward(request);
}

void dispatch_gather_rows(
    ExecutionBackend backend,
    const GatherRowsRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.row_count == 0 ||
        request.width == 0 ||
        request.row_indices.empty()) {
        throw std::invalid_argument(
            "row gather dimensions and indices must be non-empty"
        );
    }
    for (const auto row : request.row_indices) {
        if (static_cast<std::size_t>(row) >= request.row_count) {
            throw std::out_of_range(
                "gather row index is outside the table"
            );
        }
    }
    const auto table_size = checked_product({
        request.row_count,
        request.width,
    });
    const auto output_size = checked_product({
        request.row_indices.size(),
        request.width,
    });
    require_operation_storage(request.table, backend, table_size);
    require_operation_storage(request.output, backend, output_size);
    require_distinct(request.table, request.output);
    adapter.gather_rows(request);
}

void dispatch_scatter_add_rows(
    ExecutionBackend backend,
    const ScatterAddRowsRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.row_count == 0 ||
        request.width == 0 ||
        request.row_indices.empty()) {
        throw std::invalid_argument(
            "row scatter dimensions and indices must be non-empty"
        );
    }
    for (const auto row : request.row_indices) {
        if (static_cast<std::size_t>(row) >= request.row_count) {
            throw std::out_of_range(
                "scatter row index is outside the table"
            );
        }
    }
    const auto upstream_size = checked_product({
        request.row_indices.size(),
        request.width,
    });
    const auto table_size = checked_product({
        request.row_count,
        request.width,
    });
    require_operation_storage(
        request.upstream,
        backend,
        upstream_size
    );
    require_operation_storage(
        request.table_gradient,
        backend,
        table_size
    );
    require_distinct(
        request.upstream,
        request.table_gradient
    );
    adapter.scatter_add_rows(request);
}

void dispatch_layer_norm_forward(
    ExecutionBackend backend,
    const LayerNormForwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.rows == 0 || request.width == 0) {
        throw std::invalid_argument(
            "layer normalization dimensions must be positive"
        );
    }
    if (!std::isfinite(request.epsilon) ||
        request.epsilon <= 0.0F) {
        throw std::invalid_argument(
            "layer normalization epsilon must be finite and positive"
        );
    }
    const auto element_count = checked_product({
        request.rows,
        request.width,
    });
    require_operation_storage(request.input, backend, element_count);
    require_operation_storage(request.scale, backend, request.width);
    require_operation_storage(request.bias, backend, request.width);
    require_operation_storage(request.output, backend, element_count);
    require_operation_storage(request.mean, backend, request.rows);
    require_operation_storage(
        request.inverse_standard_deviation,
        backend,
        request.rows
    );
    require_output_separation(
        {
            &request.input,
            &request.scale,
            &request.bias,
        },
        {
            &request.output,
            &request.mean,
            &request.inverse_standard_deviation,
        }
    );
    adapter.layer_norm_forward(request);
}

void dispatch_layer_norm_backward(
    ExecutionBackend backend,
    const LayerNormBackwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.rows == 0 || request.width == 0) {
        throw std::invalid_argument(
            "layer normalization dimensions must be positive"
        );
    }
    const auto element_count = checked_product({
        request.rows,
        request.width,
    });
    require_operation_storage(request.input, backend, element_count);
    require_operation_storage(request.scale, backend, request.width);
    require_operation_storage(request.mean, backend, request.rows);
    require_operation_storage(
        request.inverse_standard_deviation,
        backend,
        request.rows
    );
    require_operation_storage(request.upstream, backend, element_count);
    require_operation_storage(
        request.input_gradient,
        backend,
        element_count
    );
    require_operation_storage(
        request.scale_gradient,
        backend,
        request.width
    );
    require_operation_storage(
        request.bias_gradient,
        backend,
        request.width
    );
    require_output_separation(
        {
            &request.input,
            &request.scale,
            &request.mean,
            &request.inverse_standard_deviation,
            &request.upstream,
        },
        {
            &request.input_gradient,
            &request.scale_gradient,
            &request.bias_gradient,
        }
    );
    adapter.layer_norm_backward(request);
}

void dispatch_cross_entropy_forward(
    ExecutionBackend backend,
    const CrossEntropyForwardRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.positions == 0 || request.classes == 0) {
        throw std::invalid_argument(
            "cross entropy dimensions must be positive"
        );
    }
    if (request.targets.size() != request.positions) {
        throw std::invalid_argument(
            "cross entropy target count must match positions"
        );
    }
    for (const auto target : request.targets) {
        if (static_cast<std::size_t>(target) >= request.classes) {
            throw std::out_of_range(
                "cross entropy target is outside the vocabulary"
            );
        }
    }
    const auto logits_size = checked_product({
        request.positions,
        request.classes,
    });
    require_operation_storage(request.logits, backend, logits_size);
    require_operation_storage(request.loss, backend, 1);
    require_operation_storage(
        request.base_gradient,
        backend,
        logits_size
    );
    require_distinct(request.logits, request.loss);
    require_distinct(request.logits, request.base_gradient);
    require_distinct(request.loss, request.base_gradient);
    adapter.cross_entropy_forward(request);
}

void dispatch_adam_update(
    ExecutionBackend backend,
    const AdamUpdateRequest& request
) {
    const auto& adapter = require_backend_adapter(backend);
    require_available(adapter);
    if (request.tensors.empty()) {
        throw std::invalid_argument(
            "Adam update requires at least one tensor"
        );
    }
    if (!std::isfinite(request.learning_rate) ||
        request.learning_rate <= 0.0F ||
        !std::isfinite(request.beta1) ||
        request.beta1 <= 0.0F ||
        request.beta1 >= 1.0F ||
        !std::isfinite(request.beta2) ||
        request.beta2 <= 0.0F ||
        request.beta2 >= 1.0F ||
        !std::isfinite(request.epsilon) ||
        request.epsilon <= 0.0F ||
        !std::isfinite(request.clip_scale) ||
        request.clip_scale <= 0.0 ||
        request.clip_scale > 1.0 ||
        !std::isfinite(request.first_correction) ||
        request.first_correction <= 0.0 ||
        !std::isfinite(request.second_correction) ||
        request.second_correction <= 0.0) {
        throw std::invalid_argument(
            "Adam update received invalid scalar state"
        );
    }
    for (const auto& tensor : request.tensors) {
        const auto element_count = tensor.value.size();
        if (element_count == 0 ||
            tensor.value.backend() != backend ||
            tensor.gradient.backend() != backend ||
            tensor.first_moment.backend() != backend ||
            tensor.second_moment.backend() != backend ||
            tensor.next_value.backend() != backend ||
            tensor.next_first_moment.backend() != backend ||
            tensor.next_second_moment.backend() != backend) {
            throw std::invalid_argument(
                "Adam update tensors must share its backend"
            );
        }
        if (tensor.gradient.size() != element_count ||
            tensor.first_moment.size() != element_count ||
            tensor.second_moment.size() != element_count ||
            tensor.next_value.size() != element_count ||
            tensor.next_first_moment.size() != element_count ||
            tensor.next_second_moment.size() != element_count) {
            throw std::logic_error(
                "Adam update tensor storage sizes do not match"
            );
        }
    }
    adapter.adam_update(request);
}

}  // namespace backend_detail
}  // namespace riftco_transformer
