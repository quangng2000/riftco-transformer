#include "core/backend/adapter.hpp"
#include "core/backend/adapters/tpu/runtime.hpp"
#include "core/backend/attention/reference/flash_causal.hpp"
#include "core/backend/attention/tpu/materialized_causal.hpp"
#include "core/backend/attention/tpu/paged_decode.hpp"
#include "core/backend/nn/reference/operations.hpp"
#include "core/backend/optim/adam/reference/update.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

// The public Tensor API intentionally exposes host-readable values. Keeping a
// host mirror here preserves that contract and lets the audited reference
// capabilities remain usable while selected PJRT programs stage through TPU
// memory.
class TpuHostTensorStorage final : public TensorStorage {
public:
    TpuHostTensorStorage(std::size_t element_count, float fill_value)
        : values_(element_count, fill_value) {}

    explicit TpuHostTensorStorage(std::vector<float> values)
        : values_(std::move(values)) {}

    [[nodiscard]] ExecutionBackend backend() const noexcept override {
        return ExecutionBackend::Tpu;
    }

    [[nodiscard]] std::size_t size() const noexcept override {
        return values_.size();
    }

    [[nodiscard]] std::span<float> data() noexcept override { return values_; }

    [[nodiscard]] std::span<const float> data() const noexcept override {
        return values_;
    }

    [[nodiscard]] void* native_handle() noexcept override { return nullptr; }

    [[nodiscard]] const void* native_handle() const noexcept override {
        return nullptr;
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> clone() const override {
        return std::make_unique<TpuHostTensorStorage>(values_);
    }

private:
    std::vector<float> values_;
};

class TpuBackendAdapter final : public BackendAdapter {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "tpu";
    }

    [[nodiscard]] bool is_available() const noexcept override {
        return tpu_runtime_available();
    }

    [[nodiscard]] std::string_view
    unavailability_reason() const noexcept override {
        return tpu_runtime_unavailability_reason();
    }

    [[nodiscard]] std::unique_ptr<TensorStorage>
    make_storage(std::size_t element_count, float fill_value) const override {
        return std::make_unique<TpuHostTensorStorage>(
            element_count, fill_value);
    }

    [[nodiscard]] std::unique_ptr<TensorStorage>
    make_storage(std::vector<float> values) const override {
        return std::make_unique<TpuHostTensorStorage>(std::move(values));
    }

    void matmul(const MatmulRequest& request) const override {
        tpu_runtime_matmul(request);
    }

    // Dense matmul, materialized training attention, and paged decode use
    // StableHLO/PJRT. The remaining capabilities run synchronously against the
    // authoritative host mirror. Flash remains on the reference path until its
    // TPU graph can preserve the contract's memory-linear implementation.
    void
    unary_elementwise(const UnaryElementwiseRequest& request) const override {
        nn_reference_unary_elementwise(request);
    }
    void
    binary_elementwise(const BinaryElementwiseRequest& request) const override {
        nn_reference_binary_elementwise(request);
    }
    void scale(const ScaleRequest& request) const override {
        nn_reference_scale(request);
    }
    void gelu_forward(const GeluForwardRequest& request) const override {
        nn_reference_gelu_forward(request);
    }
    void gelu_backward(const GeluBackwardRequest& request) const override {
        nn_reference_gelu_backward(request);
    }
    void reduce(const ReductionRequest& request) const override {
        nn_reference_reduce(request);
    }
    void copy(const CopyRequest& request) const override {
        nn_reference_copy(request);
    }
    void permute(const PermuteRequest& request) const override {
        nn_reference_permute(request);
    }
    void broadcast(const BroadcastRequest& request) const override {
        nn_reference_broadcast(request);
    }
    void sum_to_shape(const SumToShapeRequest& request) const override {
        nn_reference_sum_to_shape(request);
    }
    void softmax_forward(const SoftmaxForwardRequest& request) const override {
        nn_reference_softmax_forward(request);
    }
    void
    softmax_backward(const SoftmaxBackwardRequest& request) const override {
        nn_reference_softmax_backward(request);
    }
    void causal_softmax_forward(
        const CausalSoftmaxForwardRequest& request) const override {
        nn_reference_causal_softmax_forward(request);
    }
    void causal_softmax_backward(
        const CausalSoftmaxBackwardRequest& request) const override {
        nn_reference_causal_softmax_backward(request);
    }
    void gather_rows(const GatherRowsRequest& request) const override {
        nn_reference_gather_rows(request);
    }
    void scatter_add_rows(const ScatterAddRowsRequest& request) const override {
        nn_reference_scatter_add_rows(request);
    }
    void
    layer_norm_forward(const LayerNormForwardRequest& request) const override {
        nn_reference_layer_norm_forward(request);
    }
    void layer_norm_backward(
        const LayerNormBackwardRequest& request) const override {
        nn_reference_layer_norm_backward(request);
    }
    void cross_entropy_forward(
        const CrossEntropyForwardRequest& request) const override {
        nn_reference_cross_entropy_forward(request);
    }
    void materialized_causal_attention_forward(
        const MaterializedCausalAttentionForwardRequest& request)
        const override {
        tpu_materialized_causal_attention_forward(request);
    }
    void materialized_causal_attention_context_backward(
        const MaterializedCausalAttentionContextBackwardRequest& request)
        const override {
        tpu_materialized_causal_attention_context_backward(request);
    }
    void materialized_causal_attention_probabilities_backward(
        const MaterializedCausalAttentionProbabilitiesBackwardRequest& request)
        const override {
        tpu_materialized_causal_attention_probabilities_backward(request);
    }
    void flash_causal_attention_forward(
        const FlashCausalAttentionForwardRequest& request) const override {
        reference_flash_causal_attention_forward(request);
    }
    void flash_causal_attention_backward(
        const FlashCausalAttentionBackwardRequest& request) const override {
        reference_flash_causal_attention_backward(request);
    }
    void paged_decode_attention_forward(
        const PagedDecodeAttentionForwardRequest& request) const override {
        tpu_paged_decode_attention_forward(request);
    }
    void adam_update(const AdamUpdateRequest& request) const override {
        adam_reference_update(request);
    }
};

}  // namespace

const BackendAdapter& tpu_backend_adapter() noexcept {
    static const TpuBackendAdapter adapter;
    return adapter;
}

}  // namespace riftco_transformer::backend_detail
