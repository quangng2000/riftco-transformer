#include "adapter.hpp"
#include "metal_diagnostics.hpp"

#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace transformer_lab::backend_detail {
namespace {

[[noreturn]] void throw_metal_unavailable() {
    throw std::runtime_error(
        "metal execution backend is unavailable"
    );
}

class UnavailableMetalBackendAdapter final : public BackendAdapter {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "metal";
    }

    [[nodiscard]] bool is_available() const noexcept override {
        return false;
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::size_t,
        float
    ) const override {
        throw_metal_unavailable();
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::vector<float>
    ) const override {
        throw_metal_unavailable();
    }

    void matmul(const MatmulRequest&) const override {
        throw_metal_unavailable();
    }

    void unary_elementwise(
        const UnaryElementwiseRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void binary_elementwise(
        const BinaryElementwiseRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void scale(const ScaleRequest&) const override {
        throw_metal_unavailable();
    }

    void gelu_forward(const GeluForwardRequest&) const override {
        throw_metal_unavailable();
    }

    void gelu_backward(const GeluBackwardRequest&) const override {
        throw_metal_unavailable();
    }

    void reduce(const ReductionRequest&) const override {
        throw_metal_unavailable();
    }

    void copy(const CopyRequest&) const override {
        throw_metal_unavailable();
    }

    void permute(const PermuteRequest&) const override {
        throw_metal_unavailable();
    }

    void broadcast(const BroadcastRequest&) const override {
        throw_metal_unavailable();
    }

    void sum_to_shape(
        const SumToShapeRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void softmax_forward(
        const SoftmaxForwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void softmax_backward(
        const SoftmaxBackwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void causal_softmax_forward(
        const CausalSoftmaxForwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void causal_softmax_backward(
        const CausalSoftmaxBackwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void gather_rows(const GatherRowsRequest&) const override {
        throw_metal_unavailable();
    }

    void scatter_add_rows(
        const ScatterAddRowsRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void layer_norm_forward(
        const LayerNormForwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void layer_norm_backward(
        const LayerNormBackwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void cross_entropy_forward(
        const CrossEntropyForwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void materialized_causal_attention_forward(
        const MaterializedCausalAttentionForwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void materialized_causal_attention_context_backward(
        const MaterializedCausalAttentionContextBackwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void materialized_causal_attention_probabilities_backward(
        const MaterializedCausalAttentionProbabilitiesBackwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void flash_causal_attention_forward(
        const FlashCausalAttentionForwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void flash_causal_attention_backward(
        const FlashCausalAttentionBackwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void paged_decode_attention_forward(
        const PagedDecodeAttentionForwardRequest&
    ) const override {
        throw_metal_unavailable();
    }

    void adam_update(const AdamUpdateRequest&) const override {
        throw_metal_unavailable();
    }
};

}  // namespace

const BackendAdapter& metal_backend_adapter() noexcept {
    static const UnavailableMetalBackendAdapter adapter;
    return adapter;
}

void reset_metal_adam_path_counts() noexcept {}

MetalAdamPathCounts metal_adam_path_counts() noexcept {
    return {0, 0};
}

}  // namespace transformer_lab::backend_detail
