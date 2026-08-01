#pragma once

#include "adapter.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer::backend_detail {

// Shared closed-registry placeholder used when an optional backend was not
// compiled. Keeping recognized backends in the registry lets availability
// probes return false while actual construction fails with a precise error.
class UnavailableBackendAdapter final : public BackendAdapter {
public:
    explicit UnavailableBackendAdapter(std::string_view name) noexcept
        : name_(name) {}

    [[nodiscard]] std::string_view name() const noexcept override {
        return name_;
    }

    [[nodiscard]] bool is_available() const noexcept override {
        return false;
    }

    [[nodiscard]] std::string_view unavailability_reason()
        const noexcept override {
        return "support was not compiled into this build";
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::size_t,
        float
    ) const override {
        unavailable();
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::vector<float>
    ) const override {
        unavailable();
    }

    void matmul(const MatmulRequest&) const override { unavailable(); }
    void unary_elementwise(
        const UnaryElementwiseRequest&
    ) const override { unavailable(); }
    void binary_elementwise(
        const BinaryElementwiseRequest&
    ) const override { unavailable(); }
    void scale(const ScaleRequest&) const override { unavailable(); }
    void gelu_forward(
        const GeluForwardRequest&
    ) const override { unavailable(); }
    void gelu_backward(
        const GeluBackwardRequest&
    ) const override { unavailable(); }
    void reduce(const ReductionRequest&) const override { unavailable(); }
    void copy(const CopyRequest&) const override { unavailable(); }
    void permute(const PermuteRequest&) const override { unavailable(); }
    void broadcast(const BroadcastRequest&) const override { unavailable(); }
    void sum_to_shape(
        const SumToShapeRequest&
    ) const override { unavailable(); }
    void softmax_forward(
        const SoftmaxForwardRequest&
    ) const override { unavailable(); }
    void softmax_backward(
        const SoftmaxBackwardRequest&
    ) const override { unavailable(); }
    void causal_softmax_forward(
        const CausalSoftmaxForwardRequest&
    ) const override { unavailable(); }
    void causal_softmax_backward(
        const CausalSoftmaxBackwardRequest&
    ) const override { unavailable(); }
    void gather_rows(
        const GatherRowsRequest&
    ) const override { unavailable(); }
    void scatter_add_rows(
        const ScatterAddRowsRequest&
    ) const override { unavailable(); }
    void layer_norm_forward(
        const LayerNormForwardRequest&
    ) const override { unavailable(); }
    void layer_norm_backward(
        const LayerNormBackwardRequest&
    ) const override { unavailable(); }
    void cross_entropy_forward(
        const CrossEntropyForwardRequest&
    ) const override { unavailable(); }
    void materialized_causal_attention_forward(
        const MaterializedCausalAttentionForwardRequest&
    ) const override { unavailable(); }
    void materialized_causal_attention_context_backward(
        const MaterializedCausalAttentionContextBackwardRequest&
    ) const override { unavailable(); }
    void materialized_causal_attention_probabilities_backward(
        const MaterializedCausalAttentionProbabilitiesBackwardRequest&
    ) const override { unavailable(); }
    void flash_causal_attention_forward(
        const FlashCausalAttentionForwardRequest&
    ) const override { unavailable(); }
    void flash_causal_attention_backward(
        const FlashCausalAttentionBackwardRequest&
    ) const override { unavailable(); }
    void paged_decode_attention_forward(
        const PagedDecodeAttentionForwardRequest&
    ) const override { unavailable(); }
    void adam_update(
        const AdamUpdateRequest&
    ) const override { unavailable(); }

private:
    [[noreturn]] void unavailable() const {
        throw std::runtime_error(
            std::string(name_) + " execution backend is unavailable"
        );
    }

    std::string_view name_;
};

}  // namespace riftco_transformer::backend_detail
