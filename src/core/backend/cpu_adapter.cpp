#include "adapter.hpp"
#include "adam_reference.hpp"
#include "attention/reference/flash_causal.hpp"
#include "attention/reference/materialized_causal.hpp"
#include "attention/reference/paged_decode.hpp"
#include "nn_reference.hpp"

#include <cstddef>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

class CpuTensorStorage final : public TensorStorage {
public:
    CpuTensorStorage(
        std::size_t element_count,
        float fill_value
    )
        : values_(element_count, fill_value) {}

    explicit CpuTensorStorage(std::vector<float> values)
        : values_(std::move(values)) {}

    [[nodiscard]] ExecutionBackend backend() const noexcept override {
        return ExecutionBackend::Cpu;
    }

    [[nodiscard]] std::size_t size() const noexcept override {
        return values_.size();
    }

    [[nodiscard]] std::span<float> data() noexcept override {
        return values_;
    }

    [[nodiscard]] std::span<const float> data() const noexcept override {
        return values_;
    }

    [[nodiscard]] void* native_handle() noexcept override {
        return nullptr;
    }

    [[nodiscard]] const void* native_handle() const noexcept override {
        return nullptr;
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> clone() const override {
        return std::make_unique<CpuTensorStorage>(values_);
    }

private:
    std::vector<float> values_;
};

class CpuBackendAdapter final : public BackendAdapter {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "cpu";
    }

    [[nodiscard]] bool is_available() const noexcept override {
        return true;
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::size_t element_count,
        float fill_value
    ) const override {
        return std::make_unique<CpuTensorStorage>(
            element_count,
            fill_value
        );
    }

    [[nodiscard]] std::unique_ptr<TensorStorage> make_storage(
        std::vector<float> values
    ) const override {
        return std::make_unique<CpuTensorStorage>(
            std::move(values)
        );
    }

    void matmul(const MatmulRequest& request) const override {
        const auto left = request.left.data();
        const auto right = request.right.data();
        auto output = request.output.data();
        const auto& dimensions = request.dimensions;
        for (std::size_t batch = 0;
             batch < dimensions.batch_count;
             ++batch) {
            const std::size_t left_offset =
                batch * dimensions.rows * dimensions.shared;
            const std::size_t right_offset =
                batch * dimensions.shared * dimensions.columns;
            const std::size_t output_offset =
                batch * dimensions.rows * dimensions.columns;

            for (std::size_t row = 0;
                 row < dimensions.rows;
                 ++row) {
                for (std::size_t column = 0;
                     column < dimensions.columns;
                     ++column) {
                    float total = 0.0F;
                    for (std::size_t inner = 0;
                         inner < dimensions.shared;
                         ++inner) {
                        total +=
                            left[
                                left_offset +
                                row * dimensions.shared +
                                inner
                            ] *
                            right[
                                right_offset +
                                inner * dimensions.columns +
                                column
                            ];
                    }
                    output[
                        output_offset +
                        row * dimensions.columns +
                        column
                    ] = total;
                }
            }
        }
    }

    void unary_elementwise(
        const UnaryElementwiseRequest& request
    ) const override {
        nn_reference_unary_elementwise(request);
    }

    void binary_elementwise(
        const BinaryElementwiseRequest& request
    ) const override {
        nn_reference_binary_elementwise(request);
    }

    void scale(const ScaleRequest& request) const override {
        nn_reference_scale(request);
    }

    void gelu_forward(
        const GeluForwardRequest& request
    ) const override {
        nn_reference_gelu_forward(request);
    }

    void gelu_backward(
        const GeluBackwardRequest& request
    ) const override {
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

    void sum_to_shape(
        const SumToShapeRequest& request
    ) const override {
        nn_reference_sum_to_shape(request);
    }

    void softmax_forward(
        const SoftmaxForwardRequest& request
    ) const override {
        nn_reference_softmax_forward(request);
    }

    void softmax_backward(
        const SoftmaxBackwardRequest& request
    ) const override {
        nn_reference_softmax_backward(request);
    }

    void causal_softmax_forward(
        const CausalSoftmaxForwardRequest& request
    ) const override {
        nn_reference_causal_softmax_forward(request);
    }

    void causal_softmax_backward(
        const CausalSoftmaxBackwardRequest& request
    ) const override {
        nn_reference_causal_softmax_backward(request);
    }

    void gather_rows(
        const GatherRowsRequest& request
    ) const override {
        nn_reference_gather_rows(request);
    }

    void scatter_add_rows(
        const ScatterAddRowsRequest& request
    ) const override {
        nn_reference_scatter_add_rows(request);
    }

    void layer_norm_forward(
        const LayerNormForwardRequest& request
    ) const override {
        nn_reference_layer_norm_forward(request);
    }

    void layer_norm_backward(
        const LayerNormBackwardRequest& request
    ) const override {
        nn_reference_layer_norm_backward(request);
    }

    void cross_entropy_forward(
        const CrossEntropyForwardRequest& request
    ) const override {
        nn_reference_cross_entropy_forward(request);
    }

    void materialized_causal_attention_forward(
        const MaterializedCausalAttentionForwardRequest& request
    ) const override {
        reference_materialized_causal_attention_forward(request);
    }

    void materialized_causal_attention_context_backward(
        const MaterializedCausalAttentionContextBackwardRequest& request
    ) const override {
        reference_materialized_causal_attention_context_backward(request);
    }

    void materialized_causal_attention_probabilities_backward(
        const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
    ) const override {
        reference_materialized_causal_attention_probabilities_backward(
            request
        );
    }

    void flash_causal_attention_forward(
        const FlashCausalAttentionForwardRequest& request
    ) const override {
        reference_flash_causal_attention_forward(request);
    }

    void flash_causal_attention_backward(
        const FlashCausalAttentionBackwardRequest& request
    ) const override {
        reference_flash_causal_attention_backward(request);
    }

    void paged_decode_attention_forward(
        const PagedDecodeAttentionForwardRequest& request
    ) const override {
        reference_paged_decode_attention_forward(request);
    }

    void adam_update(
        const AdamUpdateRequest& request
    ) const override {
        adam_reference_update(request);
    }
};

}  // namespace

const BackendAdapter& cpu_backend_adapter() noexcept {
    static const CpuBackendAdapter adapter;
    return adapter;
}

}  // namespace riftco_transformer::backend_detail
