#include "transformer_lab/nn/loss.hpp"

#include "core/backend/adapter.hpp"
#include "transformer_lab/core/tensor_ops.hpp"

#include <memory>
#include <stdexcept>

namespace transformer_lab {

Variable
cross_entropy(const Variable& logits, std::span<const TokenId> targets) {
    if (logits.value().rank() == 0) {
        throw std::invalid_argument(
            "cross entropy logits require a vocabulary dimension"
        );
    }

    const auto vocabulary_size = logits.value().shape().back();
    const auto position_count = logits.value().numel() / vocabulary_size;
    if (targets.size() != position_count) {
        throw std::invalid_argument(
            "cross entropy target count must match leading logits positions"
        );
    }
    for (const TokenId target : targets) {
        if (static_cast<std::size_t>(target) >= vocabulary_size) {
            throw std::out_of_range(
                "cross entropy target is outside the vocabulary"
            );
        }
    }

    Tensor loss(Tensor::Shape{}, logits.value().backend());
    Tensor base_gradient(logits.value().shape(), logits.value().backend());
    backend_detail::dispatch_cross_entropy_forward(
        logits.value().backend(),
        {
            backend_detail::tensor_storage(logits.value()),
            targets,
            backend_detail::tensor_storage(loss),
            backend_detail::tensor_storage(base_gradient),
            position_count,
            vocabulary_size,
        }
    );

    const auto logits_node = logits.node_;
    auto saved_base_gradient =
        std::make_shared<Tensor>(std::move(base_gradient));
    return Variable::from_operation(
        std::move(loss),
        {logits_node},
        [logits_node, saved_base_gradient](const Tensor& upstream) {
            Variable::accumulate_gradient(
                logits_node,
                tensor_ops::scale(*saved_base_gradient, upstream.flat(0))
            );
        }
    );
}

}  // namespace transformer_lab
