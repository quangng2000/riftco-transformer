#include "riftco_transformer/nn/loss.hpp"

#include "core/backend/adapter.hpp"
#include "core/backend/nn/dispatch.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace riftco_transformer {

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

Variable cross_entropy_time_range(
    const Variable& logits,
    std::span<const TokenId> targets,
    std::size_t time_offset,
    std::size_t time_count
) {
    if (logits.value().rank() != 3) {
        throw std::invalid_argument(
            "time-range cross entropy requires "
            "[batch, time, vocabulary] logits"
        );
    }
    const auto& shape = logits.value().shape();
    const std::size_t batch = shape[0];
    const std::size_t time = shape[1];
    const std::size_t vocabulary = shape[2];
    if (batch == 0 || time == 0 || vocabulary == 0 || time_count == 0) {
        throw std::invalid_argument(
            "time-range cross entropy dimensions must be positive"
        );
    }
    if (time_offset > time || time_count > time - time_offset) {
        throw std::out_of_range(
            "time-range cross entropy selection is outside logits"
        );
    }
    if (batch > std::numeric_limits<std::size_t>::max() / time) {
        throw std::overflow_error(
            "time-range cross entropy target count overflows"
        );
    }
    const std::size_t expected_targets = batch * time;
    if (targets.size() != expected_targets) {
        throw std::invalid_argument(
            "time-range cross entropy targets must match [batch, time]"
        );
    }
    if (batch > std::numeric_limits<std::size_t>::max() / time_count) {
        throw std::overflow_error(
            "time-range cross entropy selected count overflows"
        );
    }

    const std::size_t selected_count = batch * time_count;
    std::vector<std::size_t> selected_rows;
    std::vector<TokenId> selected_targets;
    selected_rows.reserve(selected_count);
    selected_targets.reserve(selected_count);
    for (std::size_t row = 0; row < batch; ++row) {
        for (std::size_t offset = 0; offset < time_count; ++offset) {
            const std::size_t index = row * time + time_offset + offset;
            selected_rows.push_back(index);
            selected_targets.push_back(targets[index]);
        }
    }
    const Variable selected_logits = gather_rows(
        reshape(logits, {expected_targets, vocabulary}),
        selected_rows,
        {batch, time_count}
    );
    return cross_entropy(selected_logits, selected_targets);
}

}  // namespace riftco_transformer
