#include "riftco_transformer/stages/post_training/evaluation.hpp"

#include "riftco_transformer/data/token_batch.hpp"
#include "riftco_transformer/nn/loss.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::stages::post_training {
namespace {

struct CausalChunk {
    std::vector<TokenId> inputs;
    std::vector<TokenId> targets;
};

void checked_increment(
    std::size_t& value,
    std::size_t increment,
    const char* description
) {
    if (value > std::numeric_limits<std::size_t>::max() - increment) {
        throw std::overflow_error(description);
    }
    value += increment;
}

double loss_perplexity(double loss) {
    const double maximum_exponent =
        std::log(std::numeric_limits<double>::max());
    return loss > maximum_exponent
               ? std::numeric_limits<double>::infinity()
               : std::exp(loss);
}

}  // namespace

CausalEvaluationMetrics evaluate_causal_sequences(
    DecoderOnlyTransformer& model,
    std::span<const std::vector<TokenId>> sequences,
    std::size_t context_size,
    std::size_t batch_size
) {
    if (sequences.empty()) {
        throw std::invalid_argument(
            "causal evaluation requires at least one example"
        );
    }
    if (context_size == 0 || batch_size == 0) {
        throw std::invalid_argument(
            "causal evaluation context and batch sizes must be positive"
        );
    }
    if (context_size > model.dimensions().maximum_context) {
        throw std::invalid_argument(
            "causal evaluation context exceeds the model maximum context"
        );
    }

    CausalEvaluationMetrics result;
    result.example_count = sequences.size();
    long double weighted_loss_sum = 0.0L;
    std::map<std::size_t, std::vector<CausalChunk>> pending_by_width;
    std::size_t pending_chunk_count = 0;

    const auto evaluate_pending = [&](std::vector<CausalChunk>& pending) {
        if (pending.empty()) {
            return;
        }
        const std::size_t width = pending.front().inputs.size();
        const std::size_t row_count = pending.size();
        if (row_count >
            std::numeric_limits<std::size_t>::max() / width) {
            throw std::overflow_error(
                "causal evaluation batch shape exceeds addressable size"
            );
        }
        const std::size_t evaluated_tokens = row_count * width;
        std::vector<TokenId> inputs;
        std::vector<TokenId> targets;
        inputs.reserve(evaluated_tokens);
        targets.reserve(evaluated_tokens);
        for (const auto& chunk : pending) {
            if (chunk.inputs.size() != width ||
                chunk.targets.size() != width) {
                throw std::logic_error(
                    "causal evaluation grouped incompatible chunk widths"
                );
            }
            inputs.insert(
                inputs.end(),
                chunk.inputs.begin(),
                chunk.inputs.end()
            );
            targets.insert(
                targets.end(),
                chunk.targets.begin(),
                chunk.targets.end()
            );
        }

        const TokenBatch batch(
            row_count,
            width,
            std::move(inputs),
            std::move(targets)
        );
        const Variable logits = model.forward(
            batch.inputs(),
            {batch.batch_size(), batch.context_size()}
        );
        const Variable loss = cross_entropy(logits, batch.targets());
        if (loss.value().rank() != 0 || loss.value().numel() != 1) {
            throw std::logic_error(
                "causal evaluation loss must be scalar"
            );
        }
        const double loss_value =
            static_cast<double>(loss.value().flat(0));
        if (!std::isfinite(loss_value)) {
            throw std::domain_error(
                "causal evaluation loss must be finite"
            );
        }
        weighted_loss_sum +=
            static_cast<long double>(loss_value) *
            static_cast<long double>(evaluated_tokens);
        checked_increment(
            result.forward_batch_count,
            1,
            "causal evaluation forward-batch count overflow"
        );
        if (pending_chunk_count < row_count) {
            throw std::logic_error(
                "causal evaluation pending count is inconsistent"
            );
        }
        pending_chunk_count -= row_count;
        pending.clear();
    };

    for (const auto& sequence : sequences) {
        if (sequence.size() < 2) {
            checked_increment(
                result.skipped_example_count,
                1,
                "causal evaluation skipped-example count overflow"
            );
            continue;
        }
        checked_increment(
            result.usable_example_count,
            1,
            "causal evaluation usable-example count overflow"
        );
        for (std::size_t start = 0;
             start < sequence.size() - 1;
             start += context_size) {
            const std::size_t remaining_targets =
                sequence.size() - 1 - start;
            const std::size_t target_count =
                std::min(context_size, remaining_targets);
            const auto input_begin = sequence.begin() +
                static_cast<std::ptrdiff_t>(start);
            const auto target_begin = input_begin + 1;
            CausalChunk chunk{
                std::vector<TokenId>(
                    input_begin,
                    input_begin +
                        static_cast<std::ptrdiff_t>(target_count)
                ),
                std::vector<TokenId>(
                    target_begin,
                    target_begin +
                        static_cast<std::ptrdiff_t>(target_count)
                ),
            };
            auto& pending = pending_by_width[target_count];
            pending.push_back(std::move(chunk));
            checked_increment(
                pending_chunk_count,
                1,
                "causal evaluation pending-chunk count overflow"
            );
            if (pending.size() == batch_size) {
                evaluate_pending(pending);
            } else if (pending_chunk_count >= batch_size) {
                // Bound ragged-tail storage instead of retaining a partial
                // batch for every possible chunk width.
                const auto fullest = std::max_element(
                    pending_by_width.begin(),
                    pending_by_width.end(),
                    [](const auto& left, const auto& right) {
                        return left.second.size() < right.second.size();
                    }
                );
                evaluate_pending(fullest->second);
            }
            checked_increment(
                result.target_token_count,
                target_count,
                "causal evaluation target-token count overflow"
            );
            checked_increment(
                result.chunk_count,
                1,
                "causal evaluation chunk count overflow"
            );
        }
    }

    for (auto& [width, pending] : pending_by_width) {
        static_cast<void>(width);
        evaluate_pending(pending);
    }
    if (result.target_token_count == 0) {
        throw std::invalid_argument(
            "causal evaluation has no example with a target token"
        );
    }
    if (result.usable_example_count + result.skipped_example_count !=
        result.example_count) {
        throw std::logic_error(
            "causal evaluation example counts are inconsistent"
        );
    }
    result.loss = static_cast<double>(
        weighted_loss_sum /
        static_cast<long double>(result.target_token_count)
    );
    result.perplexity = loss_perplexity(result.loss);
    return result;
}

}  // namespace riftco_transformer::stages::post_training
