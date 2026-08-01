#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer::stages::post_training {

struct InstructionExample {
    std::string prompt;
    std::string response;

    void validate() const;
};

// Explicit disjoint inputs for a generalization-aware post-training run.
// The training split is the only split sampled by the optimizer. Validation
// and test are reserved for deterministic read-only evaluation.
struct InstructionSplits {
    std::vector<InstructionExample> train;
    std::vector<InstructionExample> validation;
    std::vector<InstructionExample> test;

    InstructionSplits(
        std::vector<InstructionExample> training_examples,
        std::vector<InstructionExample> validation_examples,
        std::vector<InstructionExample> test_examples
    );

    void validate() const;
};

// Strategy boundary for chat templates. A post-training stage can change the
// presentation format without changing batching, autograd, or optimization.
class InstructionFormatter {
public:
    virtual ~InstructionFormatter() = default;

    [[nodiscard]] virtual std::string format(
        const InstructionExample& example
    ) const = 0;
};

class PlainChatFormatter final : public InstructionFormatter {
public:
    [[nodiscard]] std::string format(
        const InstructionExample& example
    ) const override;
};

inline constexpr std::string_view kFullSequenceCausalObjective =
    "full_sequence_causal_sft";

}  // namespace riftco_transformer::stages::post_training
