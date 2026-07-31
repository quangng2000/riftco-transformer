#pragma once

#include <string>
#include <string_view>

namespace transformer_lab::stages::post_training {

struct InstructionExample {
    std::string prompt;
    std::string response;

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

}  // namespace transformer_lab::stages::post_training
