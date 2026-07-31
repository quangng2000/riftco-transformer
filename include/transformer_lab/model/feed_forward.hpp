#pragma once

#include "transformer_lab/nn/linear.hpp"
#include "transformer_lab/nn/module.hpp"

#include <cstddef>
#include <random>

namespace transformer_lab {

class DecoderOnlyTransformer;

class FeedForward : public Module {
public:
    FeedForward(
        std::size_t model_width,
        std::size_t hidden_width,
        std::mt19937& random
    );

    [[nodiscard]] std::size_t model_width() const noexcept;
    [[nodiscard]] std::size_t hidden_width() const noexcept;
    [[nodiscard]] Variable forward(const Variable& input) const;
    // Transfers parameters in place. Call before building a forward graph.
    void to(ExecutionBackend backend);

    [[nodiscard]] ParameterList parameters();
    [[nodiscard]] ParameterList lora_parameters();

private:
    Linear expand_;
    Linear project_;

    friend class DecoderOnlyTransformer;
};

}  // namespace transformer_lab
