#pragma once

#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <cstdint>
#include <random>

namespace riftco_transformer {

class DecoderOnlyTransformer;

enum class FeedForwardActivation : std::uint8_t {
  Gelu,
  Relu,
};

class FeedForward : public Module {
public:
    FeedForward(
        std::size_t model_width,
        std::size_t hidden_width,
        std::mt19937& random
    );
    FeedForward(std::size_t model_width, std::size_t hidden_width,
                std::mt19937 &random, FeedForwardActivation activation);

    [[nodiscard]] std::size_t model_width() const noexcept;
    [[nodiscard]] std::size_t hidden_width() const noexcept;
    [[nodiscard]] FeedForwardActivation activation() const noexcept;
    [[nodiscard]] Variable forward(const Variable& input) const;
    // Transfers parameters in place. Call before building a forward graph.
    void to(ExecutionBackend backend);

    [[nodiscard]] ParameterList parameters();
    [[nodiscard]] ParameterList lora_parameters();

private:
  FeedForwardActivation activation_;
  Linear expand_;
  Linear project_;

  friend class DecoderOnlyTransformer;
};

}  // namespace riftco_transformer
