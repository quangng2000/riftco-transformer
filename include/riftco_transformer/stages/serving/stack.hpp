#pragma once

#include "riftco_transformer/artifacts/state.hpp"
#include "riftco_transformer/stages/serving/config.hpp"
#include "riftco_transformer/stages/serving/generation.hpp"

#include <memory>
#include <string_view>

namespace riftco_transformer {
class DecoderOnlyTransformer;
class TokenizerStrategy;
}

namespace riftco_transformer::stages::serving {

// Stage-3 composition root. It restores only the model/tokenizer inference
// state and deliberately has no dependency on training or optimizers.
class ServingStack final {
public:
    explicit ServingStack(
        const artifacts::ModelSnapshot& snapshot,
        ServingConfig config = {}
    );
    ~ServingStack();

    ServingStack(const ServingStack&) = delete;
    ServingStack& operator=(const ServingStack&) = delete;
    ServingStack(ServingStack&&) = delete;
    ServingStack& operator=(ServingStack&&) = delete;

    [[nodiscard]] const ServingConfig& config() const noexcept;
    [[nodiscard]] DecoderOnlyTransformer& model() noexcept;
    [[nodiscard]] const DecoderOnlyTransformer& model() const noexcept;
    [[nodiscard]] TokenizerStrategy& tokenizer() noexcept;
    [[nodiscard]] const TokenizerStrategy& tokenizer() const noexcept;

    [[nodiscard]] GenerationResult generate(
        std::string_view prompt,
        GenerationConfig generation = {}
    );
    [[nodiscard]] GenerationResult generate(
        std::string_view prompt,
        SamplingStrategy& sampler,
        GenerationConfig generation = {}
    );

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;

    void validate_request(const GenerationConfig& generation) const;
};

}  // namespace riftco_transformer::stages::serving
