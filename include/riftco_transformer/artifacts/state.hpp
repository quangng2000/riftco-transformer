#pragma once

#include "riftco_transformer/data/tokenizer.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace riftco_transformer::artifacts {

// A backend-neutral copy of one named trainable parameter.
struct ParameterState {
    std::string name;
    Tensor::Shape shape;
    std::vector<float> values;
};

// The complete model state needed to reproduce inference parameters.
struct ModelState {
    TransformerDimensions dimensions;
    float layer_norm_epsilon = 1.0e-5F;
    std::vector<ParameterState> parameters;
};

// The exact token-ID definition for either built-in tokenizer strategy.
struct TokenizerState {
    TokenizerMethod method{TokenizerMethod::CorpusByte};
    std::vector<std::uint8_t> byte_vocabulary;
    std::vector<BpeMergeRule> bpe_merges;
};

// One in-memory handoff between training stages. This deliberately contains
// no optimizer, random-engine, persistence, or artifact-identity state.
struct ModelSnapshot {
    ModelState model;
    TokenizerState tokenizer;
};

[[nodiscard]] ModelState capture_model_state(
    DecoderOnlyTransformer& model
);

// Validates the whole state and allocates every replacement before committing
// any parameter. Values stay on each target parameter's existing backend.
void load_model_state(
    DecoderOnlyTransformer& model,
    const ModelState& state
);

[[nodiscard]] TokenizerState capture_tokenizer_state(
    const TokenizerStrategy& tokenizer
);

[[nodiscard]] std::unique_ptr<TokenizerStrategy> restore_tokenizer(
    const TokenizerState& state
);

[[nodiscard]] ModelSnapshot capture_snapshot(
    DecoderOnlyTransformer& model,
    const TokenizerStrategy& tokenizer
);

}  // namespace riftco_transformer::artifacts
