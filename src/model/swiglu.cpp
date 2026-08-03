#include "riftco_transformer/model/swiglu.hpp"

#include "riftco_transformer/nn/activations.hpp"

namespace riftco_transformer {

SwiGLU::SwiGLU(
    std::size_t model_width,
    std::size_t hidden_width,
    std::mt19937& random
) : gate_(model_width, hidden_width, random, false),
    up_(model_width, hidden_width, random, false),
    down_(hidden_width, model_width, random, false) {
    register_module("gate", gate_);
    register_module("up", up_);
    register_module("down", down_);
}

std::size_t SwiGLU::model_width() const noexcept {
    return gate_.input_width();
}

std::size_t SwiGLU::hidden_width() const noexcept {
    return gate_.output_width();
}

Variable SwiGLU::forward(const Variable& input) const {
    return down_.forward(silu(gate_.forward(input)) * up_.forward(input));
}

void SwiGLU::to(ExecutionBackend backend) {
    Module::to(backend);
}

ParameterList SwiGLU::parameters() {
    return Module::parameters();
}

}  // namespace riftco_transformer
