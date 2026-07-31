#include "riftco_transformer/model/feed_forward.hpp"

#include "riftco_transformer/nn/activations.hpp"

namespace riftco_transformer {

FeedForward::FeedForward(
    std::size_t model_width,
    std::size_t hidden_width,
    std::mt19937& random
)
    : expand_(model_width, hidden_width, random),
      project_(hidden_width, model_width, random) {
    register_module("expand", expand_);
    register_module("project", project_);
}

std::size_t FeedForward::model_width() const noexcept {
    return expand_.input_width();
}

std::size_t FeedForward::hidden_width() const noexcept {
    return expand_.output_width();
}

Variable FeedForward::forward(const Variable& input) const {
    return project_.forward(gelu(expand_.forward(input)));
}

void FeedForward::to(ExecutionBackend backend) {
    Module::to(backend);
}

ParameterList FeedForward::parameters() {
    return Module::parameters();
}

ParameterList FeedForward::lora_parameters() {
    ParameterList result;
    append_parameter_group(
        result,
        "expand",
        expand_.lora_parameters()
    );
    append_parameter_group(
        result,
        "project",
        project_.lora_parameters()
    );
    return result;
}

}  // namespace riftco_transformer
