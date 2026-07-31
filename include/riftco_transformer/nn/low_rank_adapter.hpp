#pragma once

#include "riftco_transformer/nn/module.hpp"

#include <cstddef>
#include <random>

namespace riftco_transformer {

// A bias-free LoRA branch for a Linear projection. Its matrices use the
// conventional orientations A=[rank,input] and B=[output,rank].
class LowRankAdapter final : public Module {
public:
    LowRankAdapter(
        std::size_t input_width,
        std::size_t output_width,
        std::size_t rank,
        float alpha,
        std::mt19937& random,
        ExecutionBackend backend
    );

    LowRankAdapter(const LowRankAdapter&) = delete;
    LowRankAdapter& operator=(const LowRankAdapter&) = delete;
    LowRankAdapter(LowRankAdapter&&) = delete;
    LowRankAdapter& operator=(LowRankAdapter&&) = delete;

    [[nodiscard]] std::size_t input_width() const noexcept;
    [[nodiscard]] std::size_t output_width() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] float alpha() const noexcept;
    [[nodiscard]] float scale() const noexcept;

    [[nodiscard]] Variable forward(const Variable& input) const;
    [[nodiscard]] Tensor weight_delta() const;
    void to(ExecutionBackend backend);

    [[nodiscard]] const Parameter& a() const noexcept;
    [[nodiscard]] const Parameter& b() const noexcept;
    [[nodiscard]] ParameterList parameters();

private:
    class WeightModule final : public Module {
    public:
        explicit WeightModule(Parameter& parameter) {
            register_parameter("weight", parameter);
        }
    };

    std::size_t input_width_;
    std::size_t output_width_;
    std::size_t rank_;
    float alpha_;
    float scale_;
    Parameter a_;
    Parameter b_;
    WeightModule lora_a_;
    WeightModule lora_b_;
};

}  // namespace riftco_transformer
