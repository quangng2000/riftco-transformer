#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace riftco_transformer {

struct Config {
    std::filesystem::path corpus_path;
    std::filesystem::path results_dir;

    std::uint32_t seed = 42;
    std::size_t context_size = 16;
    std::size_t batch_size = 4;
    std::size_t d_model = 32;
    std::size_t n_heads = 4;
    std::size_t n_layers = 2;
    std::size_t d_ff = 64;
    std::size_t training_steps = 500;
    std::size_t sample_every = 100;
    std::size_t sample_length = 120;

    float learning_rate = 0.001F;
    float adam_beta1 = 0.9F;
    float adam_beta2 = 0.999F;
    float adam_epsilon = 1.0e-8F;
    float gradient_clip = 1.0F;

    [[nodiscard]] static Config load(const std::filesystem::path& path);
    void validate() const;
    [[nodiscard]] std::string summary() const;
};

}  // namespace riftco_transformer
