#include "riftco_transformer/config.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("expected path to tiny.conf");
        }

        const auto config = riftco_transformer::Config::load(argv[1]);
        require(config.context_size == 16, "unexpected maximum context");
        require(config.d_model == 32, "unexpected model width");
        require(config.n_heads == 4, "unexpected attention head count");
        require(config.n_layers == 2, "unexpected transformer block count");
        require(config.d_ff == 64, "unexpected feed-forward width");
        require(config.d_model % config.n_heads == 0, "invalid head width");
        require(config.corpus_path.filename() == "tiny_corpus.txt",
                "corpus path was not resolved");
        require(std::filesystem::is_regular_file(config.corpus_path),
                "tiny corpus is missing");

        auto invalid = config;
        invalid.n_heads = 5;
        bool rejected = false;
        try {
            invalid.validate();
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "invalid head dimensions were accepted");

        invalid = config;
        invalid.learning_rate =
            std::numeric_limits<float>::quiet_NaN();
        rejected = false;
        try {
            invalid.validate();
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "NaN learning rate was accepted");

        invalid = config;
        invalid.gradient_clip =
            std::numeric_limits<float>::infinity();
        rejected = false;
        try {
            invalid.validate();
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "infinite gradient clip was accepted");

        std::cout << "config tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
