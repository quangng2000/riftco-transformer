#include "transformer_lab/config.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace transformer_lab {
namespace {

std::string trim(std::string value) {
    constexpr std::string_view whitespace = " \t\r\n";
    const auto first = value.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

std::size_t parse_size(const std::string& key, const std::string& value) {
    std::size_t result = 0;
    const auto* begin = value.data();
    const auto* end = begin + value.size();
    const auto [cursor, error] = std::from_chars(begin, end, result);
    if (error != std::errc{} || cursor != end) {
        throw std::runtime_error("invalid integer for '" + key + "': " + value);
    }
    return result;
}

std::uint32_t parse_seed(const std::string& key, const std::string& value) {
    std::uint32_t result = 0;
    const auto* begin = value.data();
    const auto* end = begin + value.size();
    const auto [cursor, error] = std::from_chars(begin, end, result);
    if (error != std::errc{} || cursor != end) {
        throw std::runtime_error("invalid seed for '" + key + "': " + value);
    }
    return result;
}

float parse_float(const std::string& key, const std::string& value) {
    float result = 0.0F;
    std::istringstream input(value);
    input.imbue(std::locale::classic());
    input >> std::noskipws >> result;
    if (!input || input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid number for '" + key + "': " + value);
    }
    return result;
}

std::filesystem::path resolve_path(
    const std::filesystem::path& project_root,
    const std::string& value
) {
    const std::filesystem::path candidate(value);
    if (candidate.is_absolute()) {
        return candidate.lexically_normal();
    }
    return (project_root / candidate).lexically_normal();
}

}  // namespace

Config Config::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("could not open config: " + path.string());
    }

    Config config;
    const auto project_root = std::filesystem::absolute(path)
                                  .parent_path()
                                  .parent_path();
    std::unordered_set<std::string> seen_keys;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error(
                "expected key=value at " + path.string() + ":" +
                std::to_string(line_number)
            );
        }

        const auto key = trim(line.substr(0, separator));
        const auto value = trim(line.substr(separator + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                "empty key or value at " + path.string() + ":" +
                std::to_string(line_number)
            );
        }
        if (!seen_keys.insert(key).second) {
            throw std::runtime_error("duplicate config key: " + key);
        }

        if (key == "corpus") {
            config.corpus_path = resolve_path(project_root, value);
        } else if (key == "results") {
            config.results_dir = resolve_path(project_root, value);
        } else if (key == "seed") {
            config.seed = parse_seed(key, value);
        } else if (key == "context_size") {
            config.context_size = parse_size(key, value);
        } else if (key == "batch_size") {
            config.batch_size = parse_size(key, value);
        } else if (key == "d_model") {
            config.d_model = parse_size(key, value);
        } else if (key == "n_heads") {
            config.n_heads = parse_size(key, value);
        } else if (key == "n_layers") {
            config.n_layers = parse_size(key, value);
        } else if (key == "d_ff") {
            config.d_ff = parse_size(key, value);
        } else if (key == "training_steps") {
            config.training_steps = parse_size(key, value);
        } else if (key == "sample_every") {
            config.sample_every = parse_size(key, value);
        } else if (key == "sample_length") {
            config.sample_length = parse_size(key, value);
        } else if (key == "learning_rate") {
            config.learning_rate = parse_float(key, value);
        } else if (key == "adam_beta1") {
            config.adam_beta1 = parse_float(key, value);
        } else if (key == "adam_beta2") {
            config.adam_beta2 = parse_float(key, value);
        } else if (key == "adam_epsilon") {
            config.adam_epsilon = parse_float(key, value);
        } else if (key == "gradient_clip") {
            config.gradient_clip = parse_float(key, value);
        } else {
            throw std::runtime_error("unknown config key: " + key);
        }
    }

    config.validate();
    return config;
}

void Config::validate() const {
    if (corpus_path.empty()) {
        throw std::runtime_error("corpus path is required");
    }
    if (results_dir.empty()) {
        throw std::runtime_error("results directory is required");
    }
    if (context_size == 0 || batch_size == 0 || d_model == 0 ||
        n_heads == 0 || n_layers == 0 || d_ff == 0) {
        throw std::runtime_error("model dimensions must be greater than zero");
    }
    if (d_model % n_heads != 0) {
        throw std::runtime_error("d_model must be divisible by n_heads");
    }
    if (training_steps == 0 || sample_every == 0 || sample_length == 0) {
        throw std::runtime_error("training and sampling counts must be positive");
    }
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0F) {
        throw std::runtime_error(
            "learning_rate must be finite and positive"
        );
    }
    if (!std::isfinite(adam_beta1) ||
        adam_beta1 <= 0.0F ||
        adam_beta1 >= 1.0F ||
        !std::isfinite(adam_beta2) ||
        adam_beta2 <= 0.0F ||
        adam_beta2 >= 1.0F) {
        throw std::runtime_error(
            "Adam beta values must be finite and between zero and one"
        );
    }
    if (!std::isfinite(adam_epsilon) ||
        adam_epsilon <= 0.0F ||
        !std::isfinite(gradient_clip) ||
        gradient_clip <= 0.0F) {
        throw std::runtime_error(
            "Adam epsilon and gradient_clip must be finite and positive"
        );
    }
}

std::string Config::summary() const {
    std::ostringstream output;
    output << "corpus:       " << corpus_path << '\n'
           << "results:      " << results_dir << '\n'
           << "seed:         " << seed << '\n'
           << "context:      " << context_size << '\n'
           << "batch size:   " << batch_size << '\n'
           << "model width:  " << d_model << '\n'
           << "heads:        " << n_heads << '\n'
           << "layers:       " << n_layers << '\n'
           << "MLP width:    " << d_ff << '\n'
           << "steps:        " << training_steps << '\n'
           << "Adam lr:      " << learning_rate;
    return output.str();
}

}  // namespace transformer_lab
