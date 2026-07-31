#include "transformer_lab/stages/post_training/instruction.hpp"

#include <algorithm>
#include <stdexcept>

namespace transformer_lab::stages::post_training {
namespace {

// Instruction files are UTF-8 byte strings. Restricting template trimming to
// ASCII whitespace avoids locale-dependent treatment of multibyte text.
bool is_ascii_whitespace(unsigned char byte) noexcept {
    return byte == static_cast<unsigned char>(' ') ||
           byte == static_cast<unsigned char>('\t') ||
           byte == static_cast<unsigned char>('\n') ||
           byte == static_cast<unsigned char>('\r') ||
           byte == static_cast<unsigned char>('\f') ||
           byte == static_cast<unsigned char>('\v');
}

bool is_blank(std::string_view value) {
    return std::all_of(
        value.begin(),
        value.end(),
        [](char character) {
            return is_ascii_whitespace(
                static_cast<unsigned char>(character)
            );
        }
    );
}

std::string trim(std::string_view value) {
    const auto first = std::find_if_not(
        value.begin(),
        value.end(),
        [](char character) {
            return is_ascii_whitespace(
                static_cast<unsigned char>(character)
            );
        }
    );
    const auto last = std::find_if_not(
        value.rbegin(),
        value.rend(),
        [](char character) {
            return is_ascii_whitespace(
                static_cast<unsigned char>(character)
            );
        }
    ).base();
    return first < last ? std::string(first, last) : std::string{};
}

}  // namespace

void InstructionExample::validate() const {
    if (prompt.empty() || is_blank(prompt)) {
        throw std::invalid_argument(
            "instruction prompt must not be blank"
        );
    }
    if (response.empty() || is_blank(response)) {
        throw std::invalid_argument(
            "instruction response must not be blank"
        );
    }
}

std::string PlainChatFormatter::format(
    const InstructionExample& example
) const {
    example.validate();
    return "### User:\n" + trim(example.prompt) +
           "\n### Assistant:\n" + trim(example.response) + "\n";
}

}  // namespace transformer_lab::stages::post_training
