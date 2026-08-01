#include "riftco_transformer/stages/post_training/instruction.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::stages::post_training {
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

InstructionSplits::InstructionSplits(
    std::vector<InstructionExample> training_examples,
    std::vector<InstructionExample> validation_examples,
    std::vector<InstructionExample> test_examples
)
    : train(std::move(training_examples)),
      validation(std::move(validation_examples)),
      test(std::move(test_examples)) {
    validate();
}

void InstructionSplits::validate() const {
    const auto validate_split = [](
        const std::vector<InstructionExample>& examples,
        const char* name
    ) {
        if (examples.empty()) {
            throw std::invalid_argument(
                std::string("post-training ") + name +
                " split must not be empty"
            );
        }
        for (const auto& example : examples) {
            example.validate();
        }
    };
    validate_split(train, "train");
    validate_split(validation, "validation");
    validate_split(test, "test");

    using Record = std::pair<std::string, std::string>;
    const auto records = [](const std::vector<InstructionExample>& examples) {
        std::set<Record> result;
        for (const auto& example : examples) {
            result.emplace(example.prompt, example.response);
        }
        return result;
    };
    const std::set<Record> training_records = records(train);
    const std::set<Record> validation_records = records(validation);
    const std::set<Record> test_records = records(test);

    const auto reject_overlap = [](
        const std::set<Record>& left,
        const char* left_name,
        const std::set<Record>& right,
        const char* right_name
    ) {
        const auto& smaller = left.size() <= right.size() ? left : right;
        const auto& larger = left.size() <= right.size() ? right : left;
        const bool found = std::any_of(
            smaller.begin(),
            smaller.end(),
            [&](const Record& record) { return larger.contains(record); }
        );
        if (found) {
            throw std::invalid_argument(
                std::string("post-training ") + left_name + " and " +
                right_name + " splits overlap by an exact instruction record"
            );
        }
    };
    reject_overlap(
        training_records,
        "train",
        validation_records,
        "validation"
    );
    reject_overlap(training_records, "train", test_records, "test");
    reject_overlap(
        validation_records,
        "validation",
        test_records,
        "test"
    );
}

std::string PlainChatFormatter::format(
    const InstructionExample& example
) const {
    example.validate();
    return "### User:\n" + trim(example.prompt) +
           "\n### Assistant:\n" + trim(example.response) + "\n";
}

}  // namespace riftco_transformer::stages::post_training
