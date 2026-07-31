#include "riftco_transformer/data/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace riftco_transformer {
namespace {

static_assert(
    std::numeric_limits<unsigned char>::max() == 255,
    "ByteTokenizer requires eight-bit bytes"
);

using TokenPair = std::pair<TokenId, TokenId>;

[[nodiscard]] std::vector<TokenId> bytes_as_base_tokens(
    std::string_view bytes
) {
    std::vector<TokenId> tokens;
    tokens.reserve(bytes.size());
    for (const char character : bytes) {
        tokens.push_back(
            static_cast<TokenId>(
                static_cast<unsigned char>(character)
            )
        );
    }
    return tokens;
}

[[nodiscard]] std::vector<TokenId> apply_merge(
    std::span<const TokenId> tokens,
    TokenId left,
    TokenId right,
    TokenId result
) {
    std::vector<TokenId> merged;
    merged.reserve(tokens.size());

    std::size_t index = 0;
    while (index < tokens.size()) {
        if (index + 1 < tokens.size() &&
            tokens[index] == left &&
            tokens[index + 1] == right) {
            merged.push_back(result);
            index += 2;
        } else {
            merged.push_back(tokens[index]);
            ++index;
        }
    }
    return merged;
}

[[nodiscard]] std::vector<std::uint8_t> concatenate_token_pieces(
    std::span<const std::uint8_t> left,
    std::span<const std::uint8_t> right
) {
    if (right.size() >
        std::numeric_limits<std::size_t>::max() - left.size()) {
        throw std::overflow_error("BPE token piece size overflow");
    }
    const std::size_t result_size = left.size() + right.size();

    std::vector<std::uint8_t> result;
    if (result_size > result.max_size()) {
        throw std::overflow_error(
            "BPE token piece exceeds the maximum supported size"
        );
    }
    result.reserve(result_size);
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());
    return result;
}

}  // namespace

std::string read_file_bytes(const std::filesystem::path& path) {
    std::error_code file_error;
    if (!std::filesystem::is_regular_file(path, file_error) ||
        file_error) {
        throw std::runtime_error(
            "byte input is not a readable regular file: " + path.string()
        );
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "could not open file for byte reading: " + path.string()
        );
    }

    const std::streamoff byte_count = input.tellg();
    if (byte_count < 0) {
        throw std::runtime_error(
            "could not determine byte input size: " + path.string()
        );
    }
    const auto unsigned_byte_count =
        static_cast<std::uintmax_t>(byte_count);
    if (unsigned_byte_count >
            std::numeric_limits<std::size_t>::max() ||
        unsigned_byte_count >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::streamsize>::max()
            )) {
        throw std::overflow_error(
            "byte input is too large to read: " + path.string()
        );
    }

    std::string bytes(
        static_cast<std::size_t>(unsigned_byte_count),
        '\0'
    );
    input.seekg(0, std::ios::beg);
    if (!input) {
        throw std::runtime_error(
            "could not seek to start of byte input: " + path.string()
        );
    }
    if (!bytes.empty()) {
        const auto expected =
            static_cast<std::streamsize>(bytes.size());
        input.read(bytes.data(), expected);
        if (input.gcount() != expected || input.fail()) {
            throw std::runtime_error(
                "failed while reading file bytes: " + path.string()
            );
        }
    }
    return bytes;
}

ByteTokenizer::ByteTokenizer(std::string_view corpus) {
    if (corpus.empty()) {
        throw std::invalid_argument(
            "tokenizer corpus must contain at least one byte"
        );
    }

    byte_to_token_.fill(-1);
    std::array<bool, 256> present{};
    for (const char character : corpus) {
        const auto byte = static_cast<unsigned char>(character);
        present[byte] = true;
    }

    for (std::size_t byte = 0; byte < present.size(); ++byte) {
        if (!present[byte]) {
            continue;
        }
        const auto token = static_cast<TokenId>(token_to_byte_.size());
        byte_to_token_[byte] = static_cast<std::int32_t>(token);
        token_to_byte_.push_back(static_cast<std::uint8_t>(byte));
    }
}

ByteTokenizer::ByteTokenizer(
    std::span<const std::uint8_t> ordered_vocabulary
) {
    if (ordered_vocabulary.empty()) {
        throw std::invalid_argument(
            "tokenizer vocabulary must contain at least one byte"
        );
    }

    byte_to_token_.fill(-1);
    token_to_byte_.reserve(ordered_vocabulary.size());
    for (const std::uint8_t byte : ordered_vocabulary) {
        if (byte_to_token_[byte] >= 0) {
            throw std::invalid_argument(
                "tokenizer vocabulary contains a duplicate byte"
            );
        }
        const auto token = static_cast<TokenId>(token_to_byte_.size());
        byte_to_token_[byte] = static_cast<std::int32_t>(token);
        token_to_byte_.push_back(byte);
    }
}

TokenizerMethod ByteTokenizer::method() const noexcept {
    return TokenizerMethod::CorpusByte;
}

std::size_t ByteTokenizer::vocab_size() const noexcept {
    return token_to_byte_.size();
}

std::span<const std::uint8_t> ByteTokenizer::vocabulary() const noexcept {
    return token_to_byte_;
}

std::span<const std::uint8_t> ByteTokenizer::token_bytes(
    TokenId token
) const {
    const auto index = static_cast<std::size_t>(token);
    if (index >= token_to_byte_.size()) {
        throw std::out_of_range(
            "token ID is outside the tokenizer vocabulary"
        );
    }
    return std::span<const std::uint8_t>(
        &token_to_byte_[index],
        1
    );
}

std::vector<TokenId> ByteTokenizer::encode(std::string_view text) const {
    std::vector<TokenId> tokens;
    tokens.reserve(text.size());

    for (const char character : text) {
        const auto byte = static_cast<unsigned char>(character);
        const auto token = byte_to_token_[byte];
        if (token < 0) {
            throw std::invalid_argument(
                "byte " + std::to_string(byte) +
                " is absent from the tokenizer vocabulary"
            );
        }
        tokens.push_back(static_cast<TokenId>(token));
    }
    return tokens;
}

std::string ByteTokenizer::decode(
    std::span<const TokenId> tokens
) const {
    std::string text;
    text.reserve(tokens.size());

    for (const TokenId token : tokens) {
        const auto index = static_cast<std::size_t>(token);
        if (index >= token_to_byte_.size()) {
            throw std::out_of_range(
                "token ID is outside the tokenizer vocabulary"
            );
        }
        text.push_back(std::bit_cast<char>(token_to_byte_[index]));
    }
    return text;
}

BytePairTokenizer::BytePairTokenizer(
    std::string_view corpus,
    std::size_t vocabulary_size,
    std::size_t minimum_pair_frequency
) {
    if (corpus.empty()) {
        throw std::invalid_argument(
            "tokenizer corpus must contain at least one byte"
        );
    }
    if (vocabulary_size < 256) {
        throw std::invalid_argument(
            "BPE vocabulary size must be at least 256"
        );
    }
    if (vocabulary_size >
        static_cast<std::size_t>(
            std::numeric_limits<TokenId>::max()
        )) {
        throw std::invalid_argument(
            "BPE vocabulary size exceeds the TokenId range"
        );
    }
    if (minimum_pair_frequency == 0) {
        throw std::invalid_argument(
            "BPE minimum pair frequency must be at least one"
        );
    }

    const std::size_t maximum_merge_count = std::min(
        vocabulary_size - 256,
        corpus.size() - 1
    );
    token_pieces_.reserve(256 + maximum_merge_count);
    for (std::size_t byte = 0; byte < 256; ++byte) {
        token_pieces_.push_back(
            {static_cast<std::uint8_t>(byte)}
        );
    }
    merge_rules_.reserve(maximum_merge_count);

    std::vector<TokenId> corpus_tokens = bytes_as_base_tokens(corpus);
    while (token_pieces_.size() < vocabulary_size &&
           corpus_tokens.size() >= 2) {
        std::map<TokenPair, std::size_t> pair_frequencies;
        for (std::size_t index = 0;
             index + 1 < corpus_tokens.size();
             ++index) {
            const TokenPair pair{
                corpus_tokens[index],
                corpus_tokens[index + 1],
            };
            ++pair_frequencies[pair];
        }

        TokenPair selected_pair{};
        std::size_t selected_frequency = 0;
        for (const auto& [pair, frequency] : pair_frequencies) {
            // std::map iteration supplies the lexicographically smallest
            // pair first, so equal-frequency candidates keep that winner.
            if (frequency > selected_frequency) {
                selected_pair = pair;
                selected_frequency = frequency;
            }
        }
        if (selected_frequency < minimum_pair_frequency) {
            break;
        }

        const auto result =
            static_cast<TokenId>(token_pieces_.size());
        const auto& left_piece =
            token_pieces_[static_cast<std::size_t>(selected_pair.first)];
        const auto& right_piece =
            token_pieces_[static_cast<std::size_t>(selected_pair.second)];
        std::vector<std::uint8_t> result_piece =
            concatenate_token_pieces(
                left_piece,
                right_piece
            );
        token_pieces_.push_back(std::move(result_piece));
        merge_rules_.push_back(
            {
                selected_pair.first,
                selected_pair.second,
                result,
            }
        );
        corpus_tokens = apply_merge(
            corpus_tokens,
            selected_pair.first,
            selected_pair.second,
            result
        );
    }
}

BytePairTokenizer::BytePairTokenizer(
    std::span<const BpeMergeRule> ordered_merge_rules
) {
    constexpr std::size_t base_vocabulary_size = 256;
    constexpr std::size_t maximum_rule_count =
        static_cast<std::size_t>(
            std::numeric_limits<TokenId>::max()
        ) -
        base_vocabulary_size +
        1;

    if (ordered_merge_rules.size() > maximum_rule_count ||
        ordered_merge_rules.size() >
            std::numeric_limits<std::size_t>::max() -
                base_vocabulary_size) {
        throw std::invalid_argument(
            "BPE merge rules exceed the TokenId range"
        );
    }
    const std::size_t restored_vocabulary_size =
        base_vocabulary_size + ordered_merge_rules.size();

    std::vector<std::size_t> token_piece_sizes;
    if (restored_vocabulary_size > token_piece_sizes.max_size() ||
        restored_vocabulary_size > token_pieces_.max_size()) {
        throw std::invalid_argument(
            "BPE merge rules exceed the supported vocabulary size"
        );
    }
    token_piece_sizes.reserve(restored_vocabulary_size);
    token_piece_sizes.insert(
        token_piece_sizes.end(),
        base_vocabulary_size,
        1
    );

    std::set<TokenPair> seen_pairs;
    const std::size_t maximum_piece_size =
        std::vector<std::uint8_t>{}.max_size();
    for (std::size_t index = 0;
         index < ordered_merge_rules.size();
         ++index) {
        const BpeMergeRule& rule = ordered_merge_rules[index];
        const auto expected_result = static_cast<TokenId>(
            base_vocabulary_size + index
        );
        if (rule.result != expected_result) {
            throw std::invalid_argument(
                "BPE merge result IDs must be sequential from 256"
            );
        }
        if (rule.left >= rule.result || rule.right >= rule.result) {
            throw std::invalid_argument(
                "BPE merge operands must precede their result token"
            );
        }
        if (!seen_pairs.emplace(rule.left, rule.right).second) {
            throw std::invalid_argument(
                "BPE merge rules contain a duplicate token pair"
            );
        }

        const std::size_t left_size =
            token_piece_sizes[static_cast<std::size_t>(rule.left)];
        const std::size_t right_size =
            token_piece_sizes[static_cast<std::size_t>(rule.right)];
        if (right_size >
            std::numeric_limits<std::size_t>::max() - left_size) {
            throw std::overflow_error("BPE token piece size overflow");
        }
        const std::size_t result_size = left_size + right_size;
        if (result_size > maximum_piece_size) {
            throw std::overflow_error(
                "BPE token piece exceeds the maximum supported size"
            );
        }
        token_piece_sizes.push_back(result_size);
    }

    token_pieces_.reserve(restored_vocabulary_size);
    for (std::size_t byte = 0; byte < base_vocabulary_size; ++byte) {
        token_pieces_.push_back(
            {static_cast<std::uint8_t>(byte)}
        );
    }
    merge_rules_.reserve(ordered_merge_rules.size());
    for (const BpeMergeRule& rule : ordered_merge_rules) {
        const auto& left_piece =
            token_pieces_[static_cast<std::size_t>(rule.left)];
        const auto& right_piece =
            token_pieces_[static_cast<std::size_t>(rule.right)];
        token_pieces_.push_back(
            concatenate_token_pieces(left_piece, right_piece)
        );
        merge_rules_.push_back(rule);
    }
}

TokenizerMethod BytePairTokenizer::method() const noexcept {
    return TokenizerMethod::BytePair;
}

std::size_t BytePairTokenizer::vocab_size() const noexcept {
    return token_pieces_.size();
}

std::span<const std::uint8_t> BytePairTokenizer::token_bytes(
    TokenId token
) const {
    const auto index = static_cast<std::size_t>(token);
    if (index >= token_pieces_.size()) {
        throw std::out_of_range(
            "token ID is outside the tokenizer vocabulary"
        );
    }
    return token_pieces_[index];
}

std::span<const BpeMergeRule> BytePairTokenizer::merge_rules() const noexcept {
    return merge_rules_;
}

std::vector<TokenId> BytePairTokenizer::encode(
    std::string_view text
) const {
    std::vector<TokenId> tokens = bytes_as_base_tokens(text);
    for (const BpeMergeRule& rule : merge_rules_) {
        tokens = apply_merge(
            tokens,
            rule.left,
            rule.right,
            rule.result
        );
    }
    return tokens;
}

std::string BytePairTokenizer::decode(
    std::span<const TokenId> tokens
) const {
    std::size_t byte_count = 0;
    for (const TokenId token : tokens) {
        const auto piece = token_bytes(token);
        if (piece.size() >
            std::numeric_limits<std::size_t>::max() - byte_count) {
            throw std::overflow_error(
                "decoded tokenizer output is too large"
            );
        }
        byte_count += piece.size();
    }

    std::string text;
    text.reserve(byte_count);
    for (const TokenId token : tokens) {
        for (const std::uint8_t byte : token_bytes(token)) {
            text.push_back(std::bit_cast<char>(byte));
        }
    }
    return text;
}

std::unique_ptr<TokenizerStrategy> make_tokenizer(
    std::string_view corpus,
    const TokenizerOptions& options
) {
    switch (options.method) {
        case TokenizerMethod::CorpusByte:
            return std::make_unique<ByteTokenizer>(corpus);
        case TokenizerMethod::BytePair:
            return std::make_unique<BytePairTokenizer>(
                corpus,
                options.vocabulary_size,
                options.minimum_pair_frequency
            );
        default:
            throw std::invalid_argument("unknown tokenizer method");
    }
}

}  // namespace riftco_transformer
