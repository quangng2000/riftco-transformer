#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace riftco_transformer {

using TokenId = std::uint32_t;

enum class TokenizerMethod {
    CorpusByte = 0,
    BytePair = 1,
};

struct TokenizerOptions {
    TokenizerMethod method{TokenizerMethod::CorpusByte};
    std::size_t vocabulary_size{512};
    std::size_t minimum_pair_frequency{2};
};

// Reads a file without newline conversion so tokenization can reproduce its
// exact bytes.
[[nodiscard]] std::string read_file_bytes(
    const std::filesystem::path& path
);

// The common interface for interchangeable tokenizer algorithms.
class TokenizerStrategy {
public:
    virtual ~TokenizerStrategy() = default;

    [[nodiscard]] virtual TokenizerMethod method() const noexcept = 0;
    [[nodiscard]] virtual std::size_t vocab_size() const noexcept = 0;
    [[nodiscard]] virtual std::span<const std::uint8_t> token_bytes(
        TokenId token
    ) const = 0;
    [[nodiscard]] virtual std::vector<TokenId> encode(
        std::string_view text
    ) const = 0;
    [[nodiscard]] virtual std::string decode(
        std::span<const TokenId> tokens
    ) const = 0;
};

// A corpus-derived byte tokenizer. Token IDs are assigned in unsigned-byte
// order, making the vocabulary independent of corpus encounter order.
class ByteTokenizer final : public TokenizerStrategy {
public:
    explicit ByteTokenizer(std::string_view corpus);
    // Restores the exact token-ID ordering of a serialized vocabulary.
    explicit ByteTokenizer(
        std::span<const std::uint8_t> ordered_vocabulary
    );

    [[nodiscard]] TokenizerMethod method() const noexcept override;
    [[nodiscard]] std::size_t vocab_size() const noexcept override;
    [[nodiscard]] std::span<const std::uint8_t> vocabulary() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> token_bytes(
        TokenId token
    ) const override;

    [[nodiscard]] std::vector<TokenId> encode(
        std::string_view text
    ) const override;
    [[nodiscard]] std::string decode(
        std::span<const TokenId> tokens
    ) const override;

private:
    // Negative entries represent bytes absent from the corpus vocabulary.
    std::array<std::int32_t, 256> byte_to_token_;
    std::vector<std::uint8_t> token_to_byte_;
};

struct BpeMergeRule {
    TokenId left;
    TokenId right;
    TokenId result;

    [[nodiscard]] bool operator==(const BpeMergeRule&) const = default;
};

// A byte-level BPE tokenizer. IDs 0..255 always represent their corresponding
// byte, and later IDs represent byte sequences learned from the corpus.
class BytePairTokenizer final : public TokenizerStrategy {
public:
    explicit BytePairTokenizer(
        std::string_view corpus,
        std::size_t vocabulary_size = 512,
        std::size_t minimum_pair_frequency = 2
    );
    // Restores a byte-level BPE tokenizer from serialized merge order.
    explicit BytePairTokenizer(
        std::span<const BpeMergeRule> ordered_merge_rules
    );

    [[nodiscard]] TokenizerMethod method() const noexcept override;
    [[nodiscard]] std::size_t vocab_size() const noexcept override;
    [[nodiscard]] std::span<const std::uint8_t> token_bytes(
        TokenId token
    ) const override;
    [[nodiscard]] std::span<const BpeMergeRule> merge_rules() const noexcept;

    [[nodiscard]] std::vector<TokenId> encode(
        std::string_view text
    ) const override;
    [[nodiscard]] std::string decode(
        std::span<const TokenId> tokens
    ) const override;

private:
    std::vector<std::vector<std::uint8_t>> token_pieces_;
    std::vector<BpeMergeRule> merge_rules_;
};

[[nodiscard]] std::unique_ptr<TokenizerStrategy> make_tokenizer(
    std::string_view corpus,
    const TokenizerOptions& options = {}
);

}  // namespace riftco_transformer
