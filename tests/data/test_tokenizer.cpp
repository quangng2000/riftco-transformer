#include "riftco_transformer/data/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using riftco_transformer::BpeMergeRule;
using riftco_transformer::BytePairTokenizer;
using riftco_transformer::ByteTokenizer;
using riftco_transformer::TokenId;
using riftco_transformer::TokenizerMethod;
using riftco_transformer::TokenizerOptions;
using riftco_transformer::TokenizerStrategy;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

void require_tokens(
    std::span<const TokenId> actual,
    const std::vector<TokenId>& expected,
    const std::string& message
) {
    require(actual.size() == expected.size(), message + ": size mismatch");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(
            actual[index] == expected[index],
            message + " at index " + std::to_string(index)
        );
    }
}

void require_bytes(
    std::span<const std::uint8_t> actual,
    std::span<const std::uint8_t> expected,
    const std::string& message
) {
    require(actual.size() == expected.size(), message + ": size mismatch");
    require(
        std::equal(
            actual.begin(),
            actual.end(),
            expected.begin(),
            expected.end()
        ),
        message + ": byte mismatch"
    );
}

void test_deterministic_vocabulary_and_round_trip() {
    const std::string corpus = "cab\ncab";
    const ByteTokenizer tokenizer(corpus);

    require(tokenizer.vocab_size() == 4, "unexpected vocabulary size");
    require(
        tokenizer.method() == TokenizerMethod::CorpusByte,
        "legacy tokenizer should identify its method"
    );
    const std::array<std::uint8_t, 4> expected_vocabulary{
        10,
        97,
        98,
        99,
    };
    require(
        std::equal(
            tokenizer.vocabulary().begin(),
            tokenizer.vocabulary().end(),
            expected_vocabulary.begin(),
            expected_vocabulary.end()
        ),
        "vocabulary should be sorted by unsigned byte value"
    );

    const auto encoded = tokenizer.encode(corpus);
    require_tokens(
        encoded,
        {3, 1, 2, 0, 3, 1, 2},
        "corpus token IDs"
    );
    require(
        tokenizer.decode(encoded) == corpus,
        "decoded corpus should match its original bytes"
    );
    const std::array<std::uint8_t, 1> expected_c_byte{99};
    require_bytes(
        tokenizer.token_bytes(3),
        expected_c_byte,
        "legacy token byte introspection"
    );

    const ByteTokenizer different_order("\nbacc");
    require(
        std::equal(
            tokenizer.vocabulary().begin(),
            tokenizer.vocabulary().end(),
            different_order.vocabulary().begin(),
            different_order.vocabulary().end()
        ),
        "vocabulary should not depend on encounter order"
    );

    require(tokenizer.encode("").empty(), "empty text should encode");
    require(
        tokenizer.decode(std::span<const TokenId>{}).empty(),
        "empty tokens should decode"
    );
}

void test_byte_vocabulary_restoration() {
    const ByteTokenizer trained("cab\ncab");
    const std::vector<std::uint8_t> serialized_vocabulary(
        trained.vocabulary().begin(),
        trained.vocabulary().end()
    );
    const ByteTokenizer restored{
        std::span<const std::uint8_t>(serialized_vocabulary)
    };

    require(
        std::equal(
            restored.vocabulary().begin(),
            restored.vocabulary().end(),
            serialized_vocabulary.begin(),
            serialized_vocabulary.end()
        ),
        "restored byte vocabulary should preserve serialized token order"
    );
    const auto trained_tokens = trained.encode("cab\n");
    const auto restored_tokens = restored.encode("cab\n");
    require_tokens(
        restored_tokens,
        trained_tokens,
        "restored byte vocabulary token IDs"
    );
    require(
        restored.decode(restored_tokens) == "cab\n",
        "restored byte vocabulary should round-trip"
    );

    const std::array<std::uint8_t, 4> custom_order{
        99,
        10,
        97,
        98,
    };
    const ByteTokenizer reordered{
        std::span<const std::uint8_t>(custom_order)
    };
    require_tokens(
        reordered.encode("c\nab"),
        {0, 1, 2, 3},
        "restoration should use the supplied byte order as token IDs"
    );
}

void test_strategy_factory_and_swapping() {
    const std::string corpus = "abababab";

    const TokenizerOptions default_options;
    std::unique_ptr<TokenizerStrategy> tokenizer =
        riftco_transformer::make_tokenizer(corpus, default_options);
    require(
        tokenizer->method() == TokenizerMethod::CorpusByte,
        "default factory method should preserve corpus-byte behavior"
    );
    require(
        tokenizer->decode(tokenizer->encode(corpus)) == corpus,
        "default strategy should round-trip"
    );

    TokenizerOptions bpe_options;
    bpe_options.method = TokenizerMethod::BytePair;
    bpe_options.vocabulary_size = 258;
    bpe_options.minimum_pair_frequency = 2;
    tokenizer = riftco_transformer::make_tokenizer(corpus, bpe_options);
    require(
        tokenizer->method() == TokenizerMethod::BytePair,
        "factory should select the BPE strategy"
    );
    require(
        tokenizer->vocab_size() == 258,
        "BPE factory options should reach the requested target"
    );
    require_tokens(
        tokenizer->encode(corpus),
        {257, 257},
        "strategy-polymorphic BPE encoding"
    );
    require(
        tokenizer->decode(tokenizer->encode(corpus)) == corpus,
        "swapped BPE strategy should round-trip"
    );

    TokenizerOptions irrelevant_byte_options;
    irrelevant_byte_options.vocabulary_size = 1;
    irrelevant_byte_options.minimum_pair_frequency = 0;
    tokenizer =
        riftco_transformer::make_tokenizer("abc", irrelevant_byte_options);
    require(
        tokenizer->vocab_size() == 3,
        "BPE-only settings should not alter corpus-byte construction"
    );

    TokenizerOptions unknown_options;
    unknown_options.method = static_cast<TokenizerMethod>(99);
    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::make_tokenizer(corpus, unknown_options)
            );
        },
        "factory should reject an unknown tokenizer method"
    );
}

void test_bpe_deterministic_merges_and_compression() {
    const BytePairTokenizer tokenizer("abababab", 258, 2);

    require(
        tokenizer.method() == TokenizerMethod::BytePair,
        "BPE tokenizer should identify its method"
    );
    require(tokenizer.vocab_size() == 258, "unexpected BPE vocabulary size");
    const auto rules = tokenizer.merge_rules();
    require(rules.size() == 2, "expected exactly two learned BPE merges");
    require(
        rules[0] == BpeMergeRule{97, 98, 256},
        "first merge should learn the most frequent 'ab' pair"
    );
    require(
        rules[1] == BpeMergeRule{256, 256, 257},
        "second merge should combine adjacent learned 'ab' tokens"
    );

    const std::array<std::uint8_t, 2> expected_ab{97, 98};
    const std::array<std::uint8_t, 4> expected_abab{97, 98, 97, 98};
    require_bytes(
        tokenizer.token_bytes(256),
        expected_ab,
        "first learned token bytes"
    );
    require_bytes(
        tokenizer.token_bytes(257),
        expected_abab,
        "second learned token bytes"
    );

    const auto encoded = tokenizer.encode("abababab");
    require_tokens(encoded, {257, 257}, "compressed BPE tokens");
    require(
        encoded.size() < std::string_view("abababab").size(),
        "learned BPE vocabulary should compress its training pattern"
    );
    require(
        tokenizer.decode(encoded) == "abababab",
        "compressed BPE tokens should decode exactly"
    );

    const BytePairTokenizer tied_pairs("abac", 257, 1);
    require(
        tied_pairs.merge_rules().size() == 1,
        "target should limit the tie test to one merge"
    );
    require(
        tied_pairs.merge_rules()[0] == BpeMergeRule{97, 98, 256},
        "equal-frequency pairs should use lexicographic ID tie-breaking"
    );

    const BytePairTokenizer overlaps("aaaaa", 257, 1);
    require_tokens(
        overlaps.encode("aaaaa"),
        {256, 256, 97},
        "merge replacement should be non-overlapping and left-to-right"
    );

    const BytePairTokenizer repeat("abababab", 258, 2);
    require(
        std::equal(
            tokenizer.merge_rules().begin(),
            tokenizer.merge_rules().end(),
            repeat.merge_rules().begin(),
            repeat.merge_rules().end()
        ),
        "BPE training should be deterministic"
    );
}

void test_bpe_merge_rule_restoration() {
    const BytePairTokenizer trained("abababab", 258, 2);
    const std::vector<BpeMergeRule> serialized_rules(
        trained.merge_rules().begin(),
        trained.merge_rules().end()
    );
    const BytePairTokenizer restored{
        std::span<const BpeMergeRule>(serialized_rules)
    };

    require(
        restored.vocab_size() == trained.vocab_size(),
        "restored BPE vocabulary size should match training"
    );
    require(
        std::equal(
            restored.merge_rules().begin(),
            restored.merge_rules().end(),
            serialized_rules.begin(),
            serialized_rules.end()
        ),
        "restored BPE rules should preserve serialized order"
    );
    for (std::size_t token = 0; token < trained.vocab_size(); ++token) {
        require_bytes(
            restored.token_bytes(static_cast<TokenId>(token)),
            trained.token_bytes(static_cast<TokenId>(token)),
            "restored BPE token piece"
        );
    }

    const std::string text("abab\0z", 6);
    const auto trained_tokens = trained.encode(text);
    const auto restored_tokens = restored.encode(text);
    require_tokens(
        restored_tokens,
        trained_tokens,
        "restored BPE token IDs"
    );
    require(
        restored.decode(restored_tokens) == text,
        "restored BPE tokenizer should round-trip binary text"
    );

    const BytePairTokenizer base_only(
        std::span<const BpeMergeRule>{}
    );
    require(
        base_only.vocab_size() == 256 &&
            base_only.merge_rules().empty(),
        "an empty serialized merge list should restore base-byte BPE"
    );
    require_tokens(
        base_only.encode("ab"),
        {97, 98},
        "base-byte BPE restoration"
    );
}

void test_invalid_restored_definitions() {
    require_throws(
        [] {
            static_cast<void>(
                ByteTokenizer(std::span<const std::uint8_t>{})
            );
        },
        "byte restoration should reject an empty vocabulary"
    );

    const std::array<std::uint8_t, 3> duplicate_bytes{97, 98, 97};
    require_throws(
        [&] {
            static_cast<void>(
                ByteTokenizer(
                    std::span<const std::uint8_t>(duplicate_bytes)
                )
            );
        },
        "byte restoration should reject duplicate vocabulary entries"
    );

    const std::array<BpeMergeRule, 1> nonsequential_result{
        BpeMergeRule{97, 98, 257},
    };
    require_throws(
        [&] {
            static_cast<void>(
                BytePairTokenizer(
                    std::span<const BpeMergeRule>(nonsequential_result)
                )
            );
        },
        "BPE restoration should reject nonsequential result IDs"
    );

    const std::array<BpeMergeRule, 1> forward_operand{
        BpeMergeRule{256, 98, 256},
    };
    require_throws(
        [&] {
            static_cast<void>(
                BytePairTokenizer(
                    std::span<const BpeMergeRule>(forward_operand)
                )
            );
        },
        "BPE restoration should reject operands not preceding the result"
    );

    const std::array<BpeMergeRule, 2> duplicate_pair{
        BpeMergeRule{97, 98, 256},
        BpeMergeRule{97, 98, 257},
    };
    require_throws(
        [&] {
            static_cast<void>(
                BytePairTokenizer(
                    std::span<const BpeMergeRule>(duplicate_pair)
                )
            );
        },
        "BPE restoration should reject duplicate token pairs"
    );

    std::vector<BpeMergeRule> overflowing_pieces;
    overflowing_pieces.reserve(
        std::numeric_limits<std::size_t>::digits
    );
    TokenId repeated_operand = 0;
    for (std::size_t index = 0;
         index < std::numeric_limits<std::size_t>::digits;
         ++index) {
        const auto result = static_cast<TokenId>(256 + index);
        overflowing_pieces.push_back(
            {repeated_operand, repeated_operand, result}
        );
        repeated_operand = result;
    }
    require_throws(
        [&] {
            static_cast<void>(
                BytePairTokenizer(
                    std::span<const BpeMergeRule>(overflowing_pieces)
                )
            );
        },
        "BPE restoration should detect token-piece size overflow safely"
    );
}

void test_bpe_universal_bytes_and_binary_round_trip() {
    const BytePairTokenizer tokenizer("abababab", 258, 2);
    require(tokenizer.encode("").empty(), "empty BPE input should encode");
    require(
        tokenizer.decode(std::span<const TokenId>{}).empty(),
        "empty BPE token input should decode"
    );

    const std::string unseen("\0\xFFz", 3);
    const auto unseen_tokens = tokenizer.encode(unseen);
    require_tokens(
        unseen_tokens,
        {0, 255, 122},
        "unseen bytes should retain their universal base IDs"
    );
    require(
        tokenizer.decode(unseen_tokens) == unseen,
        "bytes absent from the training corpus should round-trip"
    );

    std::string all_bytes;
    all_bytes.reserve(256);
    for (std::size_t byte = 0; byte < 256; ++byte) {
        all_bytes.push_back(
            static_cast<char>(static_cast<unsigned char>(byte))
        );
    }
    const auto all_tokens = tokenizer.encode(all_bytes);
    require(
        tokenizer.decode(all_tokens) == all_bytes,
        "all 256 byte values should round-trip after unrelated BPE training"
    );

    const std::string utf8_and_nul(
        "\0caf\xC3\xA9 \xF0\x9F\x99\x82\0",
        12
    );
    const auto binary_tokens = tokenizer.encode(utf8_and_nul);
    require(
        tokenizer.decode(binary_tokens) == utf8_and_nul,
        "UTF-8, high bytes, and embedded NULs should round-trip exactly"
    );

    const std::array<std::uint8_t, 1> expected_ff{255};
    require_bytes(
        tokenizer.token_bytes(255),
        expected_ff,
        "base token 255 should expose byte 255"
    );
}

void test_bpe_stopping_conditions() {
    const BytePairTokenizer frequency_stop("abcabc", 300, 3);
    require(
        frequency_stop.vocab_size() == 256,
        "training should stop when no pair meets minimum frequency"
    );
    require(
        frequency_stop.merge_rules().empty(),
        "frequency stopping should not add a merge"
    );

    const BytePairTokenizer target_stop("abababab", 256, 1);
    require(
        target_stop.vocab_size() == 256,
        "base-only target should stop before learning merges"
    );
    require(
        target_stop.merge_rules().empty(),
        "base-only target should have no rules"
    );
    require_tokens(
        target_stop.encode("ab"),
        {97, 98},
        "base-only BPE encoding"
    );

    const BytePairTokenizer one_merge("abababab", 257, 1);
    require(
        one_merge.vocab_size() == 257 &&
            one_merge.merge_rules().size() == 1,
        "vocabulary target should stop after one merge"
    );

    const BytePairTokenizer exhausted("abc", 300, 1);
    require(
        exhausted.vocab_size() == 258,
        "training should stop once the corpus becomes one token"
    );
    require(
        exhausted.encode("abc").size() == 1,
        "exhausted corpus merges should still compress correctly"
    );

    const std::size_t largest_token_id_target =
        static_cast<std::size_t>(
            std::numeric_limits<TokenId>::max()
        );
    const BytePairTokenizer large_target(
        "ab",
        largest_token_id_target,
        1
    );
    require(
        large_target.vocab_size() == 257,
        "a large valid target should reserve only achievable merges"
    );
}

void test_arbitrary_byte_round_trip() {
    const std::string bytes(
        "\0\xC3\xA9\xF0\x9F\x99\x82",
        7
    );
    const ByteTokenizer tokenizer(bytes);
    const auto encoded = tokenizer.encode(bytes);

    require(encoded.size() == bytes.size(), "one token should represent a byte");
    require(
        tokenizer.decode(encoded) == bytes,
        "UTF-8 and embedded NUL bytes should round-trip exactly"
    );
}

void test_tokenizer_errors() {
    require_throws(
        [] { static_cast<void>(ByteTokenizer("")); },
        "an empty vocabulary corpus should throw"
    );

    const ByteTokenizer tokenizer("abc");
    require_throws(
        [&] { static_cast<void>(tokenizer.encode("d")); },
        "an unknown byte should throw"
    );

    const std::array<TokenId, 1> invalid_token{3};
    require_throws(
        [&] { static_cast<void>(tokenizer.decode(invalid_token)); },
        "an invalid token ID should throw"
    );
    require_throws(
        [&] { static_cast<void>(tokenizer.token_bytes(3)); },
        "legacy token byte lookup should reject an invalid ID"
    );

    require_throws(
        [] { static_cast<void>(BytePairTokenizer("", 256, 1)); },
        "BPE should reject an empty training corpus"
    );
    require_throws(
        [] { static_cast<void>(BytePairTokenizer("abc", 255, 1)); },
        "BPE should reject a vocabulary below the base-byte count"
    );
    require_throws(
        [] { static_cast<void>(BytePairTokenizer("abc", 256, 0)); },
        "BPE should reject a zero minimum pair frequency"
    );

    if constexpr (
        std::numeric_limits<std::size_t>::max() >
        std::numeric_limits<TokenId>::max()
    ) {
        require_throws(
            [] {
                const std::size_t invalid_target =
                    static_cast<std::size_t>(
                        std::numeric_limits<TokenId>::max()
                    ) + 1;
                static_cast<void>(
                    BytePairTokenizer("abc", invalid_target, 1)
                );
            },
            "BPE should reject a vocabulary outside the TokenId range"
        );
    }

    const BytePairTokenizer bpe("abababab", 258, 2);
    const std::array<TokenId, 1> invalid_bpe_token{258};
    require_throws(
        [&] { static_cast<void>(bpe.token_bytes(258)); },
        "BPE token byte lookup should reject an invalid ID"
    );
    require_throws(
        [&] { static_cast<void>(bpe.decode(invalid_bpe_token)); },
        "BPE decode should reject an invalid ID"
    );
}

void test_corpus_file(const std::filesystem::path& corpus_path) {
    const std::string corpus =
        riftco_transformer::read_file_bytes(corpus_path);
    require(corpus.size() == 438, "unexpected tiny corpus byte count");

    const ByteTokenizer tokenizer(corpus);
    require(tokenizer.vocab_size() == 27, "unexpected tiny corpus vocabulary");
    const auto tokens = tokenizer.encode(corpus);
    require(tokens.size() == corpus.size(), "corpus token count mismatch");
    require(
        tokenizer.decode(tokens) == corpus,
        "full corpus encode/decode should reproduce the file"
    );

    const BytePairTokenizer bpe(corpus, 300, 2);
    const auto bpe_tokens = bpe.encode(corpus);
    require(
        bpe.vocab_size() > 256 && bpe.vocab_size() <= 300,
        "file-trained BPE vocabulary should respect its target"
    );
    require(
        bpe_tokens.size() < corpus.size(),
        "file-trained BPE should compress the tiny corpus"
    );
    require(
        bpe.decode(bpe_tokens) == corpus,
        "file-trained BPE should reproduce exact corpus bytes"
    );

    require_throws(
        [&] {
            static_cast<void>(riftco_transformer::read_file_bytes(
                corpus_path.parent_path() / "missing-corpus.txt"
            ));
        },
        "reading a missing corpus should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::read_file_bytes(
                    corpus_path.parent_path()
                )
            );
        },
        "reading a directory as a corpus should throw"
    );
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("expected the tiny corpus path");
        }
        test_deterministic_vocabulary_and_round_trip();
        test_byte_vocabulary_restoration();
        test_strategy_factory_and_swapping();
        test_bpe_deterministic_merges_and_compression();
        test_bpe_merge_rule_restoration();
        test_invalid_restored_definitions();
        test_bpe_universal_bytes_and_binary_round_trip();
        test_bpe_stopping_conditions();
        test_arbitrary_byte_round_trip();
        test_tokenizer_errors();
        test_corpus_file(argv[1]);
        std::cout << "tokenizer tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
