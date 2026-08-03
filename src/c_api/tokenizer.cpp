#include "internal/bridge.hpp"

using namespace riftco_transformer::c_api::detail;

namespace {

std::string_view checked_bytes(
    const std::uint8_t* values,
    std::uint64_t value_count,
    const char* description
) {
    const std::size_t count =
        checked_size(value_count, description);
    if (count != 0 && values == nullptr) {
        throw std::invalid_argument(
            std::string(description) +
            " pointer must not be null"
        );
    }
    if (count == 0) {
        return {};
    }
    return {
        reinterpret_cast<const char*>(values),
        count,
    };
}

void require_tokenizer(const rt_tokenizer* tokenizer) {
    if (tokenizer == nullptr || tokenizer->value == nullptr) {
        throw std::invalid_argument(
            "tokenizer handle must not be null"
        );
    }
}

riftco_transformer::TokenizerMethod checked_tokenizer_method(
    rt_tokenizer_method method
) {
    switch (method) {
        case RT_TOKENIZER_METHOD_BYTE:
            return riftco_transformer::TokenizerMethod::CorpusByte;
        case RT_TOKENIZER_METHOD_BPE:
            return riftco_transformer::TokenizerMethod::BytePair;
        default:
            throw std::invalid_argument(
                "unknown C API tokenizer method"
            );
    }
}

rt_tokenizer_method c_tokenizer_method(
    riftco_transformer::TokenizerMethod method
) {
    switch (method) {
        case riftco_transformer::TokenizerMethod::CorpusByte:
            return RT_TOKENIZER_METHOD_BYTE;
        case riftco_transformer::TokenizerMethod::BytePair:
            return RT_TOKENIZER_METHOD_BPE;
    }
    throw std::invalid_argument("unknown native tokenizer method");
}

riftco_transformer::TokenizerOptions checked_tokenizer_options(
    const rt_tokenizer_options* options
) {
    if (options == nullptr) {
        return {};
    }
    constexpr std::size_t minimum_size =
        offsetof(rt_tokenizer_options, minimum_pair_frequency) +
        sizeof(std::uint64_t);
    checked_structure_size(
        options->struct_size,
        minimum_size,
        "tokenizer options structure"
    );
    if (options->reserved != 0) {
        throw std::invalid_argument(
            "tokenizer options reserved field must be zero"
        );
    }

    const auto method = checked_tokenizer_method(options->method);
    if (method == riftco_transformer::TokenizerMethod::CorpusByte) {
        return {
            method,
            riftco_transformer::TokenizerOptions{}.vocabulary_size,
            riftco_transformer::TokenizerOptions{}.minimum_pair_frequency,
        };
    }

    if (options->vocabulary_size <
        static_cast<std::uint64_t>(256)) {
        throw std::invalid_argument(
            "BPE tokenizer vocabulary size must be at least 256"
        );
    }
    if (options->vocabulary_size >
        static_cast<std::uint64_t>(
            std::numeric_limits<riftco_transformer::TokenId>::max()
        )) {
        throw std::overflow_error(
            "tokenizer vocabulary size exceeds token ID range"
        );
    }
    if (options->minimum_pair_frequency == 0) {
        throw std::invalid_argument(
            "tokenizer minimum pair frequency must be positive"
        );
    }

    return {
        method,
        checked_size(
            options->vocabulary_size,
            "tokenizer vocabulary size"
        ),
        checked_size(
            options->minimum_pair_frequency,
            "tokenizer minimum pair frequency"
        ),
    };
}

}  // namespace


extern "C" {

rt_status RT_CALL rt_tokenizer_options_init(
    rt_tokenizer_options* options,
    uint64_t options_size
) {
    return guard([&] {
        if (options == nullptr) {
            throw std::invalid_argument(
                "tokenizer options must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(
                rt_tokenizer_options,
                minimum_pair_frequency
            ) +
            sizeof(std::uint64_t);
        checked_structure_size(
            options_size,
            minimum_size,
            "tokenizer options structure"
        );
        *options = {
            options_size,
            RT_TOKENIZER_METHOD_BYTE,
            0,
            512,
            2,
        };
    });
}

rt_status RT_CALL rt_tokenizer_create(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    rt_tokenizer** output
) {
    return rt_tokenizer_create_with_options(
        corpus_bytes,
        corpus_size,
        nullptr,
        output
    );
}

rt_status RT_CALL rt_tokenizer_create_with_options(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    const rt_tokenizer_options* options,
    rt_tokenizer** output
) {
    return guard([&] {
        require_output(output);
        auto result = std::make_unique<rt_tokenizer>(
            riftco_transformer::make_tokenizer(
                checked_bytes(
                    corpus_bytes,
                    corpus_size,
                    "tokenizer corpus"
                ),
                checked_tokenizer_options(options)
            )
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_tokenizer_create_from_byte_vocabulary(
    const uint8_t* ordered_vocabulary,
    uint64_t vocabulary_size,
    rt_tokenizer** output
) {
    return guard([&] {
        require_output(output);
        const std::size_t count = checked_size(
            vocabulary_size,
            "tokenizer byte-vocabulary size"
        );
        if (count != 0 && ordered_vocabulary == nullptr) {
            throw std::invalid_argument(
                "tokenizer byte vocabulary must not be null"
            );
        }
        auto result = std::make_unique<rt_tokenizer>(
            std::make_unique<riftco_transformer::ByteTokenizer>(
                std::span<const std::uint8_t>(
                    ordered_vocabulary,
                    count
                )
            )
        );
        *output = result.release();
    });
}

rt_status RT_CALL rt_tokenizer_create_from_bpe_merges(
    const rt_bpe_merge_rule* ordered_merge_rules,
    uint64_t merge_count,
    rt_tokenizer** output
) {
    return guard([&] {
        require_output(output);
        const std::size_t count = checked_size(
            merge_count,
            "BPE merge-rule count"
        );
        if (count != 0 && ordered_merge_rules == nullptr) {
            throw std::invalid_argument(
                "BPE merge rules must not be null"
            );
        }
        std::vector<riftco_transformer::BpeMergeRule> rules;
        rules.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto& rule = ordered_merge_rules[index];
            rules.push_back({
                rule.left,
                rule.right,
                rule.result,
            });
        }
        auto result = std::make_unique<rt_tokenizer>(
            std::make_unique<riftco_transformer::BytePairTokenizer>(
                std::span<const riftco_transformer::BpeMergeRule>(
                    rules.data(),
                    rules.size()
                )
            )
        );
        *output = result.release();
    });
}

void RT_CALL rt_tokenizer_release(rt_tokenizer* tokenizer) {
    delete tokenizer;
}

rt_status RT_CALL rt_tokenizer_get_method(
    const rt_tokenizer* tokenizer,
    rt_tokenizer_method* output
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (output == nullptr) {
            throw std::invalid_argument(
                "tokenizer method output must not be null"
            );
        }
        *output = c_tokenizer_method(
            tokenizer->value->method()
        );
    });
}

rt_status RT_CALL rt_tokenizer_vocabulary_size(
    const rt_tokenizer* tokenizer,
    uint64_t* output
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (output == nullptr) {
            throw std::invalid_argument(
                "tokenizer vocabulary-size output must not be null"
            );
        }
        *output = checked_u64(
            tokenizer->value->vocab_size(),
            "tokenizer vocabulary size"
        );
    });
}

rt_status RT_CALL rt_tokenizer_bpe_merge_count(
    const rt_tokenizer* tokenizer,
    uint64_t* output
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (output == nullptr) {
            throw std::invalid_argument(
                "BPE merge-count output must not be null"
            );
        }
        const auto* byte_pair =
            dynamic_cast<const riftco_transformer::BytePairTokenizer*>(
                tokenizer->value.get()
            );
        if (byte_pair == nullptr) {
            throw std::invalid_argument(
                "BPE merge rules require a BPE tokenizer"
            );
        }
        *output = checked_u64(
            byte_pair->merge_rules().size(),
            "BPE merge-rule count"
        );
    });
}

rt_status RT_CALL rt_tokenizer_bpe_merge_rule(
    const rt_tokenizer* tokenizer,
    uint64_t index,
    rt_bpe_merge_rule* output
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (output == nullptr) {
            throw std::invalid_argument(
                "BPE merge-rule output must not be null"
            );
        }
        const auto* byte_pair =
            dynamic_cast<const riftco_transformer::BytePairTokenizer*>(
                tokenizer->value.get()
            );
        if (byte_pair == nullptr) {
            throw std::invalid_argument(
                "BPE merge rules require a BPE tokenizer"
            );
        }
        const std::size_t native_index = checked_size(
            index,
            "BPE merge-rule index"
        );
        const auto rules = byte_pair->merge_rules();
        if (native_index >= rules.size()) {
            throw std::out_of_range(
                "BPE merge-rule index is outside the tokenizer"
            );
        }
        const auto& rule = rules[native_index];
        *output = {
            rule.left,
            rule.right,
            rule.result,
        };
    });
}

rt_status RT_CALL rt_tokenizer_vocabulary(
    const rt_tokenizer* tokenizer,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        if (tokenizer->value->method() !=
            riftco_transformer::TokenizerMethod::CorpusByte) {
            throw std::invalid_argument(
                "tokenizer vocabulary is only available for "
                "the corpus-byte method; use "
                "rt_tokenizer_token_bytes for BPE"
            );
        }

        std::vector<std::uint8_t> vocabulary;
        vocabulary.reserve(tokenizer->value->vocab_size());
        for (std::size_t index = 0;
             index < tokenizer->value->vocab_size();
             ++index) {
            const auto bytes = tokenizer->value->token_bytes(
                static_cast<riftco_transformer::TokenId>(index)
            );
            if (bytes.size() != 1) {
                throw std::logic_error(
                    "corpus-byte tokenizer produced a non-byte token"
                );
            }
            vocabulary.push_back(bytes.front());
        }
        copy_sized_output(
            std::span<const std::uint8_t>(
                vocabulary.data(),
                vocabulary.size()
            ),
            output,
            capacity,
            required_count,
            "tokenizer vocabulary"
        );
    });
}

rt_status RT_CALL rt_tokenizer_token_bytes(
    const rt_tokenizer* tokenizer,
    uint32_t token_id,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        const auto bytes = tokenizer->value->token_bytes(token_id);
        copy_sized_output(
            bytes,
            output,
            capacity,
            required_count,
            "tokenizer token bytes"
        );
    });
}

rt_status RT_CALL rt_tokenizer_encode(
    const rt_tokenizer* tokenizer,
    const uint8_t* text,
    uint64_t text_size,
    uint32_t* output,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        const auto tokens = tokenizer->value->encode(
            checked_bytes(
                text,
                text_size,
                "tokenizer text"
            )
        );
        copy_sized_output(
            std::span<const riftco_transformer::TokenId>(
                tokens.data(),
                tokens.size()
            ),
            output,
            capacity,
            required_count,
            "encoded token"
        );
    });
}

rt_status RT_CALL rt_tokenizer_decode(
    const rt_tokenizer* tokenizer,
    const uint32_t* tokens,
    uint64_t token_count,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_tokenizer(tokenizer);
        const std::string text = tokenizer->value->decode(
            checked_token_ids(
                tokens,
                token_count,
                "encoded token"
            )
        );
        copy_sized_output(
            std::span<const char>(
                text.data(),
                text.size()
            ),
            output,
            capacity,
            required_count,
            "decoded byte"
        );
    });
}

}  // extern "C"
