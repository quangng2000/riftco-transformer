#include "riftco_transformer/data/token_batch.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using riftco_transformer::TokenBatch;
using riftco_transformer::TokenId;

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

void test_shifted_next_token_windows() {
    const std::vector<TokenId> tokens{
        0, 1, 2, 3, 4, 5, 6, 7, 8,
    };
    const std::array<std::size_t, 2> starts{0, 3};
    const TokenBatch batch = riftco_transformer::make_next_token_batch(
        tokens,
        starts,
        3
    );

    require(
        batch.shape() == TokenBatch::Shape{2, 3},
        "token batch shape mismatch"
    );
    require(batch.batch_size() == 2, "batch size mismatch");
    require(batch.context_size() == 3, "context size mismatch");
    require_tokens(
        batch.inputs(),
        {0, 1, 2, 3, 4, 5},
        "row-major batch inputs"
    );
    require_tokens(
        batch.targets(),
        {1, 2, 3, 4, 5, 6},
        "shifted batch targets"
    );

    for (std::size_t row = 0; row < starts.size(); ++row) {
        for (std::size_t time = 0; time < batch.context_size(); ++time) {
            require(
                batch.input_at(row, time) ==
                    tokens[starts[row] + time],
                "input window mapping mismatch"
            );
            require(
                batch.target_at(row, time) ==
                    tokens[starts[row] + time + 1],
                "target should be input shifted by one token"
            );
        }
    }
}

void test_explicit_overlapping_and_boundary_windows() {
    const std::vector<TokenId> tokens{
        10, 11, 12, 13, 14, 15, 16, 17, 18,
    };
    const std::array<std::size_t, 3> starts{2, 2, 1};
    const TokenBatch overlapping =
        riftco_transformer::make_next_token_batch(tokens, starts, 2);

    require_tokens(
        overlapping.inputs(),
        {12, 13, 12, 13, 11, 12},
        "overlapping inputs"
    );
    require_tokens(
        overlapping.targets(),
        {13, 14, 13, 14, 12, 13},
        "overlapping targets"
    );

    const std::array<std::size_t, 1> boundary_start{6};
    const TokenBatch boundary =
        riftco_transformer::make_next_token_batch(
            tokens,
            boundary_start,
            2
        );
    require_tokens(boundary.inputs(), {16, 17}, "boundary inputs");
    require_tokens(boundary.targets(), {17, 18}, "boundary targets");
}

void test_seeded_random_sampling_is_deterministic() {
    const std::vector<TokenId> tokens{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    };
    constexpr std::size_t batch_size = 6;
    constexpr std::size_t context_size = 3;
    constexpr unsigned int seed = 1729U;

    std::mt19937 expected_random(seed);
    std::uniform_int_distribution<std::size_t> start_distribution(
        0,
        tokens.size() - context_size - 1
    );
    std::vector<std::size_t> expected_starts;
    expected_starts.reserve(batch_size);
    for (std::size_t row = 0; row < batch_size; ++row) {
        expected_starts.push_back(start_distribution(expected_random));
    }
    const TokenBatch expected = riftco_transformer::make_next_token_batch(
        tokens,
        expected_starts,
        context_size
    );

    std::mt19937 random(seed);
    const TokenBatch sampled =
        riftco_transformer::sample_next_token_batch(
            tokens,
            batch_size,
            context_size,
            random
        );
    require(
        sampled.shape() == TokenBatch::Shape{batch_size, context_size},
        "seeded sample shape mismatch"
    );
    require_tokens(
        sampled.inputs(),
        std::vector<TokenId>(
            expected.inputs().begin(),
            expected.inputs().end()
        ),
        "seeded sample inputs"
    );
    require_tokens(
        sampled.targets(),
        std::vector<TokenId>(
            expected.targets().begin(),
            expected.targets().end()
        ),
        "seeded sample targets"
    );

    std::mt19937 replay_random(seed);
    const TokenBatch replay =
        riftco_transformer::sample_next_token_batch(
            tokens,
            batch_size,
            context_size,
            replay_random
        );
    require_tokens(
        replay.inputs(),
        std::vector<TokenId>(
            sampled.inputs().begin(),
            sampled.inputs().end()
        ),
        "same seed should replay sampled inputs"
    );
    require_tokens(
        replay.targets(),
        std::vector<TokenId>(
            sampled.targets().begin(),
            sampled.targets().end()
        ),
        "same seed should replay sampled targets"
    );
}

void test_random_sampling_exact_window_uses_replacement() {
    const std::vector<TokenId> tokens{20, 21, 22, 23};
    std::mt19937 random(99U);
    const TokenBatch batch =
        riftco_transformer::sample_next_token_batch(
            tokens,
            4,
            3,
            random
        );

    require(
        batch.shape() == TokenBatch::Shape{4, 3},
        "single-window replacement batch shape"
    );
    require_tokens(
        batch.inputs(),
        {
            20, 21, 22,
            20, 21, 22,
            20, 21, 22,
            20, 21, 22,
        },
        "single valid window is sampled with replacement"
    );
    require_tokens(
        batch.targets(),
        {
            21, 22, 23,
            21, 22, 23,
            21, 22, 23,
            21, 22, 23,
        },
        "replacement targets remain shifted by one token"
    );
}

void test_random_sampling_errors() {
    const std::vector<TokenId> tokens{0, 1, 2, 3};
    std::mt19937 random(7U);

    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::sample_next_token_batch(
                    tokens,
                    0,
                    2,
                    random
                )
            );
        },
        "zero random batch size should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::sample_next_token_batch(
                    tokens,
                    2,
                    0,
                    random
                )
            );
        },
        "zero random context size should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::sample_next_token_batch(
                    tokens,
                    2,
                    tokens.size(),
                    random
                )
            );
        },
        "a corpus without the final target token should throw"
    );

    const std::vector<TokenId> empty_tokens;
    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::sample_next_token_batch(
                    empty_tokens,
                    1,
                    1,
                    random
                )
            );
        },
        "an empty corpus should be too short to sample"
    );
}

void test_batch_errors() {
    const std::vector<TokenId> tokens{0, 1, 2, 3, 4};
    const std::array<std::size_t, 1> valid_start{2};
    const std::array<std::size_t, 1> too_late_start{3};
    const std::array<std::size_t, 1> beyond_end_start{6};
    const std::vector<std::size_t> no_starts;

    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::make_next_token_batch(
                    tokens,
                    no_starts,
                    2
                )
            );
        },
        "an empty set of batch windows should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::make_next_token_batch(
                    tokens,
                    valid_start,
                    0
                )
            );
        },
        "zero context should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::make_next_token_batch(
                    tokens,
                    too_late_start,
                    2
                )
            );
        },
        "a window without its final target should throw"
    );
    require_throws(
        [&] {
            static_cast<void>(
                riftco_transformer::make_next_token_batch(
                    tokens,
                    beyond_end_start,
                    1
                )
            );
        },
        "a start beyond the corpus should throw"
    );

    require_throws(
        [] {
            static_cast<void>(TokenBatch(1, 2, {1}, {2}));
        },
        "batch data count mismatch should throw"
    );
    require_throws(
        [] {
            static_cast<void>(TokenBatch(0, 2, {}, {}));
        },
        "zero batch dimension should throw"
    );
    require_throws(
        [] {
            static_cast<void>(TokenBatch(
                std::numeric_limits<std::size_t>::max(),
                2,
                {},
                {}
            ));
        },
        "batch shape overflow should throw"
    );

    const TokenBatch valid(1, 2, {1, 2}, {2, 3});
    require_throws(
        [&] { static_cast<void>(valid.input_at(1, 0)); },
        "out-of-range batch row should throw"
    );
    require_throws(
        [&] { static_cast<void>(valid.target_at(0, 2)); },
        "out-of-range time position should throw"
    );
}

}  // namespace

int main() {
    try {
        test_shifted_next_token_windows();
        test_explicit_overlapping_and_boundary_windows();
        test_seeded_random_sampling_is_deterministic();
        test_random_sampling_exact_window_uses_replacement();
        test_random_sampling_errors();
        test_batch_errors();
        std::cout << "token batch tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
