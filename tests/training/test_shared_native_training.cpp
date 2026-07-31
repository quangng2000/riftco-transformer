#include "riftco_transformer/core/backend.hpp"
#include "riftco_transformer/core/tensor.hpp"
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/nn/parameter.hpp"
#include "riftco_transformer/optim/adam.hpp"
#include "riftco_transformer/training/training.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using riftco_transformer::Adam;
using riftco_transformer::AdamOptions;
using riftco_transformer::DecoderOnlyTransformer;
using riftco_transformer::ExecutionBackend;
using riftco_transformer::ParameterList;
using riftco_transformer::ScopedExecutionBackend;
using riftco_transformer::Tensor;
using riftco_transformer::TokenBatch;
using riftco_transformer::TokenId;
using riftco_transformer::global_gradient_norm;
using riftco_transformer::training::AdamOptimizerAdapter;
using riftco_transformer::training::BatchSource;
using riftco_transformer::training::CausalLanguageModelTrainer;
using riftco_transformer::training::OptimizerStepMetrics;
using riftco_transformer::training::OptimizerStrategy;
using riftco_transformer::training::RandomWindowBatchSource;
using riftco_transformer::training::SequenceWindowBatchSource;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(
    Function&& function,
    const std::string& message
) {
    bool threw_expected = false;
    try {
        function();
    } catch (const Exception&) {
        threw_expected = true;
    }
    require(threw_expected, message);
}

void require_same_batch(
    const TokenBatch& left,
    const TokenBatch& right,
    const std::string& message
) {
    require(left.shape() == right.shape(), message + ": shape");
    require(
        std::vector<TokenId>(
            left.inputs().begin(),
            left.inputs().end()
        ) ==
            std::vector<TokenId>(
                right.inputs().begin(),
                right.inputs().end()
            ),
        message + ": inputs"
    );
    require(
        std::vector<TokenId>(
            left.targets().begin(),
            left.targets().end()
        ) ==
            std::vector<TokenId>(
                right.targets().begin(),
                right.targets().end()
            ),
        message + ": targets"
    );
}

void require_shifted_rows(
    const TokenBatch& batch,
    const std::string& message
) {
    for (std::size_t row = 0;
         row < batch.batch_size();
         ++row) {
        for (std::size_t time = 1;
             time < batch.context_size();
             ++time) {
            require(
                batch.input_at(row, time) ==
                    batch.target_at(row, time - 1),
                message
            );
        }
    }
}

std::vector<float> parameter_values(
    const ParameterList& parameters
) {
    std::vector<float> result;
    for (const auto& named_parameter : parameters) {
        const auto values = named_parameter.parameter->value().data();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}

class BackendOnlyOptimizer final : public OptimizerStrategy {
public:
    explicit BackendOnlyOptimizer(ExecutionBackend backend)
        : backend_(backend) {}

    [[nodiscard]] ExecutionBackend backend() const noexcept override {
        return backend_;
    }

    [[nodiscard]] std::size_t step_count() const noexcept override {
        return step_count_;
    }

    [[nodiscard]] OptimizerStepMetrics step() override {
        ++step_count_;
        return {step_count_, 0.0, 1.0};
    }

    void zero_gradients() override {}

private:
    ExecutionBackend backend_;
    std::size_t step_count_ = 0;
};

void test_random_window_source_owns_and_replays_tokens() {
    std::vector<TokenId> caller_tokens{
        0, 1, 2, 3, 4, 5, 6, 7, 8,
    };
    RandomWindowBatchSource first(caller_tokens, 5, 3, 1729U);
    RandomWindowBatchSource replay(caller_tokens, 5, 3, 1729U);
    caller_tokens.assign(caller_tokens.size(), 99);

    require(first.batch_size() == 5, "random source batch size");
    require(first.context_size() == 3, "random source context size");
    require(
        first.valid_window_count() == 6,
        "random source valid window count"
    );
    require(
        first.tokens().front() == 0 &&
            first.tokens().back() == 8,
        "random source owns a detached token copy"
    );

    BatchSource& polymorphic = first;
    for (std::size_t index = 0; index < 4; ++index) {
        const TokenBatch actual = polymorphic.next_batch();
        const TokenBatch expected = replay.next_batch();
        require_same_batch(
            actual,
            expected,
            "seeded random source replay"
        );
        require_shifted_rows(
            actual,
            "random source rows must be shifted"
        );
    }

    require_throws<std::invalid_argument>(
        [] {
            RandomWindowBatchSource source({0, 1, 2}, 0, 2, 1U);
            static_cast<void>(source);
        },
        "random source rejects zero batch size"
    );
    require_throws<std::invalid_argument>(
        [] {
            RandomWindowBatchSource source({0, 1, 2}, 1, 0, 1U);
            static_cast<void>(source);
        },
        "random source rejects zero context size"
    );
    require_throws<std::invalid_argument>(
        [] {
            RandomWindowBatchSource source({0, 1}, 1, 2, 1U);
            static_cast<void>(source);
        },
        "random source rejects data without a target token"
    );
}

void test_sequence_source_is_deterministic_and_never_crosses() {
    std::vector<std::vector<TokenId>> caller_sequences{
        {0, 1, 2, 3, 4},
        {100, 101, 102, 103, 104},
        {50},
    };
    SequenceWindowBatchSource first(
        caller_sequences,
        64,
        2,
        991U
    );
    SequenceWindowBatchSource replay(
        caller_sequences,
        64,
        2,
        991U
    );
    caller_sequences[0][0] = 999;

    require(
        first.valid_window_count() == 6,
        "sequence source valid window count"
    );
    require(
        first.sequences().size() == 3 &&
            first.sequences()[0][0] == 0,
        "sequence source owns detached sequences"
    );

    for (std::size_t iteration = 0;
         iteration < 3;
         ++iteration) {
        const TokenBatch actual = first.next_batch();
        const TokenBatch expected = replay.next_batch();
        require_same_batch(
            actual,
            expected,
            "seeded sequence source replay"
        );
        require_shifted_rows(
            actual,
            "sequence source rows must be shifted"
        );
        for (std::size_t row = 0;
             row < actual.batch_size();
             ++row) {
            const bool low_sequence =
                actual.input_at(row, 0) < 10 &&
                actual.input_at(row, 1) < 10 &&
                actual.target_at(row, 0) < 10 &&
                actual.target_at(row, 1) < 10;
            const bool high_sequence =
                actual.input_at(row, 0) >= 100 &&
                actual.input_at(row, 1) >= 100 &&
                actual.target_at(row, 0) >= 100 &&
                actual.target_at(row, 1) >= 100;
            require(
                low_sequence || high_sequence,
                "sampled row crossed a sequence boundary"
            );
        }
    }

    require_throws<std::invalid_argument>(
        [] {
            SequenceWindowBatchSource source({}, 1, 2, 1U);
            static_cast<void>(source);
        },
        "sequence source rejects an empty collection"
    );
    require_throws<std::invalid_argument>(
        [] {
            SequenceWindowBatchSource source(
                {{0, 1}, {2}},
                1,
                2,
                1U
            );
            static_cast<void>(source);
        },
        "sequence source rejects collections without valid windows"
    );
    require_throws<std::invalid_argument>(
        [] {
            SequenceWindowBatchSource source({{0, 1, 2}}, 0, 2, 1U);
            static_cast<void>(source);
        },
        "sequence source rejects zero batch size"
    );
}

void test_adam_adapter_and_trainer() {
    ScopedExecutionBackend cpu(ExecutionBackend::Cpu);
    std::mt19937 random(137U);
    DecoderOnlyTransformer model(
        {
            5,
            3,
            4,
            2,
            1,
            8,
        },
        random
    );
    const ParameterList parameters = model.parameters();
    AdamOptions options;
    options.learning_rate = 1.0e-2F;
    Adam adam(parameters, options);
    AdamOptimizerAdapter optimizer(adam);

    require(&optimizer.adam() == &adam, "Adam adapter identity");
    require(
        optimizer.backend() == ExecutionBackend::Cpu,
        "Adam adapter backend"
    );
    require(optimizer.step_count() == 0, "Adam adapter initial step");

    CausalLanguageModelTrainer trainer(model, optimizer);
    require(&trainer.model() == &model, "trainer model identity");
    require(&trainer.optimizer() == &optimizer, "trainer optimizer identity");

    const TokenBatch batch(
        2,
        3,
        {
            0, 1, 2,
            1, 2, 3,
        },
        {
            1, 2, 3,
            2, 3, 4,
        }
    );
    const float initial_loss = trainer.evaluate_loss(batch);
    require(
        std::isfinite(initial_loss) && initial_loss > 0.0F,
        "trainer evaluation loss"
    );
    require(
        optimizer.step_count() == 0,
        "evaluation must not advance optimizer"
    );

    const std::vector<float> before = parameter_values(parameters);
    const auto metrics = trainer.train_step(batch);
    const std::vector<float> after = parameter_values(parameters);
    require(metrics.step == 1, "trainer step number");
    require(
        std::isfinite(metrics.loss) && metrics.loss > 0.0F,
        "trainer finite training loss"
    );
    require(
        std::isfinite(metrics.gradient_norm) &&
            metrics.gradient_norm >= 0.0,
        "trainer finite gradient norm"
    );
    require(
        std::isfinite(metrics.clip_scale) &&
            metrics.clip_scale > 0.0 &&
            metrics.clip_scale <= 1.0,
        "trainer valid clip scale"
    );
    require(
        optimizer.step_count() == 1 &&
            adam.step_count() == 1,
        "trainer advances wrapped Adam once"
    );
    require(before != after, "trainer updates model parameters");

    const float updated_loss = trainer.evaluate_loss(batch);
    require(
        std::isfinite(updated_loss),
        "updated model evaluation loss"
    );
    require(
        optimizer.step_count() == 1,
        "evaluation after training does not advance optimizer"
    );
    optimizer.zero_gradients();
    require(
        global_gradient_norm(parameters) == 0.0,
        "adapter zero_gradients clears all gradients"
    );

    const TokenBatch excessive_context(
        1,
        4,
        {0, 1, 2, 3},
        {1, 2, 3, 4}
    );
    require_throws<std::invalid_argument>(
        [&] {
            static_cast<void>(
                trainer.evaluate_loss(excessive_context)
            );
        },
        "trainer rejects context beyond model capacity"
    );
    const TokenBatch invalid_input(
        1,
        2,
        {0, 5},
        {1, 2}
    );
    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                trainer.evaluate_loss(invalid_input)
            );
        },
        "trainer rejects input outside vocabulary"
    );
    const TokenBatch invalid_target(
        1,
        2,
        {0, 1},
        {1, 5}
    );
    require_throws<std::out_of_range>(
        [&] {
            static_cast<void>(
                trainer.evaluate_loss(invalid_target)
            );
        },
        "trainer rejects target outside vocabulary"
    );

    BackendOnlyOptimizer wrong_backend(ExecutionBackend::Metal);
    require_throws<std::invalid_argument>(
        [&] {
            CausalLanguageModelTrainer invalid(model, wrong_backend);
            static_cast<void>(invalid);
        },
        "trainer rejects model/optimizer backend mismatch"
    );

    Tensor nonfinite(parameters.front().parameter->value());
    nonfinite.flat(0) = std::numeric_limits<float>::quiet_NaN();
    parameters.front().parameter->set_value(std::move(nonfinite));
    require_throws<std::domain_error>(
        [&] {
            static_cast<void>(trainer.evaluate_loss(batch));
        },
        "trainer rejects non-finite evaluation loss"
    );
}

}  // namespace

int main() {
    try {
        test_random_window_source_owns_and_replays_tokens();
        test_sequence_source_is_deterministic_and_never_crosses();
        test_adam_adapter_and_trainer();
        std::cout << "shared native training tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shared native training test failure: "
                  << error.what() << '\n';
        return 1;
    }
}
