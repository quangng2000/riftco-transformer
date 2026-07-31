#include "transformer_lab/config.hpp"
#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/data/tokenizer.hpp"
#include "transformer_lab/model/decoder_only_transformer.hpp"
#include "transformer_lab/nn/parameter.hpp"
#include "transformer_lab/stages/pretraining/stack.hpp"

#include <charconv>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr const char* kUsage =
    "usage: transformer_lab [--config path] [--steps count] "
    "[--metrics path] [--backend cpu|metal] "
    "[--attention materialized|flash] "
    "[--activation-checkpointing disabled|block]";

struct CommandLineOptions {
    std::filesystem::path config_path = "configs/tiny.conf";
    std::optional<std::size_t> training_steps;
    std::optional<std::filesystem::path> metrics_path;
    transformer_lab::ExecutionBackend backend =
        transformer_lab::ExecutionBackend::Cpu;
    transformer_lab::FullSequenceAttentionKind attention =
        transformer_lab::FullSequenceAttentionKind::Materialized;
    transformer_lab::ActivationCheckpointingKind
        activation_checkpointing =
            transformer_lab::ActivationCheckpointingKind::Disabled;
};

std::size_t parse_positive_size(const std::string& value) {
    std::size_t result = 0;
    const char* const begin = value.data();
    const char* const end = begin + value.size();
    const auto [cursor, error] =
        std::from_chars(begin, end, result);
    if (error != std::errc{} || cursor != end || result == 0) {
        throw std::runtime_error(
            "--steps requires a positive integer"
        );
    }
    return result;
}

transformer_lab::ExecutionBackend parse_backend(
    const std::string& value
) {
    if (value == "cpu") {
        return transformer_lab::ExecutionBackend::Cpu;
    }
    if (value == "metal") {
        return transformer_lab::ExecutionBackend::Metal;
    }
    throw std::runtime_error(
        "--backend must be 'cpu' or 'metal'"
    );
}

transformer_lab::FullSequenceAttentionKind parse_attention(
    const std::string& value
) {
    if (value == "materialized") {
        return transformer_lab::FullSequenceAttentionKind::Materialized;
    }
    if (value == "flash") {
        return transformer_lab::FullSequenceAttentionKind::Flash;
    }
    throw std::runtime_error(
        "--attention must be 'materialized' or 'flash'"
    );
}

transformer_lab::ActivationCheckpointingKind
parse_activation_checkpointing(const std::string& value) {
    if (value == "disabled") {
        return transformer_lab::ActivationCheckpointingKind::Disabled;
    }
    if (value == "block") {
        return transformer_lab::ActivationCheckpointingKind::
            TransformerBlock;
    }
    throw std::runtime_error(
        "--activation-checkpointing must be 'disabled' or 'block'"
    );
}

CommandLineOptions command_line_options(int argc, char** argv) {
    CommandLineOptions result;
    bool saw_config = false;
    bool saw_steps = false;
    bool saw_metrics = false;
    bool saw_backend = false;
    bool saw_attention = false;
    bool saw_activation_checkpointing = false;

    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        if (index + 1 >= argc) {
            throw std::runtime_error(kUsage);
        }
        const std::string value(argv[++index]);

        if (option == "--config" && !saw_config) {
            result.config_path = value;
            saw_config = true;
        } else if (option == "--steps" && !saw_steps) {
            result.training_steps = parse_positive_size(value);
            saw_steps = true;
        } else if (option == "--metrics" && !saw_metrics) {
            if (value.empty()) {
                throw std::runtime_error(
                    "--metrics requires a non-empty path"
                );
            }
            result.metrics_path = value;
            saw_metrics = true;
        } else if (option == "--backend" && !saw_backend) {
            result.backend = parse_backend(value);
            saw_backend = true;
        } else if (option == "--attention" && !saw_attention) {
            result.attention = parse_attention(value);
            saw_attention = true;
        } else if (option == "--activation-checkpointing" &&
                   !saw_activation_checkpointing) {
            result.activation_checkpointing =
                parse_activation_checkpointing(value);
            saw_activation_checkpointing = true;
        } else {
            throw std::runtime_error(kUsage);
        }
    }
    return result;
}

class MetricsCsv {
public:
    explicit MetricsCsv(std::filesystem::path path)
        : path_(std::move(path)) {
        if (path_.empty()) {
            throw std::invalid_argument(
                "metrics CSV path must not be empty"
            );
        }
        const std::filesystem::path parent = path_.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        output_.open(path_, std::ios::out | std::ios::trunc);
        if (!output_) {
            throw std::runtime_error(
                "could not open metrics CSV: " + path_.string()
            );
        }
        output_ << "step,loss,gradient_norm,clip_scale\n"
                << std::setprecision(
                       std::numeric_limits<double>::max_digits10
                   );
        require_output("write metrics CSV header");
    }

    MetricsCsv(const MetricsCsv&) = delete;
    MetricsCsv& operator=(const MetricsCsv&) = delete;

    void write(
        const transformer_lab::training::TrainingStepMetrics& metrics
    ) {
        output_ << metrics.step << ','
                << metrics.loss << ','
                << metrics.gradient_norm << ','
                << metrics.clip_scale << '\n';
        output_.flush();
        require_output("write metrics CSV row");
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
    std::ofstream output_;

    void require_output(const std::string& operation) const {
        if (!output_) {
            throw std::runtime_error(
                "failed to " + operation + ": " + path_.string()
            );
        }
    }
};

bool should_report(
    std::size_t step,
    std::size_t training_steps,
    std::size_t interval
) {
    return step == 1 ||
           step == training_steps ||
           step % interval == 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CommandLineOptions command_line =
            command_line_options(argc, argv);
        const transformer_lab::Config config =
            transformer_lab::Config::load(command_line.config_path);
        transformer_lab::set_execution_backend(
            command_line.backend
        );
        const std::size_t training_steps =
            command_line.training_steps.value_or(
                config.training_steps
            );
        const std::filesystem::path metrics_path =
            command_line.metrics_path.value_or(
                config.results_dir / "metrics.csv"
            );

        const std::string corpus =
            transformer_lab::read_file_bytes(config.corpus_path);
        transformer_lab::stages::pretraining::PretrainingConfig
            stage_config;
        stage_config.steps = training_steps;
        stage_config.context_size = config.context_size;
        stage_config.batch_size = config.batch_size;
        stage_config.model_width = config.d_model;
        stage_config.head_count = config.n_heads;
        stage_config.block_count = config.n_layers;
        stage_config.feed_forward_width = config.d_ff;
        stage_config.tokenizer = {
            transformer_lab::TokenizerMethod::CorpusByte,
            256,
            1,
        };
        stage_config.optimizer = {
            config.learning_rate,
            config.adam_beta1,
            config.adam_beta2,
            config.adam_epsilon,
            config.gradient_clip,
        };
        stage_config.model_seed = config.seed;
        stage_config.batch_seed = config.seed;
        stage_config.backend = command_line.backend;
        stage_config.attention = command_line.attention;
        stage_config.activation_checkpointing =
            command_line.activation_checkpointing;

        transformer_lab::stages::pretraining::PretrainingStack
            training(corpus, stage_config);
        const transformer_lab::ParameterList parameters =
            training.model().parameters();
        MetricsCsv metrics_csv(metrics_path);

        std::cout << "Riftco Transformer training run\n\n"
                  << config.summary() << "\n"
                  << "Run steps:     " << training_steps << '\n'
                  << "Corpus bytes:  " << corpus.size() << '\n'
                  << "Vocabulary:    "
                  << training.tokenizer().vocab_size() << '\n'
                  << "Parameters:    "
                  << transformer_lab::parameter_count(parameters)
                  << '\n'
                  << "Backend:       "
                  << transformer_lab::execution_backend_name(
                         transformer_lab::execution_backend()
                     )
                  << '\n'
                  << "Attention:     "
                  << (
                         command_line.attention ==
                                 transformer_lab::
                                     FullSequenceAttentionKind::Flash
                             ? "flash"
                             : "materialized"
                     )
                  << '\n'
                  << "Checkpointing: "
                  << (
                         command_line.activation_checkpointing ==
                                 transformer_lab::
                                     ActivationCheckpointingKind::
                                         TransformerBlock
                             ? "block"
                             : "disabled"
                     )
                  << '\n'
                  << "Metrics CSV:   " << metrics_csv.path()
                  << "\n\n";

        const auto result = training.run(
            [&](const transformer_lab::training::TrainingStepMetrics&
                    metrics) {
                metrics_csv.write(metrics);
                if (should_report(
                        metrics.step,
                        training_steps,
                        config.sample_every
                    )) {
                    std::cout << "step " << metrics.step
                              << '/' << training_steps
                              << " loss=" << metrics.loss
                              << " gradient_norm="
                              << metrics.gradient_norm
                              << " clip_scale="
                              << metrics.clip_scale
                              << '\n';
                }
            }
        );
        std::cout << "\nTraining loop: complete.\n"
                  << "First-batch loss before training: "
                  << result.first_batch_loss_before_training
                  << '\n'
                  << "First-batch loss after training:  "
                  << result.first_batch_loss_after_training
                  << '\n'
                  << "Adam steps: "
                  << result.metrics.back().step << '\n'
                  << "Native stage snapshot: ready for "
                     "post-training or serving.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
