#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/core/tensor_ops.hpp"
#include "transformer_lab/model/causal_self_attention.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using transformer_lab::CausalSelfAttention;
using transformer_lab::ExecutionBackend;
using transformer_lab::FullSequenceAttentionKind;
using transformer_lab::Tensor;
using transformer_lab::Variable;
namespace tensor_ops = transformer_lab::tensor_ops;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_parameter_backend(const transformer_lab::ParameterList &parameters,
                               ExecutionBackend backend,
                               const std::string &message) {
  require(!parameters.empty(), message + ": empty parameter list");
  for (const auto &named_parameter : parameters) {
    require(named_parameter.parameter != nullptr,
            message + ": null parameter " + named_parameter.name);
    require(named_parameter.parameter->value().backend() == backend,
            message + ": value backend for " + named_parameter.name);
    require(named_parameter.parameter->gradient().backend() == backend,
            message + ": gradient backend for " + named_parameter.name);
  }
}

void require_close(float actual, float expected, const std::string &message,
                   float tolerance = 1.0e-5F) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

void require_tensor_close(const Tensor &actual,
                          const Tensor::Shape &expected_shape,
                          const std::vector<float> &expected_values,
                          const std::string &message,
                          float tolerance = 1.0e-5F) {
  require(actual.shape() == expected_shape, message + ": shape mismatch");
  require(actual.numel() == expected_values.size(),
          message + ": value count mismatch");
  for (std::size_t index = 0; index < expected_values.size(); ++index) {
    require_close(actual.flat(index), expected_values[index],
                  message + " at flat index " + std::to_string(index),
                  tolerance);
  }
}

void require_tensor_close(const Tensor &actual, const Tensor &expected,
                          const std::string &message,
                          float tolerance = 1.0e-5F) {
  require_tensor_close(
      actual, expected.shape(),
      std::vector<float>(expected.data().begin(), expected.data().end()),
      message, tolerance);
}

template <typename Function>
void require_throws(Function &&function, const std::string &message) {
  bool threw = false;
  try {
    function();
  } catch (const std::exception &) {
    threw = true;
  }
  require(threw, message);
}

float tensor_dot(const Tensor &left, const Tensor &right) {
  return tensor_ops::sum(tensor_ops::multiply(left, right)).flat(0);
}

Tensor identity_matrix(std::size_t width) {
  Tensor result({width, width}, 0.0F);
  for (std::size_t index = 0; index < width; ++index) {
    result.at({index, index}) = 1.0F;
  }
  return result;
}

struct ReferenceAttentionResult {
  Tensor context;
  Tensor probabilities;
};

ReferenceAttentionResult reference_causal_attention(const Tensor &queries,
                                                    const Tensor &keys,
                                                    const Tensor &values) {
  require(queries.rank() == 4 && queries.shape() == keys.shape() &&
              queries.shape() == values.shape(),
          "reference attention input shape");

  const auto batch = queries.shape()[0];
  const auto heads = queries.shape()[1];
  const auto time = queries.shape()[2];
  const auto width = queries.shape()[3];
  const double scale = 1.0 / std::sqrt(static_cast<double>(width));
  Tensor context(queries.shape());
  Tensor probabilities({batch, heads, time, time}, 0.0F);

  for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::size_t head = 0; head < heads; ++head) {
      for (std::size_t query_time = 0; query_time < time; ++query_time) {
        std::vector<double> scores(query_time + 1, 0.0);
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
          double score = 0.0;
          for (std::size_t channel = 0; channel < width; ++channel) {
            score += static_cast<double>(queries.at({
                         batch_index,
                         head,
                         query_time,
                         channel,
                     })) *
                     static_cast<double>(keys.at({
                         batch_index,
                         head,
                         key_time,
                         channel,
                     }));
          }
          score *= scale;
          scores[key_time] = score;
          if (score > maximum) {
            maximum = score;
          }
        }

        double denominator = 0.0;
        for (double &score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }

        for (std::size_t channel = 0; channel < width; ++channel) {
          double total = 0.0;
          for (std::size_t key_time = 0; key_time <= query_time; ++key_time) {
            const double probability = scores[key_time] / denominator;
            probabilities.at({
                batch_index,
                head,
                query_time,
                key_time,
            }) = static_cast<float>(probability);
            total += probability * static_cast<double>(values.at({
                                       batch_index,
                                       head,
                                       key_time,
                                       channel,
                                   }));
          }
          context.at({
              batch_index,
              head,
              query_time,
              channel,
          }) = static_cast<float>(total);
        }
      }
    }
  }
  return {
      std::move(context),
      std::move(probabilities),
  };
}

Tensor reference_linear(const Tensor &input, const Tensor &weight,
                        const Tensor &bias) {
  require(input.rank() == 3 && weight.rank() == 2 && bias.rank() == 1 &&
              input.shape()[2] == weight.shape()[1] &&
              weight.shape()[0] == bias.shape()[0],
          "reference linear input shape");
  const auto batch = input.shape()[0];
  const auto time = input.shape()[1];
  const auto input_width = input.shape()[2];
  const auto output_width = weight.shape()[0];
  Tensor result({batch, time, output_width}, 0.0F);
  for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::size_t time_index = 0; time_index < time; ++time_index) {
      for (std::size_t output_channel = 0; output_channel < output_width;
           ++output_channel) {
        float total = bias.at({output_channel});
        for (std::size_t input_channel = 0; input_channel < input_width;
             ++input_channel) {
          total += input.at({
                       batch_index,
                       time_index,
                       input_channel,
                   }) *
                   weight.at({
                       output_channel,
                       input_channel,
                   });
        }
        result.at({
            batch_index,
            time_index,
            output_channel,
        }) = total;
      }
    }
  }
  return result;
}

Tensor reference_split_heads(const Tensor &input, std::size_t head_count) {
  require(input.rank() == 3 && head_count > 0 &&
              input.shape()[2] % head_count == 0,
          "reference split-head input shape");
  const auto batch = input.shape()[0];
  const auto time = input.shape()[1];
  const auto head_width = input.shape()[2] / head_count;
  Tensor result({batch, head_count, time, head_width}, 0.0F);
  for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::size_t head = 0; head < head_count; ++head) {
      for (std::size_t time_index = 0; time_index < time; ++time_index) {
        for (std::size_t channel = 0; channel < head_width; ++channel) {
          result.at({
              batch_index,
              head,
              time_index,
              channel,
          }) = input.at({
              batch_index,
              time_index,
              head * head_width + channel,
          });
        }
      }
    }
  }
  return result;
}

Tensor reference_merge_heads(const Tensor &input) {
  require(input.rank() == 4, "reference merge-head input shape");
  const auto batch = input.shape()[0];
  const auto head_count = input.shape()[1];
  const auto time = input.shape()[2];
  const auto head_width = input.shape()[3];
  Tensor result({batch, time, head_count * head_width}, 0.0F);
  for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::size_t time_index = 0; time_index < time; ++time_index) {
      for (std::size_t head = 0; head < head_count; ++head) {
        for (std::size_t channel = 0; channel < head_width; ++channel) {
          result.at({
              batch_index,
              time_index,
              head * head_width + channel,
          }) = input.at({
              batch_index,
              head,
              time_index,
              channel,
          });
        }
      }
    }
  }
  return result;
}

Tensor reference_attention_module(
    const Tensor &input, std::size_t head_count, const Tensor &query_weight,
    const Tensor &query_bias, const Tensor &key_weight, const Tensor &key_bias,
    const Tensor &value_weight, const Tensor &value_bias,
    const Tensor &output_weight, const Tensor &output_bias) {
  const Tensor queries = reference_split_heads(
      reference_linear(input, query_weight, query_bias), head_count);
  const Tensor keys = reference_split_heads(
      reference_linear(input, key_weight, key_bias), head_count);
  const Tensor values = reference_split_heads(
      reference_linear(input, value_weight, value_bias), head_count);
  const auto attention = reference_causal_attention(queries, keys, values);
  return reference_linear(reference_merge_heads(attention.context),
                          output_weight, output_bias);
}

void set_named_parameter(CausalSelfAttention &attention,
                         const std::string &name, const Tensor &value) {
  for (auto &named_parameter : attention.parameters()) {
    if (named_parameter.name == name) {
      named_parameter.parameter->set_value(value);
      return;
    }
  }
  throw std::runtime_error("missing attention parameter: " + name);
}

void test_split_and_merge_heads() {
  const Tensor input_values({2, 3, 4},
                            {
                                0.0F,   1.0F,   2.0F,   3.0F,   10.0F,  11.0F,
                                12.0F,  13.0F,  20.0F,  21.0F,  22.0F,  23.0F,
                                100.0F, 101.0F, 102.0F, 103.0F, 110.0F, 111.0F,
                                112.0F, 113.0F, 120.0F, 121.0F, 122.0F, 123.0F,
                            });
  const Variable input(input_values);
  const Variable split = transformer_lab::split_attention_heads(input, 2);
  require_tensor_close(split.value(), {2, 2, 3, 2},
                       {
                           0.0F,   1.0F,   10.0F,  11.0F,  20.0F,  21.0F,
                           2.0F,   3.0F,   12.0F,  13.0F,  22.0F,  23.0F,
                           100.0F, 101.0F, 110.0F, 111.0F, 120.0F, 121.0F,
                           102.0F, 103.0F, 112.0F, 113.0F, 122.0F, 123.0F,
                       },
                       "split-head ordering");
  require_tensor_close(transformer_lab::merge_attention_heads(split).value(),
                       input_values.shape(),
                       std::vector<float>(input_values.data().begin(),
                                          input_values.data().end()),
                       "merge should invert split");

  split.backward(Tensor(
      {2, 2, 3, 2}, {
                        1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
                        9.0F,  10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F,
                        17.0F, 18.0F, 19.0F, 20.0F, 21.0F, 22.0F, 23.0F, 24.0F,
                    }));
  require_tensor_close(input.gradient(), {2, 3, 4},
                       {
                           1.0F,  2.0F,  7.0F,  8.0F,  3.0F,  4.0F,
                           9.0F,  10.0F, 5.0F,  6.0F,  11.0F, 12.0F,
                           13.0F, 14.0F, 19.0F, 20.0F, 15.0F, 16.0F,
                           21.0F, 22.0F, 17.0F, 18.0F, 23.0F, 24.0F,
                       },
                       "split-head backward ordering");

  require_throws(
      [] {
        static_cast<void>(transformer_lab::split_attention_heads(
            Variable(Tensor({2, 4}), false), 2));
      },
      "split heads should reject non-rank-three input");
  require_throws(
      [&] {
        static_cast<void>(transformer_lab::split_attention_heads(input, 0));
      },
      "split heads should reject zero heads");
  require_throws(
      [&] {
        static_cast<void>(transformer_lab::split_attention_heads(input, 3));
      },
      "split heads should reject indivisible width");
  require_throws(
      [] {
        static_cast<void>(transformer_lab::merge_attention_heads(
            Variable(Tensor({2, 3, 4}), false)));
      },
      "merge heads should reject non-rank-four input");
}

void test_scaled_causal_attention_forward() {
  const float log_two = std::log(2.0F);
  const float log_four = std::log(4.0F);
  const Variable queries(Tensor({1, 1, 3, 4}, {
                                                  2.0F,
                                                  0.0F,
                                                  0.0F,
                                                  0.0F,
                                                  2.0F,
                                                  0.0F,
                                                  0.0F,
                                                  0.0F,
                                                  2.0F,
                                                  0.0F,
                                                  0.0F,
                                                  0.0F,
                                              }));
  const Variable keys(Tensor({1, 1, 3, 4}, {
                                               0.0F,
                                               0.0F,
                                               0.0F,
                                               0.0F,
                                               log_two,
                                               0.0F,
                                               0.0F,
                                               0.0F,
                                               log_four,
                                               0.0F,
                                               0.0F,
                                               0.0F,
                                           }));
  const Variable values(Tensor({1, 1, 3, 4}, {
                                                 10.0F,
                                                 1.0F,
                                                 0.0F,
                                                 -1.0F,
                                                 20.0F,
                                                 2.0F,
                                                 1.0F,
                                                 0.0F,
                                                 40.0F,
                                                 4.0F,
                                                 2.0F,
                                                 1.0F,
                                             }));

  const auto attention = transformer_lab::causal_scaled_dot_product_attention(
      queries, keys, values);
  require_tensor_close(attention.probabilities.value(), {1, 1, 3, 3},
                       {
                           1.0F,
                           0.0F,
                           0.0F,
                           1.0F / 3.0F,
                           2.0F / 3.0F,
                           0.0F,
                           1.0F / 7.0F,
                           2.0F / 7.0F,
                           4.0F / 7.0F,
                       },
                       "scaled causal probabilities", 2.0e-6F);
  require_tensor_close(attention.context.value(), {1, 1, 3, 4},
                       {
                           10.0F,
                           1.0F,
                           0.0F,
                           -1.0F,
                           50.0F / 3.0F,
                           5.0F / 3.0F,
                           2.0F / 3.0F,
                           -1.0F / 3.0F,
                           30.0F,
                           3.0F,
                           10.0F / 7.0F,
                           3.0F / 7.0F,
                       },
                       "scaled causal context", 5.0e-6F);

  for (std::size_t query_time = 0; query_time < 3; ++query_time) {
    float row_sum = 0.0F;
    for (std::size_t key_time = 0; key_time < 3; ++key_time) {
      const float probability = attention.probabilities.value().at({
          0,
          0,
          query_time,
          key_time,
      });
      row_sum += probability;
      if (key_time > query_time) {
        require(probability == 0.0F,
                "future attention probability must be exactly zero");
      }
    }
    require_close(row_sum, 1.0F, "attention probability row sum");
  }

  const auto single = transformer_lab::causal_scaled_dot_product_attention(
      Variable(Tensor({1, 1, 1, 2}, {0.2F, -0.3F})),
      Variable(Tensor({1, 1, 1, 2}, {0.4F, 0.5F})),
      Variable(Tensor({1, 1, 1, 2}, {7.0F, -2.0F})));
  require_tensor_close(single.probabilities.value(), {1, 1, 1, 1}, {1.0F},
                       "single-token attention probability");
  require_tensor_close(single.context.value(), {1, 1, 1, 2}, {7.0F, -2.0F},
                       "single-token attention context");
}

void test_multi_batch_attention() {
  const Tensor::Shape shape{2, 2, 3, 2};
  Tensor query_values(shape);
  Tensor key_values(shape);
  Tensor value_values(shape);
  for (std::size_t index = 0; index < query_values.numel(); ++index) {
    const int query_pattern = static_cast<int>(index % 9) - 4;
    const int key_pattern = static_cast<int>((index * 3) % 11) - 5;
    const int value_pattern = static_cast<int>((index * 5) % 13) - 6;
    query_values.flat(index) = static_cast<float>(query_pattern) * 0.17F;
    key_values.flat(index) = static_cast<float>(key_pattern) * 0.11F;
    value_values.flat(index) = static_cast<float>(value_pattern) * 0.23F;
  }

  const auto actual = transformer_lab::causal_scaled_dot_product_attention(
      Variable(query_values, false), Variable(key_values, false),
      Variable(value_values, false));
  const auto expected =
      reference_causal_attention(query_values, key_values, value_values);
  require_tensor_close(actual.context.value(), expected.context,
                       "multi-batch attention context", 3.0e-6F);
  require_tensor_close(actual.probabilities.value(), expected.probabilities,
                       "multi-batch attention probabilities", 3.0e-6F);

  Tensor changed_queries = query_values;
  Tensor changed_keys = key_values;
  Tensor changed_values = value_values;
  const std::size_t values_per_batch = 2 * 3 * 2;
  for (std::size_t index = values_per_batch; index < changed_queries.numel();
       ++index) {
    changed_queries.flat(index) += 1.25F;
    changed_keys.flat(index) -= 0.75F;
    changed_values.flat(index) += 2.0F + static_cast<float>(index) * 0.01F;
  }
  const auto changed = transformer_lab::causal_scaled_dot_product_attention(
      Variable(changed_queries, false), Variable(changed_keys, false),
      Variable(changed_values, false));
  for (std::size_t index = 0; index < values_per_batch; ++index) {
    require_close(changed.context.value().flat(index),
                  actual.context.value().flat(index),
                  "second batch changed first-batch context", 0.0F);
  }
  const std::size_t probabilities_per_batch = 2 * 3 * 3;
  for (std::size_t index = 0; index < probabilities_per_batch; ++index) {
    require_close(changed.probabilities.value().flat(index),
                  actual.probabilities.value().flat(index),
                  "second batch changed first-batch probabilities", 0.0F);
  }

  bool second_batch_changed = false;
  for (std::size_t index = values_per_batch;
       index < changed.context.value().numel(); ++index) {
    if (std::fabs(changed.context.value().flat(index) -
                  actual.context.value().flat(index)) > 1.0e-4F) {
      second_batch_changed = true;
    }
  }
  require(second_batch_changed,
          "changing the second batch should change its own context");
}

void test_functional_attention_gradients() {
  const Tensor query_values({1, 2, 3, 2}, {
                                              0.2F,
                                              -0.3F,
                                              0.5F,
                                              0.7F,
                                              -0.4F,
                                              0.9F,
                                              0.6F,
                                              -0.2F,
                                              -0.5F,
                                              0.3F,
                                              0.8F,
                                              0.1F,
                                          });
  const Tensor key_values({1, 2, 3, 2}, {
                                            -0.1F,
                                            0.4F,
                                            0.8F,
                                            -0.6F,
                                            0.3F,
                                            0.5F,
                                            0.7F,
                                            0.2F,
                                            -0.4F,
                                            0.9F,
                                            0.1F,
                                            -0.8F,
                                        });
  const Tensor value_values({1, 2, 3, 2}, {
                                              1.0F,
                                              -0.5F,
                                              0.2F,
                                              0.7F,
                                              -0.3F,
                                              1.2F,
                                              0.6F,
                                              -1.0F,
                                              0.9F,
                                              0.4F,
                                              -0.7F,
                                              0.3F,
                                          });
  const Tensor output_weights({1, 2, 3, 2}, {
                                                0.5F,
                                                -1.0F,
                                                1.5F,
                                                0.25F,
                                                -0.75F,
                                                0.6F,
                                                0.4F,
                                                -1.2F,
                                                0.9F,
                                                0.3F,
                                                -0.2F,
                                                1.1F,
                                            });

  const Variable queries(query_values);
  const Variable keys(key_values);
  const Variable values(value_values);
  const auto attention = transformer_lab::causal_scaled_dot_product_attention(
      queries, keys, values);
  const auto reference =
      reference_causal_attention(query_values, key_values, value_values);
  require_tensor_close(attention.context.value(), reference.context,
                       "attention reference forward", 2.0e-6F);
  require_tensor_close(attention.probabilities.value(), reference.probabilities,
                       "attention probability reference forward", 2.0e-6F);
  transformer_lab::sum(attention.context * Variable(output_weights, false))
      .backward();

  const auto evaluate = [&](const Tensor &candidate_queries,
                            const Tensor &candidate_keys,
                            const Tensor &candidate_values) {
    return tensor_dot(reference_causal_attention(
                          candidate_queries, candidate_keys, candidate_values)
                          .context,
                      output_weights);
  };

  constexpr float epsilon = 1.0e-2F;
  constexpr float tolerance = 8.0e-3F;
  for (std::size_t index = 0; index < query_values.numel(); ++index) {
    Tensor plus = query_values;
    Tensor minus = query_values;
    plus.flat(index) += epsilon;
    minus.flat(index) -= epsilon;
    require_close(queries.gradient().flat(index),
                  (evaluate(plus, key_values, value_values) -
                   evaluate(minus, key_values, value_values)) /
                      (2.0F * epsilon),
                  "attention query finite difference", tolerance);
  }
  for (std::size_t index = 0; index < key_values.numel(); ++index) {
    Tensor plus = key_values;
    Tensor minus = key_values;
    plus.flat(index) += epsilon;
    minus.flat(index) -= epsilon;
    require_close(keys.gradient().flat(index),
                  (evaluate(query_values, plus, value_values) -
                   evaluate(query_values, minus, value_values)) /
                      (2.0F * epsilon),
                  "attention key finite difference", tolerance);
  }
  for (std::size_t index = 0; index < value_values.numel(); ++index) {
    Tensor plus = value_values;
    Tensor minus = value_values;
    plus.flat(index) += epsilon;
    minus.flat(index) -= epsilon;
    require_close(values.gradient().flat(index),
                  (evaluate(query_values, key_values, plus) -
                   evaluate(query_values, key_values, minus)) /
                      (2.0F * epsilon),
                  "attention value finite difference", tolerance);
  }

  const Variable probability_queries(query_values);
  const Variable probability_keys(key_values);
  const Variable probability_values(value_values);
  const auto probability_attention =
      transformer_lab::causal_scaled_dot_product_attention(
          probability_queries, probability_keys, probability_values);
  const Tensor probability_seed({1, 2, 3, 3}, {
                                                  0.5F,
                                                  20.0F,
                                                  -30.0F,
                                                  -1.0F,
                                                  0.25F,
                                                  40.0F,
                                                  0.75F,
                                                  -0.5F,
                                                  1.25F,
                                                  -0.2F,
                                                  50.0F,
                                                  -60.0F,
                                                  0.4F,
                                                  1.1F,
                                                  70.0F,
                                                  -0.8F,
                                                  0.3F,
                                                  0.9F,
                                              });
  probability_attention.probabilities.backward(probability_seed);
  const auto evaluate_probabilities = [&](const Tensor &candidate_queries,
                                          const Tensor &candidate_keys) {
    return tensor_dot(reference_causal_attention(candidate_queries,
                                                 candidate_keys, value_values)
                          .probabilities,
                      probability_seed);
  };
  for (std::size_t index = 0; index < query_values.numel(); ++index) {
    Tensor plus = query_values;
    Tensor minus = query_values;
    plus.flat(index) += epsilon;
    minus.flat(index) -= epsilon;
    require_close(probability_queries.gradient().flat(index),
                  (evaluate_probabilities(plus, key_values) -
                   evaluate_probabilities(minus, key_values)) /
                      (2.0F * epsilon),
                  "attention probability query finite difference", tolerance);
  }
  for (std::size_t index = 0; index < key_values.numel(); ++index) {
    Tensor plus = key_values;
    Tensor minus = key_values;
    plus.flat(index) += epsilon;
    minus.flat(index) -= epsilon;
    require_close(probability_keys.gradient().flat(index),
                  (evaluate_probabilities(query_values, plus) -
                   evaluate_probabilities(query_values, minus)) /
                      (2.0F * epsilon),
                  "attention probability key finite difference", tolerance);
  }
  require_tensor_close(
      probability_values.gradient(), probability_values.value().shape(),
      std::vector<float>(probability_values.value().numel(), 0.0F),
      "attention probabilities must not depend on values");

  const Variable causal_queries(query_values);
  const Variable causal_keys(key_values);
  const Variable causal_values(value_values);
  const auto causal = transformer_lab::causal_scaled_dot_product_attention(
      causal_queries, causal_keys, causal_values);
  Tensor first_position_seed(query_values.shape(), 0.0F);
  for (std::size_t head = 0; head < 2; ++head) {
    for (std::size_t channel = 0; channel < 2; ++channel) {
      first_position_seed.at({0, head, 0, channel}) = 1.0F;
    }
  }
  causal.context.backward(first_position_seed);
  for (std::size_t head = 0; head < 2; ++head) {
    for (std::size_t time = 1; time < 3; ++time) {
      for (std::size_t channel = 0; channel < 2; ++channel) {
        require(causal_queries.gradient().at({
                    0,
                    head,
                    time,
                    channel,
                }) == 0.0F,
                "future query gradient must be zero");
        require(causal_keys.gradient().at({
                    0,
                    head,
                    time,
                    channel,
                }) == 0.0F,
                "future key gradient must be zero");
        require(causal_values.gradient().at({
                    0,
                    head,
                    time,
                    channel,
                }) == 0.0F,
                "future value gradient must be zero");
      }
    }
  }

  const Variable second_queries(query_values);
  const Variable second_keys(key_values);
  const Variable second_values(value_values);
  const auto second_causal =
      transformer_lab::causal_scaled_dot_product_attention(
          second_queries, second_keys, second_values);
  Tensor second_position_seed(query_values.shape(), 0.0F);
  for (std::size_t head = 0; head < 2; ++head) {
    for (std::size_t channel = 0; channel < 2; ++channel) {
      second_position_seed.at({0, head, 1, channel}) =
          0.5F + static_cast<float>(head + channel);
    }
  }
  second_causal.context.backward(second_position_seed);
  for (std::size_t head = 0; head < 2; ++head) {
    for (std::size_t channel = 0; channel < 2; ++channel) {
      require(second_keys.gradient().at({
                  0,
                  head,
                  2,
                  channel,
              }) == 0.0F,
              "time-one output reached a future key");
      require(second_values.gradient().at({
                  0,
                  head,
                  2,
                  channel,
              }) == 0.0F,
              "time-one output reached a future value");
    }
  }
}

void set_identity_projections(CausalSelfAttention &attention) {
  for (auto &named_parameter : attention.parameters()) {
    if (named_parameter.name.ends_with(".weight")) {
      named_parameter.parameter->set_value(identity_matrix(4));
    } else {
      named_parameter.parameter->set_value(Tensor::zeros({4}));
    }
  }
}

void test_module_context_only_matches_explicit_attention() {
  std::mt19937 random(251U);
  CausalSelfAttention attention(4, 2, random);
  set_identity_projections(attention);

  const Tensor input_values({1, 3, 4}, {
                                           0.2F,
                                           -0.3F,
                                           0.5F,
                                           0.7F,
                                           1.0F,
                                           0.4F,
                                           -0.2F,
                                           0.6F,
                                           -0.6F,
                                           0.8F,
                                           0.15F,
                                           -0.9F,
                                       });
  const Tensor seed({1, 3, 4}, {
                                   0.5F,
                                   -1.0F,
                                   0.25F,
                                   1.5F,
                                   0.75F,
                                   -0.4F,
                                   -0.6F,
                                   0.2F,
                                   1.25F,
                                   0.9F,
                                   -1.1F,
                                   0.35F,
                               });

  const Variable module_input(input_values);
  const Variable module_output = attention.forward(module_input);

  const Variable functional_input(input_values);
  const Variable heads =
      transformer_lab::split_attention_heads(functional_input, 2);
  const auto explicit_attention =
      transformer_lab::causal_scaled_dot_product_attention(heads, heads, heads);
  const Variable explicit_output =
      transformer_lab::merge_attention_heads(explicit_attention.context);

  require_tensor_close(module_output.value(), explicit_output.value(),
                       "context-only module forward parity", 3.0e-6F);

  module_output.backward(seed);
  explicit_output.backward(seed);
  require_tensor_close(module_input.gradient(), functional_input.gradient(),
                       "context-only module gradient parity", 2.0e-5F);

  require(explicit_attention.probabilities.value().shape() ==
              Tensor::Shape({1, 2, 3, 3}),
          "explicit attention must still return probabilities");
}

void test_module_flash_attention_policy_and_parity() {
  std::mt19937 random(263U);
  CausalSelfAttention attention(4, 2, random);
  require(attention.full_sequence_attention_kind() ==
              FullSequenceAttentionKind::Materialized,
          "attention should default to materialized full-sequence attention");

  const Tensor input_values({1, 3, 4}, {
                                           0.2F,
                                           -0.3F,
                                           0.5F,
                                           0.7F,
                                           1.0F,
                                           0.4F,
                                           -0.2F,
                                           0.6F,
                                           -0.6F,
                                           0.8F,
                                           0.15F,
                                           -0.9F,
                                       });
  const Tensor seed({1, 3, 4}, {
                                   0.5F,
                                   -1.0F,
                                   0.25F,
                                   1.5F,
                                   0.75F,
                                   -0.4F,
                                   -0.6F,
                                   0.2F,
                                   1.25F,
                                   0.9F,
                                   -1.1F,
                                   0.35F,
                               });

  const Variable materialized_input(input_values);
  const Variable materialized_output = attention.forward(materialized_input);
  materialized_output.backward(seed);

  attention.set_full_sequence_attention_kind(
      FullSequenceAttentionKind::Flash);
  require(attention.full_sequence_attention_kind() ==
              FullSequenceAttentionKind::Flash,
          "attention selector should report Flash");

  const Variable flash_input(input_values);
  const Variable flash_output = attention.forward(flash_input);
  flash_output.backward(seed);

  require_tensor_close(flash_output.value(), materialized_output.value(),
                       "Flash attention module forward parity", 3.0e-5F);
  require_tensor_close(flash_input.gradient(),
                       materialized_input.gradient(),
                       "Flash attention module input-gradient parity",
                       5.0e-5F);

  require_throws(
      [&] {
        attention.set_full_sequence_attention_kind(
            static_cast<FullSequenceAttentionKind>(99));
      },
      "attention selector should reject unknown kinds");
  require(attention.full_sequence_attention_kind() ==
              FullSequenceAttentionKind::Flash,
          "invalid attention selection should preserve the prior policy");
}

void test_module_identity_and_causality() {
  std::mt19937 random(7);
  CausalSelfAttention attention(4, 2, random);
  set_identity_projections(attention);

  const Tensor input_values({1, 2, 4}, {
                                           1.0F,
                                           0.0F,
                                           0.0F,
                                           1.0F,
                                           0.0F,
                                           1.0F,
                                           1.0F,
                                           0.0F,
                                       });
  const Tensor output =
      attention.forward(Variable(input_values, false)).value();
  const float scaled_one = 1.0F / std::sqrt(2.0F);
  const float previous_probability = 1.0F / (1.0F + std::exp(scaled_one));
  require_tensor_close(output, {1, 2, 4},
                       {
                           1.0F,
                           0.0F,
                           0.0F,
                           1.0F,
                           previous_probability,
                           1.0F - previous_probability,
                           1.0F - previous_probability,
                           previous_probability,
                       },
                       "identity-projection multi-head attention", 2.0e-6F);

  Tensor changed_values = input_values;
  changed_values.flat(4) = 9.0F;
  changed_values.flat(5) = -7.0F;
  changed_values.flat(6) = 5.0F;
  changed_values.flat(7) = 11.0F;
  const Tensor changed_output =
      attention.forward(Variable(changed_values, false)).value();
  for (std::size_t channel = 0; channel < 4; ++channel) {
    require_close(changed_output.at({0, 0, channel}),
                  output.at({0, 0, channel}),
                  "future token changed an earlier output", 0.0F);
  }
  bool final_position_changed = false;
  for (std::size_t channel = 0; channel < 4; ++channel) {
    if (std::fabs(changed_output.at({0, 1, channel}) -
                  output.at({0, 1, channel})) > 1.0e-4F) {
      final_position_changed = true;
    }
  }
  require(final_position_changed,
          "changing the current token should change its attention output");

  const Variable trainable_input(input_values);
  const Variable trainable_output = attention.forward(trainable_input);
  Tensor first_position_seed({1, 2, 4}, 0.0F);
  for (std::size_t channel = 0; channel < 4; ++channel) {
    first_position_seed.at({0, 0, channel}) = 1.0F;
  }
  trainable_output.backward(first_position_seed);
  for (std::size_t channel = 0; channel < 4; ++channel) {
    require(trainable_input.gradient().at({0, 1, channel}) == 0.0F,
            "future input gradient must be exactly zero");
  }

  const Tensor longer_input_values({1, 3, 4}, {
                                                  1.0F,
                                                  0.0F,
                                                  0.0F,
                                                  1.0F,
                                                  0.0F,
                                                  1.0F,
                                                  1.0F,
                                                  0.0F,
                                                  4.0F,
                                                  -3.0F,
                                                  2.0F,
                                                  5.0F,
                                              });
  const Variable longer_input(longer_input_values);
  const Variable longer_output = attention.forward(longer_input);
  Tensor second_position_seed({1, 3, 4}, 0.0F);
  second_position_seed.at({0, 1, 0}) = 0.5F;
  second_position_seed.at({0, 1, 1}) = -1.0F;
  second_position_seed.at({0, 1, 2}) = 1.5F;
  second_position_seed.at({0, 1, 3}) = 0.25F;
  longer_output.backward(second_position_seed);
  for (std::size_t channel = 0; channel < 4; ++channel) {
    require(longer_input.gradient().at({0, 2, channel}) == 0.0F,
            "time-one module output reached a future input");
  }
}

void test_module_distinct_projections() {
  std::mt19937 random(13);
  CausalSelfAttention attention(4, 2, random);

  const Tensor query_weight({4, 4}, {
                                        1.0F,
                                        0.2F,
                                        -0.1F,
                                        0.0F,
                                        0.3F,
                                        0.7F,
                                        0.0F,
                                        0.1F,
                                        -0.2F,
                                        0.0F,
                                        0.9F,
                                        0.4F,
                                        0.0F,
                                        -0.3F,
                                        0.2F,
                                        0.8F,
                                    });
  const Tensor key_weight({4, 4}, {
                                      0.5F,
                                      -0.4F,
                                      0.1F,
                                      0.0F,
                                      0.2F,
                                      1.1F,
                                      0.3F,
                                      -0.2F,
                                      0.0F,
                                      0.2F,
                                      0.6F,
                                      -0.5F,
                                      0.4F,
                                      0.0F,
                                      0.7F,
                                      0.9F,
                                  });
  const Tensor value_weight({4, 4}, {
                                        0.8F,
                                        0.0F,
                                        0.3F,
                                        -0.2F,
                                        -0.1F,
                                        0.6F,
                                        0.4F,
                                        0.0F,
                                        0.2F,
                                        -0.5F,
                                        1.2F,
                                        0.1F,
                                        0.3F,
                                        0.2F,
                                        -0.4F,
                                        0.7F,
                                    });
  const Tensor output_weight({4, 4}, {
                                         0.7F,
                                         0.1F,
                                         -0.2F,
                                         0.4F,
                                         0.0F,
                                         0.9F,
                                         0.3F,
                                         -0.1F,
                                         0.5F,
                                         -0.3F,
                                         0.8F,
                                         0.2F,
                                         -0.2F,
                                         0.4F,
                                         0.0F,
                                         1.1F,
                                     });
  const Tensor query_bias({4}, {0.05F, -0.1F, 0.15F, 0.2F});
  const Tensor key_bias({4}, {-0.2F, 0.1F, 0.0F, 0.05F});
  const Tensor value_bias({4}, {0.3F, -0.15F, 0.2F, -0.05F});
  const Tensor output_bias({4}, {-0.1F, 0.25F, -0.2F, 0.15F});

  set_named_parameter(attention, "query.weight", query_weight);
  set_named_parameter(attention, "query.bias", query_bias);
  set_named_parameter(attention, "key.weight", key_weight);
  set_named_parameter(attention, "key.bias", key_bias);
  set_named_parameter(attention, "value.weight", value_weight);
  set_named_parameter(attention, "value.bias", value_bias);
  set_named_parameter(attention, "output.weight", output_weight);
  set_named_parameter(attention, "output.bias", output_bias);

  const Tensor input_values({1, 3, 4}, {
                                           0.2F,
                                           -0.3F,
                                           0.5F,
                                           0.7F,
                                           1.0F,
                                           0.4F,
                                           -0.2F,
                                           0.6F,
                                           -0.6F,
                                           0.8F,
                                           0.15F,
                                           -0.9F,
                                       });
  const Tensor actual =
      attention.forward(Variable(input_values, false)).value();
  const Tensor expected = reference_attention_module(
      input_values, 2, query_weight, query_bias, key_weight, key_bias,
      value_weight, value_bias, output_weight, output_bias);
  require_tensor_close(actual, expected, "distinct-projection attention module",
                       1.0e-5F);
}

void test_module_parameters_and_gradients() {
  std::mt19937 random(19);
  CausalSelfAttention attention(4, 2, random);
  require(attention.model_width() == 4, "attention model width");
  require(attention.head_count() == 2, "attention head count");
  require(attention.head_width() == 2, "attention head width");

  auto parameters = attention.parameters();
  const std::vector<std::string> expected_names{
      "query.weight", "query.bias", "key.weight",    "key.bias",
      "value.weight", "value.bias", "output.weight", "output.bias",
  };
  require(parameters.size() == expected_names.size(),
          "attention parameter count");
  std::set<const transformer_lab::Parameter *> parameter_addresses;
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    parameter_addresses.insert(parameters[index].parameter);
    require(parameters[index].name == expected_names[index],
            "attention parameter order");
    const Tensor::Shape expected_shape =
        parameters[index].name.ends_with(".weight") ? Tensor::Shape({4, 4})
                                                    : Tensor::Shape({4});
    require(parameters[index].parameter->value().shape() == expected_shape,
            "attention parameter shape");
  }
  require(parameter_addresses.size() == parameters.size(),
          "attention parameters must have distinct identities");
  require(transformer_lab::parameter_count(parameters) == 80,
          "attention scalar parameter count");

  const Tensor input_values({1, 3, 4}, {
                                           0.2F,
                                           -0.3F,
                                           0.5F,
                                           0.7F,
                                           1.0F,
                                           0.4F,
                                           -0.2F,
                                           0.6F,
                                           -0.6F,
                                           0.8F,
                                           0.15F,
                                           -0.9F,
                                       });
  const Tensor output_weights({1, 3, 4}, {
                                             0.5F,
                                             -1.0F,
                                             0.25F,
                                             1.5F,
                                             0.75F,
                                             -0.4F,
                                             -0.6F,
                                             0.2F,
                                             1.25F,
                                             0.9F,
                                             -1.1F,
                                             0.35F,
                                         });
  const Variable input(input_values);
  const Variable output = attention.forward(input);
  require(output.value().shape() == input_values.shape(),
          "attention output shape");
  transformer_lab::sum(output * Variable(output_weights, false)).backward();

  const Tensor input_gradient = input.gradient();
  std::vector<Tensor> parameter_values;
  std::vector<Tensor> parameter_gradients;
  parameter_values.reserve(parameters.size());
  parameter_gradients.reserve(parameters.size());
  for (const auto &named_parameter : parameters) {
    parameter_values.push_back(named_parameter.parameter->value());
    parameter_gradients.push_back(named_parameter.parameter->gradient());
  }

  const auto evaluate = [&](const Tensor &candidate_input) {
    return tensor_dot(
        attention.forward(Variable(candidate_input, false)).value(),
        output_weights);
  };

  constexpr float epsilon = 1.0e-2F;
  constexpr float tolerance = 1.5e-2F;
  for (std::size_t index = 0; index < input_values.numel(); ++index) {
    Tensor plus = input_values;
    Tensor minus = input_values;
    plus.flat(index) += epsilon;
    minus.flat(index) -= epsilon;
    require_close(input_gradient.flat(index),
                  (evaluate(plus) - evaluate(minus)) / (2.0F * epsilon),
                  "attention module input finite difference", tolerance);
  }

  for (std::size_t parameter_index = 0; parameter_index < parameters.size();
       ++parameter_index) {
    const Tensor &original = parameter_values[parameter_index];
    for (std::size_t index = 0; index < original.numel(); ++index) {
      Tensor plus = original;
      Tensor minus = original;
      plus.flat(index) += epsilon;
      minus.flat(index) -= epsilon;

      parameters[parameter_index].parameter->set_value(plus);
      const float plus_loss = evaluate(input_values);
      parameters[parameter_index].parameter->set_value(minus);
      const float minus_loss = evaluate(input_values);
      require_close(parameter_gradients[parameter_index].flat(index),
                    (plus_loss - minus_loss) / (2.0F * epsilon),
                    "attention " + parameters[parameter_index].name +
                        " finite difference",
                    tolerance);
    }
    parameters[parameter_index].parameter->set_value(original);
  }

  require_throws(
      [] {
        std::mt19937 invalid_random(1);
        static_cast<void>(CausalSelfAttention(0, 1, invalid_random));
      },
      "attention should reject zero model width");
  require_throws(
      [] {
        std::mt19937 invalid_random(1);
        static_cast<void>(CausalSelfAttention(4, 0, invalid_random));
      },
      "attention should reject zero heads");
  require_throws(
      [] {
        std::mt19937 invalid_random(1);
        static_cast<void>(CausalSelfAttention(4, 3, invalid_random));
      },
      "attention should reject indivisible head width");
  require_throws(
      [&] {
        static_cast<void>(attention.forward(Variable(Tensor({3, 4}), false)));
      },
      "attention should reject rank-two input");
  require_throws(
      [&] {
        static_cast<void>(
            attention.forward(Variable(Tensor({1, 3, 5}), false)));
      },
      "attention should reject wrong model width");
  require_throws(
      [] {
        static_cast<void>(transformer_lab::causal_scaled_dot_product_attention(
            Variable(Tensor({1, 2, 3}), false),
            Variable(Tensor({1, 2, 3}), false),
            Variable(Tensor({1, 2, 3}), false)));
      },
      "functional attention should reject non-rank-four input");
  require_throws(
      [] {
        static_cast<void>(transformer_lab::causal_scaled_dot_product_attention(
            Variable(Tensor({1, 1, 2, 2}), false),
            Variable(Tensor({1, 1, 3, 2}), false),
            Variable(Tensor({1, 1, 2, 2}), false)));
      },
      "functional attention should reject shape mismatch");
}

void test_module_device_transfer_and_forward() {
  const transformer_lab::ScopedExecutionBackend cpu_backend(
      ExecutionBackend::Cpu);
  std::mt19937 random(223U);
  CausalSelfAttention attention(4, 2, random);
  const Tensor input({1, 2, 4}, {
                                    0.1F,
                                    -0.2F,
                                    0.3F,
                                    -0.4F,
                                    0.5F,
                                    0.6F,
                                    -0.7F,
                                    0.8F,
                                });
  const Tensor expected = attention.forward(Variable(input, false)).value();

  attention.to(ExecutionBackend::Cpu);
  require_parameter_backend(attention.parameters(), ExecutionBackend::Cpu,
                            "attention CPU transfer");

  if (!transformer_lab::execution_backend_available(ExecutionBackend::Metal)) {
    return;
  }

  attention.to(ExecutionBackend::Metal);
  require_parameter_backend(attention.parameters(), ExecutionBackend::Metal,
                            "attention Metal transfer");
  const Tensor actual =
      attention.forward(Variable(input.to(ExecutionBackend::Metal), false))
          .value();
  require(actual.backend() == ExecutionBackend::Metal,
          "attention forward should preserve the transferred backend");
  require_tensor_close(actual, expected, "attention transfer forward parity",
                       2.0e-4F);

  attention.to(ExecutionBackend::Cpu);
  require_parameter_backend(attention.parameters(), ExecutionBackend::Cpu,
                            "attention CPU round trip");
}

} // namespace

int main() {
  try {
    test_split_and_merge_heads();
    test_scaled_causal_attention_forward();
    test_multi_batch_attention();
    test_functional_attention_gradients();
    test_module_context_only_matches_explicit_attention();
    test_module_flash_attention_policy_and_parity();
    test_module_identity_and_causality();
    test_module_distinct_projections();
    test_module_parameters_and_gradients();
    test_module_device_transfer_and_forward();
    std::cout << "causal self-attention tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
