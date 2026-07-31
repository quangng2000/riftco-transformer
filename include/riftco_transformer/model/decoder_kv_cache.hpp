#pragma once

#include "riftco_transformer/core/tensor.hpp"

#include <cstddef>

namespace riftco_transformer {

// Serving-time adapter used by the decoder's one-token inference path.
//
// A token append is transactional across all transformer layers:
// begin_token() reserves one logical position, append_and_attend() stores the
// layer's K/V vectors and returns attention for that position, and
// commit_token() makes the position visible. abort_token() must leave size()
// unchanged. Training forward/backward never depends on this interface.
class DecoderKeyValueCache {
public:
    virtual ~DecoderKeyValueCache() = default;

    [[nodiscard]] virtual ExecutionBackend backend() const noexcept = 0;
    [[nodiscard]] virtual std::size_t layer_count() const noexcept = 0;
    [[nodiscard]] virtual std::size_t head_count() const noexcept = 0;
    [[nodiscard]] virtual std::size_t head_width() const noexcept = 0;
    [[nodiscard]] virtual std::size_t capacity() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;

    virtual void begin_token() = 0;

    // query, key, value: [1, head_count, 1, head_width].
    // Returns a detached tensor with the same shape as query.
    [[nodiscard]] virtual Tensor append_and_attend(
        std::size_t layer,
        const Tensor& query,
        const Tensor& key,
        const Tensor& value
    ) = 0;

    virtual void commit_token() = 0;
    virtual void abort_token() noexcept = 0;
    virtual void reset() noexcept = 0;
};

}  // namespace riftco_transformer
