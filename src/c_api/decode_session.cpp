#include "internal/bridge.hpp"

using namespace riftco_transformer::c_api::detail;

namespace {

struct CheckedDecodeSessionOptions {
    rt_kv_cache_kind kind;
    std::size_t block_size;
};

CheckedDecodeSessionOptions checked_decode_session_options(
    const rt_decode_session_options* options
) {
    if (options == nullptr) {
        return {
            RT_KV_CACHE_PAGED,
            16,
        };
    }
    constexpr std::size_t minimum_size =
        offsetof(rt_decode_session_options, block_size) +
        sizeof(std::uint64_t);
    checked_structure_size(
        options->struct_size,
        minimum_size,
        "decode-session options structure"
    );
    if (options->reserved != 0) {
        throw std::invalid_argument(
            "decode-session options reserved field must be zero"
        );
    }
    switch (options->kind) {
        case RT_KV_CACHE_CONTIGUOUS:
        case RT_KV_CACHE_PAGED:
            break;
        default:
            throw std::invalid_argument(
                "unknown C API KV-cache kind"
            );
    }
    const std::size_t block_size = checked_size(
        options->block_size,
        "decode-session block size"
    );
    if (block_size == 0) {
        throw std::invalid_argument(
            "decode-session block size must be positive"
        );
    }
    return {
        options->kind,
        block_size,
    };
}

void require_decode_session(const rt_decode_session* session) {
    if (session == nullptr ||
        session->owner == nullptr ||
        session->cache == nullptr) {
        throw std::invalid_argument(
            "decode-session handle must not be null"
        );
    }
}

void require_current_decode_session(
    const rt_decode_session* session
) {
    require_decode_session(session);
    if (session->parameter_epoch !=
        session->owner->parameter_epoch.load(
            std::memory_order_relaxed
        )) {
        throw std::logic_error(
            "decode session cannot use stale model parameters"
        );
    }
}

}  // namespace


extern "C" {

rt_status RT_CALL rt_decode_session_options_init(
    rt_decode_session_options* options,
    uint64_t options_size
) {
    return guard([&] {
        if (options == nullptr) {
            throw std::invalid_argument(
                "decode-session options must not be null"
            );
        }
        constexpr std::size_t minimum_size =
            offsetof(rt_decode_session_options, block_size) +
            sizeof(std::uint64_t);
        checked_structure_size(
            options_size,
            minimum_size,
            "decode-session options structure"
        );
        *options = {
            options_size,
            RT_KV_CACHE_PAGED,
            0,
            16,
        };
    });
}

rt_status RT_CALL rt_model_decode_session_create(
    const rt_model* model,
    const rt_decode_session_options* options,
    rt_decode_session** output
) {
    return guard([&] {
        require_output(output);
        require_model(model);
        const auto configured =
            checked_decode_session_options(options);
        const auto& dimensions =
            model->state->value.dimensions();
        const auto backend = model->state->value.backend();

        std::unique_ptr<riftco_transformer::DecoderKeyValueCache>
            cache;
        std::size_t block_size = 0;
        if (configured.kind == RT_KV_CACHE_CONTIGUOUS) {
            const riftco_transformer::stages::serving::
                ContiguousKvCacheFactory factory(
                    dimensions,
                    backend
                );
            cache = factory.create();
            block_size = dimensions.maximum_context;
        } else {
            const riftco_transformer::stages::serving::
                PagedKvCachePool pool(
                    dimensions,
                    backend,
                    configured.block_size
                );
            cache = pool.create();
            block_size = pool.block_size();
        }

        auto result = std::make_unique<rt_decode_session>(
            model->state,
            std::move(cache),
            configured.kind,
            block_size,
            model->state->parameter_epoch.load(
                std::memory_order_relaxed
            )
        );
        *output = result.release();
    });
}

void RT_CALL rt_decode_session_release(
    rt_decode_session* session
) {
    delete session;
}

rt_status RT_CALL rt_decode_session_reset(
    rt_decode_session* session
) {
    return guard([&] {
        require_current_decode_session(session);
        session->cache->reset();
    });
}

rt_status RT_CALL rt_decode_session_size(
    const rt_decode_session* session,
    uint64_t* output
) {
    return guard([&] {
        require_decode_session(session);
        if (output == nullptr) {
            throw std::invalid_argument(
                "decode-session size output must not be null"
            );
        }
        *output = checked_u64(
            session->cache->size(),
            "decode-session size"
        );
    });
}

rt_status RT_CALL rt_decode_session_capacity(
    const rt_decode_session* session,
    uint64_t* output
) {
    return guard([&] {
        require_decode_session(session);
        if (output == nullptr) {
            throw std::invalid_argument(
                "decode-session capacity output must not be null"
            );
        }
        *output = checked_u64(
            session->cache->capacity(),
            "decode-session capacity"
        );
    });
}

rt_status RT_CALL rt_decode_session_cache_kind(
    const rt_decode_session* session,
    rt_kv_cache_kind* output
) {
    return guard([&] {
        require_decode_session(session);
        if (output == nullptr) {
            throw std::invalid_argument(
                "decode-session cache-kind output must not be null"
            );
        }
        *output = session->kind;
    });
}

rt_status RT_CALL rt_decode_session_block_size(
    const rt_decode_session* session,
    uint64_t* output
) {
    return guard([&] {
        require_decode_session(session);
        if (output == nullptr) {
            throw std::invalid_argument(
                "decode-session block-size output must not be null"
            );
        }
        *output = checked_u64(
            session->block_size,
            "decode-session block size"
        );
    });
}

rt_status RT_CALL rt_decode_session_step(
    rt_decode_session* session,
    uint32_t token_id,
    float* output_logits,
    uint64_t capacity,
    uint64_t* required_count
) {
    return guard([&] {
        require_decode_session(session);
        if (required_count == nullptr) {
            throw std::invalid_argument(
                "decode-session required-count output must not be null"
            );
        }
        const std::size_t vocabulary_size =
            session->owner->value.dimensions().vocabulary_size;
        *required_count = checked_u64(
            vocabulary_size,
            "decode-session logits"
        );
        require_current_decode_session(session);
        if (static_cast<std::size_t>(token_id) >=
            vocabulary_size) {
            throw std::out_of_range(
                "decode-session token is outside the model vocabulary"
            );
        }
        const std::size_t output_capacity = checked_size(
            capacity,
            "decode-session logits capacity"
        );
        if (output_logits == nullptr) {
            throw std::invalid_argument(
                "decode-session logits output must not be null"
            );
        }
        if (output_capacity < vocabulary_size) {
            throw std::out_of_range(
                "decode-session logits output capacity is too small"
            );
        }
        if (session->cache->size() >=
            session->cache->capacity()) {
            throw std::out_of_range(
                "decode-session cache capacity is exhausted"
            );
        }

        const riftco_transformer::Tensor logits =
            session->owner->value.decode_token(
                token_id,
                *session->cache
            );
        const riftco_transformer::Tensor::Shape expected_shape{
            1,
            1,
            vocabulary_size,
        };
        if (logits.shape() != expected_shape ||
            logits.numel() != vocabulary_size) {
            throw std::logic_error(
                "decode session returned an unexpected logit shape"
            );
        }
        std::copy(
            logits.data().begin(),
            logits.data().end(),
            output_logits
        );
    });
}

}  // extern "C"
