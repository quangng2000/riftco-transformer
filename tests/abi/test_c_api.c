#include "riftco_transformer/c_api.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA
#define RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA 0
#endif

#define RT_LEGACY_ADAM_OPTIONS_SIZE UINT64_C(32)
#define RT_LEGACY_QUANTIZED_MEMORY_STATS_SIZE UINT64_C(56)

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

static void require_condition(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "C API test failure: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

static void require_status(rt_status status, const char* operation) {
    if (status != RT_STATUS_OK) {
        fprintf(
            stderr,
            "C API test failure: %s: %s (%s)\n",
            operation,
            rt_status_string(status),
            rt_last_error()
        );
        exit(EXIT_FAILURE);
    }
}

static void require_close(
    float actual,
    float expected,
    const char* message
) {
    if (!isfinite(actual) || fabsf(actual - expected) > 1.0e-5F) {
        fprintf(
            stderr,
            "C API test failure: %s: expected %.7g, got %.7g\n",
            message,
            expected,
            actual
        );
        exit(EXIT_FAILURE);
    }
}

static void require_tensor_backend(
    const rt_tensor* tensor,
    rt_backend expected,
    const char* operation
) {
    rt_backend actual = (rt_backend)-1;
    require_status(
        rt_tensor_backend(tensor, &actual),
        operation
    );
    require_condition(actual == expected, operation);
}

typedef struct extended_transformer_config {
    rt_transformer_config value;
    uint64_t tail[2];
} extended_transformer_config;

typedef struct extended_lora_config {
    rt_lora_config value;
    uint64_t tail[2];
} extended_lora_config;

typedef struct extended_decode_session_options {
    rt_decode_session_options value;
    uint64_t tail[2];
} extended_decode_session_options;

typedef struct extended_adam_options {
    rt_adam_options value;
    uint64_t tail[2];
} extended_adam_options;

typedef struct extended_adam_step_stats {
    rt_adam_step_stats value;
    uint64_t tail[2];
} extended_adam_step_stats;

typedef struct extended_tokenizer_options {
    rt_tokenizer_options value;
    uint64_t tail[2];
} extended_tokenizer_options;

_Static_assert(
    sizeof(rt_tokenizer_options) == 32,
    "rt_tokenizer_options ABI layout"
);
_Static_assert(
    sizeof(rt_lora_config) == 40,
    "rt_lora_config ABI layout"
);
_Static_assert(
    sizeof(rt_decode_session_options) == 24,
    "rt_decode_session_options ABI layout"
);
_Static_assert(
    offsetof(rt_adam_options, state_storage) ==
        RT_LEGACY_ADAM_OPTIONS_SIZE,
    "legacy rt_adam_options ABI prefix"
);
_Static_assert(
    offsetof(rt_quantized_memory_stats, fp32_scale_bytes) ==
        RT_LEGACY_QUANTIZED_MEMORY_STATS_SIZE,
    "legacy rt_quantized_memory_stats ABI prefix"
);
_Static_assert(
    sizeof(rt_full_sequence_attention_kind) == 4,
    "rt_full_sequence_attention_kind ABI layout"
);
_Static_assert(
    sizeof(rt_activation_checkpointing_kind) == 4,
    "rt_activation_checkpointing_kind ABI layout"
);

typedef struct error_thread_result {
    int started_with_empty_error;
    rt_status status;
    char error[256];
} error_thread_result;

static void copy_last_error(char* output, size_t capacity) {
    const char* source = rt_last_error();
    size_t length = strlen(source);
    if (length >= capacity) {
        length = capacity - 1;
    }
    memcpy(output, source, length);
    output[length] = '\0';
}

#if defined(_WIN32)
static DWORD WINAPI record_thread_local_error(LPVOID argument)
#else
static void* record_thread_local_error(void* argument)
#endif
{
    error_thread_result* result = (error_thread_result*)argument;
    rt_backend backend = (rt_backend)-1;
    result->started_with_empty_error = rt_last_error()[0] == '\0';
    result->status = rt_context_backend(NULL, &backend);
    copy_last_error(result->error, sizeof(result->error));
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

static void test_thread_local_errors(void) {
    int32_t available = -1;
    require_condition(
        rt_backend_is_available((rt_backend)77, &available) ==
            RT_STATUS_INVALID_ARGUMENT,
        "main thread records an invalid-backend error"
    );

    char main_error[256];
    copy_last_error(main_error, sizeof(main_error));
    require_condition(
        strstr(main_error, "unknown C API backend") != NULL,
        "main thread error text"
    );

    error_thread_result result;
    memset(&result, 0, sizeof(result));
#if defined(_WIN32)
    HANDLE thread = CreateThread(
        NULL,
        0,
        record_thread_local_error,
        &result,
        0,
        NULL
    );
    require_condition(thread != NULL, "create error-isolation thread");
    require_condition(
        WaitForSingleObject(thread, INFINITE) == WAIT_OBJECT_0,
        "join error-isolation thread"
    );
    require_condition(
        CloseHandle(thread) != 0,
        "release error-isolation thread"
    );
#else
    pthread_t thread;
    require_condition(
        pthread_create(
            &thread,
            NULL,
            record_thread_local_error,
            &result
        ) == 0,
        "create error-isolation thread"
    );
    require_condition(
        pthread_join(thread, NULL) == 0,
        "join error-isolation thread"
    );
#endif

    require_condition(
        result.started_with_empty_error,
        "new thread starts with an empty error"
    );
    require_condition(
        result.status == RT_STATUS_INVALID_ARGUMENT,
        "worker thread records its own status"
    );
    require_condition(
        strstr(result.error, "context handle must not be null") != NULL,
        "worker thread records its own error text"
    );
    require_condition(
        strcmp(result.error, main_error) != 0,
        "worker and main thread errors are distinct"
    );
    require_condition(
        strcmp(rt_last_error(), main_error) == 0,
        "worker call does not overwrite the main thread error"
    );

    require_status(
        rt_backend_is_available(RT_BACKEND_CPU, &available),
        "clear main thread error"
    );
    require_condition(
        rt_last_error()[0] == '\0',
        "successful main-thread call clears only its error"
    );
}

static void test_tokenizer_api(void) {
    static const uint8_t original_corpus[] = {
        0x63, 0x00, 0xc3, 0xa9, 0xf0, 0x9f, 0x99,
        0x82, 0x61, 0x62, 0x0a, 0xff, 0x63, 0x00,
    };
    static const uint8_t expected_vocabulary[] = {
        0x00, 0x0a, 0x61, 0x62, 0x63, 0x82,
        0x99, 0x9f, 0xa9, 0xc3, 0xf0, 0xff,
    };
    static const uint8_t reordered_corpus[] = {
        0xff, 0x0a, 0x62, 0x61, 0x82, 0x99,
        0x9f, 0xf0, 0xa9, 0xc3, 0x00, 0x63,
    };
    static const uint8_t round_trip_bytes[] = {
        0x00, 0xc3, 0xa9, 0xf0, 0x9f,
        0x99, 0x82, 0x63, 0x0a, 0xff,
    };
    static const uint32_t expected_tokens[] = {
        0, 9, 8, 10, 7, 6, 5, 4, 1, 11,
    };

    rt_tokenizer* tokenizer = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create(NULL, 1, &tokenizer) ==
            RT_STATUS_INVALID_ARGUMENT,
        "tokenizer creation rejects null nonempty corpus"
    );
    require_condition(
        tokenizer == NULL,
        "failed tokenizer creation clears output"
    );

    tokenizer = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create(NULL, 0, &tokenizer) ==
            RT_STATUS_INVALID_ARGUMENT,
        "tokenizer creation rejects empty corpus"
    );
    require_condition(
        tokenizer == NULL,
        "empty tokenizer corpus clears output"
    );

    tokenizer = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create(
            original_corpus,
            0,
            &tokenizer
        ) == RT_STATUS_INVALID_ARGUMENT,
        "zero corpus size is empty even with a nonnull pointer"
    );
    require_condition(
        tokenizer == NULL,
        "zero-size tokenizer corpus clears output"
    );
    require_condition(
        rt_tokenizer_create(
            original_corpus,
            (uint64_t)sizeof(original_corpus),
            NULL
        ) == RT_STATUS_INVALID_ARGUMENT,
        "tokenizer creation requires an output pointer"
    );

    uint8_t mutable_corpus[sizeof(original_corpus)];
    memcpy(
        mutable_corpus,
        original_corpus,
        sizeof(mutable_corpus)
    );
    require_status(
        rt_tokenizer_create(
            mutable_corpus,
            (uint64_t)sizeof(mutable_corpus),
            &tokenizer
        ),
        "create binary tokenizer"
    );
    require_condition(tokenizer != NULL, "binary tokenizer handle");
    require_condition(
        rt_last_error()[0] == '\0',
        "successful tokenizer creation clears the prior error"
    );

    rt_tokenizer_method method = (rt_tokenizer_method)-1;
    require_status(
        rt_tokenizer_get_method(tokenizer, &method),
        "query legacy tokenizer method"
    );
    require_condition(
        method == RT_TOKENIZER_METHOD_BYTE,
        "legacy creation selects corpus-byte tokenization"
    );
    require_condition(
        rt_tokenizer_get_method(tokenizer, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "tokenizer method requires an output"
    );

    // The tokenizer derives and owns its vocabulary during creation.
    memset(mutable_corpus, 0xee, sizeof(mutable_corpus));

    uint64_t vocabulary_size = 0;
    require_status(
        rt_tokenizer_vocabulary_size(
            tokenizer,
            &vocabulary_size
        ),
        "query tokenizer vocabulary size"
    );
    require_condition(
        vocabulary_size ==
            (uint64_t)sizeof(expected_vocabulary),
        "binary tokenizer vocabulary size"
    );
    require_condition(
        rt_tokenizer_vocabulary_size(tokenizer, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "vocabulary size requires an output"
    );

    uint64_t required = UINT64_MAX;
    require_status(
        rt_tokenizer_vocabulary(
            tokenizer,
            NULL,
            0,
            &required
        ),
        "query tokenizer vocabulary output size"
    );
    require_condition(
        required == vocabulary_size,
        "vocabulary size query count"
    );

    uint8_t legacy_piece[2] = {0x7e, 0x6d};
    required = UINT64_MAX;
    require_status(
        rt_tokenizer_token_bytes(
            tokenizer,
            0,
            legacy_piece,
            1,
            &required
        ),
        "copy legacy tokenizer token bytes"
    );
    require_condition(
        required == 1 &&
            legacy_piece[0] == expected_vocabulary[0] &&
            legacy_piece[1] == 0x6d,
        "legacy token piece and trailing canary"
    );

    uint8_t zero_capacity_byte = 0x5a;
    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_vocabulary(
            tokenizer,
            &zero_capacity_byte,
            0,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "nonnull zero vocabulary capacity is insufficient"
    );
    require_condition(
        required == vocabulary_size &&
            zero_capacity_byte == 0x5a,
        "zero-capacity vocabulary call preserves output"
    );

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_vocabulary(
            tokenizer,
            NULL,
            vocabulary_size,
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "null vocabulary output requires zero capacity"
    );
    require_condition(
        required == vocabulary_size,
        "null vocabulary output still reports required count"
    );

    uint8_t vocabulary_canary[16];
    memset(vocabulary_canary, 0xa5, sizeof(vocabulary_canary));
    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_vocabulary(
            tokenizer,
            vocabulary_canary,
            vocabulary_size - 1,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "vocabulary rejects insufficient capacity"
    );
    require_condition(
        required == vocabulary_size,
        "insufficient vocabulary reports required count"
    );
    for (size_t index = 0;
         index < sizeof(vocabulary_canary);
         ++index) {
        require_condition(
            vocabulary_canary[index] == 0xa5,
            "insufficient vocabulary must not modify output"
        );
    }

    require_status(
        rt_tokenizer_vocabulary(
            tokenizer,
            vocabulary_canary,
            vocabulary_size,
            &required
        ),
        "copy tokenizer vocabulary"
    );
    require_condition(
        memcmp(
            vocabulary_canary,
            expected_vocabulary,
            sizeof(expected_vocabulary)
        ) == 0,
        "vocabulary is sorted by unsigned byte"
    );
    for (size_t index = sizeof(expected_vocabulary);
         index < sizeof(vocabulary_canary);
         ++index) {
        require_condition(
            vocabulary_canary[index] == 0xa5,
            "vocabulary copy preserves trailing canary"
        );
    }

    uint8_t untouched_vocabulary[16];
    memset(
        untouched_vocabulary,
        0x3c,
        sizeof(untouched_vocabulary)
    );
    require_condition(
        rt_tokenizer_vocabulary(
            tokenizer,
            untouched_vocabulary,
            vocabulary_size,
            NULL
        ) == RT_STATUS_INVALID_ARGUMENT,
        "vocabulary requires a required-count output"
    );
    for (size_t index = 0;
         index < sizeof(untouched_vocabulary);
         ++index) {
        require_condition(
            untouched_vocabulary[index] == 0x3c,
            "missing vocabulary count must not modify output"
        );
    }

    rt_tokenizer* reordered = NULL;
    require_status(
        rt_tokenizer_create(
            reordered_corpus,
            (uint64_t)sizeof(reordered_corpus),
            &reordered
        ),
        "create reordered tokenizer"
    );
    uint8_t reordered_vocabulary[sizeof(expected_vocabulary)];
    required = 0;
    require_status(
        rt_tokenizer_vocabulary(
            reordered,
            reordered_vocabulary,
            (uint64_t)sizeof(reordered_vocabulary),
            &required
        ),
        "copy reordered vocabulary"
    );
    require_condition(
        required == (uint64_t)sizeof(expected_vocabulary) &&
            memcmp(
                reordered_vocabulary,
                expected_vocabulary,
                sizeof(expected_vocabulary)
            ) == 0,
        "vocabulary is independent of corpus encounter order"
    );

    required = UINT64_MAX;
    require_status(
        rt_tokenizer_encode(
            tokenizer,
            round_trip_bytes,
            (uint64_t)sizeof(round_trip_bytes),
            NULL,
            0,
            &required
        ),
        "query encoded token count"
    );
    require_condition(
        required == (uint64_t)sizeof(round_trip_bytes),
        "encoded token count equals byte count"
    );

    uint32_t encoded_canary[12];
    for (size_t index = 0;
         index < sizeof(encoded_canary) / sizeof(encoded_canary[0]);
         ++index) {
        encoded_canary[index] = UINT32_C(0xdeadbeef);
    }
    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_encode(
            tokenizer,
            round_trip_bytes,
            (uint64_t)sizeof(round_trip_bytes),
            encoded_canary,
            0,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "nonnull zero token capacity is insufficient"
    );
    require_condition(
        required == (uint64_t)sizeof(round_trip_bytes),
        "zero token capacity reports required count"
    );
    for (size_t index = 0;
         index < sizeof(encoded_canary) / sizeof(encoded_canary[0]);
         ++index) {
        require_condition(
            encoded_canary[index] == UINT32_C(0xdeadbeef),
            "zero token capacity preserves output"
        );
    }

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_encode(
            tokenizer,
            round_trip_bytes,
            (uint64_t)sizeof(round_trip_bytes),
            NULL,
            1,
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "null token output requires zero capacity"
    );
    require_condition(
        required == (uint64_t)sizeof(round_trip_bytes),
        "null token output reports required count"
    );

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_encode(
            tokenizer,
            round_trip_bytes,
            (uint64_t)sizeof(round_trip_bytes),
            encoded_canary,
            (uint64_t)sizeof(round_trip_bytes) - 1,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "encode rejects insufficient capacity"
    );
    require_condition(
        required == (uint64_t)sizeof(round_trip_bytes),
        "insufficient encode reports required count"
    );
    for (size_t index = 0;
         index < sizeof(encoded_canary) / sizeof(encoded_canary[0]);
         ++index) {
        require_condition(
            encoded_canary[index] == UINT32_C(0xdeadbeef),
            "insufficient encode must not modify output"
        );
    }

    require_status(
        rt_tokenizer_encode(
            tokenizer,
            round_trip_bytes,
            (uint64_t)sizeof(round_trip_bytes),
            encoded_canary,
            (uint64_t)sizeof(round_trip_bytes),
            &required
        ),
        "encode binary and UTF-8 bytes"
    );
    require_condition(
        memcmp(
            encoded_canary,
            expected_tokens,
            sizeof(expected_tokens)
        ) == 0,
        "encoded token IDs"
    );
    require_condition(
        encoded_canary[10] == UINT32_C(0xdeadbeef) &&
            encoded_canary[11] == UINT32_C(0xdeadbeef),
        "encode preserves trailing token canary"
    );

    uint32_t untouched_token = UINT32_C(0x13579bdf);
    require_condition(
        rt_tokenizer_encode(
            tokenizer,
            round_trip_bytes,
            (uint64_t)sizeof(round_trip_bytes),
            &untouched_token,
            1,
            NULL
        ) == RT_STATUS_INVALID_ARGUMENT,
        "encode requires a required-count output"
    );
    require_condition(
        untouched_token == UINT32_C(0x13579bdf),
        "missing encode count must not modify output"
    );

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_encode(
            tokenizer,
            NULL,
            1,
            &untouched_token,
            1,
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "encode rejects null nonempty text"
    );
    require_condition(
        required == UINT64_MAX &&
            untouched_token == UINT32_C(0x13579bdf),
        "invalid text leaves count and output untouched"
    );

    required = UINT64_MAX;
    require_status(
        rt_tokenizer_encode(
            tokenizer,
            NULL,
            0,
            NULL,
            0,
            &required
        ),
        "encode empty text"
    );
    require_condition(required == 0, "empty encode count");
    require_status(
        rt_tokenizer_encode(
            tokenizer,
            NULL,
            0,
            &untouched_token,
            0,
            &required
        ),
        "encode empty text into zero capacity"
    );
    require_condition(
        required == 0 &&
            untouched_token == UINT32_C(0x13579bdf),
        "empty encode preserves output"
    );

    static const uint8_t unknown_text[] = {0x61, 0x7f};
    uint32_t unknown_output[2] = {
        UINT32_C(0xaaaaaaaa),
        UINT32_C(0xbbbbbbbb),
    };
    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_encode(
            tokenizer,
            unknown_text,
            (uint64_t)sizeof(unknown_text),
            NULL,
            0,
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "encode size query validates every input byte"
    );
    require_condition(
        required == UINT64_MAX,
        "invalid encode size query leaves count untouched"
    );

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_encode(
            tokenizer,
            unknown_text,
            (uint64_t)sizeof(unknown_text),
            unknown_output,
            2,
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "encode rejects a byte absent from the vocabulary"
    );
    require_condition(
        required == UINT64_MAX &&
            unknown_output[0] == UINT32_C(0xaaaaaaaa) &&
            unknown_output[1] == UINT32_C(0xbbbbbbbb),
        "unknown-byte encode leaves count and output untouched"
    );
    require_condition(
        strstr(rt_last_error(), "absent") != NULL,
        "unknown-byte encode diagnostic"
    );

    required = UINT64_MAX;
    require_status(
        rt_tokenizer_decode(
            tokenizer,
            expected_tokens,
            (uint64_t)(
                sizeof(expected_tokens) /
                sizeof(expected_tokens[0])
            ),
            NULL,
            0,
            &required
        ),
        "query decoded byte count"
    );
    require_condition(
        required == (uint64_t)sizeof(round_trip_bytes),
        "decoded byte count equals token count"
    );

    uint8_t decoded_canary[12];
    memset(decoded_canary, 0x6d, sizeof(decoded_canary));
    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_decode(
            tokenizer,
            expected_tokens,
            (uint64_t)(
                sizeof(expected_tokens) /
                sizeof(expected_tokens[0])
            ),
            decoded_canary,
            0,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "nonnull zero decode capacity is insufficient"
    );
    require_condition(
        required == (uint64_t)sizeof(round_trip_bytes),
        "zero decode capacity reports required count"
    );
    for (size_t index = 0;
         index < sizeof(decoded_canary);
         ++index) {
        require_condition(
            decoded_canary[index] == 0x6d,
            "zero decode capacity preserves output"
        );
    }

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_decode(
            tokenizer,
            expected_tokens,
            (uint64_t)(
                sizeof(expected_tokens) /
                sizeof(expected_tokens[0])
            ),
            NULL,
            1,
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "null decoded output requires zero capacity"
    );
    require_condition(
        required == (uint64_t)sizeof(round_trip_bytes),
        "null decoded output reports required count"
    );

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_decode(
            tokenizer,
            expected_tokens,
            (uint64_t)(
                sizeof(expected_tokens) /
                sizeof(expected_tokens[0])
            ),
            decoded_canary,
            (uint64_t)sizeof(round_trip_bytes) - 1,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "decode rejects insufficient capacity"
    );
    require_condition(
        required == (uint64_t)sizeof(round_trip_bytes),
        "insufficient decode reports required count"
    );
    for (size_t index = 0;
         index < sizeof(decoded_canary);
         ++index) {
        require_condition(
            decoded_canary[index] == 0x6d,
            "insufficient decode must not modify output"
        );
    }

    require_status(
        rt_tokenizer_decode(
            tokenizer,
            expected_tokens,
            (uint64_t)(
                sizeof(expected_tokens) /
                sizeof(expected_tokens[0])
            ),
            decoded_canary,
            (uint64_t)sizeof(round_trip_bytes),
            &required
        ),
        "decode binary and UTF-8 bytes"
    );
    require_condition(
        memcmp(
            decoded_canary,
            round_trip_bytes,
            sizeof(round_trip_bytes)
        ) == 0,
        "decoded bytes round-trip NUL and UTF-8"
    );
    require_condition(
        decoded_canary[10] == 0x6d &&
            decoded_canary[11] == 0x6d,
        "decode preserves trailing byte canary"
    );

    uint8_t untouched_decode = 0x4e;
    require_condition(
        rt_tokenizer_decode(
            tokenizer,
            expected_tokens,
            1,
            &untouched_decode,
            1,
            NULL
        ) == RT_STATUS_INVALID_ARGUMENT,
        "decode requires a required-count output"
    );
    require_condition(
        untouched_decode == 0x4e,
        "missing decode count must not modify output"
    );

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_decode(
            tokenizer,
            NULL,
            1,
            &untouched_decode,
            1,
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "decode rejects null nonempty tokens"
    );
    require_condition(
        required == UINT64_MAX &&
            untouched_decode == 0x4e,
        "invalid tokens leave count and output untouched"
    );

    required = UINT64_MAX;
    require_status(
        rt_tokenizer_decode(
            tokenizer,
            NULL,
            0,
            NULL,
            0,
            &required
        ),
        "decode empty token sequence"
    );
    require_condition(required == 0, "empty decode count");
    require_status(
        rt_tokenizer_decode(
            tokenizer,
            NULL,
            0,
            &untouched_decode,
            0,
            &required
        ),
        "decode empty tokens into zero capacity"
    );
    require_condition(
        required == 0 && untouched_decode == 0x4e,
        "empty decode preserves output"
    );

    const uint32_t invalid_tokens[] = {
        0,
        1,
        UINT32_MAX,
    };
    uint8_t invalid_decode[3] = {0x21, 0x43, 0x65};
    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_decode(
            tokenizer,
            invalid_tokens,
            (uint64_t)(
                sizeof(invalid_tokens) /
                sizeof(invalid_tokens[0])
            ),
            NULL,
            0,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "decode size query validates every token ID"
    );
    require_condition(
        required == UINT64_MAX,
        "invalid decode size query leaves count untouched"
    );

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_decode(
            tokenizer,
            invalid_tokens,
            (uint64_t)(
                sizeof(invalid_tokens) /
                sizeof(invalid_tokens[0])
            ),
            invalid_decode,
            (uint64_t)sizeof(invalid_decode),
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "decode rejects an invalid token ID"
    );
    require_condition(
        required == UINT64_MAX &&
            invalid_decode[0] == 0x21 &&
            invalid_decode[1] == 0x43 &&
            invalid_decode[2] == 0x65,
        "invalid-token decode leaves count and output untouched"
    );
    require_condition(
        strstr(rt_last_error(), "outside") != NULL,
        "invalid-token decode diagnostic"
    );

    require_status(
        rt_tokenizer_vocabulary_size(
            tokenizer,
            &vocabulary_size
        ),
        "successful tokenizer call after an error"
    );
    require_condition(
        rt_last_error()[0] == '\0',
        "successful tokenizer call clears its prior error"
    );
    require_condition(
        rt_tokenizer_vocabulary_size(NULL, &vocabulary_size) ==
            RT_STATUS_INVALID_ARGUMENT,
        "vocabulary size rejects a null tokenizer"
    );

    // Exhaust the full byte domain: every byte maps to its own numeric ID.
    uint8_t all_bytes[256];
    for (size_t index = 0; index < sizeof(all_bytes); ++index) {
        all_bytes[index] = (uint8_t)index;
    }
    rt_tokenizer* all_byte_tokenizer = NULL;
    require_status(
        rt_tokenizer_create(
            all_bytes,
            (uint64_t)sizeof(all_bytes),
            &all_byte_tokenizer
        ),
        "create all-byte tokenizer"
    );
    uint32_t all_tokens[256];
    required = 0;
    require_status(
        rt_tokenizer_encode(
            all_byte_tokenizer,
            all_bytes,
            (uint64_t)sizeof(all_bytes),
            all_tokens,
            (uint64_t)(
                sizeof(all_tokens) /
                sizeof(all_tokens[0])
            ),
            &required
        ),
        "encode every byte value"
    );
    require_condition(
        required == (uint64_t)sizeof(all_bytes),
        "all-byte token count"
    );
    for (size_t index = 0; index < sizeof(all_bytes); ++index) {
        require_condition(
            all_tokens[index] == (uint32_t)index,
            "all-byte tokenizer ID ordering"
        );
    }
    uint8_t all_decoded[256];
    require_status(
        rt_tokenizer_decode(
            all_byte_tokenizer,
            all_tokens,
            (uint64_t)(
                sizeof(all_tokens) /
                sizeof(all_tokens[0])
            ),
            all_decoded,
            (uint64_t)sizeof(all_decoded),
            &required
        ),
        "decode every byte value"
    );
    require_condition(
        memcmp(all_decoded, all_bytes, sizeof(all_bytes)) == 0,
        "all byte values round-trip"
    );

    rt_tokenizer_release(all_byte_tokenizer);
    rt_tokenizer_release(reordered);
    rt_tokenizer_release(tokenizer);
    rt_tokenizer_release(NULL);
}

static void test_tokenizer_options_and_bpe_api(void) {
    const uint64_t first_canary = UINT64_C(0x123456789abcdef0);
    const uint64_t second_canary = UINT64_C(0xfedcba9876543210);

    require_condition(
        rt_tokenizer_options_init(
            NULL,
            (uint64_t)sizeof(rt_tokenizer_options)
        ) == RT_STATUS_INVALID_ARGUMENT,
        "tokenizer options initializer rejects null"
    );

    rt_tokenizer_options undersized;
    memset(&undersized, 0x5a, sizeof(undersized));
    const rt_tokenizer_options original_undersized = undersized;
    require_condition(
        rt_tokenizer_options_init(
            &undersized,
            (uint64_t)sizeof(undersized) - 1
        ) == RT_STATUS_INVALID_ARGUMENT,
        "tokenizer options initializer rejects undersized storage"
    );
    require_condition(
        memcmp(
            &undersized,
            &original_undersized,
            sizeof(undersized)
        ) == 0,
        "failed tokenizer options initialization is atomic"
    );

    extended_tokenizer_options extended;
    memset(&extended, 0xa5, sizeof(extended));
    extended.tail[0] = first_canary;
    extended.tail[1] = second_canary;
    require_status(
        rt_tokenizer_options_init(
            &extended.value,
            (uint64_t)sizeof(extended)
        ),
        "initialize oversized tokenizer options"
    );
    require_condition(
        extended.value.struct_size == (uint64_t)sizeof(extended) &&
            extended.value.method == RT_TOKENIZER_METHOD_BYTE &&
            extended.value.reserved == 0 &&
            extended.value.vocabulary_size == 512 &&
            extended.value.minimum_pair_frequency == 2,
        "tokenizer options defaults"
    );
    require_condition(
        extended.tail[0] == first_canary &&
            extended.tail[1] == second_canary,
        "tokenizer options initializer preserves extension tail"
    );

    static const uint8_t byte_corpus[] = {'z', 'a', 'z'};
    rt_tokenizer* extended_tokenizer = NULL;
    require_status(
        rt_tokenizer_create_with_options(
            byte_corpus,
            (uint64_t)sizeof(byte_corpus),
            &extended.value,
            &extended_tokenizer
        ),
        "consume oversized tokenizer options"
    );
    require_condition(
        extended.tail[0] == first_canary &&
            extended.tail[1] == second_canary,
        "tokenizer creation preserves options extension tail"
    );
    rt_tokenizer_release(extended_tokenizer);

    rt_tokenizer* default_tokenizer = NULL;
    require_status(
        rt_tokenizer_create_with_options(
            byte_corpus,
            (uint64_t)sizeof(byte_corpus),
            NULL,
            &default_tokenizer
        ),
        "create tokenizer with null options"
    );
    rt_tokenizer_method method = (rt_tokenizer_method)-1;
    require_status(
        rt_tokenizer_get_method(default_tokenizer, &method),
        "query null-options tokenizer method"
    );
    require_condition(
        method == RT_TOKENIZER_METHOD_BYTE,
        "null tokenizer options select legacy bytes"
    );
    uint64_t vocabulary_size = UINT64_MAX;
    require_status(
        rt_tokenizer_vocabulary_size(
            default_tokenizer,
            &vocabulary_size
        ),
        "query null-options vocabulary"
    );
    require_condition(
        vocabulary_size == 2,
        "null options preserve corpus-derived byte vocabulary"
    );
    rt_tokenizer_release(default_tokenizer);

    rt_tokenizer_options options;
    require_status(
        rt_tokenizer_options_init(
            &options,
            (uint64_t)sizeof(options)
        ),
        "initialize exact tokenizer options"
    );
    static const uint8_t bpe_corpus[] = {
        'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b',
    };

    rt_tokenizer* rejected = (rt_tokenizer*)(uintptr_t)1;
    options.struct_size = (uint64_t)sizeof(options) - 1;
    require_condition(
        rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &options,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT,
        "tokenizer rejects undersized options"
    );
    require_condition(
        rejected == NULL,
        "undersized tokenizer options clear output"
    );

    require_status(
        rt_tokenizer_options_init(
            &options,
            (uint64_t)sizeof(options)
        ),
        "reset tokenizer options after size rejection"
    );
    options.reserved = 1;
    rejected = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &options,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT,
        "tokenizer rejects nonzero reserved options"
    );
    require_condition(
        rejected == NULL,
        "reserved tokenizer options clear output"
    );

    options.reserved = 0;
    options.method = (rt_tokenizer_method)99;
    rejected = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &options,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT,
        "tokenizer rejects unknown methods"
    );
    require_condition(
        rejected == NULL,
        "unknown tokenizer method clears output"
    );

    options.method = RT_TOKENIZER_METHOD_BPE;
    options.vocabulary_size = 255;
    rejected = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &options,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT,
        "BPE rejects a vocabulary smaller than the byte domain"
    );
    require_condition(
        rejected == NULL,
        "small BPE vocabulary clears output"
    );

    options.vocabulary_size = UINT64_MAX;
    rejected = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &options,
            &rejected
        ) == RT_STATUS_OVERFLOW,
        "BPE rejects a vocabulary beyond the token-ID range"
    );
    require_condition(
        rejected == NULL,
        "overflowing BPE vocabulary clears output"
    );

    static const uint8_t no_merge_corpus[] = {'x', 'y'};
    options.vocabulary_size = UINT32_MAX;
    options.minimum_pair_frequency = 2;
    rt_tokenizer* maximum_target = NULL;
    require_status(
        rt_tokenizer_create_with_options(
            no_merge_corpus,
            (uint64_t)sizeof(no_merge_corpus),
            &options,
            &maximum_target
        ),
        "BPE accepts the maximum token-ID-sized target"
    );
    vocabulary_size = UINT64_MAX;
    require_status(
        rt_tokenizer_vocabulary_size(
            maximum_target,
            &vocabulary_size
        ),
        "query maximum-target BPE vocabulary"
    );
    require_condition(
        vocabulary_size == 256,
        "maximum BPE target stops when no merge qualifies"
    );
    rt_tokenizer_release(maximum_target);

    options.vocabulary_size = 258;
    options.minimum_pair_frequency = 0;
    rejected = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &options,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT,
        "BPE rejects a zero minimum pair frequency"
    );
    require_condition(
        rejected == NULL,
        "zero BPE frequency clears output"
    );

    options.minimum_pair_frequency = 1;
    rt_tokenizer* minimum_frequency = NULL;
    require_status(
        rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &options,
            &minimum_frequency
        ),
        "BPE accepts a minimum pair frequency of one"
    );
    rt_tokenizer_release(minimum_frequency);

    options.minimum_pair_frequency = 2;
    require_condition(
        rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &options,
            NULL
        ) == RT_STATUS_INVALID_ARGUMENT,
        "BPE creation requires an output pointer"
    );

    uint8_t mutable_corpus[sizeof(bpe_corpus)];
    memcpy(mutable_corpus, bpe_corpus, sizeof(mutable_corpus));
    rt_tokenizer* tokenizer = NULL;
    require_status(
        rt_tokenizer_create_with_options(
            mutable_corpus,
            (uint64_t)sizeof(mutable_corpus),
            &options,
            &tokenizer
        ),
        "create deterministic BPE tokenizer"
    );
    require_condition(tokenizer != NULL, "BPE tokenizer handle");
    memset(mutable_corpus, 0xee, sizeof(mutable_corpus));

    method = (rt_tokenizer_method)-1;
    require_status(
        rt_tokenizer_get_method(tokenizer, &method),
        "query BPE tokenizer method"
    );
    require_condition(
        method == RT_TOKENIZER_METHOD_BPE,
        "BPE tokenizer reports its method"
    );
    require_condition(
        rt_tokenizer_get_method(NULL, &method) ==
            RT_STATUS_INVALID_ARGUMENT,
        "tokenizer method rejects a null handle"
    );

    vocabulary_size = 0;
    require_status(
        rt_tokenizer_vocabulary_size(tokenizer, &vocabulary_size),
        "query BPE vocabulary size"
    );
    require_condition(
        vocabulary_size == 258,
        "BPE learns deterministic target vocabulary"
    );

    uint8_t legacy_vocabulary[4] = {0x11, 0x22, 0x33, 0x44};
    uint64_t required = UINT64_MAX;
    require_condition(
        rt_tokenizer_vocabulary(
            tokenizer,
            legacy_vocabulary,
            (uint64_t)sizeof(legacy_vocabulary),
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "legacy vocabulary API clearly rejects BPE"
    );
    require_condition(
        required == UINT64_MAX &&
            legacy_vocabulary[0] == 0x11 &&
            legacy_vocabulary[1] == 0x22 &&
            legacy_vocabulary[2] == 0x33 &&
            legacy_vocabulary[3] == 0x44,
        "rejected BPE legacy vocabulary output is atomic"
    );
    require_condition(
        strstr(rt_last_error(), "corpus-byte") != NULL,
        "BPE legacy vocabulary rejection has a clear diagnostic"
    );

    required = UINT64_MAX;
    require_status(
        rt_tokenizer_token_bytes(
            tokenizer,
            UINT32_C(257),
            NULL,
            0,
            &required
        ),
        "query learned BPE token size"
    );
    require_condition(required == 4, "learned BPE token byte count");

    uint8_t token_canary[6];
    memset(token_canary, 0xa7, sizeof(token_canary));
    require_condition(
        rt_tokenizer_token_bytes(
            tokenizer,
            UINT32_C(257),
            token_canary,
            0,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "nonnull zero token-piece capacity is insufficient"
    );
    require_condition(
        required == 4,
        "zero token-piece capacity reports required bytes"
    );
    for (size_t index = 0; index < sizeof(token_canary); ++index) {
        require_condition(
            token_canary[index] == 0xa7,
            "zero token-piece capacity preserves output"
        );
    }

    require_condition(
        rt_tokenizer_token_bytes(
            tokenizer,
            UINT32_C(257),
            NULL,
            4,
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "null token-piece output requires zero capacity"
    );
    require_condition(
        required == 4,
        "null token-piece output reports required bytes"
    );

    require_condition(
        rt_tokenizer_token_bytes(
            tokenizer,
            UINT32_C(257),
            token_canary,
            3,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "token-piece output rejects insufficient capacity"
    );
    for (size_t index = 0; index < sizeof(token_canary); ++index) {
        require_condition(
            token_canary[index] == 0xa7,
            "undersized token-piece output is atomic"
        );
    }

    require_status(
        rt_tokenizer_token_bytes(
            tokenizer,
            UINT32_C(257),
            token_canary,
            4,
            &required
        ),
        "copy learned BPE token bytes"
    );
    require_condition(
        required == 4 &&
            memcmp(token_canary, "abab", 4) == 0 &&
            token_canary[4] == 0xa7 &&
            token_canary[5] == 0xa7,
        "learned BPE token bytes and trailing canary"
    );

    uint8_t required_canary = 0x6c;
    require_condition(
        rt_tokenizer_token_bytes(
            tokenizer,
            UINT32_C(256),
            &required_canary,
            1,
            NULL
        ) == RT_STATUS_INVALID_ARGUMENT,
        "token-piece output requires a required-count pointer"
    );
    require_condition(
        required_canary == 0x6c,
        "missing token-piece count preserves output"
    );

    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_token_bytes(
            tokenizer,
            UINT32_MAX,
            token_canary,
            (uint64_t)sizeof(token_canary),
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "token-piece output rejects an invalid token"
    );
    require_condition(
        required == UINT64_MAX,
        "invalid token piece leaves required count untouched"
    );
    require_condition(
        rt_tokenizer_token_bytes(
            NULL,
            0,
            token_canary,
            1,
            &required
        ) == RT_STATUS_INVALID_ARGUMENT,
        "token-piece output rejects a null handle"
    );

    static const uint32_t expected_tokens[] = {257, 257};
    required = UINT64_MAX;
    require_status(
        rt_tokenizer_encode(
            tokenizer,
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            NULL,
            0,
            &required
        ),
        "query deterministic BPE encoding size"
    );
    require_condition(required == 2, "BPE compresses repeated corpus");

    uint32_t encoded_canary[3] = {
        UINT32_C(0xaaaaaaaa),
        UINT32_C(0xbbbbbbbb),
        UINT32_C(0xcccccccc),
    };
    require_condition(
        rt_tokenizer_encode(
            tokenizer,
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            encoded_canary,
            1,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "BPE encode rejects insufficient capacity"
    );
    require_condition(
        required == 2 &&
            encoded_canary[0] == UINT32_C(0xaaaaaaaa) &&
            encoded_canary[1] == UINT32_C(0xbbbbbbbb) &&
            encoded_canary[2] == UINT32_C(0xcccccccc),
        "undersized BPE encode is atomic"
    );
    require_status(
        rt_tokenizer_encode(
            tokenizer,
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            encoded_canary,
            2,
            &required
        ),
        "encode deterministic BPE corpus"
    );
    require_condition(
        memcmp(
            encoded_canary,
            expected_tokens,
            sizeof(expected_tokens)
        ) == 0 &&
            encoded_canary[2] == UINT32_C(0xcccccccc),
        "deterministic BPE token IDs and trailing canary"
    );

    required = UINT64_MAX;
    require_status(
        rt_tokenizer_decode(
            tokenizer,
            expected_tokens,
            2,
            NULL,
            0,
            &required
        ),
        "query BPE decoded size"
    );
    require_condition(
        required == (uint64_t)sizeof(bpe_corpus),
        "BPE decoded byte count"
    );
    uint8_t decoded_canary[9];
    memset(decoded_canary, 0x4d, sizeof(decoded_canary));
    require_condition(
        rt_tokenizer_decode(
            tokenizer,
            expected_tokens,
            2,
            decoded_canary,
            (uint64_t)sizeof(bpe_corpus) - 1,
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "BPE decode rejects insufficient capacity"
    );
    for (size_t index = 0; index < sizeof(decoded_canary); ++index) {
        require_condition(
            decoded_canary[index] == 0x4d,
            "undersized BPE decode is atomic"
        );
    }
    require_status(
        rt_tokenizer_decode(
            tokenizer,
            expected_tokens,
            2,
            decoded_canary,
            (uint64_t)sizeof(bpe_corpus),
            &required
        ),
        "decode BPE corpus"
    );
    require_condition(
        memcmp(
            decoded_canary,
            bpe_corpus,
            sizeof(bpe_corpus)
        ) == 0 &&
            decoded_canary[8] == 0x4d,
        "BPE decode round-trip and trailing canary"
    );

    static const uint8_t universal_bytes[] = {
        0x00, 0xff, 0x78,
    };
    static const uint32_t universal_tokens[] = {
        0, 255, 120,
    };
    uint32_t actual_universal_tokens[3] = {0, 0, 0};
    require_status(
        rt_tokenizer_encode(
            tokenizer,
            universal_bytes,
            (uint64_t)sizeof(universal_bytes),
            actual_universal_tokens,
            3,
            &required
        ),
        "BPE encodes unseen NUL, high-bit, and ASCII bytes"
    );
    require_condition(
        required == 3 &&
            memcmp(
                actual_universal_tokens,
                universal_tokens,
                sizeof(universal_tokens)
            ) == 0,
        "BPE base-byte IDs cover the universal byte domain"
    );
    uint8_t decoded_universal[3] = {0, 0, 0};
    require_status(
        rt_tokenizer_decode(
            tokenizer,
            actual_universal_tokens,
            3,
            decoded_universal,
            3,
            &required
        ),
        "decode unseen universal bytes"
    );
    require_condition(
        required == 3 &&
            memcmp(
                decoded_universal,
                universal_bytes,
                sizeof(universal_bytes)
            ) == 0,
        "unseen NUL and high-bit bytes round-trip"
    );

    const uint32_t invalid_tokens[] = {257, 258};
    uint8_t invalid_decode[8];
    memset(invalid_decode, 0x72, sizeof(invalid_decode));
    required = UINT64_MAX;
    require_condition(
        rt_tokenizer_decode(
            tokenizer,
            invalid_tokens,
            2,
            invalid_decode,
            (uint64_t)sizeof(invalid_decode),
            &required
        ) == RT_STATUS_OUT_OF_RANGE,
        "BPE decode rejects an invalid token ID"
    );
    require_condition(
        required == UINT64_MAX,
        "invalid BPE token leaves decoded size untouched"
    );
    for (size_t index = 0; index < sizeof(invalid_decode); ++index) {
        require_condition(
            invalid_decode[index] == 0x72,
            "invalid BPE decode is atomic"
        );
    }

    rt_tokenizer* deterministic_copy = NULL;
    require_status(
        rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &options,
            &deterministic_copy
        ),
        "create second deterministic BPE tokenizer"
    );
    uint32_t copied_tokens[2] = {0, 0};
    require_status(
        rt_tokenizer_encode(
            deterministic_copy,
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            copied_tokens,
            2,
            &required
        ),
        "encode with second deterministic BPE tokenizer"
    );
    require_condition(
        memcmp(
            copied_tokens,
            expected_tokens,
            sizeof(expected_tokens)
        ) == 0,
        "repeated BPE training produces identical token IDs"
    );

    uint64_t merge_count = UINT64_MAX;
    require_status(
        rt_tokenizer_bpe_merge_count(tokenizer, &merge_count),
        "query serialized BPE merge count"
    );
    require_condition(
        merge_count == 2,
        "serialized BPE merge count"
    );
    rt_bpe_merge_rule merge_rules[2] = {{0, 0, 0}, {0, 0, 0}};
    require_status(
        rt_tokenizer_bpe_merge_rule(tokenizer, 0, &merge_rules[0]),
        "copy first BPE merge rule"
    );
    require_status(
        rt_tokenizer_bpe_merge_rule(tokenizer, 1, &merge_rules[1]),
        "copy second BPE merge rule"
    );
    require_condition(
        merge_rules[0].left == 97 &&
            merge_rules[0].right == 98 &&
            merge_rules[0].result == 256 &&
            merge_rules[1].left == 256 &&
            merge_rules[1].right == 256 &&
            merge_rules[1].result == 257,
        "serialized BPE rules preserve learned order"
    );
    require_condition(
        rt_tokenizer_bpe_merge_rule(tokenizer, 2, &merge_rules[0]) ==
            RT_STATUS_OUT_OF_RANGE,
        "BPE merge export rejects an out-of-range index"
    );
    require_condition(
        rt_tokenizer_bpe_merge_rule(tokenizer, 0, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "BPE merge export requires output storage"
    );

    rt_tokenizer* restored_bpe = NULL;
    require_status(
        rt_tokenizer_create_from_bpe_merges(
            merge_rules,
            2,
            &restored_bpe
        ),
        "restore BPE tokenizer from merge rules"
    );
    uint32_t restored_tokens[2] = {0, 0};
    require_status(
        rt_tokenizer_encode(
            restored_bpe,
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            restored_tokens,
            2,
            &required
        ),
        "encode with restored BPE tokenizer"
    );
    require_condition(
        required == 2 &&
            memcmp(
                restored_tokens,
                expected_tokens,
                sizeof(expected_tokens)
            ) == 0,
        "restored BPE tokenizer preserves token IDs"
    );

    rt_tokenizer* base_bpe = NULL;
    require_status(
        rt_tokenizer_create_from_bpe_merges(NULL, 0, &base_bpe),
        "restore base-only BPE tokenizer"
    );
    vocabulary_size = 0;
    require_status(
        rt_tokenizer_vocabulary_size(base_bpe, &vocabulary_size),
        "query base-only BPE vocabulary"
    );
    require_condition(
        vocabulary_size == 256,
        "base-only BPE contains every byte"
    );

    rt_bpe_merge_rule invalid_rule = {97, 98, 257};
    rejected = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create_from_bpe_merges(
            &invalid_rule,
            1,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT,
        "BPE restoration rejects a nonsequential result"
    );
    require_condition(
        rejected == NULL,
        "failed BPE restoration clears output"
    );
    rejected = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create_from_bpe_merges(NULL, 1, &rejected) ==
            RT_STATUS_INVALID_ARGUMENT,
        "BPE restoration rejects null nonempty rules"
    );
    require_condition(
        rejected == NULL,
        "null BPE rules clear output"
    );

    static const uint8_t restored_byte_vocabulary[] = {'z', 'a'};
    rt_tokenizer* restored_byte = NULL;
    require_status(
        rt_tokenizer_create_from_byte_vocabulary(
            restored_byte_vocabulary,
            2,
            &restored_byte
        ),
        "restore ordered byte tokenizer vocabulary"
    );
    static const uint8_t restored_byte_text[] = {'a', 'z'};
    uint32_t restored_byte_tokens[2] = {UINT32_MAX, UINT32_MAX};
    require_status(
        rt_tokenizer_encode(
            restored_byte,
            restored_byte_text,
            2,
            restored_byte_tokens,
            2,
            &required
        ),
        "encode with restored byte tokenizer"
    );
    require_condition(
        required == 2 &&
            restored_byte_tokens[0] == 1 &&
            restored_byte_tokens[1] == 0,
        "restored byte tokenizer preserves vocabulary order"
    );
    require_condition(
        rt_tokenizer_bpe_merge_count(restored_byte, &merge_count) ==
            RT_STATUS_INVALID_ARGUMENT,
        "BPE merge export rejects a byte tokenizer"
    );
    static const uint8_t duplicate_bytes[] = {'a', 'a'};
    rejected = (rt_tokenizer*)(uintptr_t)1;
    require_condition(
        rt_tokenizer_create_from_byte_vocabulary(
            duplicate_bytes,
            2,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT,
        "byte restoration rejects duplicate vocabulary entries"
    );
    require_condition(
        rejected == NULL,
        "failed byte restoration clears output"
    );
    require_condition(
        rt_tokenizer_create_from_byte_vocabulary(
            restored_byte_vocabulary,
            2,
            NULL
        ) == RT_STATUS_INVALID_ARGUMENT,
        "byte restoration requires an output pointer"
    );

    rt_tokenizer_release(restored_byte);
    rt_tokenizer_release(base_bpe);
    rt_tokenizer_release(restored_bpe);
    rt_tokenizer_release(deterministic_copy);
    rt_tokenizer_release(tokenizer);
}

static void test_versioned_structures(void) {
    const uint64_t first_canary = UINT64_C(0x0123456789abcdef);
    const uint64_t second_canary = UINT64_C(0xfedcba9876543210);

    extended_transformer_config config;
    memset(&config, 0xa5, sizeof(config));
    config.tail[0] = first_canary;
    config.tail[1] = second_canary;
    require_status(
        rt_transformer_config_init(
            &config.value,
            (uint64_t)sizeof(config)
        ),
        "initialize oversized transformer config"
    );
    require_condition(
        config.value.struct_size == (uint64_t)sizeof(config),
        "oversized transformer config reports caller size"
    );
    require_condition(
        config.tail[0] == first_canary &&
            config.tail[1] == second_canary,
        "transformer config initializer preserves extension tail"
    );

    config.value.vocabulary_size = 5;
    config.value.maximum_context = 4;
    config.value.model_width = 4;
    config.value.head_count = 2;
    config.value.block_count = 1;
    config.value.feed_forward_width = 8;

    rt_model* model = NULL;
    require_status(
        rt_model_create(&config.value, &model),
        "consume oversized transformer config"
    );
    require_condition(
        config.tail[0] == first_canary &&
            config.tail[1] == second_canary,
        "model creation preserves transformer config extension tail"
    );

    rt_parameter_list* parameters = NULL;
    require_status(
        rt_model_parameters(model, &parameters),
        "create parameters for versioned Adam structures"
    );

    rt_adam_options legacy_options;
    memset(&legacy_options, 0xa5, sizeof(legacy_options));
    unsigned char legacy_extension[
        sizeof(legacy_options) - RT_LEGACY_ADAM_OPTIONS_SIZE
    ];
    memcpy(
        legacy_extension,
        (const unsigned char*)&legacy_options +
            RT_LEGACY_ADAM_OPTIONS_SIZE,
        sizeof(legacy_extension)
    );
    require_status(
        rt_adam_options_init(
            &legacy_options,
            RT_LEGACY_ADAM_OPTIONS_SIZE
        ),
        "initialize legacy 32-byte Adam options"
    );
    require_condition(
        legacy_options.struct_size == RT_LEGACY_ADAM_OPTIONS_SIZE &&
            legacy_options.reserved == 0,
        "legacy Adam options initialize the supported prefix"
    );
    require_condition(
        memcmp(
            (const unsigned char*)&legacy_options +
                RT_LEGACY_ADAM_OPTIONS_SIZE,
            legacy_extension,
            sizeof(legacy_extension)
        ) == 0,
        "legacy Adam options initializer preserves extension canaries"
    );
    rt_adam* legacy_adam = NULL;
    require_status(
        rt_adam_create(parameters, &legacy_options, &legacy_adam),
        "consume legacy 32-byte Adam options"
    );
    rt_adam_state_storage_kind legacy_state_storage =
        (rt_adam_state_storage_kind)-1;
    uint64_t legacy_page_size = 0;
    require_status(
        rt_adam_state_storage(legacy_adam, &legacy_state_storage),
        "query legacy Adam state storage"
    );
    require_status(
        rt_adam_state_page_size(legacy_adam, &legacy_page_size),
        "query legacy Adam page size"
    );
    require_condition(
        legacy_state_storage == RT_ADAM_STATE_CONTIGUOUS &&
            legacy_page_size == UINT64_C(4096),
        "legacy Adam options select extension defaults"
    );
    require_condition(
        memcmp(
            (const unsigned char*)&legacy_options +
                RT_LEGACY_ADAM_OPTIONS_SIZE,
            legacy_extension,
            sizeof(legacy_extension)
        ) == 0,
        "legacy Adam creation preserves extension canaries"
    );
    rt_adam_release(legacy_adam);

    extended_adam_options options;
    memset(&options, 0x5a, sizeof(options));
    options.tail[0] = second_canary;
    options.tail[1] = first_canary;
    require_status(
        rt_adam_options_init(
            &options.value,
            (uint64_t)sizeof(options)
        ),
        "initialize oversized Adam options"
    );
    require_condition(
        options.value.struct_size == (uint64_t)sizeof(options),
        "oversized Adam options report caller size"
    );
    require_condition(
        options.value.reserved == 0 &&
            options.value.reserved2 == 0,
        "Adam options initializer clears reserved fields"
    );
    require_condition(
        options.value.state_storage == RT_ADAM_STATE_CONTIGUOUS &&
            options.value.page_size == UINT64_C(4096),
        "Adam options initializer selects contiguous default state"
    );
    require_condition(
        options.tail[0] == second_canary &&
            options.tail[1] == first_canary,
        "Adam options initializer preserves extension tail"
    );

    options.value.reserved = 1;
    rt_adam* rejected_adam = (rt_adam*)(uintptr_t)1;
    require_condition(
        rt_adam_create(
            parameters,
            &options.value,
            &rejected_adam
        ) == RT_STATUS_INVALID_ARGUMENT,
        "Adam rejects a nonzero reserved option"
    );
    require_condition(
        rejected_adam == NULL,
        "reserved-option rejection clears Adam output"
    );
    require_condition(
        strstr(rt_last_error(), "reserved field must be zero") != NULL,
        "reserved-option rejection records a diagnostic"
    );
    require_condition(
        options.tail[0] == second_canary &&
            options.tail[1] == first_canary,
        "rejected Adam creation preserves options extension tail"
    );

    options.value.reserved = 0;
    options.value.reserved2 = 1;
    rejected_adam = (rt_adam*)(uintptr_t)1;
    require_condition(
        rt_adam_create(parameters, &options.value, &rejected_adam) ==
            RT_STATUS_INVALID_ARGUMENT &&
            rejected_adam == NULL,
        "Adam rejects a nonzero second reserved option"
    );

    options.value.reserved2 = 0;
    options.value.state_storage = (rt_adam_state_storage_kind)99;
    rejected_adam = (rt_adam*)(uintptr_t)1;
    require_condition(
        rt_adam_create(parameters, &options.value, &rejected_adam) ==
            RT_STATUS_INVALID_ARGUMENT &&
            rejected_adam == NULL,
        "Adam rejects an unknown state storage kind"
    );

    options.value.state_storage = RT_ADAM_STATE_PAGED;
    options.value.page_size = 0;
    rejected_adam = (rt_adam*)(uintptr_t)1;
    require_condition(
        rt_adam_create(parameters, &options.value, &rejected_adam) ==
            RT_STATUS_INVALID_ARGUMENT &&
            rejected_adam == NULL,
        "Adam rejects a zero state page size"
    );

    options.value.page_size = 2;
    rt_adam* adam = NULL;
    require_status(
        rt_adam_create(parameters, &options.value, &adam),
        "consume oversized Adam options"
    );
    require_condition(
        options.tail[0] == second_canary &&
            options.tail[1] == first_canary,
        "Adam creation preserves options extension tail"
    );
    rt_adam_state_storage_kind state_storage =
        (rt_adam_state_storage_kind)-1;
    uint64_t state_page_size = 0;
    uint64_t state_page_count = 0;
    uint64_t state_payload_bytes = 0;
    require_status(
        rt_adam_state_storage(adam, &state_storage),
        "query paged Adam state storage"
    );
    require_status(
        rt_adam_state_page_size(adam, &state_page_size),
        "query paged Adam page size"
    );
    require_status(
        rt_adam_state_page_count(adam, &state_page_count),
        "query paged Adam page count"
    );
    require_status(
        rt_adam_state_payload_bytes(adam, &state_payload_bytes),
        "query paged Adam payload bytes"
    );
    require_condition(
        state_storage == RT_ADAM_STATE_PAGED &&
            state_page_size == 2 &&
            state_page_count > 0 &&
            state_payload_bytes > 0,
        "paged Adam reports its state layout"
    );

    rt_adam_step_stats undersized = {
        (uint64_t)sizeof(rt_adam_step_stats) - 1,
        UINT64_C(0x1122334455667788),
        -123.0,
        -456.0,
    };
    const rt_adam_step_stats original_undersized = undersized;
    require_condition(
        rt_adam_step(adam, &undersized) ==
            RT_STATUS_INVALID_ARGUMENT,
        "Adam rejects undersized step statistics"
    );
    require_condition(
        memcmp(
            &undersized,
            &original_undersized,
            sizeof(undersized)
        ) == 0,
        "undersized statistics remain untouched on failure"
    );
    require_condition(
        strstr(rt_last_error(), "structure is too small") != NULL,
        "undersized statistics record a diagnostic"
    );

    uint64_t step_count = UINT64_MAX;
    require_status(
        rt_adam_step_count(adam, &step_count),
        "query Adam after rejected statistics"
    );
    require_condition(
        step_count == 0,
        "rejected statistics do not advance Adam"
    );

    extended_adam_step_stats stats;
    memset(&stats, 0xc3, sizeof(stats));
    stats.value.struct_size = (uint64_t)sizeof(stats);
    stats.value.step = UINT64_MAX;
    stats.value.gradient_norm = -1.0;
    stats.value.clip_scale = -1.0;
    stats.tail[0] = first_canary;
    stats.tail[1] = second_canary;
    require_status(
        rt_adam_step(adam, &stats.value),
        "write oversized Adam step statistics"
    );
    require_condition(
        stats.value.struct_size == (uint64_t)sizeof(stats),
        "Adam statistics preserve caller size"
    );
    require_condition(
        stats.value.step == 1,
        "Adam statistics report the successful step"
    );
    require_condition(
        isfinite(stats.value.gradient_norm) &&
            stats.value.gradient_norm >= 0.0,
        "Adam statistics report a valid gradient norm"
    );
    require_condition(
        isfinite(stats.value.clip_scale) &&
            stats.value.clip_scale > 0.0 &&
            stats.value.clip_scale <= 1.0,
        "Adam statistics report a valid clip scale"
    );
    require_condition(
        stats.tail[0] == first_canary &&
            stats.tail[1] == second_canary,
        "Adam statistics writer preserves extension tail"
    );

    rt_adam_release(adam);
    rt_parameter_list_release(parameters);
    rt_model_release(model);
}

static void test_full_sequence_attention_api(void) {
    rt_transformer_config config;
    require_status(
        rt_transformer_config_init(
            &config,
            (uint64_t)sizeof(config)
        ),
        "initialize attention-selector model config"
    );
    config.vocabulary_size = 5;
    config.maximum_context = 4;
    config.model_width = 4;
    config.head_count = 2;
    config.block_count = 1;
    config.feed_forward_width = 8;

    rt_model* model = NULL;
    require_status(
        rt_model_create(&config, &model),
        "create attention-selector model"
    );

    rt_full_sequence_attention_kind kind =
        (rt_full_sequence_attention_kind)-1;
    require_status(
        rt_model_full_sequence_attention(model, &kind),
        "query default full-sequence attention"
    );
    require_condition(
        kind == RT_FULL_SEQUENCE_ATTENTION_MATERIALIZED,
        "model defaults to materialized full-sequence attention"
    );

    require_status(
        rt_model_set_full_sequence_attention(
            model,
            RT_FULL_SEQUENCE_ATTENTION_FLASH
        ),
        "select Flash full-sequence attention"
    );
    require_status(
        rt_model_full_sequence_attention(model, &kind),
        "query selected full-sequence attention"
    );
    require_condition(
        kind == RT_FULL_SEQUENCE_ATTENTION_FLASH,
        "model reports selected Flash full-sequence attention"
    );

    require_condition(
        rt_model_set_full_sequence_attention(
            model,
            (rt_full_sequence_attention_kind)99
        ) == RT_STATUS_INVALID_ARGUMENT,
        "attention selector rejects unknown kinds"
    );
    require_status(
        rt_model_full_sequence_attention(model, &kind),
        "query attention after invalid selection"
    );
    require_condition(
        kind == RT_FULL_SEQUENCE_ATTENTION_FLASH,
        "invalid attention selection preserves the prior policy"
    );
    require_condition(
        rt_model_full_sequence_attention(model, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "attention getter rejects a null output"
    );
    require_condition(
        rt_model_set_full_sequence_attention(
            NULL,
            RT_FULL_SEQUENCE_ATTENTION_FLASH
        ) == RT_STATUS_INVALID_ARGUMENT,
        "attention setter rejects a null model"
    );

    rt_model_release(model);
}

static void test_activation_checkpointing_api(void) {
    rt_transformer_config config;
    require_status(
        rt_transformer_config_init(
            &config,
            (uint64_t)sizeof(config)
        ),
        "initialize checkpointing model config"
    );
    config.vocabulary_size = 5;
    config.maximum_context = 4;
    config.model_width = 4;
    config.head_count = 2;
    config.block_count = 2;
    config.feed_forward_width = 8;

    rt_model* model = NULL;
    require_status(
        rt_model_create(&config, &model),
        "create checkpointing model"
    );

    rt_activation_checkpointing_kind kind =
        (rt_activation_checkpointing_kind)-1;
    require_status(
        rt_model_activation_checkpointing(model, &kind),
        "query default activation checkpointing"
    );
    require_condition(
        kind == RT_ACTIVATION_CHECKPOINTING_DISABLED,
        "model defaults to disabled activation checkpointing"
    );
    require_status(
        rt_model_set_activation_checkpointing(
            model,
            RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK
        ),
        "select transformer-block activation checkpointing"
    );
    require_status(
        rt_model_activation_checkpointing(model, &kind),
        "query selected activation checkpointing"
    );
    require_condition(
        kind == RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK,
        "model reports transformer-block activation checkpointing"
    );
    require_condition(
        rt_model_set_activation_checkpointing(
            model,
            (rt_activation_checkpointing_kind)99
        ) == RT_STATUS_INVALID_ARGUMENT,
        "activation checkpointing rejects unknown kinds"
    );
    require_status(
        rt_model_activation_checkpointing(model, &kind),
        "query checkpointing after invalid selection"
    );
    require_condition(
        kind == RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK,
        "invalid checkpointing selection preserves the prior policy"
    );
    require_condition(
        rt_model_activation_checkpointing(model, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "checkpointing getter rejects a null output"
    );
    require_condition(
        rt_model_set_activation_checkpointing(
            NULL,
            RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK
        ) == RT_STATUS_INVALID_ARGUMENT,
        "checkpointing setter rejects a null model"
    );

    rt_decode_session* session = NULL;
    require_status(
        rt_model_decode_session_create(model, NULL, &session),
        "create decode session while testing checkpointing"
    );
    require_status(
        rt_model_set_activation_checkpointing(
            model,
            RT_ACTIVATION_CHECKPOINTING_DISABLED
        ),
        "change checkpointing while a decode session is alive"
    );
    require_status(
        rt_model_set_activation_checkpointing(
            model,
            RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK
        ),
        "restore checkpointing while a decode session is alive"
    );
    rt_decode_session_release(session);

    const uint32_t tokens[] = {0, 1, 2};
    const uint32_t targets[] = {1, 2, 3};
    rt_variable* logits = NULL;
    require_status(
        rt_model_forward(
            model,
            tokens,
            3,
            1,
            3,
            &logits
        ),
        "checkpointed C API forward"
    );
    rt_variable* loss = NULL;
    require_status(
        rt_cross_entropy(logits, targets, 3, &loss),
        "checkpointed C API loss"
    );
    rt_variable_release(logits);
    rt_model_release(model);
    require_status(
        rt_variable_backward(loss),
        "checkpointed C API backward after model handle release"
    );
    rt_variable_release(loss);
}

static void test_lora_api(void) {
    const uint64_t first_canary = UINT64_C(0x13579bdf2468ace0);
    const uint64_t second_canary = UINT64_C(0xfdb97531eca86420);

    require_condition(
        rt_lora_config_init(
            NULL,
            (uint64_t)sizeof(rt_lora_config)
        ) == RT_STATUS_INVALID_ARGUMENT,
        "LoRA config initializer rejects null"
    );

    rt_lora_config undersized;
    memset(&undersized, 0x5a, sizeof(undersized));
    const rt_lora_config original_undersized = undersized;
    require_condition(
        rt_lora_config_init(
            &undersized,
            (uint64_t)sizeof(undersized) - 1
        ) == RT_STATUS_INVALID_ARGUMENT,
        "LoRA config initializer rejects undersized storage"
    );
    require_condition(
        memcmp(
            &undersized,
            &original_undersized,
            sizeof(undersized)
        ) == 0,
        "failed LoRA config initialization is atomic"
    );

    extended_lora_config extended;
    memset(&extended, 0xa5, sizeof(extended));
    extended.tail[0] = first_canary;
    extended.tail[1] = second_canary;
    require_status(
        rt_lora_config_init(
            &extended.value,
            (uint64_t)sizeof(extended)
        ),
        "initialize oversized LoRA config"
    );
    require_condition(
        extended.value.struct_size == (uint64_t)sizeof(extended) &&
            extended.value.rank > 0 &&
            isfinite(extended.value.alpha) &&
            extended.value.alpha > 0.0F &&
            extended.value.targets == RT_LORA_TARGET_DEFAULT &&
            extended.value.reserved == 0,
        "LoRA config defaults"
    );
    require_condition(
        extended.tail[0] == first_canary &&
            extended.tail[1] == second_canary,
        "LoRA config initializer preserves extension tail"
    );

    rt_transformer_config model_config;
    require_status(
        rt_transformer_config_init(
            &model_config,
            (uint64_t)sizeof(model_config)
        ),
        "initialize LoRA model config"
    );
    model_config.vocabulary_size = 5;
    model_config.maximum_context = 3;
    model_config.model_width = 4;
    model_config.head_count = 2;
    model_config.block_count = 1;
    model_config.feed_forward_width = 8;
    model_config.random_seed = 401;

    rt_model* model = NULL;
    require_status(
        rt_model_create(&model_config, &model),
        "create LoRA model"
    );
    uint64_t epoch_before_attachment = UINT64_MAX;
    require_status(
        rt_test_model_parameter_epoch(
            model,
            &epoch_before_attachment
        ),
        "query epoch before LoRA attachment"
    );
    int32_t attached = -1;
    require_status(
        rt_model_has_lora(model, &attached),
        "query initial LoRA state"
    );
    require_condition(attached == 0, "model initially has no LoRA");

    rt_lora_config unavailable_config;
    memset(&unavailable_config, 0x3c, sizeof(unavailable_config));
    unavailable_config.struct_size =
        (uint64_t)sizeof(unavailable_config);
    const uint64_t unavailable_rank = unavailable_config.rank;
    require_condition(
        rt_model_lora_config(model, &unavailable_config) ==
            RT_STATUS_INVALID_ARGUMENT,
        "LoRA config query requires an attached adapter"
    );
    require_condition(
        unavailable_config.rank == unavailable_rank,
        "failed LoRA config query preserves output"
    );

    rt_parameter_list* base_parameters = NULL;
    require_status(
        rt_model_parameters(model, &base_parameters),
        "query base parameters before LoRA"
    );
    uint64_t base_count = 0;
    require_status(
        rt_parameter_list_count(base_parameters, &base_count),
        "count base parameters before LoRA"
    );

    const uint32_t token_ids[] = {0, 1};
    rt_variable* before = NULL;
    require_status(
        rt_model_forward(model, token_ids, 2, 1, 2, &before),
        "forward before LoRA attachment"
    );
    float before_values[10];
    require_status(
        rt_variable_copy_to_host_f32(before, before_values, 10),
        "copy output before LoRA attachment"
    );
    rt_variable_release(before);

    rt_lora_config config;
    require_status(
        rt_lora_config_init(
            &config,
            (uint64_t)sizeof(config)
        ),
        "initialize exact LoRA config"
    );
    config.rank = 2;
    config.alpha = 4.0F;
    config.random_seed = 409;
    config.targets =
        RT_LORA_TARGET_ATTENTION_QUERY |
        RT_LORA_TARGET_ATTENTION_VALUE;

    rt_lora_config invalid_config = config;
    invalid_config.rank = 0;
    require_condition(
        rt_model_attach_lora(model, &invalid_config) ==
            RT_STATUS_INVALID_ARGUMENT,
        "LoRA attachment rejects zero rank"
    );
    invalid_config = config;
    invalid_config.targets = UINT64_C(1) << 63;
    require_condition(
        rt_model_attach_lora(model, &invalid_config) ==
            RT_STATUS_INVALID_ARGUMENT,
        "LoRA attachment rejects unknown target bits"
    );
    invalid_config = config;
    invalid_config.reserved = 1;
    require_condition(
        rt_model_attach_lora(model, &invalid_config) ==
            RT_STATUS_INVALID_ARGUMENT,
        "LoRA attachment rejects reserved fields"
    );

    require_status(
        rt_model_attach_lora(model, &config),
        "attach LoRA"
    );
    uint64_t epoch_after_attachment = UINT64_MAX;
    require_status(
        rt_test_model_parameter_epoch(
            model,
            &epoch_after_attachment
        ),
        "query epoch after LoRA attachment"
    );
    require_condition(
        epoch_after_attachment == epoch_before_attachment + 1,
        "LoRA attachment increments the parameter epoch once"
    );
    require_status(
        rt_model_has_lora(model, &attached),
        "query attached LoRA state"
    );
    require_condition(attached == 1, "model reports attached LoRA");
    require_condition(
        rt_model_attach_lora(model, &config) ==
            RT_STATUS_INVALID_ARGUMENT,
        "model rejects a second LoRA attachment"
    );

    extended_lora_config observed_config;
    memset(&observed_config, 0xc3, sizeof(observed_config));
    observed_config.value.struct_size =
        (uint64_t)sizeof(observed_config);
    observed_config.tail[0] = first_canary;
    observed_config.tail[1] = second_canary;
    require_status(
        rt_model_lora_config(model, &observed_config.value),
        "query oversized attached LoRA config"
    );
    require_condition(
        observed_config.value.struct_size ==
                (uint64_t)sizeof(observed_config) &&
            observed_config.value.rank == config.rank &&
            observed_config.value.alpha == config.alpha &&
            observed_config.value.random_seed ==
                config.random_seed &&
            observed_config.value.targets == config.targets &&
            observed_config.value.reserved == 0,
        "attached LoRA config round trip"
    );
    require_condition(
        observed_config.tail[0] == first_canary &&
            observed_config.tail[1] == second_canary,
        "LoRA config query preserves extension tail"
    );

    rt_parameter_list* base_after_attachment = NULL;
    require_status(
        rt_model_parameters(model, &base_after_attachment),
        "query base parameters after LoRA"
    );
    uint64_t base_count_after_attachment = 0;
    require_status(
        rt_parameter_list_count(
            base_after_attachment,
            &base_count_after_attachment
        ),
        "count base parameters after LoRA"
    );
    require_condition(
        base_count_after_attachment == base_count,
        "LoRA preserves base parameter count"
    );

    rt_parameter_list* adapters = NULL;
    require_status(
        rt_model_lora_parameters(model, &adapters),
        "query LoRA adapter parameters"
    );
    uint64_t adapter_count = 0;
    require_status(
        rt_parameter_list_count(adapters, &adapter_count),
        "count LoRA adapter parameters"
    );
    require_condition(
        adapter_count == 4,
        "query/value LoRA exposes A and B parameters"
    );
    static const char* expected_names[] = {
        "blocks.0.attention.query.lora_a.weight",
        "blocks.0.attention.query.lora_b.weight",
        "blocks.0.attention.value.lora_a.weight",
        "blocks.0.attention.value.lora_b.weight",
    };
    static const uint64_t expected_shapes[][2] = {
        {2, 4},
        {4, 2},
        {2, 4},
        {4, 2},
    };
    for (uint64_t index = 0; index < adapter_count; ++index) {
        char name[96];
        uint64_t required_capacity = 0;
        require_status(
            rt_parameter_list_name(
                adapters,
                index,
                name,
                (uint64_t)sizeof(name),
                &required_capacity
            ),
            "query LoRA parameter name"
        );
        require_condition(
            strcmp(name, expected_names[index]) == 0,
            "LoRA parameter name and ordering"
        );
        uint64_t rank = 0;
        uint64_t shape[2] = {0, 0};
        require_status(
            rt_parameter_list_rank(adapters, index, &rank),
            "query LoRA parameter rank"
        );
        require_status(
            rt_parameter_list_shape(adapters, index, shape, 2),
            "query LoRA parameter shape"
        );
        require_condition(
            rank == 2 &&
                shape[0] == expected_shapes[index][0] &&
                shape[1] == expected_shapes[index][1],
            "LoRA parameter shape"
        );
    }
    uint64_t adapter_numel = 0;
    require_status(
        rt_parameter_list_total_numel(adapters, &adapter_numel),
        "count LoRA adapter values"
    );
    require_condition(
        adapter_numel == 32,
        "LoRA adapter flattened value count"
    );

    rt_variable* attached_output = NULL;
    require_status(
        rt_model_forward(
            model,
            token_ids,
            2,
            1,
            2,
            &attached_output
        ),
        "forward after LoRA attachment"
    );
    float attached_values[10];
    require_status(
        rt_variable_copy_to_host_f32(
            attached_output,
            attached_values,
            10
        ),
        "copy output after LoRA attachment"
    );
    for (size_t index = 0; index < 10; ++index) {
        require_close(
            attached_values[index],
            before_values[index],
            "zero-initialized LoRA preserves output"
        );
    }
    require_condition(
        rt_model_merge_lora(model) == RT_STATUS_INVALID_ARGUMENT,
        "LoRA merge rejects a live variable graph"
    );
    rt_variable_release(attached_output);

    require_condition(
        rt_model_merge_lora(model) == RT_STATUS_INVALID_ARGUMENT,
        "LoRA merge rejects a live adapter parameter list"
    );
    rt_adam* optimizer = NULL;
    require_status(
        rt_adam_create(adapters, NULL, &optimizer),
        "create Adam for LoRA adapter parameters"
    );
    rt_parameter_list_release(adapters);
    adapters = NULL;
    require_condition(
        rt_model_merge_lora(model) == RT_STATUS_INVALID_ARGUMENT,
        "LoRA merge rejects a live optimizer"
    );
    rt_adam_release(optimizer);

    require_status(
        rt_test_model_set_parameter_epoch(model, UINT64_MAX),
        "set LoRA epoch to overflow boundary"
    );
    require_condition(
        rt_model_merge_lora(model) == RT_STATUS_OVERFLOW,
        "LoRA merge rejects parameter epoch overflow"
    );
    require_status(
        rt_model_has_lora(model, &attached),
        "query LoRA state after rejected overflow merge"
    );
    require_condition(
        attached == 1,
        "rejected overflow merge preserves attached LoRA"
    );
    require_status(
        rt_test_model_set_parameter_epoch(
            model,
            epoch_after_attachment
        ),
        "restore LoRA parameter epoch"
    );

    uint64_t epoch_before_merge = UINT64_MAX;
    require_status(
        rt_test_model_parameter_epoch(model, &epoch_before_merge),
        "query epoch before LoRA merge"
    );
    require_status(
        rt_model_merge_lora(model),
        "merge LoRA"
    );
    uint64_t epoch_after_merge = UINT64_MAX;
    require_status(
        rt_test_model_parameter_epoch(model, &epoch_after_merge),
        "query epoch after LoRA merge"
    );
    require_condition(
        epoch_after_merge == epoch_before_merge + 1,
        "LoRA merge increments the parameter epoch once"
    );
    require_status(
        rt_model_has_lora(model, &attached),
        "query merged LoRA state"
    );
    require_condition(attached == 0, "merged LoRA is detached");

    rt_parameter_list* rejected =
        (rt_parameter_list*)(uintptr_t)1;
    require_condition(
        rt_model_lora_parameters(model, &rejected) ==
            RT_STATUS_INVALID_ARGUMENT,
        "adapter parameter query rejects a merged model"
    );
    require_condition(
        rejected == NULL,
        "failed adapter parameter query clears output"
    );
    require_condition(
        rt_model_merge_lora(model) == RT_STATUS_INVALID_ARGUMENT,
        "model rejects a second LoRA merge"
    );

    rt_variable* merged_output = NULL;
    require_status(
        rt_model_forward(
            model,
            token_ids,
            2,
            1,
            2,
            &merged_output
        ),
        "forward after LoRA merge"
    );
    float merged_values[10];
    require_status(
        rt_variable_copy_to_host_f32(
            merged_output,
            merged_values,
            10
        ),
        "copy output after LoRA merge"
    );
    for (size_t index = 0; index < 10; ++index) {
        require_close(
            merged_values[index],
            attached_values[index],
            "merged LoRA preserves attached output"
        );
    }

    rt_variable_release(merged_output);
    rt_parameter_list_release(base_after_attachment);
    rt_parameter_list_release(base_parameters);
    rt_model_release(model);

    require_condition(
        rt_model_attach_lora(NULL, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "LoRA attachment rejects a null model"
    );
    require_condition(
        rt_model_has_lora(NULL, &attached) ==
            RT_STATUS_INVALID_ARGUMENT,
        "LoRA state query rejects a null model"
    );
}

static void test_qlora_api(void) {
    rt_transformer_config config;
    require_status(
        rt_transformer_config_init(
            &config,
            (uint64_t)sizeof(config)
        ),
        "initialize QLoRA model config"
    );
    config.vocabulary_size = 5;
    config.maximum_context = 3;
    config.model_width = 4;
    config.head_count = 2;
    config.block_count = 1;
    config.feed_forward_width = 8;
    config.random_seed = 443;

    rt_model* model = NULL;
    require_status(
        rt_model_create(&config, &model),
        "create QLoRA model"
    );
    int32_t quantized = -1;
    require_status(
        rt_model_has_quantized_linear_weights(model, &quantized),
        "query initial quantized state"
    );
    require_condition(quantized == 0, "model initially is not quantized");

    require_condition(
        rt_model_quantize_linear_weights_nf4(model, 16) ==
            RT_STATUS_INVALID_ARGUMENT,
        "NF4 conversion rejects an unsupported block size"
    );
    require_status(
        rt_model_has_quantized_linear_weights(model, &quantized),
        "query quantized state after invalid block size"
    );
    require_condition(
        quantized == 0,
        "invalid NF4 block size leaves the model unquantized"
    );

    const uint32_t token_ids[] = {0, 1};
    const uint32_t targets[] = {1, 2};
    rt_variable* live_logits = NULL;
    require_status(
        rt_model_forward(model, token_ids, 2, 1, 2, &live_logits),
        "forward before rejected live-variable NF4 conversion"
    );
    require_condition(
        rt_model_quantize_linear_weights_nf4(model, 64) ==
            RT_STATUS_INVALID_ARGUMENT,
        "NF4 conversion rejects a live forward variable"
    );
    require_status(
        rt_model_has_quantized_linear_weights(model, &quantized),
        "query quantized state after live-variable rejection"
    );
    require_condition(
        quantized == 0,
        "live-variable rejection leaves the model unquantized"
    );
    float live_logits_values[10];
    require_status(
        rt_variable_copy_to_host_f32(
            live_logits,
            live_logits_values,
            10
        ),
        "copy the live forward variable after rejected conversion"
    );
    for (size_t index = 0; index < 10; ++index) {
        require_condition(
            isfinite(live_logits_values[index]),
            "rejected conversion preserves the live forward variable"
        );
    }
    rt_variable_release(live_logits);

    rt_parameter_list* optimizer_parameters = NULL;
    require_status(
        rt_model_parameters(model, &optimizer_parameters),
        "query parameters for the live-Adam NF4 guard"
    );
    rt_adam* live_optimizer = NULL;
    require_status(
        rt_adam_create(optimizer_parameters, NULL, &live_optimizer),
        "create live Adam before rejected NF4 conversion"
    );
    rt_parameter_list_release(optimizer_parameters);
    require_condition(
        rt_model_quantize_linear_weights_nf4(model, 64) ==
            RT_STATUS_INVALID_ARGUMENT,
        "NF4 conversion rejects a live Adam optimizer"
    );
    require_status(
        rt_model_has_quantized_linear_weights(model, &quantized),
        "query quantized state after live-Adam rejection"
    );
    require_condition(
        quantized == 0,
        "live-Adam rejection leaves the model unquantized"
    );
    uint64_t live_optimizer_step_count = UINT64_MAX;
    require_status(
        rt_adam_step_count(live_optimizer, &live_optimizer_step_count),
        "query live Adam after rejected conversion"
    );
    require_condition(
        live_optimizer_step_count == 0,
        "rejected conversion preserves the live Adam optimizer"
    );
    rt_adam_release(live_optimizer);

    rt_decode_session* live_session = NULL;
    require_status(
        rt_model_decode_session_create(model, NULL, &live_session),
        "create live decode session before rejected NF4 conversion"
    );
    require_condition(
        rt_model_quantize_linear_weights_nf4(model, 64) ==
            RT_STATUS_INVALID_ARGUMENT,
        "NF4 conversion rejects a live decode session"
    );
    require_status(
        rt_model_has_quantized_linear_weights(model, &quantized),
        "query quantized state after live-decode rejection"
    );
    require_condition(
        quantized == 0,
        "live-decode rejection leaves the model unquantized"
    );
    float decode_logits[5];
    uint64_t decode_logits_count = UINT64_MAX;
    require_status(
        rt_decode_session_step(
            live_session,
            token_ids[0],
            decode_logits,
            5,
            &decode_logits_count
        ),
        "use live decode session after rejected conversion"
    );
    require_condition(
        decode_logits_count == 5,
        "rejected conversion preserves the live decode session"
    );
    for (size_t index = 0; index < 5; ++index) {
        require_condition(
            isfinite(decode_logits[index]),
            "live decode output remains finite after rejected conversion"
        );
    }
    rt_decode_session_release(live_session);

    rt_parameter_list* dense_parameters = NULL;
    require_status(
        rt_model_parameters(model, &dense_parameters),
        "query dense parameters before QLoRA"
    );
    uint64_t dense_count = 0;
    require_status(
        rt_parameter_list_count(dense_parameters, &dense_count),
        "count dense parameters before QLoRA"
    );
    require_condition(
        rt_model_quantize_linear_weights_nf4(model, 64) ==
            RT_STATUS_INVALID_ARGUMENT,
        "NF4 conversion rejects a live parameter list"
    );
    rt_parameter_list_release(dense_parameters);

    require_status(
        rt_model_quantize_linear_weights_nf4_double_quantized(
            model,
            64,
            256
        ),
        "double-quantize model linear weights as NF4"
    );
    require_status(
        rt_model_has_quantized_linear_weights(model, &quantized),
        "query quantized state"
    );
    require_condition(quantized == 1, "model reports packed linear weights");
    require_condition(
        rt_model_quantize_linear_weights_nf4(model, 64) ==
            RT_STATUS_INVALID_ARGUMENT,
        "model rejects a second NF4 conversion"
    );

    rt_quantized_memory_stats memory;
    memset(&memory, 0, sizeof(memory));
    memory.struct_size = (uint64_t)sizeof(memory);
    require_status(
        rt_model_quantized_memory_stats(model, &memory),
        "query QLoRA packed-memory statistics"
    );
    require_condition(
        memory.weight_count == 7 &&
            memory.packed_code_bytes == 74 &&
            memory.scale_bytes == 63 &&
            memory.logical_payload_bytes == 137 &&
            memory.resident_payload_bytes == 137 &&
            memory.fp32_equivalent_bytes == 592 &&
            memory.fp32_scale_bytes == 0 &&
            memory.scale_code_bytes == 7 &&
            memory.second_level_scale_bytes == 28 &&
            memory.scale_offset_bytes == 28 &&
            memory.double_quantized_weight_count == 7,
        "QLoRA memory statistics describe double-quantized NF4 payloads"
    );

    rt_quantized_memory_stats legacy_memory;
    memset(&legacy_memory, 0xa5, sizeof(legacy_memory));
    legacy_memory.struct_size =
        RT_LEGACY_QUANTIZED_MEMORY_STATS_SIZE;
    unsigned char legacy_memory_extension[
        sizeof(legacy_memory) - RT_LEGACY_QUANTIZED_MEMORY_STATS_SIZE
    ];
    memcpy(
        legacy_memory_extension,
        (const unsigned char*)&legacy_memory +
            RT_LEGACY_QUANTIZED_MEMORY_STATS_SIZE,
        sizeof(legacy_memory_extension)
    );
    require_status(
        rt_model_quantized_memory_stats(model, &legacy_memory),
        "query legacy 56-byte packed-memory statistics"
    );
    require_condition(
        legacy_memory.struct_size ==
                RT_LEGACY_QUANTIZED_MEMORY_STATS_SIZE &&
            legacy_memory.weight_count == 7 &&
            legacy_memory.packed_code_bytes == 74 &&
            legacy_memory.scale_bytes == 63 &&
            legacy_memory.logical_payload_bytes == 137 &&
            legacy_memory.resident_payload_bytes == 137 &&
            legacy_memory.fp32_equivalent_bytes == 592,
        "legacy packed-memory statistics expose the stable prefix"
    );
    require_condition(
        memcmp(
            (const unsigned char*)&legacy_memory +
                RT_LEGACY_QUANTIZED_MEMORY_STATS_SIZE,
            legacy_memory_extension,
            sizeof(legacy_memory_extension)
        ) == 0,
        "legacy packed-memory statistics preserve extension canaries"
    );

    rt_parameter_list* frozen_parameters = NULL;
    require_status(
        rt_model_parameters(model, &frozen_parameters),
        "query parameters after NF4 conversion"
    );
    uint64_t frozen_count = 0;
    require_status(
        rt_parameter_list_count(frozen_parameters, &frozen_count),
        "count parameters after NF4 conversion"
    );
    require_condition(
        frozen_count + 7 == dense_count,
        "packed weights are absent from the Parameter list"
    );
    rt_parameter_list_release(frozen_parameters);

    rt_lora_config lora;
    require_status(
        rt_lora_config_init(&lora, (uint64_t)sizeof(lora)),
        "initialize QLoRA adapter config"
    );
    lora.rank = 2;
    lora.alpha = 4.0F;
    lora.random_seed = 449;
    require_status(
        rt_model_attach_lora(model, &lora),
        "attach adapters to packed model"
    );
    rt_parameter_list* adapters = NULL;
    require_status(
        rt_model_lora_parameters(model, &adapters),
        "query QLoRA adapter parameters"
    );
    rt_adam_options qlora_adam_options;
    require_status(
        rt_adam_options_init(
            &qlora_adam_options,
            (uint64_t)sizeof(qlora_adam_options)
        ),
        "initialize paged QLoRA Adam options"
    );
    qlora_adam_options.state_storage = RT_ADAM_STATE_PAGED;
    qlora_adam_options.page_size = 2;
    rt_adam* optimizer = NULL;
    require_status(
        rt_adam_create(adapters, &qlora_adam_options, &optimizer),
        "create paged adapter-only QLoRA Adam"
    );
    rt_adam_state_storage_kind qlora_state_storage =
        (rt_adam_state_storage_kind)-1;
    uint64_t qlora_state_page_size = 0;
    uint64_t qlora_state_page_count = 0;
    uint64_t qlora_state_payload_bytes = 0;
    require_status(
        rt_adam_state_storage(optimizer, &qlora_state_storage),
        "query QLoRA Adam state storage"
    );
    require_status(
        rt_adam_state_page_size(optimizer, &qlora_state_page_size),
        "query QLoRA Adam page size"
    );
    require_status(
        rt_adam_state_page_count(optimizer, &qlora_state_page_count),
        "query QLoRA Adam page count"
    );
    require_status(
        rt_adam_state_payload_bytes(
            optimizer,
            &qlora_state_payload_bytes
        ),
        "query QLoRA Adam state payload"
    );
    require_condition(
        qlora_state_storage == RT_ADAM_STATE_PAGED &&
            qlora_state_page_size == 2 &&
            qlora_state_page_count > 0 &&
            qlora_state_payload_bytes > 0,
        "QLoRA Adam reports paged adapter-only state"
    );
    rt_variable* logits = NULL;
    require_status(
        rt_model_forward(model, token_ids, 2, 1, 2, &logits),
        "QLoRA model forward"
    );
    rt_variable* loss = NULL;
    require_status(
        rt_cross_entropy(logits, targets, 2, &loss),
        "QLoRA loss"
    );
    rt_variable_release(logits);
    require_status(rt_variable_backward(loss), "QLoRA backward");
    rt_variable_release(loss);
    rt_adam_step_stats step = {
        (uint64_t)sizeof(step),
        0,
        0.0,
        0.0,
    };
    require_status(rt_adam_step(optimizer, &step), "QLoRA Adam step");
    require_condition(step.step == 1, "QLoRA optimizer advances once");
    rt_adam_release(optimizer);
    rt_parameter_list_release(adapters);

    rt_quantized_memory_stats after_step;
    memset(&after_step, 0, sizeof(after_step));
    after_step.struct_size = (uint64_t)sizeof(after_step);
    require_status(
        rt_model_quantized_memory_stats(model, &after_step),
        "query packed memory after QLoRA step"
    );
    require_condition(
        memcmp(&memory, &after_step, sizeof(memory)) == 0,
        "QLoRA training preserves packed base storage accounting"
    );

    require_status(rt_model_merge_lora(model), "export QLoRA model to FP32");
    require_status(
        rt_model_has_quantized_linear_weights(model, &quantized),
        "query quantized state after export merge"
    );
    require_condition(quantized == 0, "QLoRA export removes packed runtime state");
    rt_parameter_list* exported_parameters = NULL;
    require_status(
        rt_model_parameters(model, &exported_parameters),
        "query exported FP32 parameters"
    );
    uint64_t exported_count = 0;
    require_status(
        rt_parameter_list_count(exported_parameters, &exported_count),
        "count exported FP32 parameters"
    );
    require_condition(
        exported_count == dense_count,
        "QLoRA export restores the ordinary parameter schema"
    );
    rt_parameter_list_release(exported_parameters);
    rt_model_release(model);

    require_condition(
        rt_model_quantize_linear_weights_nf4(NULL, 64) ==
            RT_STATUS_INVALID_ARGUMENT,
        "NF4 conversion rejects a null model"
    );
}

static void copy_last_model_logits(
    const rt_model* model,
    const uint32_t* tokens,
    uint64_t token_count,
    float* output,
    uint64_t vocabulary_size
) {
    rt_variable* logits = NULL;
    require_status(
        rt_model_forward(
            model,
            tokens,
            token_count,
            1,
            token_count,
            &logits
        ),
        "forward reference decode logits"
    );
    const uint64_t value_count = token_count * vocabulary_size;
    float* values =
        (float*)malloc((size_t)value_count * sizeof(float));
    require_condition(
        values != NULL,
        "allocate reference decode logits"
    );
    require_status(
        rt_variable_copy_to_host_f32(
            logits,
            values,
            value_count
        ),
        "copy reference decode logits"
    );
    memcpy(
        output,
        values + (token_count - 1) * vocabulary_size,
        (size_t)vocabulary_size * sizeof(float)
    );
    free(values);
    rt_variable_release(logits);
}

static void test_decode_session_api(void) {
    const uint64_t first_canary = UINT64_C(0x1020304050607080);
    const uint64_t second_canary = UINT64_C(0x8877665544332211);

    require_condition(
        rt_decode_session_options_init(
            NULL,
            (uint64_t)sizeof(rt_decode_session_options)
        ) == RT_STATUS_INVALID_ARGUMENT,
        "decode-session options initializer rejects null"
    );

    rt_decode_session_options undersized;
    memset(&undersized, 0x5a, sizeof(undersized));
    const rt_decode_session_options original_undersized = undersized;
    require_condition(
        rt_decode_session_options_init(
            &undersized,
            (uint64_t)sizeof(undersized) - 1
        ) == RT_STATUS_INVALID_ARGUMENT,
        "decode-session options initializer rejects undersized storage"
    );
    require_condition(
        memcmp(
            &undersized,
            &original_undersized,
            sizeof(undersized)
        ) == 0,
        "failed decode-session option initialization is atomic"
    );

    extended_decode_session_options extended;
    memset(&extended, 0xa5, sizeof(extended));
    extended.tail[0] = first_canary;
    extended.tail[1] = second_canary;
    require_status(
        rt_decode_session_options_init(
            &extended.value,
            (uint64_t)sizeof(extended)
        ),
        "initialize oversized decode-session options"
    );
    require_condition(
        extended.value.struct_size == (uint64_t)sizeof(extended) &&
            extended.value.kind == RT_KV_CACHE_PAGED &&
            extended.value.reserved == 0 &&
            extended.value.block_size == 16,
        "decode-session option defaults"
    );
    require_condition(
        extended.tail[0] == first_canary &&
            extended.tail[1] == second_canary,
        "decode-session option initializer preserves extension tail"
    );

    rt_transformer_config config;
    require_status(
        rt_transformer_config_init(
            &config,
            (uint64_t)sizeof(config)
        ),
        "initialize decode-session model config"
    );
    config.vocabulary_size = 5;
    config.maximum_context = 4;
    config.model_width = 4;
    config.head_count = 2;
    config.block_count = 1;
    config.feed_forward_width = 8;
    config.random_seed = 1013;

    rt_model* model = NULL;
    require_status(
        rt_model_create(&config, &model),
        "create decode-session model"
    );

    rt_decode_session* rejected =
        (rt_decode_session*)(uintptr_t)1;
    rt_decode_session_options invalid = extended.value;
    invalid.struct_size = (uint64_t)sizeof(invalid);
    invalid.kind = (rt_kv_cache_kind)77;
    require_condition(
        rt_model_decode_session_create(
            model,
            &invalid,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT,
        "decode session rejects unknown cache kind"
    );
    require_condition(
        rejected == NULL,
        "failed decode-session creation clears output"
    );
    invalid = extended.value;
    invalid.struct_size = (uint64_t)sizeof(invalid);
    invalid.reserved = 1;
    rejected = (rt_decode_session*)(uintptr_t)1;
    require_condition(
        rt_model_decode_session_create(
            model,
            &invalid,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT &&
            rejected == NULL,
        "decode session rejects reserved options"
    );
    invalid = extended.value;
    invalid.struct_size = (uint64_t)sizeof(invalid);
    invalid.block_size = 0;
    rejected = (rt_decode_session*)(uintptr_t)1;
    require_condition(
        rt_model_decode_session_create(
            model,
            &invalid,
            &rejected
        ) == RT_STATUS_INVALID_ARGUMENT &&
            rejected == NULL,
        "decode session rejects zero block size"
    );

    rt_decode_session* default_session = NULL;
    require_status(
        rt_model_decode_session_create(
            model,
            NULL,
            &default_session
        ),
        "create default decode session"
    );
    rt_kv_cache_kind kind = (rt_kv_cache_kind)-1;
    uint64_t block_size = 0;
    require_status(
        rt_decode_session_cache_kind(default_session, &kind),
        "query default decode cache kind"
    );
    require_status(
        rt_decode_session_block_size(
            default_session,
            &block_size
        ),
        "query default decode block size"
    );
    require_condition(
        kind == RT_KV_CACHE_PAGED && block_size == 16,
        "default decode session is paged with block size 16"
    );
    rt_decode_session_release(default_session);

    const uint32_t prefix[] = {1, 2, 3, 4};
    for (int cache_index = 0; cache_index < 2; ++cache_index) {
        rt_decode_session_options options;
        require_status(
            rt_decode_session_options_init(
                &options,
                (uint64_t)sizeof(options)
            ),
            "initialize parity decode-session options"
        );
        options.kind = cache_index == 0
            ? RT_KV_CACHE_CONTIGUOUS
            : RT_KV_CACHE_PAGED;
        options.block_size = 2;

        rt_decode_session* session = NULL;
        require_status(
            rt_model_decode_session_create(
                model,
                &options,
                &session
            ),
            "create parity decode session"
        );
        require_status(
            rt_decode_session_cache_kind(session, &kind),
            "query parity cache kind"
        );
        require_condition(
            kind == options.kind,
            "decode-session cache kind"
        );
        uint64_t capacity = 0;
        uint64_t size = UINT64_MAX;
        require_status(
            rt_decode_session_capacity(session, &capacity),
            "query decode-session capacity"
        );
        require_status(
            rt_decode_session_size(session, &size),
            "query empty decode-session size"
        );
        require_status(
            rt_decode_session_block_size(session, &block_size),
            "query parity cache block size"
        );
        require_condition(
            capacity == 4 &&
                size == 0 &&
                block_size == (
                    options.kind == RT_KV_CACHE_PAGED ? 2 : 4
                ),
            "decode-session cache introspection"
        );

        float short_output[5] = {
            -91.0F,
            -91.0F,
            -91.0F,
            -91.0F,
            -91.0F,
        };
        uint64_t required = UINT64_MAX;
        require_condition(
            rt_decode_session_step(
                session,
                prefix[0],
                short_output,
                4,
                &required
            ) == RT_STATUS_OUT_OF_RANGE,
            "decode step rejects a short output"
        );
        require_condition(
            required == 5 &&
                short_output[0] == -91.0F &&
                short_output[4] == -91.0F,
            "short decode output reports size and remains untouched"
        );
        require_status(
            rt_decode_session_size(session, &size),
            "query size after short decode output"
        );
        require_condition(
            size == 0,
            "short decode output does not mutate the cache"
        );
        required = UINT64_MAX;
        require_condition(
            rt_decode_session_step(
                session,
                prefix[0],
                NULL,
                0,
                &required
            ) == RT_STATUS_INVALID_ARGUMENT,
            "decode step does not support a null size query"
        );
        require_condition(
            required == 5,
            "null decode output still reports required size"
        );

        for (uint64_t index = 0; index < 4; ++index) {
            float expected[5];
            float actual[6] = {
                0.0F,
                0.0F,
                0.0F,
                0.0F,
                0.0F,
                1234.5F,
            };
            copy_last_model_logits(
                model,
                prefix,
                index + 1,
                expected,
                5
            );
            required = UINT64_MAX;
            require_status(
                rt_decode_session_step(
                    session,
                    prefix[index],
                    actual,
                    6,
                    &required
                ),
                "append one cached decode token"
            );
            require_condition(
                required == 5 && actual[5] == 1234.5F,
                "decode step writes exactly one vocabulary vector"
            );
            for (size_t logit = 0; logit < 5; ++logit) {
                require_close(
                    actual[logit],
                    expected[logit],
                    "cached decode matches full forward"
                );
            }
            require_status(
                rt_decode_session_size(session, &size),
                "query appended decode-session size"
            );
            require_condition(
                size == index + 1,
                "decode-session size advances once per token"
            );
        }

        required = UINT64_MAX;
        float full_output[5];
        require_condition(
            rt_decode_session_step(
                session,
                0,
                full_output,
                5,
                &required
            ) == RT_STATUS_OUT_OF_RANGE,
            "decode step rejects cache overflow"
        );
        require_status(
            rt_decode_session_size(session, &size),
            "query size after cache overflow"
        );
        require_condition(
            size == capacity,
            "cache overflow leaves decode-session size unchanged"
        );

        require_status(
            rt_decode_session_reset(session),
            "reset decode session"
        );
        require_status(
            rt_decode_session_size(session, &size),
            "query reset decode-session size"
        );
        require_condition(size == 0, "decode-session reset clears size");

        required = UINT64_MAX;
        require_condition(
            rt_decode_session_step(
                session,
                5,
                full_output,
                5,
                &required
            ) == RT_STATUS_OUT_OF_RANGE,
            "decode step rejects a token outside the vocabulary"
        );
        require_status(
            rt_decode_session_size(session, &size),
            "query size after invalid token"
        );
        require_condition(
            required == 5 && size == 0,
            "invalid token leaves decode-session state unchanged"
        );
        rt_decode_session_release(session);
    }

    rt_parameter_list* parameters = NULL;
    require_status(
        rt_model_parameters(model, &parameters),
        "query parameters for decode-session mutation guards"
    );
    uint64_t total_numel = 0;
    require_status(
        rt_parameter_list_total_numel(parameters, &total_numel),
        "query parameters for decode-session mutation guards"
    );
    float* parameter_values =
        (float*)malloc((size_t)total_numel * sizeof(float));
    require_condition(
        parameter_values != NULL,
        "allocate decode-session mutation-guard values"
    );
    require_status(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            parameter_values,
            total_numel
        ),
        "copy decode-session mutation-guard values"
    );

    rt_decode_session* guarded_session = NULL;
    require_status(
        rt_model_decode_session_create(
            model,
            NULL,
            &guarded_session
        ),
        "create guarded decode session"
    );
    require_condition(
        rt_model_to(model, RT_BACKEND_CPU) ==
            RT_STATUS_INVALID_ARGUMENT,
        "model move rejects a live decode session"
    );
    require_condition(
        rt_model_attach_lora(model, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "LoRA attachment rejects a live decode session"
    );
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            parameter_values,
            total_numel
        ) == RT_STATUS_INVALID_ARGUMENT,
        "parameter load rejects a live decode session"
    );
    rt_adam* adam = NULL;
    require_status(
        rt_adam_create(parameters, NULL, &adam),
        "create Adam beside a decode session"
    );
    rt_adam_step_stats stats = {
        (uint64_t)sizeof(rt_adam_step_stats),
        0,
        -1.0,
        -1.0,
    };
    require_condition(
        rt_adam_step(adam, &stats) ==
            RT_STATUS_INVALID_ARGUMENT,
        "Adam step rejects a live decode session"
    );
    rt_adam_release(adam);

    uint64_t epoch = 0;
    require_status(
        rt_test_model_parameter_epoch(model, &epoch),
        "query decode-session model epoch"
    );
    require_status(
        rt_test_model_set_parameter_epoch(model, epoch + 1),
        "force a stale decode-session epoch"
    );
    float stale_output[5];
    uint64_t required = UINT64_MAX;
    require_condition(
        rt_decode_session_step(
            guarded_session,
            0,
            stale_output,
            5,
            &required
        ) == RT_STATUS_RUNTIME_ERROR,
        "decode step rejects stale model parameters"
    );
    rt_decode_session_release(guarded_session);

    free(parameter_values);
    rt_parameter_list_release(parameters);
    rt_model_release(model);

    rt_model* retained_model = NULL;
    require_status(
        rt_model_create(&config, &retained_model),
        "create retained decode-session model"
    );
    rt_decode_session* retained_session = NULL;
    require_status(
        rt_model_decode_session_create(
            retained_model,
            NULL,
            &retained_session
        ),
        "create retained decode session"
    );
    rt_model_release(retained_model);
    float retained_output[5];
    required = UINT64_MAX;
    require_status(
        rt_decode_session_step(
            retained_session,
            0,
            retained_output,
            5,
            &required
        ),
        "decode session survives model-handle release"
    );
    require_condition(
        required == 5,
        "retained decode session reports vocabulary size"
    );
    rt_decode_session_release(retained_session);
    rt_decode_session_release(NULL);
}

static void test_parameter_state_api(void) {
    rt_transformer_config config;
    require_status(
        rt_transformer_config_init(
            &config,
            (uint64_t)sizeof(config)
        ),
        "initialize parameter-state model config"
    );
    config.vocabulary_size = 5;
    config.maximum_context = 4;
    config.model_width = 4;
    config.head_count = 2;
    config.block_count = 1;
    config.feed_forward_width = 8;
    config.random_seed = 911;

    rt_model* model = NULL;
    require_status(
        rt_model_create(&config, &model),
        "create parameter-state model"
    );
    rt_parameter_list* parameters = NULL;
    require_status(
        rt_model_parameters(model, &parameters),
        "create parameter-state list"
    );

    uint64_t parameter_count = 0;
    require_status(
        rt_parameter_list_count(parameters, &parameter_count),
        "query parameter-state tensor count"
    );
    require_condition(
        parameter_count == 22,
        "parameter-state tensor count"
    );

    uint64_t rank = UINT64_MAX;
    require_status(
        rt_parameter_list_rank(parameters, 0, &rank),
        "query first parameter rank"
    );
    require_condition(rank == 2, "first parameter rank");
    require_condition(
        rt_parameter_list_rank(parameters, 0, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "parameter rank requires an output"
    );
    require_condition(
        rt_parameter_list_rank(
            parameters,
            parameter_count,
            &rank
        ) == RT_STATUS_OUT_OF_RANGE,
        "parameter rank rejects an out-of-range index"
    );

    const uint64_t shape_canary =
        UINT64_C(0x123456789abcdef0);
    uint64_t shape[3] = {
        shape_canary,
        shape_canary,
        shape_canary,
    };
    require_status(
        rt_parameter_list_shape(parameters, 0, shape, 3),
        "copy first parameter shape"
    );
    require_condition(
        shape[0] == 5 &&
            shape[1] == 4 &&
            shape[2] == shape_canary,
        "first parameter shape and trailing canary"
    );

    shape[0] = shape_canary;
    shape[1] = shape_canary;
    require_condition(
        rt_parameter_list_shape(parameters, 0, shape, 1) ==
            RT_STATUS_INVALID_ARGUMENT,
        "parameter shape rejects insufficient capacity"
    );
    require_condition(
        shape[0] == shape_canary &&
            shape[1] == shape_canary,
        "failed parameter shape copy preserves output"
    );
    require_condition(
        rt_parameter_list_shape(parameters, 0, NULL, 0) ==
            RT_STATUS_INVALID_ARGUMENT,
        "non-scalar parameter shape rejects zero capacity"
    );

    uint64_t first_numel = UINT64_MAX;
    require_status(
        rt_parameter_list_numel(parameters, 0, &first_numel),
        "query first parameter element count"
    );
    require_condition(
        first_numel == 20,
        "first parameter element count"
    );
    require_condition(
        rt_parameter_list_numel(parameters, 0, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "parameter numel requires an output"
    );

    uint64_t summed_numel = 0;
    for (uint64_t index = 0;
         index < parameter_count;
         ++index) {
        uint64_t parameter_rank = UINT64_MAX;
        uint64_t parameter_shape[4] = {0, 0, 0, 0};
        uint64_t parameter_numel = UINT64_MAX;
        require_status(
            rt_parameter_list_rank(
                parameters,
                index,
                &parameter_rank
            ),
            "query indexed parameter rank"
        );
        require_condition(
            parameter_rank <= 4,
            "test model parameter rank fits test storage"
        );
        require_status(
            rt_parameter_list_shape(
                parameters,
                index,
                parameter_shape,
                4
            ),
            "copy indexed parameter shape"
        );
        require_status(
            rt_parameter_list_numel(
                parameters,
                index,
                &parameter_numel
            ),
            "query indexed parameter numel"
        );
        uint64_t shape_numel = 1;
        for (uint64_t dimension = 0;
             dimension < parameter_rank;
             ++dimension) {
            shape_numel *= parameter_shape[dimension];
        }
        require_condition(
            shape_numel == parameter_numel,
            "indexed parameter shape matches numel"
        );
        summed_numel += parameter_numel;
    }

    uint64_t total_numel = UINT64_MAX;
    require_status(
        rt_parameter_list_total_numel(
            parameters,
            &total_numel
        ),
        "query flattened parameter element count"
    );
    require_condition(
        total_numel == 241 &&
            total_numel == summed_numel,
        "flattened parameter element count"
    );
    require_condition(
        rt_parameter_list_total_numel(parameters, NULL) ==
            RT_STATUS_INVALID_ARGUMENT,
        "total parameter numel requires an output"
    );

    const size_t total = (size_t)total_numel;
    float* original =
        (float*)malloc((total + 1) * sizeof(float));
    float* candidate =
        (float*)malloc((total + 1) * sizeof(float));
    float* observed =
        (float*)malloc((total + 1) * sizeof(float));
    float* short_output =
        (float*)malloc(total * sizeof(float));
    require_condition(
        original != NULL &&
            candidate != NULL &&
            observed != NULL &&
            short_output != NULL,
        "allocate flattened parameter buffers"
    );

    const float value_canary = 12345.5F;
    original[total] = value_canary;
    require_status(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            original,
            total_numel + 1
        ),
        "copy flattened parameter snapshot"
    );
    require_condition(
        original[total] == value_canary,
        "flattened parameter copy preserves trailing canary"
    );
    for (size_t index = 0; index < total; ++index) {
        require_condition(
            isfinite(original[index]),
            "initial parameter snapshot is finite"
        );
        short_output[index] = -9876.5F;
        candidate[index] =
            ((float)((int)(index % 31) - 15)) / 64.0F;
    }
    candidate[total] = -0.125F;

    float zero_capacity_canary = value_canary;
    require_condition(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            &zero_capacity_canary,
            0
        ) == RT_STATUS_INVALID_ARGUMENT,
        "flattened copy rejects nonnull zero capacity"
    );
    require_condition(
        zero_capacity_canary == value_canary,
        "zero-capacity flattened copy preserves output"
    );
    require_condition(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            NULL,
            0
        ) == RT_STATUS_INVALID_ARGUMENT,
        "flattened copy rejects null zero capacity"
    );
    require_condition(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            short_output,
            total_numel - 1
        ) == RT_STATUS_INVALID_ARGUMENT,
        "flattened copy rejects insufficient capacity"
    );
    require_condition(
        short_output[0] == -9876.5F &&
            short_output[total - 1] == -9876.5F,
        "short flattened copy preserves output"
    );
    require_condition(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            NULL,
            total_numel
        ) == RT_STATUS_INVALID_ARGUMENT,
        "flattened copy rejects null output"
    );

    uint64_t epoch_before = UINT64_MAX;
    uint64_t epoch_after = UINT64_MAX;
    require_status(
        rt_test_model_parameter_epoch(model, &epoch_before),
        "query epoch before parameter load"
    );
    require_status(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel
        ),
        "load flattened parameter values"
    );
    require_status(
        rt_test_model_parameter_epoch(model, &epoch_after),
        "query epoch after parameter load"
    );
    require_condition(
        epoch_after == epoch_before + 1,
        "successful parameter load increments epoch once"
    );

    observed[total] = value_canary;
    require_status(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            observed,
            total_numel + 1
        ),
        "copy loaded parameter values"
    );
    require_condition(
        memcmp(
            observed,
            candidate,
            total * sizeof(float)
        ) == 0 &&
            observed[total] == value_canary,
        "loaded parameters round trip in list order"
    );

    rt_backend backend = (rt_backend)-1;
    require_status(
        rt_parameter_list_backend(parameters, &backend),
        "query backend after parameter load"
    );
    require_condition(
        backend == RT_BACKEND_CPU,
        "parameter load preserves backend"
    );
    shape[0] = 0;
    shape[1] = 0;
    require_status(
        rt_parameter_list_shape(parameters, 0, shape, 2),
        "query shape after parameter load"
    );
    require_condition(
        shape[0] == 5 && shape[1] == 4,
        "parameter load preserves shapes"
    );
    char first_name[64];
    uint64_t name_capacity = 0;
    require_status(
        rt_parameter_list_name(
            parameters,
            0,
            first_name,
            (uint64_t)sizeof(first_name),
            &name_capacity
        ),
        "query name after parameter load"
    );
    require_condition(
        strcmp(first_name, "token_embedding.weight") == 0,
        "parameter load preserves names"
    );

    require_status(
        rt_parameter_list_load_from_host_f32(
            parameters,
            original,
            total_numel
        ),
        "restore original parameter snapshot"
    );
    require_status(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            observed,
            total_numel
        ),
        "copy restored parameter snapshot"
    );
    require_condition(
        memcmp(
            observed,
            original,
            total * sizeof(float)
        ) == 0,
        "original parameter snapshot round trip"
    );

    require_status(
        rt_test_model_parameter_epoch(model, &epoch_before),
        "query epoch before rejected loads"
    );
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel - 1
        ) == RT_STATUS_INVALID_ARGUMENT,
        "parameter load rejects a short value count"
    );
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel + 1
        ) == RT_STATUS_INVALID_ARGUMENT,
        "parameter load rejects a long value count"
    );
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            NULL,
            total_numel
        ) == RT_STATUS_INVALID_ARGUMENT,
        "parameter load rejects null values"
    );

    const float saved_last_candidate = candidate[total - 1];
    candidate[total - 1] = NAN;
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel
        ) == RT_STATUS_INVALID_ARGUMENT,
        "parameter load rejects NaN"
    );
    candidate[total - 1] = INFINITY;
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel
        ) == RT_STATUS_INVALID_ARGUMENT,
        "parameter load rejects infinity"
    );
    candidate[total - 1] = saved_last_candidate;

    require_status(
        rt_test_model_parameter_epoch(model, &epoch_after),
        "query epoch after rejected loads"
    );
    require_condition(
        epoch_after == epoch_before,
        "rejected loads preserve the parameter epoch"
    );
    require_status(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            observed,
            total_numel
        ),
        "copy parameters after rejected loads"
    );
    require_condition(
        memcmp(
            observed,
            original,
            total * sizeof(float)
        ) == 0,
        "rejected loads are transactional"
    );

    const uint32_t token_ids[] = {0, 1};
    const uint32_t targets[] = {1, 2};
    rt_variable* logits = NULL;
    rt_variable* loss = NULL;
    require_status(
        rt_model_forward(
            model,
            token_ids,
            2,
            1,
            2,
            &logits
        ),
        "create graph before parameter load"
    );
    require_status(
        rt_cross_entropy(logits, targets, 2, &loss),
        "create loss before parameter load"
    );
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel
        ) == RT_STATUS_INVALID_ARGUMENT,
        "parameter load rejects a live variable graph"
    );
    require_condition(
        strstr(rt_last_error(), "variable graphs") != NULL,
        "live-graph rejection records a diagnostic"
    );
    require_status(
        rt_variable_backward(loss),
        "populate gradients before parameter load"
    );
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel
        ) == RT_STATUS_INVALID_ARGUMENT,
        "parameter load rejects a consumed graph still held alive"
    );
    rt_variable_release(loss);
    rt_variable_release(logits);

    rt_adam* adam = NULL;
    require_status(
        rt_adam_create(parameters, NULL, &adam),
        "create optimizer before parameter load"
    );
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel
        ) == RT_STATUS_INVALID_ARGUMENT,
        "parameter load rejects a live optimizer"
    );
    require_condition(
        strstr(rt_last_error(), "optimizers") != NULL,
        "live-optimizer rejection records a diagnostic"
    );
    rt_adam_release(adam);

    require_status(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            observed,
            total_numel
        ),
        "copy parameters after graph and optimizer guards"
    );
    require_condition(
        memcmp(
            observed,
            original,
            total * sizeof(float)
        ) == 0,
        "graph and optimizer guards preserve parameters"
    );

    require_status(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel
        ),
        "load parameters after graph and optimizer release"
    );

    adam = NULL;
    require_status(
        rt_adam_create(parameters, NULL, &adam),
        "create optimizer after parameter load"
    );
    rt_adam_step_stats stats = {
        (uint64_t)sizeof(rt_adam_step_stats),
        0,
        -1.0,
        -1.0,
    };
    require_status(
        rt_adam_step(adam, &stats),
        "step optimizer after parameter load"
    );
    require_condition(
        stats.gradient_norm == 0.0,
        "parameter load clears every gradient"
    );
    require_status(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            observed,
            total_numel
        ),
        "copy values after zero-gradient Adam step"
    );
    require_condition(
        memcmp(
            observed,
            candidate,
            total * sizeof(float)
        ) == 0,
        "zero-gradient Adam step preserves loaded values"
    );
    rt_adam_release(adam);

    require_status(
        rt_parameter_list_load_from_host_f32(
            parameters,
            original,
            total_numel
        ),
        "restore snapshot before epoch-overflow test"
    );
    require_status(
        rt_test_model_set_parameter_epoch(model, UINT64_MAX),
        "force maximum parameter epoch"
    );
    require_condition(
        rt_parameter_list_load_from_host_f32(
            parameters,
            candidate,
            total_numel
        ) == RT_STATUS_OVERFLOW,
        "parameter load rejects epoch overflow"
    );
    require_status(
        rt_parameter_list_copy_to_host_f32(
            parameters,
            observed,
            total_numel
        ),
        "copy parameters after epoch-overflow rejection"
    );
    require_condition(
        memcmp(
            observed,
            original,
            total * sizeof(float)
        ) == 0,
        "epoch-overflow rejection preserves parameters"
    );

    free(short_output);
    free(observed);
    free(candidate);
    free(original);
    rt_parameter_list_release(parameters);
    rt_model_release(model);
}

static void test_cuda_backend(
    int32_t* cuda_available,
    rt_tensor** retained_cuda
) {
    require_condition(
        RT_BACKEND_CUDA == (rt_backend)2,
        "CUDA backend numeric identity"
    );
    require_status(
        rt_backend_is_available(
            RT_BACKEND_CUDA,
            cuda_available
        ),
        "query CUDA availability"
    );

    rt_context* cuda_context =
        (rt_context*)(uintptr_t)1;
    const rt_status context_status = rt_context_create(
        RT_BACKEND_CUDA,
        &cuda_context
    );
    if (*cuda_available == 0) {
        require_condition(
            context_status == RT_STATUS_BACKEND_UNAVAILABLE,
            "unavailable CUDA context status"
        );
        require_condition(
            cuda_context == NULL,
            "failed CUDA creation clears output handle"
        );
        require_condition(
            strstr(rt_last_error(), "cuda") != NULL,
            "unavailable CUDA diagnostic"
        );
        return;
    }

    require_status(context_status, "create available CUDA context");
    require_condition(
        cuda_context != NULL,
        "available CUDA context handle"
    );
    rt_backend actual_backend = RT_BACKEND_CPU;
    require_status(
        rt_context_backend(cuda_context, &actual_backend),
        "query CUDA context backend"
    );
    require_condition(
        actual_backend == RT_BACKEND_CUDA,
        "CUDA context backend"
    );

    const uint64_t shape[] = {2, 2};
    const float left_values[] = {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const float right_values[] = {
        5.0F, 6.0F,
        7.0F, 8.0F,
    };
    rt_tensor* left = NULL;
    rt_tensor* right = NULL;
    require_status(
        rt_tensor_create_f32(
            cuda_context,
            shape,
            2,
            left_values,
            4,
            &left
        ),
        "create CUDA left tensor"
    );
    require_status(
        rt_tensor_create_f32(
            cuda_context,
            shape,
            2,
            right_values,
            4,
            &right
        ),
        "create CUDA right tensor"
    );
    require_tensor_backend(
        left,
        RT_BACKEND_CUDA,
        "CUDA tensor intrinsic backend"
    );

    // Tensor storage retains the CUDA allocation after its construction
    // context is released.
    rt_context_release(cuda_context);
    cuda_context = NULL;
    require_status(
        rt_tensor_matmul(left, right, retained_cuda),
        "CUDA matmul after context release"
    );
    require_tensor_backend(
        *retained_cuda,
        RT_BACKEND_CUDA,
        "CUDA matmul output backend"
    );
    float product_values[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    require_status(
        rt_tensor_copy_to_host_f32(
            *retained_cuda,
            product_values,
            4
        ),
        "copy retained CUDA product"
    );
    const float expected[] = {
        19.0F, 22.0F,
        43.0F, 50.0F,
    };
    for (size_t index = 0; index < 4; ++index) {
        require_close(
            product_values[index],
            expected[index],
            "retained CUDA product value"
        );
    }
    rt_tensor_release(right);
    rt_tensor_release(left);
}

static void test_tpu_backend(int32_t* tpu_available) {
    require_condition(
        RT_BACKEND_TPU == (rt_backend)3,
        "TPU backend numeric identity"
    );
    require_status(
        rt_backend_is_available(RT_BACKEND_TPU, tpu_available),
        "query TPU availability"
    );

    rt_context* tpu_context = (rt_context*)(uintptr_t)1;
    const rt_status context_status = rt_context_create(
        RT_BACKEND_TPU,
        &tpu_context
    );
    if (*tpu_available == 0) {
        require_condition(
            context_status == RT_STATUS_BACKEND_UNAVAILABLE,
            "unavailable TPU context status"
        );
        require_condition(
            tpu_context == NULL,
            "failed TPU creation clears output handle"
        );
        require_condition(
            strstr(rt_last_error(), "tpu") != NULL,
            "unavailable TPU diagnostic"
        );
        return;
    }

    require_status(context_status, "create available TPU context");
    require_condition(tpu_context != NULL, "available TPU context handle");
    rt_backend actual_backend = RT_BACKEND_CPU;
    require_status(
        rt_context_backend(tpu_context, &actual_backend),
        "query TPU context backend"
    );
    require_condition(
        actual_backend == RT_BACKEND_TPU,
        "TPU context backend"
    );
    rt_context_release(tpu_context);
}

static void run_model_training(rt_backend backend) {
    rt_transformer_config config;
    require_status(
        rt_transformer_config_init(
            &config,
            (uint64_t)sizeof(config)
        ),
        "initialize transformer config"
    );
    require_condition(
        config.struct_size == (uint64_t)sizeof(config),
        "transformer config size"
    );
    config.vocabulary_size = 5;
    config.maximum_context = 4;
    config.model_width = 4;
    config.head_count = 2;
    config.block_count = 1;
    config.feed_forward_width = 8;
    config.random_seed = 137;

    rt_model* model = NULL;
    require_status(
        rt_model_create(&config, &model),
        "create decoder model"
    );
    require_condition(model != NULL, "decoder model handle");
    if (backend != RT_BACKEND_CPU) {
        require_status(
            rt_model_to(model, backend),
            "move decoder model"
        );
    }

    rt_backend actual_backend = (rt_backend)-1;
    require_status(
        rt_model_backend(model, &actual_backend),
        "query decoder model backend"
    );
    require_condition(
        actual_backend == backend,
        "decoder model backend"
    );

    rt_parameter_list* parameters = NULL;
    require_status(
        rt_model_parameters(model, &parameters),
        "query decoder parameters"
    );
    uint64_t parameter_count = 0;
    require_status(
        rt_parameter_list_count(parameters, &parameter_count),
        "query decoder parameter count"
    );
    require_condition(
        parameter_count == 22,
        "one-block decoder parameter count"
    );
    require_status(
        rt_parameter_list_backend(
            parameters,
            &actual_backend
        ),
        "query parameter backend"
    );
    require_condition(
        actual_backend == backend,
        "parameter-list backend"
    );

    uint64_t name_capacity = 0;
    require_status(
        rt_parameter_list_name(
            parameters,
            0,
            NULL,
            0,
            &name_capacity
        ),
        "query first parameter name size"
    );
    require_condition(
        name_capacity > 1,
        "first parameter name capacity"
    );
    char* first_name = (char*)malloc((size_t)name_capacity);
    require_condition(
        first_name != NULL,
        "allocate first parameter name"
    );
    require_status(
        rt_parameter_list_name(
            parameters,
            0,
            first_name,
            name_capacity,
            &name_capacity
        ),
        "copy first parameter name"
    );
    require_condition(
        strcmp(first_name, "token_embedding.weight") == 0,
        "first parameter name"
    );
    free(first_name);

    rt_adam_options options;
    require_status(
        rt_adam_options_init(
            &options,
            (uint64_t)sizeof(options)
        ),
        "initialize Adam options"
    );
    require_condition(
        options.struct_size == (uint64_t)sizeof(options) &&
            options.reserved == 0,
        "Adam options layout"
    );
    options.learning_rate = 1.0e-2F;

    rt_adam* adam = NULL;
    require_status(
        rt_adam_create(parameters, &options, &adam),
        "create Adam optimizer"
    );
    require_status(
        rt_adam_backend(adam, &actual_backend),
        "query Adam backend"
    );
    require_condition(
        actual_backend == backend,
        "Adam backend"
    );
    require_status(
        rt_adam_parameter_count(adam, &parameter_count),
        "query Adam parameter count"
    );
    require_condition(
        parameter_count == 22,
        "Adam parameter count"
    );

    const uint32_t token_ids[] = {0, 1};
    rt_variable* logits = NULL;
    require_status(
        rt_model_forward(
            model,
            token_ids,
            2,
            1,
            2,
            &logits
        ),
        "decoder forward"
    );
    require_status(
        rt_variable_backend(logits, &actual_backend),
        "query logits backend"
    );
    require_condition(
        actual_backend == backend,
        "logits backend"
    );
    uint64_t logits_shape[3] = {0, 0, 0};
    require_status(
        rt_variable_shape(logits, logits_shape, 3),
        "query logits shape"
    );
    require_condition(
        logits_shape[0] == 1 &&
            logits_shape[1] == 2 &&
            logits_shape[2] == 5,
        "logits shape"
    );

    const uint32_t targets[] = {1, 2};
    rt_variable* loss = NULL;
    require_status(
        rt_cross_entropy(logits, targets, 2, &loss),
        "cross entropy"
    );
    uint64_t loss_rank = 99;
    require_status(
        rt_variable_rank(loss, &loss_rank),
        "query loss rank"
    );
    require_condition(loss_rank == 0, "loss is scalar");
    float loss_value = 0.0F;
    require_status(
        rt_variable_copy_to_host_f32(loss, &loss_value, 1),
        "copy loss value"
    );
    require_condition(
        isfinite(loss_value) && loss_value > 0.0F,
        "finite positive loss"
    );

    require_condition(
        rt_model_to(model, backend) ==
            RT_STATUS_INVALID_ARGUMENT,
        "model transfer rejects live graph and optimizer"
    );

    // Every derived handle owns the shared model state independently.
    rt_variable_release(logits);
    rt_parameter_list_release(parameters);
    rt_model_release(model);

    require_status(
        rt_variable_backward(loss),
        "loss backward after parent-handle release"
    );
    rt_adam_step_stats stats = {
        (uint64_t)sizeof(rt_adam_step_stats),
        0,
        0.0,
        0.0,
    };
    require_status(
        rt_adam_step(adam, &stats),
        "Adam step after model release"
    );
    require_condition(stats.step == 1, "Adam first step");
    require_condition(
        isfinite(stats.gradient_norm) &&
            stats.gradient_norm >= 0.0,
        "finite Adam gradient norm"
    );
    require_condition(
        isfinite(stats.clip_scale) &&
            stats.clip_scale > 0.0 &&
            stats.clip_scale <= 1.0,
        "valid Adam clip scale"
    );
    uint64_t step_count = 0;
    require_status(
        rt_adam_step_count(adam, &step_count),
        "query Adam step count"
    );
    require_condition(step_count == 1, "Adam step count");
    require_status(
        rt_adam_zero_gradients(adam),
        "zero Adam gradients"
    );

    require_condition(
        rt_variable_backward(loss) == RT_STATUS_RUNTIME_ERROR,
        "backward graph is consumed exactly once"
    );

    rt_variable_release(loss);
    rt_adam_release(adam);
}

int main(void) {
    require_condition(
        rt_abi_version() == RT_ABI_VERSION,
        "ABI version"
    );
    require_condition(
        RT_ABI_VERSION_MAJOR == UINT32_C(2) &&
            RT_ABI_VERSION_MINOR == UINT32_C(4),
        "current additive ABI version"
    );

    test_thread_local_errors();
    test_tokenizer_api();
    test_tokenizer_options_and_bpe_api();
    test_versioned_structures();
    test_full_sequence_attention_api();
    test_activation_checkpointing_api();
    test_lora_api();
    test_qlora_api();
    test_decode_session_api();
    test_parameter_state_api();

    int32_t cpu_available = 0;
    require_status(
        rt_backend_is_available(
            RT_BACKEND_CPU,
            &cpu_available
        ),
        "query CPU availability"
    );
    require_condition(cpu_available == 1, "CPU must be available");

    int32_t metal_available = 0;
    require_status(
        rt_backend_is_available(
            RT_BACKEND_METAL,
            &metal_available
        ),
        "query Metal availability"
    );

    rt_tensor* retained_metal = NULL;
    rt_context* metal_context =
        (rt_context*)(uintptr_t)1;
    const rt_status metal_context_status =
        rt_context_create(
            RT_BACKEND_METAL,
            &metal_context
        );
    if (metal_available != 0) {
        require_status(
            metal_context_status,
            "create available Metal context"
        );
        require_condition(
            metal_context != NULL,
            "available Metal context handle"
        );
        rt_backend metal_context_backend = RT_BACKEND_CPU;
        require_status(
            rt_context_backend(
                metal_context,
                &metal_context_backend
            ),
            "query Metal context backend"
        );
        require_condition(
            metal_context_backend == RT_BACKEND_METAL,
            "Metal context backend"
        );

        const uint64_t metal_shape[] = {2, 2};
        const float metal_left_values[] = {
            1.0F, 2.0F,
            3.0F, 4.0F,
        };
        const float metal_right_values[] = {
            5.0F, 6.0F,
            7.0F, 8.0F,
        };
        rt_tensor* metal_left = NULL;
        rt_tensor* metal_right = NULL;
        require_status(
            rt_tensor_create_f32(
                metal_context,
                metal_shape,
                2,
                metal_left_values,
                4,
                &metal_left
            ),
            "create Metal left tensor"
        );
        require_status(
            rt_tensor_create_f32(
                metal_context,
                metal_shape,
                2,
                metal_right_values,
                4,
                &metal_right
            ),
            "create Metal right tensor"
        );
        require_tensor_backend(
            metal_left,
            RT_BACKEND_METAL,
            "Metal tensor intrinsic backend"
        );

        // The tensor storage owns its backend resources. The context only
        // selects the backend used at construction time.
        rt_context_release(metal_context);
        metal_context = NULL;

        require_status(
            rt_tensor_matmul(
                metal_left,
                metal_right,
                &retained_metal
            ),
            "Metal matmul after context release"
        );
        require_tensor_backend(
            retained_metal,
            RT_BACKEND_METAL,
            "Metal matmul output backend"
        );
        float metal_product_values[4] = {
            0.0F, 0.0F, 0.0F, 0.0F,
        };
        require_status(
            rt_tensor_copy_to_host_f32(
                retained_metal,
                metal_product_values,
                4
            ),
            "copy retained Metal product"
        );
        const float expected_metal[] = {
            19.0F, 22.0F,
            43.0F, 50.0F,
        };
        for (size_t index = 0; index < 4; ++index) {
            require_close(
                metal_product_values[index],
                expected_metal[index],
                "retained Metal product value"
            );
        }
        rt_tensor_release(metal_right);
        rt_tensor_release(metal_left);
    } else {
        require_condition(
            metal_context_status ==
                RT_STATUS_BACKEND_UNAVAILABLE,
            "unavailable Metal context status"
        );
        require_condition(
            metal_context == NULL,
            "failed Metal creation clears output handle"
        );
    }

    int32_t cuda_available = 0;
    rt_tensor* retained_cuda = NULL;
    test_cuda_backend(&cuda_available, &retained_cuda);
#if RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA
    require_condition(
        cuda_available != 0,
        "CUDA is required for this C ABI test"
    );
#endif

    int32_t tpu_available = 0;
    test_tpu_backend(&tpu_available);
#if RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU
    require_condition(
        tpu_available != 0,
        "TPU is required for this C ABI test"
    );
#endif

    rt_context* context = NULL;
    require_status(
        rt_context_create(RT_BACKEND_CPU, &context),
        "create CPU context"
    );
    require_condition(context != NULL, "context handle");

    rt_backend context_backend = RT_BACKEND_METAL;
    require_status(
        rt_context_backend(context, &context_backend),
        "query context backend"
    );
    require_condition(
        context_backend == RT_BACKEND_CPU,
        "CPU context backend"
    );

    const uint64_t left_shape[] = {2, 3};
    float left_values[] = {
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
    };
    rt_tensor* left = NULL;
    require_status(
        rt_tensor_create_f32(
            context,
            left_shape,
            2,
            left_values,
            6,
            &left
        ),
        "create left tensor"
    );
    require_tensor_backend(
        left,
        RT_BACKEND_CPU,
        "left tensor intrinsic backend"
    );
    left_values[0] = 999.0F;

    const uint64_t right_shape[] = {3, 2};
    const float right_values[] = {
        7.0F, 8.0F,
        9.0F, 10.0F,
        11.0F, 12.0F,
    };
    rt_tensor* right = NULL;
    require_status(
        rt_tensor_create_f32(
            context,
            right_shape,
            2,
            right_values,
            6,
            &right
        ),
        "create right tensor"
    );
    require_tensor_backend(
        right,
        RT_BACKEND_CPU,
        "right tensor intrinsic backend"
    );

    const uint64_t zero_shape[] = {2, 2};
    rt_tensor* zeros = NULL;
    require_status(
        rt_tensor_zeros_f32(
            context,
            zero_shape,
            2,
            &zeros
        ),
        "create zero tensor"
    );
    require_tensor_backend(
        zeros,
        RT_BACKEND_CPU,
        "zero tensor intrinsic backend"
    );
    float zero_values[4] = {
        1.0F, 1.0F, 1.0F, 1.0F,
    };
    require_status(
        rt_tensor_copy_to_host_f32(
            zeros,
            zero_values,
            4
        ),
        "copy zero tensor"
    );
    for (size_t index = 0; index < 4; ++index) {
        require_close(zero_values[index], 0.0F, "zero value");
    }

    const float scalar_value = 3.5F;
    rt_tensor* scalar = NULL;
    require_status(
        rt_tensor_create_f32(
            context,
            NULL,
            0,
            &scalar_value,
            1,
            &scalar
        ),
        "create scalar"
    );
    require_tensor_backend(
        scalar,
        RT_BACKEND_CPU,
        "scalar tensor intrinsic backend"
    );
    uint64_t scalar_rank = 99;
    require_status(
        rt_tensor_rank(scalar, &scalar_rank),
        "query scalar rank"
    );
    require_condition(scalar_rank == 0, "scalar rank");
    require_status(
        rt_tensor_shape(scalar, NULL, 0),
        "copy scalar shape"
    );

    // Tensor storage owns its backend identity and remains usable after the
    // construction context is gone.
    rt_context_release(context);
    context = NULL;

    rt_tensor* product = NULL;
    require_status(
        rt_tensor_matmul(left, right, &product),
        "CPU matmul"
    );
    require_condition(product != NULL, "matmul output handle");
    require_tensor_backend(
        product,
        RT_BACKEND_CPU,
        "CPU matmul output intrinsic backend"
    );

    uint64_t rank = 0;
    require_status(
        rt_tensor_rank(product, &rank),
        "query product rank"
    );
    require_condition(rank == 2, "product rank");

    uint64_t product_shape[2] = {0, 0};
    require_status(
        rt_tensor_shape(product, product_shape, 2),
        "copy product shape"
    );
    require_condition(
        product_shape[0] == 2 && product_shape[1] == 2,
        "product shape"
    );

    uint64_t product_numel = 0;
    require_status(
        rt_tensor_numel(product, &product_numel),
        "query product element count"
    );
    require_condition(product_numel == 4, "product element count");

    float product_values[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    require_status(
        rt_tensor_copy_to_host_f32(
            product,
            product_values,
            4
        ),
        "copy product values"
    );
    const float expected[] = {
        58.0F, 64.0F,
        139.0F, 154.0F,
    };
    for (size_t index = 0; index < 4; ++index) {
        require_close(
            product_values[index],
            expected[index],
            "product value"
        );
    }

    float copied_left[6] = {0.0F};
    require_status(
        rt_tensor_copy_to_host_f32(left, copied_left, 6),
        "copy owned input"
    );
    require_close(
        copied_left[0],
        1.0F,
        "tensor creation copies host input"
    );

    if (retained_metal != NULL) {
        rt_tensor* mixed_product =
            (rt_tensor*)(uintptr_t)1;
        const rt_status mixed_status =
            rt_tensor_matmul(
                left,
                retained_metal,
                &mixed_product
            );
        require_condition(
            mixed_status == RT_STATUS_INVALID_ARGUMENT,
            "mixed-backend matmul status"
        );
        require_condition(
            mixed_product == NULL,
            "mixed-backend matmul clears output handle"
        );
    }
    if (retained_cuda != NULL) {
        rt_tensor* mixed_product =
            (rt_tensor*)(uintptr_t)1;
        const rt_status mixed_status =
            rt_tensor_matmul(
                left,
                retained_cuda,
                &mixed_product
            );
        require_condition(
            mixed_status == RT_STATUS_INVALID_ARGUMENT,
            "mixed CPU/CUDA matmul status"
        );
        require_condition(
            mixed_product == NULL,
            "mixed CPU/CUDA matmul clears output handle"
        );
    }

    rt_tensor* invalid =
        (rt_tensor*)(uintptr_t)1;
    const rt_status invalid_status =
        rt_tensor_create_f32(
            NULL,
            left_shape,
            2,
            left_values,
            6,
            &invalid
        );
    require_condition(
        invalid_status == RT_STATUS_INVALID_ARGUMENT,
        "null context status"
    );
    require_condition(
        invalid == NULL,
        "failed creation clears output handle"
    );
    require_condition(
        rt_last_error()[0] != '\0',
        "failed call records an error"
    );
    require_status(
        rt_backend_is_available(
            RT_BACKEND_CPU,
            &cpu_available
        ),
        "successful call after an error"
    );
    require_condition(
        rt_last_error()[0] == '\0',
        "successful status call clears the prior error"
    );

    rt_tensor_release(product);
    rt_tensor_release(retained_cuda);
    rt_tensor_release(retained_metal);
    rt_tensor_release(scalar);
    rt_tensor_release(zeros);
    rt_tensor_release(right);
    rt_tensor_release(left);
    rt_tensor_release(NULL);
    rt_context_release(NULL);

    run_model_training(RT_BACKEND_CPU);
    if (metal_available != 0) {
        run_model_training(RT_BACKEND_METAL);
    }
    if (cuda_available != 0) {
        run_model_training(RT_BACKEND_CUDA);
    }

    rt_transformer_config invalid_config;
    require_condition(
        rt_transformer_config_init(
            &invalid_config,
            (uint64_t)sizeof(invalid_config) - 1
        ) == RT_STATUS_INVALID_ARGUMENT,
        "config init rejects undersized caller storage"
    );
    rt_adam_options invalid_options;
    memset(&invalid_options, 0x5a, sizeof(invalid_options));
    const rt_adam_options original_invalid_options = invalid_options;
    require_condition(
        rt_adam_options_init(
            &invalid_options,
            RT_LEGACY_ADAM_OPTIONS_SIZE - UINT64_C(1)
        ) == RT_STATUS_INVALID_ARGUMENT,
        "Adam init rejects undersized caller storage"
    );
    require_condition(
        memcmp(
            &invalid_options,
            &original_invalid_options,
            sizeof(invalid_options)
        ) == 0,
        "undersized Adam initialization is atomic"
    );

    rt_model_release(NULL);
    rt_parameter_list_release(NULL);
    rt_variable_release(NULL);
    rt_adam_release(NULL);

    puts("C ABI tests passed");
    return EXIT_SUCCESS;
}
