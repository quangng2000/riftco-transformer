#ifndef TRANSFORMER_LAB_C_API_H
#define TRANSFORMER_LAB_C_API_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(TRANSFORMER_LAB_C_EXPORTS)
#define TL_API __declspec(dllexport)
#else
#define TL_API __declspec(dllimport)
#endif
#define TL_CALL __cdecl
#else
#define TL_API __attribute__((visibility("default")))
#define TL_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TL_ABI_VERSION_MAJOR UINT32_C(1)
#define TL_ABI_VERSION_MINOR UINT32_C(8)
#define TL_ABI_VERSION \
    ((TL_ABI_VERSION_MAJOR << 16) | TL_ABI_VERSION_MINOR)

typedef struct tl_context tl_context;
typedef struct tl_tensor tl_tensor;
typedef struct tl_tokenizer tl_tokenizer;
typedef struct tl_model tl_model;
typedef struct tl_decode_session tl_decode_session;
typedef struct tl_parameter_list tl_parameter_list;
typedef struct tl_variable tl_variable;
typedef struct tl_adam tl_adam;

// Fixed-width integer constants keep the ABI independent of C enum layout.
typedef int32_t tl_status;
#define TL_STATUS_OK ((tl_status)0)
#define TL_STATUS_INVALID_ARGUMENT ((tl_status)1)
#define TL_STATUS_OUT_OF_RANGE ((tl_status)2)
#define TL_STATUS_OVERFLOW ((tl_status)3)
#define TL_STATUS_BACKEND_UNAVAILABLE ((tl_status)4)
#define TL_STATUS_OUT_OF_MEMORY ((tl_status)5)
#define TL_STATUS_RUNTIME_ERROR ((tl_status)6)
#define TL_STATUS_UNKNOWN_ERROR ((tl_status)255)

typedef int32_t tl_backend;
#define TL_BACKEND_CPU ((tl_backend)0)
#define TL_BACKEND_METAL ((tl_backend)1)

typedef int32_t tl_full_sequence_attention_kind;
#define TL_FULL_SEQUENCE_ATTENTION_MATERIALIZED \
    ((tl_full_sequence_attention_kind)0)
#define TL_FULL_SEQUENCE_ATTENTION_FLASH \
    ((tl_full_sequence_attention_kind)1)

typedef int32_t tl_activation_checkpointing_kind;
#define TL_ACTIVATION_CHECKPOINTING_DISABLED \
    ((tl_activation_checkpointing_kind)0)
#define TL_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK \
    ((tl_activation_checkpointing_kind)1)

typedef int32_t tl_tokenizer_method;
#define TL_TOKENIZER_METHOD_BYTE ((tl_tokenizer_method)0)
#define TL_TOKENIZER_METHOD_BPE ((tl_tokenizer_method)1)

typedef int32_t tl_kv_cache_kind;
#define TL_KV_CACHE_CONTIGUOUS ((tl_kv_cache_kind)0)
#define TL_KV_CACHE_PAGED ((tl_kv_cache_kind)1)

typedef uint64_t tl_lora_target_mask;
#define TL_LORA_TARGET_ATTENTION_QUERY \
    ((tl_lora_target_mask)(UINT64_C(1) << 0))
#define TL_LORA_TARGET_ATTENTION_KEY \
    ((tl_lora_target_mask)(UINT64_C(1) << 1))
#define TL_LORA_TARGET_ATTENTION_VALUE \
    ((tl_lora_target_mask)(UINT64_C(1) << 2))
#define TL_LORA_TARGET_ATTENTION_OUTPUT \
    ((tl_lora_target_mask)(UINT64_C(1) << 3))
#define TL_LORA_TARGET_FF_EXPAND \
    ((tl_lora_target_mask)(UINT64_C(1) << 4))
#define TL_LORA_TARGET_FF_PROJECT \
    ((tl_lora_target_mask)(UINT64_C(1) << 5))
#define TL_LORA_TARGET_LM_HEAD \
    ((tl_lora_target_mask)(UINT64_C(1) << 6))
#define TL_LORA_TARGET_DEFAULT \
    (TL_LORA_TARGET_ATTENTION_QUERY | TL_LORA_TARGET_ATTENTION_VALUE)
#define TL_LORA_TARGET_ALL_LINEAR \
    ((tl_lora_target_mask)((UINT64_C(1) << 7) - UINT64_C(1)))

// Versioned value structures start with their byte size. Initialize config
// and option inputs with the corresponding tl_*_init function and
// sizeof(your_structure) before changing selected fields. Initialize an
// output structure's struct_size before passing it to the producing call.
typedef struct tl_tokenizer_options {
    uint64_t struct_size;
    tl_tokenizer_method method;
    uint32_t reserved;
    uint64_t vocabulary_size;
    uint64_t minimum_pair_frequency;
} tl_tokenizer_options;

typedef struct tl_bpe_merge_rule {
    uint32_t left;
    uint32_t right;
    uint32_t result;
} tl_bpe_merge_rule;

typedef struct tl_transformer_config {
    uint64_t struct_size;
    uint64_t vocabulary_size;
    uint64_t maximum_context;
    uint64_t model_width;
    uint64_t head_count;
    uint64_t block_count;
    uint64_t feed_forward_width;
    uint32_t random_seed;
    float layer_norm_epsilon;
} tl_transformer_config;

typedef struct tl_lora_config {
    uint64_t struct_size;
    uint64_t rank;
    float alpha;
    uint32_t random_seed;
    tl_lora_target_mask targets;
    uint64_t reserved;
} tl_lora_config;

typedef struct tl_decode_session_options {
    uint64_t struct_size;
    tl_kv_cache_kind kind;
    uint32_t reserved;
    uint64_t block_size;
} tl_decode_session_options;

typedef struct tl_adam_options {
    uint64_t struct_size;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float maximum_gradient_norm;
    uint32_t reserved;
} tl_adam_options;

typedef struct tl_adam_step_stats {
    uint64_t struct_size;
    uint64_t step;
    double gradient_norm;
    double clip_scale;
} tl_adam_step_stats;

// Raw handles do not provide lifetime synchronization. A caller must not
// release a handle while any call using that handle is in flight and must
// externally synchronize calls involving the same handle or handles derived
// from the same model. The Python wrapper provides this synchronization for
// its objects.
//
// Handles derived from a model retain its native state. Releasing tl_model
// does not invalidate an already-created decode session, parameter list,
// variable, or Adam optimizer. Each handle itself must still be released
// exactly once.
//
// A live decode session pins the model backend and parameter values. Model
// transfer, LoRA attachment/merge, parameter loading, and Adam steps are
// rejected until every decode session derived from that model is released.

TL_API uint32_t TL_CALL tl_abi_version(void);
TL_API const char* TL_CALL tl_status_string(tl_status status);

// The returned pointer is thread-local and remains valid until the next
// status-returning C API call on the same thread. Long details may be
// truncated to the implementation's fixed diagnostic-buffer capacity.
TL_API const char* TL_CALL tl_last_error(void);

// Initializes options for the legacy corpus-byte method. The defaults are a
// target vocabulary size of 512 and a minimum pair frequency of 2; these
// fields take effect when method is changed to TL_TOKENIZER_METHOD_BPE.
TL_API tl_status TL_CALL tl_tokenizer_options_init(
    tl_tokenizer_options* options,
    uint64_t options_size
);

// Tokenizers own an immutable vocabulary derived from a nonempty corpus.
// Null input data is valid only when its corresponding count is zero.
// tl_tokenizer_create preserves the ABI 1.2 corpus-byte behavior.
TL_API tl_status TL_CALL tl_tokenizer_create(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    tl_tokenizer** output
);

// Null options select the legacy corpus-byte method.
TL_API tl_status TL_CALL tl_tokenizer_create_with_options(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    const tl_tokenizer_options* options,
    tl_tokenizer** output
);

// Restores exact tokenizer state without access to its training corpus.
// The byte vocabulary is ordered by token ID and must be nonempty and unique.
TL_API tl_status TL_CALL tl_tokenizer_create_from_byte_vocabulary(
    const uint8_t* ordered_vocabulary,
    uint64_t vocabulary_size,
    tl_tokenizer** output
);

// A zero-rule BPE tokenizer is valid and contains the fixed 256 base bytes.
// Nonempty rules must use sequential result IDs beginning at 256.
TL_API tl_status TL_CALL tl_tokenizer_create_from_bpe_merges(
    const tl_bpe_merge_rule* ordered_merge_rules,
    uint64_t merge_count,
    tl_tokenizer** output
);

TL_API void TL_CALL tl_tokenizer_release(tl_tokenizer* tokenizer);

TL_API tl_status TL_CALL tl_tokenizer_get_method(
    const tl_tokenizer* tokenizer,
    tl_tokenizer_method* output
);

TL_API tl_status TL_CALL tl_tokenizer_vocabulary_size(
    const tl_tokenizer* tokenizer,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_tokenizer_bpe_merge_count(
    const tl_tokenizer* tokenizer,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_tokenizer_bpe_merge_rule(
    const tl_tokenizer* tokenizer,
    uint64_t index,
    tl_bpe_merge_rule* output
);

// Returns the ordered one-byte vocabulary of a legacy corpus-byte tokenizer.
// BPE tokenizers must use tl_tokenizer_token_bytes for individual pieces.
TL_API tl_status TL_CALL tl_tokenizer_vocabulary(
    const tl_tokenizer* tokenizer,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
);

// Variable-size output APIs validate the complete input and set
// required_count before checking output capacity. A null output with zero
// capacity is a size query. A null output with nonzero capacity is invalid.
// Insufficient capacity returns TL_STATUS_OUT_OF_RANGE without modifying the
// output buffer.
TL_API tl_status TL_CALL tl_tokenizer_token_bytes(
    const tl_tokenizer* tokenizer,
    uint32_t token_id,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
);

TL_API tl_status TL_CALL tl_tokenizer_encode(
    const tl_tokenizer* tokenizer,
    const uint8_t* text,
    uint64_t text_size,
    uint32_t* output,
    uint64_t capacity,
    uint64_t* required_count
);

TL_API tl_status TL_CALL tl_tokenizer_decode(
    const tl_tokenizer* tokenizer,
    const uint32_t* tokens,
    uint64_t token_count,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
);

TL_API tl_status TL_CALL tl_backend_is_available(
    tl_backend backend,
    int32_t* available
);

TL_API tl_status TL_CALL tl_context_create(
    tl_backend backend,
    tl_context** output
);

TL_API void TL_CALL tl_context_release(tl_context* context);

TL_API tl_status TL_CALL tl_context_backend(
    const tl_context* context,
    tl_backend* output
);

TL_API tl_status TL_CALL tl_tensor_create_f32(
    const tl_context* context,
    const uint64_t* shape,
    uint64_t rank,
    const float* values,
    uint64_t value_count,
    tl_tensor** output
);

TL_API tl_status TL_CALL tl_tensor_zeros_f32(
    const tl_context* context,
    const uint64_t* shape,
    uint64_t rank,
    tl_tensor** output
);

TL_API void TL_CALL tl_tensor_release(tl_tensor* tensor);

TL_API tl_status TL_CALL tl_tensor_backend(
    const tl_tensor* tensor,
    tl_backend* output
);

TL_API tl_status TL_CALL tl_tensor_rank(
    const tl_tensor* tensor,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_tensor_shape(
    const tl_tensor* tensor,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
);

TL_API tl_status TL_CALL tl_tensor_numel(
    const tl_tensor* tensor,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_tensor_copy_to_host_f32(
    const tl_tensor* tensor,
    float* output_values,
    uint64_t value_capacity
);

TL_API tl_status TL_CALL tl_tensor_matmul(
    const tl_tensor* left,
    const tl_tensor* right,
    tl_tensor** output
);

TL_API tl_status TL_CALL tl_transformer_config_init(
    tl_transformer_config* config,
    uint64_t config_size
);

// Initializes a paged cache with a block size of 16. A null options pointer
// passed to tl_model_decode_session_create selects the same defaults.
TL_API tl_status TL_CALL tl_decode_session_options_init(
    tl_decode_session_options* options,
    uint64_t options_size
);

// Models are initialized on CPU. Call tl_model_to before creating variables
// or an optimizer to select another available backend.
TL_API tl_status TL_CALL tl_model_create(
    const tl_transformer_config* config,
    tl_model** output
);

TL_API void TL_CALL tl_model_release(tl_model* model);

TL_API tl_status TL_CALL tl_model_to(
    tl_model* model,
    tl_backend backend
);

TL_API tl_status TL_CALL tl_model_backend(
    const tl_model* model,
    tl_backend* output
);

// Selects the implementation used by future full-sequence model forwards.
// Incremental decode sessions continue to use their configured KV-cache path.
TL_API tl_status TL_CALL tl_model_set_full_sequence_attention(
    tl_model* model,
    tl_full_sequence_attention_kind kind
);

TL_API tl_status TL_CALL tl_model_full_sequence_attention(
    const tl_model* model,
    tl_full_sequence_attention_kind* output
);

// Selects activation retention for future full-sequence training graphs.
// Incremental decode is stateful and is never checkpointed.
TL_API tl_status TL_CALL tl_model_set_activation_checkpointing(
    tl_model* model,
    tl_activation_checkpointing_kind kind
);

TL_API tl_status TL_CALL tl_model_activation_checkpointing(
    const tl_model* model,
    tl_activation_checkpointing_kind* output
);

// Initializes the native LoRA defaults. A null config passed to
// tl_model_attach_lora selects the same defaults.
TL_API tl_status TL_CALL tl_lora_config_init(
    tl_lora_config* config,
    uint64_t config_size
);

// A model accepts one LoRA attachment during its lifetime; a merged model
// cannot attach another. Attachment preserves the base parameter list and
// initially preserves model outputs.
TL_API tl_status TL_CALL tl_model_attach_lora(
    tl_model* model,
    const tl_lora_config* config
);

TL_API tl_status TL_CALL tl_model_has_lora(
    const tl_model* model,
    int32_t* output
);

// output must have struct_size initialized to sizeof(*output).
TL_API tl_status TL_CALL tl_model_lora_config(
    const tl_model* model,
    tl_lora_config* output
);

// Token shape is [batch_size, sequence_length]. token_count must equal their
// product. The returned variable has shape
// [batch_size, sequence_length, vocabulary_size].
TL_API tl_status TL_CALL tl_model_forward(
    const tl_model* model,
    const uint32_t* token_ids,
    uint64_t token_count,
    uint64_t batch_size,
    uint64_t sequence_length,
    tl_variable** output
);

// Creates a single-sequence incremental-decoding session. The session retains
// the model state and remains valid if the original model handle is released.
// Its cache capacity is the model's maximum context.
TL_API tl_status TL_CALL tl_model_decode_session_create(
    const tl_model* model,
    const tl_decode_session_options* options,
    tl_decode_session** output
);

TL_API void TL_CALL tl_decode_session_release(
    tl_decode_session* session
);

// Clears all cached tokens while preserving the allocated cache.
TL_API tl_status TL_CALL tl_decode_session_reset(
    tl_decode_session* session
);

TL_API tl_status TL_CALL tl_decode_session_size(
    const tl_decode_session* session,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_decode_session_capacity(
    const tl_decode_session* session,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_decode_session_cache_kind(
    const tl_decode_session* session,
    tl_kv_cache_kind* output
);

TL_API tl_status TL_CALL tl_decode_session_block_size(
    const tl_decode_session* session,
    uint64_t* output
);

// Appends one token at the cache's current position and returns the
// vocabulary-sized logits that predict the following token. required_count
// is set to the model vocabulary size. Unlike tokenizer size-query APIs,
// output_logits must be nonnull and capacity must be sufficient; these checks
// occur before the session is mutated.
TL_API tl_status TL_CALL tl_decode_session_step(
    tl_decode_session* session,
    uint32_t token_id,
    float* output_logits,
    uint64_t capacity,
    uint64_t* required_count
);

TL_API tl_status TL_CALL tl_model_parameters(
    tl_model* model,
    tl_parameter_list** output
);

// Returns only the attached adapter parameters. The base-model parameter
// count, ordering, and names exposed by tl_model_parameters are unchanged.
TL_API tl_status TL_CALL tl_model_lora_parameters(
    tl_model* model,
    tl_parameter_list** output
);

// Merging is transactional and invalidates the attached adapter. It is
// rejected while a variable graph, optimizer, decode session, or adapter
// parameter list is alive.
TL_API tl_status TL_CALL tl_model_merge_lora(tl_model* model);

TL_API void TL_CALL tl_parameter_list_release(
    tl_parameter_list* parameters
);

TL_API tl_status TL_CALL tl_parameter_list_count(
    const tl_parameter_list* parameters,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_parameter_list_backend(
    const tl_parameter_list* parameters,
    tl_backend* output
);

// required_capacity includes the trailing null byte. Passing a null
// output_name with zero capacity is a supported size query.
TL_API tl_status TL_CALL tl_parameter_list_name(
    const tl_parameter_list* parameters,
    uint64_t index,
    char* output_name,
    uint64_t name_capacity,
    uint64_t* required_capacity
);

TL_API tl_status TL_CALL tl_parameter_list_rank(
    const tl_parameter_list* parameters,
    uint64_t index,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_parameter_list_shape(
    const tl_parameter_list* parameters,
    uint64_t index,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
);

TL_API tl_status TL_CALL tl_parameter_list_numel(
    const tl_parameter_list* parameters,
    uint64_t index,
    uint64_t* output
);

// Returns the number of floats in the deterministic flattened parameter
// representation. Values are ordered by parameter-list index and then by
// each tensor's native flat order.
TL_API tl_status TL_CALL tl_parameter_list_total_numel(
    const tl_parameter_list* parameters,
    uint64_t* output
);

// value_capacity must be at least tl_parameter_list_total_numel(). Extra
// capacity is left untouched.
TL_API tl_status TL_CALL tl_parameter_list_copy_to_host_f32(
    const tl_parameter_list* parameters,
    float* output_values,
    uint64_t value_capacity
);

// Replaces every parameter from the deterministic flattened representation.
// value_count must exactly equal tl_parameter_list_total_numel(), and every
// value must be finite. Loading is transactional and preserves parameter
// names, shapes, and backends while clearing all parameter gradients. It is
// rejected while a variable graph or optimizer derived from the model lives.
TL_API tl_status TL_CALL tl_parameter_list_load_from_host_f32(
    tl_parameter_list* parameters,
    const float* values,
    uint64_t value_count
);

#if defined(TRANSFORMER_LAB_C_API_TESTING)
// Test-only epoch controls are not part of the installed stable ABI.
TL_API tl_status TL_CALL tl_test_model_parameter_epoch(
    const tl_model* model,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_test_model_set_parameter_epoch(
    tl_model* model,
    uint64_t value
);
#endif

TL_API void TL_CALL tl_variable_release(tl_variable* variable);

TL_API tl_status TL_CALL tl_variable_backend(
    const tl_variable* variable,
    tl_backend* output
);

TL_API tl_status TL_CALL tl_variable_rank(
    const tl_variable* variable,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_variable_shape(
    const tl_variable* variable,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
);

TL_API tl_status TL_CALL tl_variable_numel(
    const tl_variable* variable,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_variable_copy_to_host_f32(
    const tl_variable* variable,
    float* output_values,
    uint64_t value_capacity
);

// A successful backward consumes the shared computation graph. Adam steps
// invalidate every graph created before that parameter update.
TL_API tl_status TL_CALL tl_variable_backward(
    const tl_variable* variable
);

TL_API tl_status TL_CALL tl_cross_entropy(
    const tl_variable* logits,
    const uint32_t* targets,
    uint64_t target_count,
    tl_variable** output
);

TL_API tl_status TL_CALL tl_adam_options_init(
    tl_adam_options* options,
    uint64_t options_size
);

// A null options pointer selects the native Adam defaults.
TL_API tl_status TL_CALL tl_adam_create(
    const tl_parameter_list* parameters,
    const tl_adam_options* options,
    tl_adam** output
);

TL_API void TL_CALL tl_adam_release(tl_adam* adam);

TL_API tl_status TL_CALL tl_adam_backend(
    const tl_adam* adam,
    tl_backend* output
);

TL_API tl_status TL_CALL tl_adam_step_count(
    const tl_adam* adam,
    uint64_t* output
);

TL_API tl_status TL_CALL tl_adam_parameter_count(
    const tl_adam* adam,
    uint64_t* output
);

// output_stats must have struct_size initialized to sizeof(*output_stats).
TL_API tl_status TL_CALL tl_adam_step(
    tl_adam* adam,
    tl_adam_step_stats* output_stats
);

TL_API tl_status TL_CALL tl_adam_zero_gradients(
    const tl_adam* adam
);

#ifdef __cplusplus
}
#endif

#endif
