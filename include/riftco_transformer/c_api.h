#ifndef RIFTCO_TRANSFORMER_C_API_H
#define RIFTCO_TRANSFORMER_C_API_H

#include <stdint.h>

#if defined(_WIN32)
#if defined(RIFTCO_TRANSFORMER_C_EXPORTS)
#define RT_API __declspec(dllexport)
#else
#define RT_API __declspec(dllimport)
#endif
#define RT_CALL __cdecl
#else
#define RT_API __attribute__((visibility("default")))
#define RT_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RT_ABI_VERSION_MAJOR UINT32_C(2)
#define RT_ABI_VERSION_MINOR UINT32_C(4)
#define RT_ABI_VERSION \
    ((RT_ABI_VERSION_MAJOR << 16) | RT_ABI_VERSION_MINOR)

typedef struct rt_context rt_context;
typedef struct rt_tensor rt_tensor;
typedef struct rt_tokenizer rt_tokenizer;
typedef struct rt_model rt_model;
typedef struct rt_decode_session rt_decode_session;
typedef struct rt_parameter_list rt_parameter_list;
typedef struct rt_variable rt_variable;
typedef struct rt_adam rt_adam;

// Fixed-width integer constants keep the ABI independent of C enum layout.
typedef int32_t rt_status;
#define RT_STATUS_OK ((rt_status)0)
#define RT_STATUS_INVALID_ARGUMENT ((rt_status)1)
#define RT_STATUS_OUT_OF_RANGE ((rt_status)2)
#define RT_STATUS_OVERFLOW ((rt_status)3)
#define RT_STATUS_BACKEND_UNAVAILABLE ((rt_status)4)
#define RT_STATUS_OUT_OF_MEMORY ((rt_status)5)
#define RT_STATUS_RUNTIME_ERROR ((rt_status)6)
#define RT_STATUS_UNKNOWN_ERROR ((rt_status)255)

typedef int32_t rt_backend;
#define RT_BACKEND_CPU ((rt_backend)0)
#define RT_BACKEND_METAL ((rt_backend)1)
#define RT_BACKEND_CUDA ((rt_backend)2)
#define RT_BACKEND_TPU ((rt_backend)3)

typedef int32_t rt_full_sequence_attention_kind;
#define RT_FULL_SEQUENCE_ATTENTION_MATERIALIZED \
    ((rt_full_sequence_attention_kind)0)
#define RT_FULL_SEQUENCE_ATTENTION_FLASH \
    ((rt_full_sequence_attention_kind)1)

typedef int32_t rt_activation_checkpointing_kind;
#define RT_ACTIVATION_CHECKPOINTING_DISABLED \
    ((rt_activation_checkpointing_kind)0)
#define RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK \
    ((rt_activation_checkpointing_kind)1)

typedef int32_t rt_tokenizer_method;
#define RT_TOKENIZER_METHOD_BYTE ((rt_tokenizer_method)0)
#define RT_TOKENIZER_METHOD_BPE ((rt_tokenizer_method)1)

typedef int32_t rt_kv_cache_kind;
#define RT_KV_CACHE_CONTIGUOUS ((rt_kv_cache_kind)0)
#define RT_KV_CACHE_PAGED ((rt_kv_cache_kind)1)

typedef int32_t rt_adam_state_storage_kind;
#define RT_ADAM_STATE_CONTIGUOUS ((rt_adam_state_storage_kind)0)
#define RT_ADAM_STATE_PAGED ((rt_adam_state_storage_kind)1)

typedef uint64_t rt_lora_target_mask;
#define RT_LORA_TARGET_ATTENTION_QUERY \
    ((rt_lora_target_mask)(UINT64_C(1) << 0))
#define RT_LORA_TARGET_ATTENTION_KEY \
    ((rt_lora_target_mask)(UINT64_C(1) << 1))
#define RT_LORA_TARGET_ATTENTION_VALUE \
    ((rt_lora_target_mask)(UINT64_C(1) << 2))
#define RT_LORA_TARGET_ATTENTION_OUTPUT \
    ((rt_lora_target_mask)(UINT64_C(1) << 3))
#define RT_LORA_TARGET_FF_EXPAND \
    ((rt_lora_target_mask)(UINT64_C(1) << 4))
#define RT_LORA_TARGET_FF_PROJECT \
    ((rt_lora_target_mask)(UINT64_C(1) << 5))
#define RT_LORA_TARGET_LM_HEAD \
    ((rt_lora_target_mask)(UINT64_C(1) << 6))
#define RT_LORA_TARGET_DEFAULT \
    (RT_LORA_TARGET_ATTENTION_QUERY | RT_LORA_TARGET_ATTENTION_VALUE)
#define RT_LORA_TARGET_ALL_LINEAR \
    ((rt_lora_target_mask)((UINT64_C(1) << 7) - UINT64_C(1)))

// Versioned value structures start with their byte size. Initialize config
// and option inputs with the corresponding rt_*_init function and
// sizeof(your_structure) before changing selected fields. Initialize an
// output structure's struct_size before passing it to the producing call.
typedef struct rt_tokenizer_options {
    uint64_t struct_size;
    rt_tokenizer_method method;
    uint32_t reserved;
    uint64_t vocabulary_size;
    uint64_t minimum_pair_frequency;
} rt_tokenizer_options;

typedef struct rt_bpe_merge_rule {
    uint32_t left;
    uint32_t right;
    uint32_t result;
} rt_bpe_merge_rule;

typedef struct rt_transformer_config {
    uint64_t struct_size;
    uint64_t vocabulary_size;
    uint64_t maximum_context;
    uint64_t model_width;
    uint64_t head_count;
    uint64_t block_count;
    uint64_t feed_forward_width;
    uint32_t random_seed;
    float layer_norm_epsilon;
} rt_transformer_config;

typedef struct rt_lora_config {
    uint64_t struct_size;
    uint64_t rank;
    float alpha;
    uint32_t random_seed;
    rt_lora_target_mask targets;
    uint64_t reserved;
} rt_lora_config;

// Packed-memory accounting for the model's NF4 linear weights. The payload
// fields exclude C/C++ object metadata and all ordinary FP32 parameters.
typedef struct rt_quantized_memory_stats {
    uint64_t struct_size;
    uint64_t weight_count;
    uint64_t packed_code_bytes;
    uint64_t scale_bytes;
    uint64_t logical_payload_bytes;
    uint64_t resident_payload_bytes;
    uint64_t fp32_equivalent_bytes;
    uint64_t fp32_scale_bytes;
    uint64_t scale_code_bytes;
    uint64_t second_level_scale_bytes;
    uint64_t scale_offset_bytes;
    uint64_t double_quantized_weight_count;
} rt_quantized_memory_stats;

typedef struct rt_decode_session_options {
    uint64_t struct_size;
    rt_kv_cache_kind kind;
    uint32_t reserved;
    uint64_t block_size;
} rt_decode_session_options;

typedef struct rt_adam_options {
    uint64_t struct_size;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float maximum_gradient_norm;
    uint32_t reserved;
    rt_adam_state_storage_kind state_storage;
    uint32_t reserved2;
    uint64_t page_size;
} rt_adam_options;

// ABI 2.3 callers may continue to pass the original 32-byte prefix ending at
// reserved; the runtime supplies contiguous state and a page size of 4096.
// ABI 2.4 callers can select bounded-page moment storage with state_storage.

typedef struct rt_adam_step_stats {
    uint64_t struct_size;
    uint64_t step;
    double gradient_norm;
    double clip_scale;
} rt_adam_step_stats;

// Raw handles do not provide lifetime synchronization. A caller must not
// release a handle while any call using that handle is in flight and must
// externally synchronize calls involving the same handle or handles derived
// from the same model. The Python wrapper provides this synchronization for
// its objects.
//
// Handles derived from a model retain its native state. Releasing rt_model
// does not invalidate an already-created decode session, parameter list,
// variable, or Adam optimizer. Each handle itself must still be released
// exactly once.
//
// A live decode session pins the model backend and parameter values. Model
// transfer, NF4 conversion, LoRA attachment/merge, parameter loading, and Adam
// steps are rejected until every decode session derived from that model is
// released.

RT_API uint32_t RT_CALL rt_abi_version(void);
RT_API const char* RT_CALL rt_status_string(rt_status status);

// The returned pointer is thread-local and remains valid until the next
// status-returning C API call on the same thread. Long details may be
// truncated to the implementation's fixed diagnostic-buffer capacity.
RT_API const char* RT_CALL rt_last_error(void);

// Initializes options for the legacy corpus-byte method. The defaults are a
// target vocabulary size of 512 and a minimum pair frequency of 2; these
// fields take effect when method is changed to RT_TOKENIZER_METHOD_BPE.
RT_API rt_status RT_CALL rt_tokenizer_options_init(
    rt_tokenizer_options* options,
    uint64_t options_size
);

// Tokenizers own an immutable vocabulary derived from a nonempty corpus.
// Null input data is valid only when its corresponding count is zero.
// rt_tokenizer_create preserves the ABI 1.2 corpus-byte behavior.
RT_API rt_status RT_CALL rt_tokenizer_create(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    rt_tokenizer** output
);

// Null options select the legacy corpus-byte method.
RT_API rt_status RT_CALL rt_tokenizer_create_with_options(
    const uint8_t* corpus_bytes,
    uint64_t corpus_size,
    const rt_tokenizer_options* options,
    rt_tokenizer** output
);

// Restores exact tokenizer state without access to its training corpus.
// The byte vocabulary is ordered by token ID and must be nonempty and unique.
RT_API rt_status RT_CALL rt_tokenizer_create_from_byte_vocabulary(
    const uint8_t* ordered_vocabulary,
    uint64_t vocabulary_size,
    rt_tokenizer** output
);

// A zero-rule BPE tokenizer is valid and contains the fixed 256 base bytes.
// Nonempty rules must use sequential result IDs beginning at 256.
RT_API rt_status RT_CALL rt_tokenizer_create_from_bpe_merges(
    const rt_bpe_merge_rule* ordered_merge_rules,
    uint64_t merge_count,
    rt_tokenizer** output
);

RT_API void RT_CALL rt_tokenizer_release(rt_tokenizer* tokenizer);

RT_API rt_status RT_CALL rt_tokenizer_get_method(
    const rt_tokenizer* tokenizer,
    rt_tokenizer_method* output
);

RT_API rt_status RT_CALL rt_tokenizer_vocabulary_size(
    const rt_tokenizer* tokenizer,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_tokenizer_bpe_merge_count(
    const rt_tokenizer* tokenizer,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_tokenizer_bpe_merge_rule(
    const rt_tokenizer* tokenizer,
    uint64_t index,
    rt_bpe_merge_rule* output
);

// Returns the ordered one-byte vocabulary of a legacy corpus-byte tokenizer.
// BPE tokenizers must use rt_tokenizer_token_bytes for individual pieces.
RT_API rt_status RT_CALL rt_tokenizer_vocabulary(
    const rt_tokenizer* tokenizer,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
);

// Variable-size output APIs validate the complete input and set
// required_count before checking output capacity. A null output with zero
// capacity is a size query. A null output with nonzero capacity is invalid.
// Insufficient capacity returns RT_STATUS_OUT_OF_RANGE without modifying the
// output buffer.
RT_API rt_status RT_CALL rt_tokenizer_token_bytes(
    const rt_tokenizer* tokenizer,
    uint32_t token_id,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
);

RT_API rt_status RT_CALL rt_tokenizer_encode(
    const rt_tokenizer* tokenizer,
    const uint8_t* text,
    uint64_t text_size,
    uint32_t* output,
    uint64_t capacity,
    uint64_t* required_count
);

RT_API rt_status RT_CALL rt_tokenizer_decode(
    const rt_tokenizer* tokenizer,
    const uint32_t* tokens,
    uint64_t token_count,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* required_count
);

RT_API rt_status RT_CALL rt_backend_is_available(
    rt_backend backend,
    int32_t* available
);

RT_API rt_status RT_CALL rt_context_create(
    rt_backend backend,
    rt_context** output
);

RT_API void RT_CALL rt_context_release(rt_context* context);

RT_API rt_status RT_CALL rt_context_backend(
    const rt_context* context,
    rt_backend* output
);

RT_API rt_status RT_CALL rt_tensor_create_f32(
    const rt_context* context,
    const uint64_t* shape,
    uint64_t rank,
    const float* values,
    uint64_t value_count,
    rt_tensor** output
);

RT_API rt_status RT_CALL rt_tensor_zeros_f32(
    const rt_context* context,
    const uint64_t* shape,
    uint64_t rank,
    rt_tensor** output
);

RT_API void RT_CALL rt_tensor_release(rt_tensor* tensor);

RT_API rt_status RT_CALL rt_tensor_backend(
    const rt_tensor* tensor,
    rt_backend* output
);

RT_API rt_status RT_CALL rt_tensor_rank(
    const rt_tensor* tensor,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_tensor_shape(
    const rt_tensor* tensor,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
);

RT_API rt_status RT_CALL rt_tensor_numel(
    const rt_tensor* tensor,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_tensor_copy_to_host_f32(
    const rt_tensor* tensor,
    float* output_values,
    uint64_t value_capacity
);

RT_API rt_status RT_CALL rt_tensor_matmul(
    const rt_tensor* left,
    const rt_tensor* right,
    rt_tensor** output
);

RT_API rt_status RT_CALL rt_transformer_config_init(
    rt_transformer_config* config,
    uint64_t config_size
);

// Initializes a paged cache with a block size of 16. A null options pointer
// passed to rt_model_decode_session_create selects the same defaults.
RT_API rt_status RT_CALL rt_decode_session_options_init(
    rt_decode_session_options* options,
    uint64_t options_size
);

// Models are initialized on CPU. Call rt_model_to before creating variables
// or an optimizer to select another available backend.
RT_API rt_status RT_CALL rt_model_create(
    const rt_transformer_config* config,
    rt_model** output
);

RT_API void RT_CALL rt_model_release(rt_model* model);

RT_API rt_status RT_CALL rt_model_to(
    rt_model* model,
    rt_backend backend
);

RT_API rt_status RT_CALL rt_model_backend(
    const rt_model* model,
    rt_backend* output
);

// Selects the implementation used by future full-sequence model forwards.
// Incremental decode sessions continue to use their configured KV-cache path.
RT_API rt_status RT_CALL rt_model_set_full_sequence_attention(
    rt_model* model,
    rt_full_sequence_attention_kind kind
);

RT_API rt_status RT_CALL rt_model_full_sequence_attention(
    const rt_model* model,
    rt_full_sequence_attention_kind* output
);

// Selects activation retention for future full-sequence training graphs.
// Incremental decode is stateful and is never checkpointed.
RT_API rt_status RT_CALL rt_model_set_activation_checkpointing(
    rt_model* model,
    rt_activation_checkpointing_kind kind
);

RT_API rt_status RT_CALL rt_model_activation_checkpointing(
    const rt_model* model,
    rt_activation_checkpointing_kind* output
);

// Initializes the native LoRA defaults. A null config passed to
// rt_model_attach_lora selects the same defaults.
RT_API rt_status RT_CALL rt_lora_config_init(
    rt_lora_config* config,
    uint64_t config_size
);

// A model accepts one LoRA attachment during its lifetime; a merged model
// cannot attach another. Attachment preserves the base parameter list and
// initially preserves model outputs.
RT_API rt_status RT_CALL rt_model_attach_lora(
    rt_model* model,
    const rt_lora_config* config
);

RT_API rt_status RT_CALL rt_model_has_lora(
    const rt_model* model,
    int32_t* output
);

// output must have struct_size initialized to sizeof(*output).
RT_API rt_status RT_CALL rt_model_lora_config(
    const rt_model* model,
    rt_lora_config* output
);

// Replaces every eligible FP32 Linear base weight with immutable blockwise
// NF4 storage. The common block size is 64. This operation must happen before
// LoRA attachment and is rejected while model-derived handles are alive.
RT_API rt_status RT_CALL rt_model_quantize_linear_weights_nf4(
    rt_model* model,
    uint64_t block_size
);

// Also stores first-level NF4 block scales as centered uint8 values with
// FP32 second-level scales. No persistent FP32 first-level scale vector is
// retained.
RT_API rt_status RT_CALL
rt_model_quantize_linear_weights_nf4_double_quantized(
    rt_model* model,
    uint64_t block_size,
    uint64_t scale_block_size
);

RT_API rt_status RT_CALL rt_model_has_quantized_linear_weights(
    const rt_model* model,
    int32_t* output
);

// output must have struct_size initialized to sizeof(*output).
RT_API rt_status RT_CALL rt_model_quantized_memory_stats(
    const rt_model* model,
    rt_quantized_memory_stats* output
);

// Token shape is [batch_size, sequence_length]. token_count must equal their
// product. The returned variable has shape
// [batch_size, sequence_length, vocabulary_size].
RT_API rt_status RT_CALL rt_model_forward(
    const rt_model* model,
    const uint32_t* token_ids,
    uint64_t token_count,
    uint64_t batch_size,
    uint64_t sequence_length,
    rt_variable** output
);

// Creates a single-sequence incremental-decoding session. The session retains
// the model state and remains valid if the original model handle is released.
// Its cache capacity is the model's maximum context.
RT_API rt_status RT_CALL rt_model_decode_session_create(
    const rt_model* model,
    const rt_decode_session_options* options,
    rt_decode_session** output
);

RT_API void RT_CALL rt_decode_session_release(
    rt_decode_session* session
);

// Clears all cached tokens while preserving the allocated cache.
RT_API rt_status RT_CALL rt_decode_session_reset(
    rt_decode_session* session
);

RT_API rt_status RT_CALL rt_decode_session_size(
    const rt_decode_session* session,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_decode_session_capacity(
    const rt_decode_session* session,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_decode_session_cache_kind(
    const rt_decode_session* session,
    rt_kv_cache_kind* output
);

RT_API rt_status RT_CALL rt_decode_session_block_size(
    const rt_decode_session* session,
    uint64_t* output
);

// Appends one token at the cache's current position and returns the
// vocabulary-sized logits that predict the following token. required_count
// is set to the model vocabulary size. Unlike tokenizer size-query APIs,
// output_logits must be nonnull and capacity must be sufficient; these checks
// occur before the session is mutated.
RT_API rt_status RT_CALL rt_decode_session_step(
    rt_decode_session* session,
    uint32_t token_id,
    float* output_logits,
    uint64_t capacity,
    uint64_t* required_count
);

RT_API rt_status RT_CALL rt_model_parameters(
    rt_model* model,
    rt_parameter_list** output
);

// Returns only the attached adapter parameters. The base-model parameter
// count, ordering, and names exposed by rt_model_parameters are unchanged.
RT_API rt_status RT_CALL rt_model_lora_parameters(
    rt_model* model,
    rt_parameter_list** output
);

// Merging is transactional and invalidates the attached adapter. It is
// rejected while a variable graph, optimizer, decode session, or adapter
// parameter list is alive.
RT_API rt_status RT_CALL rt_model_merge_lora(rt_model* model);

RT_API void RT_CALL rt_parameter_list_release(
    rt_parameter_list* parameters
);

RT_API rt_status RT_CALL rt_parameter_list_count(
    const rt_parameter_list* parameters,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_parameter_list_backend(
    const rt_parameter_list* parameters,
    rt_backend* output
);

// required_capacity includes the trailing null byte. Passing a null
// output_name with zero capacity is a supported size query.
RT_API rt_status RT_CALL rt_parameter_list_name(
    const rt_parameter_list* parameters,
    uint64_t index,
    char* output_name,
    uint64_t name_capacity,
    uint64_t* required_capacity
);

RT_API rt_status RT_CALL rt_parameter_list_rank(
    const rt_parameter_list* parameters,
    uint64_t index,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_parameter_list_shape(
    const rt_parameter_list* parameters,
    uint64_t index,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
);

RT_API rt_status RT_CALL rt_parameter_list_numel(
    const rt_parameter_list* parameters,
    uint64_t index,
    uint64_t* output
);

// Returns the number of floats in the deterministic flattened parameter
// representation. Values are ordered by parameter-list index and then by
// each tensor's native flat order.
RT_API rt_status RT_CALL rt_parameter_list_total_numel(
    const rt_parameter_list* parameters,
    uint64_t* output
);

// value_capacity must be at least rt_parameter_list_total_numel(). Extra
// capacity is left untouched.
RT_API rt_status RT_CALL rt_parameter_list_copy_to_host_f32(
    const rt_parameter_list* parameters,
    float* output_values,
    uint64_t value_capacity
);

// Replaces every parameter from the deterministic flattened representation.
// value_count must exactly equal rt_parameter_list_total_numel(), and every
// value must be finite. Loading is transactional and preserves parameter
// names, shapes, and backends while clearing all parameter gradients. It is
// rejected while a variable graph or optimizer derived from the model lives.
RT_API rt_status RT_CALL rt_parameter_list_load_from_host_f32(
    rt_parameter_list* parameters,
    const float* values,
    uint64_t value_count
);

#if defined(RIFTCO_TRANSFORMER_C_API_TESTING)
// Test-only epoch controls are not part of the installed stable ABI.
RT_API rt_status RT_CALL rt_test_model_parameter_epoch(
    const rt_model* model,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_test_model_set_parameter_epoch(
    rt_model* model,
    uint64_t value
);
#endif

RT_API void RT_CALL rt_variable_release(rt_variable* variable);

RT_API rt_status RT_CALL rt_variable_backend(
    const rt_variable* variable,
    rt_backend* output
);

RT_API rt_status RT_CALL rt_variable_rank(
    const rt_variable* variable,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_variable_shape(
    const rt_variable* variable,
    uint64_t* output_dimensions,
    uint64_t dimension_capacity
);

RT_API rt_status RT_CALL rt_variable_numel(
    const rt_variable* variable,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_variable_copy_to_host_f32(
    const rt_variable* variable,
    float* output_values,
    uint64_t value_capacity
);

// A successful backward consumes the shared computation graph. Adam steps
// invalidate every graph created before that parameter update.
RT_API rt_status RT_CALL rt_variable_backward(
    const rt_variable* variable
);

RT_API rt_status RT_CALL rt_cross_entropy(
    const rt_variable* logits,
    const uint32_t* targets,
    uint64_t target_count,
    rt_variable** output
);

RT_API rt_status RT_CALL rt_adam_options_init(
    rt_adam_options* options,
    uint64_t options_size
);

// A null options pointer selects the native Adam defaults.
RT_API rt_status RT_CALL rt_adam_create(
    const rt_parameter_list* parameters,
    const rt_adam_options* options,
    rt_adam** output
);

RT_API void RT_CALL rt_adam_release(rt_adam* adam);

RT_API rt_status RT_CALL rt_adam_backend(
    const rt_adam* adam,
    rt_backend* output
);

RT_API rt_status RT_CALL rt_adam_step_count(
    const rt_adam* adam,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_adam_parameter_count(
    const rt_adam* adam,
    uint64_t* output
);

// Paged state divides each first/second-moment tensor into allocations of at
// most page_size elements. CUDA pages use managed allocations and may migrate
// between host and device; this interface does not promise disk-backed spill.
RT_API rt_status RT_CALL rt_adam_state_storage(
    const rt_adam* adam,
    rt_adam_state_storage_kind* output
);

RT_API rt_status RT_CALL rt_adam_state_page_size(
    const rt_adam* adam,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_adam_state_page_count(
    const rt_adam* adam,
    uint64_t* output
);

RT_API rt_status RT_CALL rt_adam_state_payload_bytes(
    const rt_adam* adam,
    uint64_t* output
);

// output_stats must have struct_size initialized to sizeof(*output_stats).
RT_API rt_status RT_CALL rt_adam_step(
    rt_adam* adam,
    rt_adam_step_stats* output_stats
);

RT_API rt_status RT_CALL rt_adam_zero_gradients(
    const rt_adam* adam
);

#ifdef __cplusplus
}
#endif

#endif
