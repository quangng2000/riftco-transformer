#include "riftco_transformer/c_api.h"

#include <stdint.h>
#include <stdlib.h>

int main(void) {
    if (rt_abi_version() != RT_ABI_VERSION ||
        RT_ABI_VERSION_MINOR != UINT32_C(5) ||
        RT_BACKEND_CUDA != (rt_backend)2 ||
        RT_BACKEND_TPU != (rt_backend)3 ||
        RT_ADAM_STATE_PAGED != (rt_adam_state_storage_kind)1) {
        return EXIT_FAILURE;
    }

    rt_lora_config lora;
    if (rt_lora_config_init(
            &lora,
            (uint64_t)sizeof(lora)
        ) != RT_STATUS_OK ||
        lora.rank != 4 ||
        lora.alpha != 8.0F ||
        lora.targets != RT_LORA_TARGET_DEFAULT) {
        return EXIT_FAILURE;
    }

    rt_quantized_memory_stats quantized = {
        (uint64_t)sizeof(rt_quantized_memory_stats),
        0,
        0,
        0,
        0,
        0,
        0,
    };
    if (quantized.struct_size !=
        (uint64_t)sizeof(rt_quantized_memory_stats)) {
        return EXIT_FAILURE;
    }

    rt_adam_options adam_options;
    if (rt_adam_options_init(
            &adam_options,
            (uint64_t)sizeof(adam_options)
        ) != RT_STATUS_OK ||
        adam_options.state_storage != RT_ADAM_STATE_CONTIGUOUS ||
        adam_options.page_size != UINT64_C(4096)) {
        return EXIT_FAILURE;
    }

    const uint8_t corpus[] = {0x00, 0x61, 0xff};
    rt_tokenizer* tokenizer = NULL;
    uint64_t vocabulary_size = 0;
    if (rt_tokenizer_create(
            corpus,
            (uint64_t)sizeof(corpus),
            &tokenizer
        ) != RT_STATUS_OK ||
        tokenizer == NULL ||
        rt_tokenizer_vocabulary_size(
            tokenizer,
            &vocabulary_size
        ) != RT_STATUS_OK ||
        vocabulary_size != 3) {
        rt_tokenizer_release(tokenizer);
        return EXIT_FAILURE;
    }
    rt_tokenizer_release(tokenizer);

    rt_tokenizer_options tokenizer_options;
    if (rt_tokenizer_options_init(
            &tokenizer_options,
            (uint64_t)sizeof(tokenizer_options)
        ) != RT_STATUS_OK) {
        return EXIT_FAILURE;
    }
    tokenizer_options.method = RT_TOKENIZER_METHOD_BPE;
    tokenizer_options.vocabulary_size = 258;
    tokenizer_options.minimum_pair_frequency = 2;

    const uint8_t bpe_corpus[] = {
        'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b',
    };
    rt_tokenizer* bpe = NULL;
    rt_tokenizer_method method = (rt_tokenizer_method)-1;
    if (rt_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &tokenizer_options,
            &bpe
        ) != RT_STATUS_OK ||
        bpe == NULL ||
        rt_tokenizer_get_method(bpe, &method) != RT_STATUS_OK ||
        method != RT_TOKENIZER_METHOD_BPE ||
        rt_tokenizer_vocabulary_size(
            bpe,
            &vocabulary_size
        ) != RT_STATUS_OK ||
        vocabulary_size != 258) {
        rt_tokenizer_release(bpe);
        return EXIT_FAILURE;
    }

    uint64_t required = 0;
    uint8_t learned_piece[4] = {0, 0, 0, 0};
    if (rt_tokenizer_token_bytes(
            bpe,
            UINT32_C(257),
            learned_piece,
            (uint64_t)sizeof(learned_piece),
            &required
        ) != RT_STATUS_OK ||
        required != 4 ||
        learned_piece[0] != 'a' ||
        learned_piece[1] != 'b' ||
        learned_piece[2] != 'a' ||
        learned_piece[3] != 'b') {
        rt_tokenizer_release(bpe);
        return EXIT_FAILURE;
    }

    const uint8_t universal_bytes[] = {0x00, 0xff, 0x78};
    uint32_t universal_tokens[3] = {0, 0, 0};
    uint8_t decoded_bytes[3] = {0, 0, 0};
    if (rt_tokenizer_encode(
            bpe,
            universal_bytes,
            (uint64_t)sizeof(universal_bytes),
            universal_tokens,
            3,
            &required
        ) != RT_STATUS_OK ||
        required != 3 ||
        universal_tokens[0] != 0 ||
        universal_tokens[1] != 255 ||
        universal_tokens[2] != 120 ||
        rt_tokenizer_decode(
            bpe,
            universal_tokens,
            3,
            decoded_bytes,
            3,
            &required
        ) != RT_STATUS_OK ||
        required != 3 ||
        decoded_bytes[0] != 0x00 ||
        decoded_bytes[1] != 0xff ||
        decoded_bytes[2] != 0x78) {
        rt_tokenizer_release(bpe);
        return EXIT_FAILURE;
    }
    rt_tokenizer_release(bpe);

    int32_t cpu_available = 0;
    if (rt_backend_is_available(
            RT_BACKEND_CPU,
            &cpu_available
        ) != RT_STATUS_OK ||
        cpu_available == 0) {
        return EXIT_FAILURE;
    }

    int32_t cuda_available = -1;
    if (rt_backend_is_available(
            RT_BACKEND_CUDA,
            &cuda_available
        ) != RT_STATUS_OK) {
        return EXIT_FAILURE;
    }
    rt_context* cuda_context =
        (rt_context*)(uintptr_t)1;
    const rt_status cuda_status = rt_context_create(
        RT_BACKEND_CUDA,
        &cuda_context
    );
    if (cuda_available != 0) {
        rt_backend selected = RT_BACKEND_CPU;
        if (cuda_status != RT_STATUS_OK ||
            cuda_context == NULL ||
            rt_context_backend(cuda_context, &selected) !=
                RT_STATUS_OK ||
            selected != RT_BACKEND_CUDA) {
            rt_context_release(cuda_context);
            return EXIT_FAILURE;
        }
        rt_context_release(cuda_context);
    } else if (cuda_status != RT_STATUS_BACKEND_UNAVAILABLE ||
               cuda_context != NULL) {
        rt_context_release(cuda_context);
        return EXIT_FAILURE;
    }

    rt_context* context = NULL;
    if (rt_context_create(
            RT_BACKEND_CPU,
            &context
        ) != RT_STATUS_OK ||
        context == NULL) {
        return EXIT_FAILURE;
    }

    rt_context_release(context);
    return EXIT_SUCCESS;
}
