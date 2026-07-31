#include "transformer_lab/c_api.h"

#include <stdint.h>
#include <stdlib.h>

int main(void) {
    if (tl_abi_version() != TL_ABI_VERSION) {
        return EXIT_FAILURE;
    }

    tl_lora_config lora;
    if (tl_lora_config_init(
            &lora,
            (uint64_t)sizeof(lora)
        ) != TL_STATUS_OK ||
        lora.rank != 4 ||
        lora.alpha != 8.0F ||
        lora.targets != TL_LORA_TARGET_DEFAULT) {
        return EXIT_FAILURE;
    }

    const uint8_t corpus[] = {0x00, 0x61, 0xff};
    tl_tokenizer* tokenizer = NULL;
    uint64_t vocabulary_size = 0;
    if (tl_tokenizer_create(
            corpus,
            (uint64_t)sizeof(corpus),
            &tokenizer
        ) != TL_STATUS_OK ||
        tokenizer == NULL ||
        tl_tokenizer_vocabulary_size(
            tokenizer,
            &vocabulary_size
        ) != TL_STATUS_OK ||
        vocabulary_size != 3) {
        tl_tokenizer_release(tokenizer);
        return EXIT_FAILURE;
    }
    tl_tokenizer_release(tokenizer);

    tl_tokenizer_options tokenizer_options;
    if (tl_tokenizer_options_init(
            &tokenizer_options,
            (uint64_t)sizeof(tokenizer_options)
        ) != TL_STATUS_OK) {
        return EXIT_FAILURE;
    }
    tokenizer_options.method = TL_TOKENIZER_METHOD_BPE;
    tokenizer_options.vocabulary_size = 258;
    tokenizer_options.minimum_pair_frequency = 2;

    const uint8_t bpe_corpus[] = {
        'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b',
    };
    tl_tokenizer* bpe = NULL;
    tl_tokenizer_method method = (tl_tokenizer_method)-1;
    if (tl_tokenizer_create_with_options(
            bpe_corpus,
            (uint64_t)sizeof(bpe_corpus),
            &tokenizer_options,
            &bpe
        ) != TL_STATUS_OK ||
        bpe == NULL ||
        tl_tokenizer_get_method(bpe, &method) != TL_STATUS_OK ||
        method != TL_TOKENIZER_METHOD_BPE ||
        tl_tokenizer_vocabulary_size(
            bpe,
            &vocabulary_size
        ) != TL_STATUS_OK ||
        vocabulary_size != 258) {
        tl_tokenizer_release(bpe);
        return EXIT_FAILURE;
    }

    uint64_t required = 0;
    uint8_t learned_piece[4] = {0, 0, 0, 0};
    if (tl_tokenizer_token_bytes(
            bpe,
            UINT32_C(257),
            learned_piece,
            (uint64_t)sizeof(learned_piece),
            &required
        ) != TL_STATUS_OK ||
        required != 4 ||
        learned_piece[0] != 'a' ||
        learned_piece[1] != 'b' ||
        learned_piece[2] != 'a' ||
        learned_piece[3] != 'b') {
        tl_tokenizer_release(bpe);
        return EXIT_FAILURE;
    }

    const uint8_t universal_bytes[] = {0x00, 0xff, 0x78};
    uint32_t universal_tokens[3] = {0, 0, 0};
    uint8_t decoded_bytes[3] = {0, 0, 0};
    if (tl_tokenizer_encode(
            bpe,
            universal_bytes,
            (uint64_t)sizeof(universal_bytes),
            universal_tokens,
            3,
            &required
        ) != TL_STATUS_OK ||
        required != 3 ||
        universal_tokens[0] != 0 ||
        universal_tokens[1] != 255 ||
        universal_tokens[2] != 120 ||
        tl_tokenizer_decode(
            bpe,
            universal_tokens,
            3,
            decoded_bytes,
            3,
            &required
        ) != TL_STATUS_OK ||
        required != 3 ||
        decoded_bytes[0] != 0x00 ||
        decoded_bytes[1] != 0xff ||
        decoded_bytes[2] != 0x78) {
        tl_tokenizer_release(bpe);
        return EXIT_FAILURE;
    }
    tl_tokenizer_release(bpe);

    int32_t cpu_available = 0;
    if (tl_backend_is_available(
            TL_BACKEND_CPU,
            &cpu_available
        ) != TL_STATUS_OK ||
        cpu_available == 0) {
        return EXIT_FAILURE;
    }

    tl_context* context = NULL;
    if (tl_context_create(
            TL_BACKEND_CPU,
            &context
        ) != TL_STATUS_OK ||
        context == NULL) {
        return EXIT_FAILURE;
    }

    tl_context_release(context);
    return EXIT_SUCCESS;
}
