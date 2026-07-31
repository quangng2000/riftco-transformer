#include "transformer_lab/c_api.h"

#include <cstdlib>

int main() {
    return tl_abi_version() == TL_ABI_VERSION &&
                   TL_ACTIVATION_CHECKPOINTING_DISABLED !=
                       TL_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
