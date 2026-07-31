#include "riftco_transformer/c_api.h"

#include <cstdlib>

int main() {
    return rt_abi_version() == RT_ABI_VERSION &&
                   RT_ACTIVATION_CHECKPOINTING_DISABLED !=
                       RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
