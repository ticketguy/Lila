#include "asi_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: test_asi_v2 <cartridge.asi>\n");
        return 2;
    }

    AsiV2Cartridge cart;
    char err[256] = {0};
    if (asi_v2_open(argv[1], &cart, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "FAIL: asi_v2_open failed: %s\n", err);
        return 1;
    }

    printf("ASI v%u Cartridge OK: arch=%s layers=%u hidden=%u tensors=%u\n",
           cart.header.version,
           asi_v2_arch_name(cart.header.arch_id),
           cart.config.n_layers,
           cart.config.hidden_size,
           cart.n_tensors);

    const AsiV2TensorEntry *t = asi_v2_find_tensor(&cart, "blk.0.attn_q.weight");
    if (!t || t->data_size == 0)
    {
        fprintf(stderr, "FAIL: expected tensor blk.0.attn_q.weight not found\n");
        asi_v2_close(&cart);
        return 1;
    }

    if (!asi_v2_find_section(&cart, ASI_SEC_CHAT_TEMPLATE))
    {
        fprintf(stderr, "FAIL: CHAT_TEMPLATE section missing\n");
        asi_v2_close(&cart);
        return 1;
    }

    /* Negative test: corrupt 1 byte in memory and verify CRC-64 rejects it */
    uint8_t *copy = (uint8_t *)malloc(cart.file_size);
    memcpy(copy, cart.base, cart.file_size);
    const AsiV2SectionEntry *cfg = asi_v2_find_section(&cart, ASI_SEC_MODEL_CONFIG);
    copy[cfg->offset] ^= 0xFF;

    AsiV2Cartridge bad;
    if (asi_v2_validate_memory(copy, cart.file_size, &bad, err, sizeof(err)) == 0)
    {
        fprintf(stderr, "FAIL: corrupted section was not rejected\n");
        free(copy);
        asi_v2_close(&cart);
        return 1;
    }
    printf("PASS: corrupted cartridge rejected (%s)\n", err);

    free(copy);
    asi_v2_close(&cart);
    printf("PASS: ASI v2 spec & C validator verified\n");
    return 0;
}
