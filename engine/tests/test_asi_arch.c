#include "asi_arch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void print_top5(const char *label, const float *logits, uint32_t vocab_size)
{
    int top_idx[5] = {-1, -1, -1, -1, -1};
    float top_val[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
    for (uint32_t i = 0; i < vocab_size; ++i)
    {
        float v = logits[i];
        for (int k = 0; k < 5; ++k)
        {
            if (v > top_val[k])
            {
                for (int s = 4; s > k; --s)
                {
                    top_val[s] = top_val[s - 1];
                    top_idx[s] = top_idx[s - 1];
                }
                top_val[k] = v;
                top_idx[k] = (int)i;
                break;
            }
        }
    }
    printf("%s top5: ", label);
    for (int k = 0; k < 5; ++k)
    {
        printf("(%d, %.5f)%s", top_idx[k], top_val[k], k < 4 ? ", " : "\n");
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: test_asi_arch <cartridge.asi>\n");
        return 2;
    }

    AsiV2Cartridge cart;
    char err[256] = {0};
    if (asi_v2_open(argv[1], &cart, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "FAIL: asi_v2_open failed: %s\n", err);
        return 1;
    }

    const AsiArchHandler *handler = asi_arch_find(cart.header.arch_id);
    if (!handler || !handler->init_session)
    {
        fprintf(stderr, "FAIL: no active architecture handler for arch_id=%u (%s)\n",
                cart.header.arch_id, asi_v2_arch_name(cart.header.arch_id));
        asi_v2_close(&cart);
        return 1;
    }

    AsiArchSession *session = NULL;
    if (handler->init_session(&cart, 2048, 4, &session, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "FAIL: init_session failed: %s\n", err);
        asi_v2_close(&cart);
        return 1;
    }

    printf("PASS: %s bound all %u layers (%u tensors) from %s\n",
           handler->name, cart.config.n_layers, cart.n_tensors, argv[1]);

    float *logits = (float *)malloc((size_t)cart.config.vocab_size * sizeof(float));
    struct timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (handler->forward_token(session, 9906, 0, logits) != 0)
    {
        fprintf(stderr, "FAIL: forward_token(9906, 0) failed\n");
        free(logits);
        handler->free_session(session);
        asi_v2_close(&cart);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms0 = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    print_top5("pos=0 token=9906", logits, cart.config.vocab_size);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (handler->forward_token(session, 11, 1, logits) != 0)
    {
        fprintf(stderr, "FAIL: forward_token(11, 1) failed\n");
        free(logits);
        handler->free_session(session);
        asi_v2_close(&cart);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms1 = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    print_top5("pos=1 token=11", logits, cart.config.vocab_size);
    printf("Timing: token0=%.1f ms, token1=%.1f ms (%.2f tok/s)\n", ms0, ms1, 1000.0 / ms1);

    /* Greedy 10-token continuation from pos=2 */
    int32_t cur = 198;
    for (uint32_t i = 0; i < (uint32_t)cart.config.vocab_size; ++i)
    {
        if (logits[i] > logits[cur]) cur = (int32_t)i;
    }
    printf("Generated IDs from [9906, 11]:");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t step = 0; step < 10; ++step)
    {
        printf(" %d", cur);
        if (handler->forward_token(session, cur, 2 + step, logits) != 0) break;
        int32_t next = 0;
        for (uint32_t i = 1; i < cart.config.vocab_size; ++i)
        {
            if (logits[i] > logits[next]) next = (int32_t)i;
        }
        cur = next;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gen_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("\n10-token decode: %.1f ms (%.2f tok/s)\n", gen_ms, 10000.0 / gen_ms);

    free(logits);
    handler->reset_state(session);
    handler->free_session(session);
    asi_v2_close(&cart);
    return 0;
}
