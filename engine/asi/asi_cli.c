#include "asi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * lila-asi — Load and run Lila from a single .asi file.
 *
 * Usage:
 *   lila-asi lila.asi                    # Interactive chat
 *   lila-asi lila.asi --info             # Print .asi contents
 *   lila-asi lila.asi --bench            # Benchmark inference
 *   lila-asi lila.asi --reload new.asi   # Hot-reload adapters
 */

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PLATFORM TIMER                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#include <windows.h>
static double get_time_sec(void)
{
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}
#endif

#define MAX_SEQ 4096
#define MAX_INPUT 4096
#define EOS_TOKEN 106 /* Gemma EOS */
#define BOS_TOKEN 2   /* Gemma BOS */

/* Forward declarations */
extern void lila_init_dispatch(void);

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  --info                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void print_info(AsiRuntime *rt)
{
    printf("\n");
    printf("  ASI File Information\n");
    printf("  ====================\n");
    printf("\n  Sections present:\n");

    const char *section_names[] = {
        NULL, "MODEL_CONFIG", "WEIGHTS", "MEMORY_FABRIC",
        "TOKENIZER", "BYTECODE", "HARNESS", "PERSONALITY", "METADATA"};

    for (int i = 1; i <= 8; i++)
    {
        if (asi_has_section(rt, (AsiSectionType)i))
        {
            uint64_t size = asi_section_size(rt, (AsiSectionType)i);
            if (size > 1024 * 1024)
                printf("    [x] %-16s  %8.2f MB\n", section_names[i], size / (1024.0 * 1024.0));
            else
                printf("    [x] %-16s  %8.2f KB\n", section_names[i], size / 1024.0);
        }
        else
        {
            printf("    [ ] %-16s  (missing)\n", section_names[i]);
        }
    }
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  --bench                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void run_bench(AsiRuntime *rt)
{
    printf("Running inference benchmark...\n");

    int tokens[MAX_SEQ];
    tokens[0] = BOS_TOKEN;
    int n_tokens = 1;

    double start = get_time_sec();
    int n_generated = 0;

    for (int i = 0; i < 20; i++)
    {
        int next = asi_generate_token(rt, tokens, n_tokens);
        if (next <= 0 || next == EOS_TOKEN)
            break;
        tokens[n_tokens++] = next;
        n_generated++;
    }

    double elapsed = get_time_sec() - start;
    printf("  Generated: %d tokens\n", n_generated);
    printf("  Time:      %.3f seconds\n", elapsed);
    if (elapsed > 0 && n_generated > 0)
        printf("  Speed:     %.2f tokens/sec\n", n_generated / elapsed);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  INTERACTIVE CHAT                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void interactive(AsiRuntime *rt)
{
    printf("\n  Lila is ready. Type to talk. 'quit' to exit.\n\n");

    char input[MAX_INPUT];
    int tokens[MAX_SEQ];
    int n_tokens = 0;

    tokens[0] = BOS_TOKEN;
    n_tokens = 1;

    while (1)
    {
        printf("Sammie: ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin))
            break;
        input[strcspn(input, "\n")] = 0;
        if (strlen(input) == 0)
            continue;
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0)
            break;

        /* ── Tokenize input ── */
        int n_new = asi_tokenize(rt, input, tokens + n_tokens, MAX_SEQ - n_tokens);
        if (n_new > 0)
        {
            n_tokens += n_new;
        }
        else
        {
            fprintf(stderr, "[warn] Tokenizer returned 0 tokens for input\n");
        }

        /* ── Generate ── */
        printf("Lila: ");
        fflush(stdout);

        double t_start = get_time_sec();
        int n_gen = 0;

        for (int i = 0; i < 512; i++)
        {
            if (n_tokens >= MAX_SEQ - 1)
                break;

            int next = asi_generate_token(rt, tokens, n_tokens);

            if (next <= 0 || next == EOS_TOKEN)
                break;

            tokens[n_tokens++] = next;
            n_gen++;

            /* Decode token to text */
            const char *piece = asi_decode_token(rt, next);
            if (piece)
            {
                printf("%s", piece);
                fflush(stdout);
            }
            else
            {
                printf("[%d]", next);
                fflush(stdout);
            }
        }

        double elapsed = get_time_sec() - t_start;
        printf("\n");
        if (n_gen > 0 && elapsed > 0)
            printf("  [%.1f tok/s]\n", n_gen / elapsed);
        printf("\n");
    }

    printf("\n  Lila is resting. Goodbye.\n");
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  MAIN                                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("\n  lila-asi -- Load Lila from an Active System Image\n\n");
        printf("  Usage:\n");
        printf("    lila-asi <file.asi>              # Chat with Lila\n");
        printf("    lila-asi <file.asi> --info       # Inspect .asi contents\n");
        printf("    lila-asi <file.asi> --bench      # Benchmark inference\n");
        printf("    lila-asi <file.asi> --reload <new.asi>  # Hot-reload adapters\n");
        return 1;
    }

    const char *asi_path = argv[1];

    lila_init_dispatch();

    AsiRuntime *rt = asi_load(asi_path);
    if (!rt)
    {
        fprintf(stderr, "Failed to load: %s\n", asi_path);
        return 1;
    }

    if (argc >= 3)
    {
        if (strcmp(argv[2], "--info") == 0)
        {
            print_info(rt);
            asi_free(rt);
            return 0;
        }
        if (strcmp(argv[2], "--bench") == 0)
        {
            run_bench(rt);
            asi_free(rt);
            return 0;
        }
        if (strcmp(argv[2], "--reload") == 0 && argc >= 4)
        {
            int result = asi_reload_fabric(rt, argv[3]);
            printf(result == 0
                       ? "  Hot-reload successful. New adapters active.\n"
                       : "  Hot-reload failed.\n");
        }
    }

    interactive(rt);

    asi_free(rt);
    return 0;
}