#include "asi.h"
#include "../runtime/tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/*
 * lila-asi — Load and run Lila from a single .asi file.
 *
 * This is the end-user facing binary. One command:
 *   ./lila-asi lila.asi
 *
 * And she's alive. Everything she needs is in that one file.
 *
 * Usage:
 *   ./lila-asi lila.asi                    # Interactive chat
 *   ./lila-asi lila.asi --info             # Print .asi contents
 *   ./lila-asi lila.asi --bench            # Benchmark inference
 *   ./lila-asi lila.asi --reload new.asi   # Hot-reload adapters
 */

#define MAX_SEQ 4096
#define MAX_INPUT 4096

/* Forward declarations */
extern void lila_init_kv_cache(LilaKVCache *cache, int n_layers, int max_seq,
                                int n_kv_heads, int head_dim);
extern void lila_init_dispatch(void);

static void print_info(AsiRuntime *rt) {
    printf("\n🌸 ASI File Information\n");
    printf("═══════════════════════════════════════\n");
    
    printf("\nSections present:\n");
    const char *section_names[] = {
        NULL, "MODEL_CONFIG", "WEIGHTS", "MEMORY_FABRIC",
        "TOKENIZER", "BYTECODE", "HARNESS", "PERSONALITY", "METADATA"
    };
    
    for (int i = 1; i <= 8; i++) {
        if (asi_has_section(rt, (AsiSectionType)i)) {
            uint64_t size = asi_section_size(rt, (AsiSectionType)i);
            printf("  ✓ %-16s  %8.2f KB\n", section_names[i], size / 1024.0);
        } else {
            printf("  ✗ %-16s  (missing)\n", section_names[i]);
        }
    }
    printf("\n");
}

static void run_bench(AsiRuntime *rt) {
    printf("Running inference benchmark...\n");
    
    int tokens[MAX_SEQ];
    tokens[0] = 1;  /* BOS */
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int n_generated = 0;
    for (int i = 0; i < 100; i++) {
        int next = asi_generate_token(rt, tokens, 1 + i);
        tokens[1 + i] = next;
        n_generated++;
        if (next <= 0) break;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("  Generated: %d tokens\n", n_generated);
    printf("  Time: %.3f seconds\n", elapsed);
    if (elapsed > 0) {
        printf("  Speed: %.1f tokens/sec\n", n_generated / elapsed);
    }
}

static void interactive(AsiRuntime *rt) {
    printf("🌸 Lila is ready. Type to talk. 'quit' to exit.\n\n");
    
    char input[MAX_INPUT];
    int tokens[MAX_SEQ];
    int n_tokens = 0;
    
    /* BOS token */
    tokens[0] = 1;
    n_tokens = 1;
    
    while (1) {
        printf("Sammie: ");
        fflush(stdout);
        
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;
        if (strlen(input) == 0) continue;
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;
        
        /* Generate response */
        printf("Lila: ");
        fflush(stdout);
        
        int max_new = 256;
        for (int i = 0; i < max_new; i++) {
            int next = asi_generate_token(rt, tokens, n_tokens);
            if (next <= 0) break;
            tokens[n_tokens++] = next;
            
            /* Print token (would use embedded tokenizer) */
            printf("[%d]", next);
            fflush(stdout);
            
            if (n_tokens >= MAX_SEQ - 1) break;
        }
        printf("\n\n");
    }
    
    printf("\n🌸 Lila is resting. Goodbye.\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("🌸 lila-asi — Load Lila from an Active System Image\n\n");
        printf("Usage:\n");
        printf("  lila-asi <file.asi>              # Chat with Lila\n");
        printf("  lila-asi <file.asi> --info       # Inspect .asi contents\n");
        printf("  lila-asi <file.asi> --bench      # Benchmark inference\n");
        printf("  lila-asi <file.asi> --reload <new.asi>  # Hot-reload adapters\n");
        return 1;
    }
    
    const char *asi_path = argv[1];
    
    /* Initialize native kernel dispatch */
    lila_init_dispatch();
    
    /* Load .asi */
    AsiRuntime *rt = asi_load(asi_path);
    if (!rt) {
        fprintf(stderr, "Failed to load: %s\n", asi_path);
        return 1;
    }
    
    /* Handle flags */
    if (argc >= 3) {
        if (strcmp(argv[2], "--info") == 0) {
            print_info(rt);
            asi_free(rt);
            return 0;
        }
        if (strcmp(argv[2], "--bench") == 0) {
            run_bench(rt);
            asi_free(rt);
            return 0;
        }
        if (strcmp(argv[2], "--reload") == 0 && argc >= 4) {
            int result = asi_reload_fabric(rt, argv[3]);
            if (result == 0) {
                printf("✅ Hot-reload successful. New adapters active.\n");
            } else {
                printf("❌ Hot-reload failed.\n");
            }
            /* Continue to interactive mode with new adapters */
        }
    }
    
    /* Interactive mode */
    interactive(rt);
    
    asi_free(rt);
    return 0;
}
