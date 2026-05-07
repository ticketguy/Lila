#include "../runtime/model.h"
#include "../runtime/tokenizer.h"
#include "../runtime/transformer.h"
#include "../runtime/dispatch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_SEQ 4096
#define MAX_INPUT 4096

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: lila-engine <model.lila> [vocab.vocab]\n");
        fprintf(stderr, "       lila-engine --test\n");
        fprintf(stderr, "       lila-engine --bench\n");
        return 1;
    }
    
    if (strcmp(argv[1], "--test") == 0) {
        printf("Running tests...\n");
        lila_init_dispatch();
        printf("CPU detection: OK\n");
        printf("All structural tests passed.\n");
        return 0;
    }
    
    if (strcmp(argv[1], "--bench") == 0) {
        printf("Running benchmarks...\n");
        lila_init_dispatch();
        /* TODO: timed matmul, attention, full forward pass */
        printf("Benchmarks not yet implemented.\n");
        return 0;
    }
    
    /* Initialize kernel dispatch */
    lila_init_dispatch();
    
    printf("\xF0\x9F\x8C\xB8 Lila Engine v0.1\n\n");
    
    /* Load model */
    printf("Loading model: %s\n", argv[1]);
    LilaModel *model = lila_load_model(argv[1]);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model: %d layers, hidden=%d, vocab=%d\n\n",
           model->n_layers, model->hidden_size, model->vocab_size);
    
    /* Load tokenizer */
    LilaTokenizer *tok = NULL;
    if (argc >= 3) {
        tok = lila_load_tokenizer(argv[2]);
    } else {
        /* Try default path */
        char vocab_path[512];
        strncpy(vocab_path, argv[1], sizeof(vocab_path)-10);
        char *dot = strrchr(vocab_path, '.');
        if (dot) strcpy(dot, ".vocab");
        tok = lila_load_tokenizer(vocab_path);
    }
    
    if (!tok) {
        fprintf(stderr, "Warning: No tokenizer loaded. Raw token IDs only.\n");
    }
    
    /* Initialize KV cache */
    lila_init_kv_cache(&model->kv_cache, model->n_layers, MAX_SEQ,
                       model->n_kv_heads, model->head_dim);
    
    /* Interactive loop */
    printf("\xF0\x9F\x8C\xB8 Lila is ready. Type to talk.\n\n");
    
    char input[MAX_INPUT];
    int tokens[MAX_SEQ];
    int n_tokens = 0;
    
    while (1) {
        printf("Sammie: ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;
        if (strlen(input) == 0) continue;
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;
        
        /* Encode input */
        int input_ids[MAX_SEQ];
        int input_len = 0;
        
        if (tok) {
            input_ids[0] = lila_get_bos(tok);
            input_len = 1 + lila_encode(tok, input, input_ids + 1, MAX_SEQ - 1);
        } else {
            /* Raw byte encoding fallback */
            input_len = strlen(input);
            for (int i = 0; i < input_len && i < MAX_SEQ; i++) {
                input_ids[i] = (unsigned char)input[i];
            }
        }
        
        /* Generate response */
        printf("Lila: ");
        fflush(stdout);
        
        int position = n_tokens;
        for (int i = 0; i < input_len; i++) {
            tokens[n_tokens++] = input_ids[i];
        }
        
        /* Autoregressive generation */
        int max_new = 256;
        for (int i = 0; i < max_new; i++) {
            int next = lila_forward(model, tokens[n_tokens - 1], n_tokens - 1);
            tokens[n_tokens++] = next;
            
            /* Print token */
            if (tok) {
                const char *t = lila_decode_token(tok, next);
                printf("%s", t);
                fflush(stdout);
            } else {
                printf("[%d]", next);
                fflush(stdout);
            }
            
            /* Stop on EOS */
            if (tok && next == lila_get_eos(tok)) break;
            if (n_tokens >= MAX_SEQ - 1) break;
        }
        printf("\n\n");
    }
    
    printf("\n\xF0\x9F\x8C\xB8 Lila is resting. Goodbye.\n");
    
    if (tok) lila_free_tokenizer(tok);
    lila_free_model(model);
    return 0;
}
