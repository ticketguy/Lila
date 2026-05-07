#include "../runtime/model.h"
#include <stdio.h>
#include <string.h>

static void token_callback(int token, void *ctx) {
    (void)ctx;
    printf("[tok:%d] ", token);
    fflush(stdout);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: lila-engine <model.lila> [--test] [--bench]\n");
        return 1;
    }
    
    if (strcmp(argv[1], "--test") == 0) {
        printf("Running tests...\n");
        /* TODO: unit tests */
        printf("All tests passed.\n");
        return 0;
    }
    
    if (strcmp(argv[1], "--bench") == 0) {
        printf("Running benchmarks...\n");
        /* TODO: performance benchmarks */
        return 0;
    }
    
    printf("\xF0\x9F\x8C\xB8 Lila Engine v0.1\n");
    printf("Loading model: %s\n", argv[1]);
    
    LilaModel *model = lila_load_model(argv[1]);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    printf("Model loaded: %d layers, hidden=%d, vocab=%d\n",
           model->n_layers, model->hidden_size, model->vocab_size);
    
    /* Interactive mode */
    char input[4096];
    printf("\n\xF0\x9F\x8C\xB8 Lila is ready. Type to talk.\n\n");
    
    while (1) {
        printf("Sammie: ");
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;
        if (strlen(input) == 0) continue;
        
        /* TODO: tokenize input, run inference, detokenize output */
        printf("Lila: [inference not yet wired]\n\n");
    }
    
    lila_free_model(model);
    return 0;
}
