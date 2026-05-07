#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * BPE Tokenizer for Gemma/LLaMA-family models.
 * Loads sentencepiece vocabulary and performs encoding/decoding.
 *
 * For full functionality, this would need:
 * 1. Load .model file (protobuf) or vocab.json
 * 2. BPE merge rules
 * 3. Byte-fallback for unknown characters
 *
 * Phase 1: Load vocab from a simple text format (one token per line).
 * Phase 4: Full sentencepiece compatibility.
 */

#define MAX_VOCAB 128000
#define MAX_TOKEN_LEN 128

typedef struct {
    char **tokens;      /* Array of token strings */
    int vocab_size;
    /* TODO: merge rules, scores */
} LilaTokenizer;

LilaTokenizer *lila_load_tokenizer(const char *vocab_path) {
    LilaTokenizer *tok = calloc(1, sizeof(LilaTokenizer));
    tok->tokens = calloc(MAX_VOCAB, sizeof(char *));
    
    FILE *f = fopen(vocab_path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open vocab: %s\n", vocab_path);
        free(tok->tokens);
        free(tok);
        return NULL;
    }
    
    char line[MAX_TOKEN_LEN];
    int i = 0;
    while (fgets(line, sizeof(line), f) && i < MAX_VOCAB) {
        line[strcspn(line, "\n")] = 0;
        tok->tokens[i] = strdup(line);
        i++;
    }
    tok->vocab_size = i;
    fclose(f);
    
    fprintf(stderr, "Tokenizer loaded: %d tokens\n", tok->vocab_size);
    return tok;
}

/* Decode token ID to string */
const char *lila_decode_token(LilaTokenizer *tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) return "<unk>";
    return tok->tokens[token_id];
}

/* Simple encode (character-level fallback — full BPE in Phase 4) */
int lila_encode_char(LilaTokenizer *tok, char c) {
    /* Search for single-character token */
    char target[2] = {c, 0};
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->tokens[i] && strcmp(tok->tokens[i], target) == 0) {
            return i;
        }
    }
    return 0; /* unknown → first token */
}

void lila_free_tokenizer(LilaTokenizer *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) {
        free(tok->tokens[i]);
    }
    free(tok->tokens);
    free(tok);
}
