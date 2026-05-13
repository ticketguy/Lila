#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * BPE Tokenizer — encodes text into token IDs and decodes back.
 *
 * Encoding strategy (simplified BPE):
 * 1. Convert input to bytes (UTF-8)
 * 2. Start with each byte as a separate token
 * 3. Iteratively merge the most frequent pair (using merge rules)
 * 4. Return final token IDs
 *
 * For Phase 1: greedy longest-match against vocabulary.
 * This is not perfect BPE but produces reasonable tokenization
 * for testing the inference pipeline end-to-end.
 */

#define MAX_VOCAB 524288
#define MAX_TOKEN_LEN 256
#define MAX_INPUT_LEN 65536

struct LilaTokenizer {
    char **tokens;
    float *scores;      /* Token scores for BPE priority */
    int vocab_size;
    int bos_id;
    int eos_id;
    int pad_id;
};

LilaTokenizer *lila_load_tokenizer(const char *vocab_path) {
    LilaTokenizer *tok = calloc(1, sizeof(LilaTokenizer));
    tok->tokens = calloc(MAX_VOCAB, sizeof(char *));
    tok->scores = calloc(MAX_VOCAB, sizeof(float));
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->pad_id = 0;
    
    FILE *f = fopen(vocab_path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open vocab: %s\n", vocab_path);
        free(tok->tokens);
        free(tok->scores);
        free(tok);
        return NULL;
    }
    
    char line[MAX_TOKEN_LEN];
    int i = 0;
    while (fgets(line, sizeof(line), f) && i < MAX_VOCAB) {
        line[strcspn(line, "\n")] = 0;
        tok->tokens[i] = strdup(line);
        tok->scores[i] = (float)(MAX_VOCAB - i);  /* Higher score = more common */
        i++;
    }
    tok->vocab_size = i;
    fclose(f);
    
    fprintf(stderr, "Tokenizer: %d tokens loaded\n", tok->vocab_size);
    return tok;
}

const char *lila_decode_token(LilaTokenizer *tok, int token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) return "";
    if (!tok->tokens[token_id]) return "";
    return tok->tokens[token_id];
}

/* Decode a sequence of token IDs to a string */
char *lila_decode_sequence(LilaTokenizer *tok, const int *tokens, int n_tokens) {
    /* Estimate output size */
    size_t total_len = 0;
    for (int i = 0; i < n_tokens; i++) {
        const char *t = lila_decode_token(tok, tokens[i]);
        total_len += strlen(t);
    }
    
    char *output = malloc(total_len + 1);
    output[0] = 0;
    
    for (int i = 0; i < n_tokens; i++) {
        const char *t = lila_decode_token(tok, tokens[i]);
        /* Handle sentencepiece-style tokens: replace ▁ with space */
        if (t[0] == (char)0xE2 && t[1] == (char)0x96 && t[2] == (char)0x81) {
            strcat(output, " ");
            strcat(output, t + 3);
        } else {
            strcat(output, t);
        }
    }
    
    return output;
}

/* Encode text → token IDs (greedy longest match) */
int lila_encode(LilaTokenizer *tok, const char *text, int *output_ids, int max_tokens) {
    int n_tokens = 0;
    int text_len = strlen(text);
    int pos = 0;
    
    while (pos < text_len && n_tokens < max_tokens) {
        int best_id = -1;
        int best_len = 0;
        
        /* Find longest matching token starting at pos */
        for (int i = 0; i < tok->vocab_size && i < 100000; i++) {
            if (!tok->tokens[i]) continue;
            int tlen = strlen(tok->tokens[i]);
            if (tlen <= 0 || tlen > text_len - pos) continue;
            if (tlen <= best_len) continue;
            
            if (strncmp(text + pos, tok->tokens[i], tlen) == 0) {
                best_id = i;
                best_len = tlen;
            }
        }
        
        if (best_id >= 0) {
            output_ids[n_tokens++] = best_id;
            pos += best_len;
        } else {
            /* Byte fallback — encode as raw byte token */
            /* Skip this character */
            pos++;
        }
    }
    
    return n_tokens;
}

int lila_get_bos(LilaTokenizer *tok) { return tok ? tok->bos_id : 1; }
int lila_get_eos(LilaTokenizer *tok) { return tok ? tok->eos_id : 2; }
int lila_get_vocab_size(LilaTokenizer *tok) { return tok ? tok->vocab_size : 0; }

/*
 * Load tokenizer directly from binary memory (no temp file).
 * Data format: sequence of [uint16_t length][bytes...] entries.
 * This avoids newline corruption that breaks the file-based loader.
 */
LilaTokenizer *lila_load_tokenizer_from_memory(const uint8_t *data, size_t data_size,
                                                int vocab_size, int bos_id, int eos_id, int pad_id) {
    LilaTokenizer *tok = calloc(1, sizeof(LilaTokenizer));
    if (!tok) return NULL;
    
    int max_vocab = vocab_size < 512000 ? vocab_size : 512000;
    tok->tokens = calloc(max_vocab, sizeof(char *));
    tok->scores = calloc(max_vocab, sizeof(float));
    tok->bos_id = bos_id;
    tok->eos_id = eos_id;
    tok->pad_id = pad_id;
    
    const uint8_t *ptr = data;
    const uint8_t *end = data + data_size;
    int i = 0;
    
    while (i < max_vocab && ptr + 2 <= end) {
        uint16_t token_len;
        memcpy(&token_len, ptr, 2);
        ptr += 2;
        
        if (ptr + token_len > end) break;
        
        if (token_len > 0 && token_len < 4096) {
            tok->tokens[i] = malloc(token_len + 1);
            if (tok->tokens[i]) {
                memcpy(tok->tokens[i], ptr, token_len);
                tok->tokens[i][token_len] = '\0';
            }
        } else {
            /* Empty or oversized token — store placeholder */
            tok->tokens[i] = strdup("");
        }
        tok->scores[i] = (float)(max_vocab - i);
        ptr += token_len;
        i++;
    }
    
    tok->vocab_size = i;
    fprintf(stderr, "Tokenizer: %d tokens loaded (from memory)\n", tok->vocab_size);
    return tok;
}

void lila_free_tokenizer(LilaTokenizer *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) free(tok->tokens[i]);
    free(tok->tokens);
    free(tok->scores);
    free(tok);
}
