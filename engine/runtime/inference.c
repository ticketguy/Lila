#include "model.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * Core inference loop.
 * For each new token:
 *   1. Embed token
 *   2. For each layer: attention + MLP (with Memory Fabric)
 *   3. Final norm
 *   4. LM head → logits
 *   5. Sample next token
 */

/* RMSNorm — will be replaced by assembly kernel */
static void rmsnorm(float *out, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
}

/* SiLU activation */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Softmax */
static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

/* Matrix-vector multiply — THE hot path. Will be assembly. */
void matvec(float *out, const float *mat, const float *vec, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += mat[i * cols + j] * vec[j];
        }
        out[i] = sum;
    }
}

/* INT4 dequant + matvec — fused for cache efficiency */
void dequant_matvec(float *out, const LilaQuantWeight *w, const float *vec) {
    int rows = w->rows;
    int cols = w->cols;
    
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            int flat_idx = i * cols + j;
            int group_idx = flat_idx / LILA_GROUP_SIZE;
            int byte_idx = flat_idx / 2;
            int nibble = (flat_idx % 2 == 0) 
                ? (w->indices[byte_idx] & 0x0F)
                : ((w->indices[byte_idx] >> 4) & 0x0F);
            
            /* Dequant: codebook[nibble] * scale */
            float scale = (float)w->scales[group_idx]; /* TODO: FP16 decode */
            float val = w->codebook[nibble] * scale;
            sum += val * vec[j];
        }
        out[i] = sum;
    }
}

/* Sample from logits (temperature + top-p) */
static int sample_token(float *logits, int vocab_size, float temperature, float top_p) {
    /* Apply temperature */
    if (temperature > 0.0f) {
        for (int i = 0; i < vocab_size; i++) logits[i] /= temperature;
    }
    
    softmax(logits, vocab_size);
    
    /* Top-p sampling */
    /* For now: greedy (argmax) */
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) { max_val = logits[i]; max_idx = i; }
    }
    return max_idx;
}

/* Generate one token */
int lila_generate_token(LilaModel *model, int *tokens, int n_tokens) {
    /* TODO: full transformer forward pass */
    /* This is the structural skeleton — actual compute dispatches to kernels */
    (void)model; (void)tokens; (void)n_tokens;
    return 0; /* placeholder */
}

/* Generate sequence */
void lila_generate(LilaModel *model, int *tokens, int n_tokens, int max_new_tokens,
                   void (*callback)(int token, void *ctx), void *ctx) {
    for (int i = 0; i < max_new_tokens; i++) {
        int next = lila_generate_token(model, tokens, n_tokens + i);
        tokens[n_tokens + i] = next;
        if (callback) callback(next, ctx);
        if (next == 0) break; /* EOS */
    }
}
