#include "model.h"
#include "q4k.h"
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
 *
 * Supports both FigQuant INT4 and native Q4_K (GGUF) weights.
 */

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  MATH PRIMITIVES                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Softmax */
static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

/* Matrix-vector multiply (FP32 dense) */
void matvec(float *out, const float *mat, const float *vec, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += mat[i * cols + j] * vec[j];
        }
        out[i] = sum;
    }
}

/* FigQuant INT4 dequant + matvec */
static void figquant_matvec(float *out, const LilaQuantWeight *w, const float *vec) {
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
            
            float scale = w->scales[group_idx];
            float val = w->codebook[nibble] * scale;
            sum += val * vec[j];
        }
        out[i] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  UNIFIED WEIGHT DISPATCH                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * weight_matvec — THE core compute function.
 * Dispatches to the correct dequant based on the weight's quant_type.
 * Called by attention.c and transformer.c for all projections.
 */
void weight_matvec(float *out, const LilaQuantWeight *w, const float *vec) {
    if (!w || !w->data || w->rows == 0 || w->cols == 0) {
        return; /* Empty weight — skip */
    }
    
    switch (w->quant_type) {
    case QUANT_Q4_K:
        /* Native GGUF Q4_K — best quality, no re-quantization */
        q4k_matvec(out, w->data, vec, w->rows, w->cols);
        break;
    
    case QUANT_FIGQUANT:
        /* FigQuant INT4 — codebook + scales + packed */
        figquant_matvec(out, w, vec);
        break;
    
    case QUANT_NONE:
        /* FP32 dense — used for embedding/lm_head when not quantized */
        matvec(out, (const float *)w->data, vec, w->rows, w->cols);
        break;
    
    default:
        /* Unknown quant type — zero output */
        memset(out, 0, w->rows * sizeof(float));
        break;
    }
}

/* Legacy name — redirect to unified dispatch */
void dequant_matvec(float *out, const LilaQuantWeight *w, const float *vec) {
    weight_matvec(out, w, vec);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  SAMPLING                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int sample_token(float *logits, int vocab_size, float temperature) {
    if (temperature > 0.0f && temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++) logits[i] /= temperature;
    }
    
    softmax(logits, vocab_size);
    
    /* Greedy argmax */
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) { max_val = logits[i]; max_idx = i; }
    }
    return max_idx;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  GENERATE (stub — actual forward pass is in transformer.c)                */
/* ═══════════════════════════════════════════════════════════════════════════ */

int lila_generate_token(LilaModel *model, int *tokens, int n_tokens) {
    (void)model; (void)tokens; (void)n_tokens;
    return 0; /* Not used — asi_runtime calls lila_forward() directly */
}

void lila_generate(LilaModel *model, int *tokens, int n_tokens, int max_new_tokens,
                   void (*callback)(int token, void *ctx), void *ctx) {
    for (int i = 0; i < max_new_tokens; i++) {
        int next = lila_generate_token(model, tokens, n_tokens + i);
        tokens[n_tokens + i] = next;
        if (callback) callback(next, ctx);
        if (next == 0) break;
    }
}
