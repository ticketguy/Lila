#include "model.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Multi-Head Attention with Rotary Position Embeddings (RoPE)
 * and KV Cache for efficient autoregressive generation.
 *
 * For Gemma 4B: n_heads=16, n_kv_heads=8 (GQA), head_dim=256
 * GQA: key/value heads are shared across query head groups
 */

/* Apply RoPE to a single head vector */
static void apply_rope(float *vec, int head_dim, int position, float theta) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / head_dim);
        float angle = position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_a - v1 * sin_a;
        vec[i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

/* Initialize KV cache */
void lila_init_kv_cache(LilaKVCache *cache, int n_layers, int max_seq,
                         int n_kv_heads, int head_dim) {
    cache->max_seq_len = max_seq;
    cache->current_pos = 0;
    
    size_t layer_size = (size_t)max_seq * n_kv_heads * head_dim * sizeof(float);
    cache->key_cache = calloc(n_layers, layer_size);
    cache->value_cache = calloc(n_layers, layer_size);
}

/* Single-token attention (for autoregressive generation) */
void lila_attention(
    float *output,          /* [hidden_size] */
    const float *input,     /* [hidden_size] */
    LilaLayer *layer,
    LilaKVCache *cache,
    int layer_idx,
    int position
) {
    int hidden = layer->hidden_size;
    int n_heads = layer->n_heads;
    int n_kv_heads = layer->n_kv_heads;
    int head_dim = layer->head_dim;
    int kv_group = n_heads / n_kv_heads;  /* GQA group size */
    
    /* Allocate scratch (TODO: pre-allocate in model struct) */
    float *q = malloc(hidden * sizeof(float));
    float *k = malloc(n_kv_heads * head_dim * sizeof(float));
    float *v = malloc(n_kv_heads * head_dim * sizeof(float));
    float *attn_out = calloc(hidden, sizeof(float));
    
    /* Project Q, K, V using quantized weights */
    /* TODO: replace with dequant_matvec from kernels */
    dequant_matvec(q, &layer->q_proj, input);
    dequant_matvec(k, &layer->k_proj, input);
    dequant_matvec(v, &layer->v_proj, input);
    
    /* Apply RoPE to Q and K */
    for (int h = 0; h < n_heads; h++) {
        apply_rope(q + h * head_dim, head_dim, position, 10000.0f);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        apply_rope(k + h * head_dim, head_dim, position, 10000.0f);
    }
    
    /* Store K, V in cache */
    size_t kv_offset = (size_t)position * n_kv_heads * head_dim;
    size_t layer_offset = (size_t)layer_idx * cache->max_seq_len * n_kv_heads * head_dim;
    memcpy(cache->key_cache + layer_offset + kv_offset, k, n_kv_heads * head_dim * sizeof(float));
    memcpy(cache->value_cache + layer_offset + kv_offset, v, n_kv_heads * head_dim * sizeof(float));
    
    /* Compute attention scores for each head */
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / kv_group;  /* GQA: which KV head this Q head uses */
        float *q_h = q + h * head_dim;
        
        /* Attention scores: dot(q, all cached keys) */
        float *scores = malloc((position + 1) * sizeof(float));
        float max_score = -1e30f;
        
        for (int t = 0; t <= position; t++) {
            float *k_t = cache->key_cache + layer_offset + (size_t)t * n_kv_heads * head_dim + kv_h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_h[d] * k_t[d];
            }
            score *= scale;
            scores[t] = score;
            if (score > max_score) max_score = score;
        }
        
        /* Softmax */
        float sum = 0.0f;
        for (int t = 0; t <= position; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        for (int t = 0; t <= position; t++) {
            scores[t] /= sum;
        }
        
        /* Weighted sum of values */
        float *out_h = attn_out + h * head_dim;
        for (int t = 0; t <= position; t++) {
            float *v_t = cache->value_cache + layer_offset + (size_t)t * n_kv_heads * head_dim + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++) {
                out_h[d] += scores[t] * v_t[d];
            }
        }
        
        free(scores);
    }
    
    /* Output projection */
    dequant_matvec(output, &layer->o_proj, attn_out);
    
    free(q);
    free(k);
    free(v);
    free(attn_out);
}

/* Forward declaration for dequant_matvec (defined in inference.c) */
extern void dequant_matvec(float *out, const LilaQuantWeight *w, const float *vec);
