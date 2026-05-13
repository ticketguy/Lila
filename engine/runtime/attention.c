#include "model.h"

/* Unified weight dispatch (handles Q4_K, FigQuant, FP32) */
extern void weight_matvec(float *out, const LilaQuantWeight *w, const float *vec);
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * Multi-Head Attention with Rotary Position Embeddings (RoPE)
 * and KV Cache for efficient autoregressive generation.
 *
 * For Gemma 4B: n_heads=16, n_kv_heads=8 (GQA), head_dim=256
 * GQA: key/value heads are shared across query head groups
 */

/* Apply RoPE to a single head vector */
static void apply_rope(float *vec, int head_dim, int position, float theta)
{
    for (int i = 0; i < head_dim; i += 2)
    {
        float freq = 1.0f / powf(theta, (float)i / head_dim);
        float angle = position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * cos_a - v1 * sin_a;
        vec[i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

/* Initialize KV cache */
void lila_init_kv_cache(LilaKVCache *cache, int n_layers, int max_seq,
                        int n_kv_heads, int head_dim)
{
    cache->max_seq_len = max_seq;
    cache->current_pos = 0;

    size_t layer_size = (size_t)max_seq * n_kv_heads * head_dim * sizeof(float);

    cache->key_cache = calloc(n_layers, layer_size);
    cache->value_cache = calloc(n_layers, layer_size);

    if (!cache->key_cache || !cache->value_cache)
    {
        fprintf(stderr, "ASI: KV cache alloc failed (%.1f GB requested)\n",
                (double)((size_t)n_layers * layer_size * 2) / (1024.0 * 1024.0 * 1024.0));
        exit(1);
    }

    fprintf(stderr, "ASI: KV cache allocated (%.1f GB)\n",
            (double)((size_t)n_layers * layer_size * 2) / (1024.0 * 1024.0 * 1024.0));
}

/* Single-token attention (for autoregressive generation) */
void lila_attention(
    float *output,      /* [hidden_size] */
    const float *input, /* [hidden_size] */
    LilaLayer *layer,
    LilaKVCache *cache,
    int layer_idx,
    int position)
{
    int hidden = layer->hidden_size;
    int n_heads = layer->n_heads;
    int n_kv_heads = layer->n_kv_heads;
    int head_dim = layer->head_dim;
    int kv_group = n_heads / n_kv_heads; /* GQA group size */
    int kv_size = n_kv_heads * head_dim;
    int q_size = hidden;
    int k_size = kv_size;
    int v_size = kv_size;
    int out_size = hidden;
    int attn_out_size = hidden;

    if (q_size < layer->q_proj.rows)
        q_size = layer->q_proj.rows;
    if (k_size < layer->k_proj.rows)
        k_size = layer->k_proj.rows;
    if (v_size < layer->v_proj.rows)
        v_size = layer->v_proj.rows;
    if (out_size < layer->o_proj.rows)
        out_size = layer->o_proj.rows;
    if (attn_out_size < layer->o_proj.cols)
        attn_out_size = layer->o_proj.cols;

    fprintf(stderr, "[DBG] attention: hidden=%d n_heads=%d n_kv_heads=%d head_dim=%d\n",
            hidden, n_heads, n_kv_heads, head_dim);
    fflush(stderr);
    fprintf(stderr, "[DBG] q_proj rows=%d cols=%d data=%p\n",
            layer->q_proj.rows, layer->q_proj.cols,
            (void *)layer->q_proj.data);
    fflush(stderr);
    fprintf(stderr, "[DBG] k_proj rows=%d cols=%d data=%p\n",
            layer->k_proj.rows, layer->k_proj.cols,
            (void *)layer->k_proj.data);
    fflush(stderr);
    fprintf(stderr, "[DBG] v_proj rows=%d cols=%d data=%p\n",
            layer->v_proj.rows, layer->v_proj.cols,
            (void *)layer->v_proj.data);
    fflush(stderr);
    fprintf(stderr, "[DBG] o_proj rows=%d cols=%d data=%p\n",
            layer->o_proj.rows, layer->o_proj.cols,
            (void *)layer->o_proj.data);
    fflush(stderr);
    fprintf(stderr, "[DBG] allocating q[%d] k[%d] v[%d] attn_out[%d] out[%d]\n",
            q_size, k_size, v_size, attn_out_size, out_size);
    fflush(stderr);

    /* Allocate scratch (TODO: pre-allocate in model struct) */
    float *q = calloc((size_t)q_size, sizeof(float));
    float *k = calloc((size_t)k_size, sizeof(float));
    float *v = calloc((size_t)v_size, sizeof(float));
    float *attn_out = calloc((size_t)attn_out_size, sizeof(float));
    float *proj_out = calloc((size_t)out_size, sizeof(float));
    if (!q || !k || !v || !attn_out || !proj_out)
    {
        fprintf(stderr, "ASI: Attention scratch allocation failed\n");
        free(q);
        free(k);
        free(v);
        free(attn_out);
        free(proj_out);
        return;
    }

    /* Project Q, K, V using quantized weights */
    fprintf(stderr, "[DBG] calling q weight_matvec...\n");
    fflush(stderr);
    weight_matvec(q, &layer->q_proj, input);
    fprintf(stderr, "[DBG] q weight_matvec returned\n");
    fflush(stderr);
    fprintf(stderr, "[DBG] calling k weight_matvec...\n");
    fflush(stderr);
    weight_matvec(k, &layer->k_proj, input);
    fprintf(stderr, "[DBG] k weight_matvec returned\n");
    fflush(stderr);
    fprintf(stderr, "[DBG] calling v weight_matvec...\n");
    fflush(stderr);
    weight_matvec(v, &layer->v_proj, input);
    fprintf(stderr, "[DBG] v weight_matvec returned\n");
    fflush(stderr);

    /* Apply RoPE to Q and K */
    for (int h = 0; h < n_heads; h++)
    {
        apply_rope(q + h * head_dim, head_dim, position, 10000.0f);
    }
    for (int h = 0; h < n_kv_heads; h++)
    {
        apply_rope(k + h * head_dim, head_dim, position, 10000.0f);
    }

    /* Store K, V in cache */
    size_t kv_offset = (size_t)position * n_kv_heads * head_dim;
    size_t layer_offset = (size_t)layer_idx * cache->max_seq_len * n_kv_heads * head_dim;
    memcpy(cache->key_cache + layer_offset + kv_offset, k, n_kv_heads * head_dim * sizeof(float));
    memcpy(cache->value_cache + layer_offset + kv_offset, v, n_kv_heads * head_dim * sizeof(float));

    /* Compute attention scores for each head */
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++)
    {
        int kv_h = h / kv_group; /* GQA: which KV head this Q head uses */
        float *q_h = q + h * head_dim;

        /* Attention scores: dot(q, all cached keys) */
        float *scores = malloc((position + 1) * sizeof(float));
        float max_score = -1e30f;

        for (int t = 0; t <= position; t++)
        {
            float *k_t = cache->key_cache + layer_offset + (size_t)t * n_kv_heads * head_dim + kv_h * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_h[d] * k_t[d];
            score *= scale;
            scores[t] = score;
            if (score > max_score)
                max_score = score;
        }

        /* Softmax */
        float sum = 0.0f;
        for (int t = 0; t <= position; t++)
        {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        for (int t = 0; t <= position; t++)
            scores[t] /= sum;

        /* Weighted sum of values */
        float *out_h = attn_out + h * head_dim;
        for (int t = 0; t <= position; t++)
        {
            float *v_t = cache->value_cache + layer_offset + (size_t)t * n_kv_heads * head_dim + kv_h * head_dim;
            for (int d = 0; d < head_dim; d++)
                out_h[d] += scores[t] * v_t[d];
        }

        free(scores);
    }

    /* Output projection */
    fprintf(stderr, "[DBG] calling o weight_matvec...\n");
    fflush(stderr);
    weight_matvec(proj_out, &layer->o_proj, attn_out);
    fprintf(stderr, "[DBG] o weight_matvec returned\n");
    fflush(stderr);

    memset(output, 0, (size_t)hidden * sizeof(float));
    int copy = hidden < layer->o_proj.rows ? hidden : layer->o_proj.rows;
    if (copy > 0)
        memcpy(output, proj_out, (size_t)copy * sizeof(float));

    free(q);
    free(k);
    free(v);
    free(attn_out);
    free(proj_out);
}
