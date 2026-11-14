#include "model.h"
#include "q4k.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/*
 * Multi-Head / Grouped-Query Attention with RoPE and FP16 KV Cache.
 * Model-agnostic: supports any n_heads, n_kv_heads, head_dim, and rope_theta.
 */

static void ensure_rope_tables(LilaScratch *scratch, int max_seq, int head_dim, float theta)
{
    if (!scratch || head_dim <= 0 || max_seq <= 0)
        return;
    if (scratch->rope_cos && scratch->rope_seq_len >= max_seq && scratch->rope_head_dim == head_dim)
        return;

    free(scratch->rope_cos);
    free(scratch->rope_sin);
    int half = head_dim / 2;
    size_t count = (size_t)max_seq * (size_t)half;
    scratch->rope_cos = (float *)malloc(count * sizeof(float));
    scratch->rope_sin = (float *)malloc(count * sizeof(float));
    if (!scratch->rope_cos || !scratch->rope_sin)
    {
        free(scratch->rope_cos);
        free(scratch->rope_sin);
        scratch->rope_cos = NULL;
        scratch->rope_sin = NULL;
        return;
    }

    if (theta <= 0.0f)
        theta = 10000.0f;

    float inv_freq_buf[512];
    float *inv_freq = (half <= 512) ? inv_freq_buf : (float *)malloc((size_t)half * sizeof(float));
    for (int i = 0; i < half; i++)
    {
        inv_freq[i] = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
    }

    for (int pos = 0; pos < max_seq; pos++)
    {
        float *c_row = scratch->rope_cos + (size_t)pos * half;
        float *s_row = scratch->rope_sin + (size_t)pos * half;
        float p = (float)pos;
        for (int i = 0; i < half; i++)
        {
            float angle = p * inv_freq[i];
            c_row[i] = cosf(angle);
            s_row[i] = sinf(angle);
        }
    }

    if (inv_freq != inv_freq_buf)
        free(inv_freq);

    scratch->rope_seq_len = max_seq;
    scratch->rope_head_dim = head_dim;
}

static inline void apply_rope_fast(float *vec, int head_dim, int position, float theta,
                                   const LilaScratch *scratch)
{
    int half = head_dim / 2;
    if (scratch && scratch->rope_cos && position < scratch->rope_seq_len && scratch->rope_head_dim == head_dim)
    {
        const float *c_row = scratch->rope_cos + (size_t)position * half;
        const float *s_row = scratch->rope_sin + (size_t)position * half;
        for (int i = 0; i < half; i++)
        {
            float v0 = vec[2 * i];
            float v1 = vec[2 * i + 1];
            float cos_a = c_row[i];
            float sin_a = s_row[i];
            vec[2 * i] = v0 * cos_a - v1 * sin_a;
            vec[2 * i + 1] = v0 * sin_a + v1 * cos_a;
        }
        return;
    }

    if (theta <= 0.0f)
        theta = 10000.0f;
    for (int i = 0; i < head_dim; i += 2)
    {
        float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
        float angle = (float)position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * cos_a - v1 * sin_a;
        vec[i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

/* Initialize FP16 KV cache */
void lila_init_kv_cache(LilaKVCache *cache, int n_layers, int max_seq,
                        int n_kv_heads, int head_dim)
{
    cache->max_seq_len = max_seq;
    cache->current_pos = 0;

    size_t layer_elems = (size_t)max_seq * (size_t)n_kv_heads * (size_t)head_dim;
    cache->key_cache = (uint16_t *)calloc((size_t)n_layers * layer_elems, sizeof(uint16_t));
    cache->value_cache = (uint16_t *)calloc((size_t)n_layers * layer_elems, sizeof(uint16_t));

    if (!cache->key_cache || !cache->value_cache)
    {
        fprintf(stderr, "ASI: KV cache alloc failed (%.1f MB requested)\n",
                (double)((size_t)n_layers * layer_elems * 2 * sizeof(uint16_t)) / (1024.0 * 1024.0));
        exit(1);
    }

    fprintf(stderr, "ASI: KV cache allocated (%.1f MB FP16)\n",
            (double)((size_t)n_layers * layer_elems * 2 * sizeof(uint16_t)) / (1024.0 * 1024.0));
}

static inline float dot_f32_f16(const float *q, const uint16_t *k_f16, int n)
{
#if defined(__AVX2__) && defined(__F16C__)
    __m256 acc = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= n; d += 8)
    {
        __m128i kh = _mm_loadu_si128((const __m128i *)(k_f16 + d));
        __m256 kf = _mm256_cvtph_ps(kh);
        __m256 qf = _mm256_loadu_ps(q + d);
        acc = _mm256_fmadd_ps(qf, kf, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; d < n; d++)
        sum += q[d] * fp16_to_fp32(k_f16[d]);
    return sum;
#else
    float sum = 0.0f;
    for (int d = 0; d < n; d++)
        sum += q[d] * fp16_to_fp32(k_f16[d]);
    return sum;
#endif
}

static inline void accum_f16_weighted(float *out, const uint16_t *v_f16, float w, int n)
{
#if defined(__AVX2__) && defined(__F16C__)
    __m256 wv = _mm256_set1_ps(w);
    int d = 0;
    for (; d + 8 <= n; d += 8)
    {
        __m128i vh = _mm_loadu_si128((const __m128i *)(v_f16 + d));
        __m256 vf = _mm256_cvtph_ps(vh);
        __m256 ov = _mm256_loadu_ps(out + d);
        ov = _mm256_fmadd_ps(wv, vf, ov);
        _mm256_storeu_ps(out + d, ov);
    }
    for (; d < n; d++)
        out[d] += w * fp16_to_fp32(v_f16[d]);
#else
    for (int d = 0; d < n; d++)
        out[d] += w * fp16_to_fp32(v_f16[d]);
#endif
}

void lila_attention_ex(
    float *output,
    const float *input,
    LilaLayer *layer,
    LilaKVCache *cache,
    LilaScratch *scratch,
    float rope_theta,
    int layer_idx,
    int position)
{
    int hidden = layer->hidden_size;
    int n_heads = layer->n_heads;
    int n_kv_heads = layer->n_kv_heads > 0 ? layer->n_kv_heads : n_heads;
    int head_dim = layer->q_proj.rows > 0 ? layer->q_proj.rows / n_heads : layer->head_dim;
    int kv_group = n_heads / n_kv_heads;
    if (kv_group < 1) kv_group = 1;
    int kv_size = n_kv_heads * head_dim;

    int q_size = hidden > layer->q_proj.rows ? hidden : layer->q_proj.rows;
    int k_size = kv_size > layer->k_proj.rows ? kv_size : layer->k_proj.rows;
    int v_size = kv_size > layer->v_proj.rows ? kv_size : layer->v_proj.rows;
    int out_size = hidden > layer->o_proj.rows ? hidden : layer->o_proj.rows;
    int attn_out_size = hidden > layer->o_proj.cols ? hidden : layer->o_proj.cols;

    int own_alloc = 0;
    float *q, *k, *v, *attn_out, *proj_out, *scores;
    if (scratch && scratch->initialized)
    {
        ensure_rope_tables(scratch, cache->max_seq_len, head_dim, rope_theta);
        q = scratch->q;
        k = scratch->k;
        v = scratch->v;
        attn_out = scratch->attn_buf;
        proj_out = scratch->proj_out;
        scores = scratch->scores;
        memset(attn_out, 0, (size_t)attn_out_size * sizeof(float));
    }
    else
    {
        own_alloc = 1;
        q = (float *)calloc((size_t)q_size, sizeof(float));
        k = (float *)calloc((size_t)k_size, sizeof(float));
        v = (float *)calloc((size_t)v_size, sizeof(float));
        attn_out = (float *)calloc((size_t)attn_out_size, sizeof(float));
        proj_out = (float *)calloc((size_t)out_size, sizeof(float));
        scores = (float *)malloc((size_t)(position + 1) * sizeof(float));
        if (!q || !k || !v || !attn_out || !proj_out || !scores)
        {
            free(q); free(k); free(v); free(attn_out); free(proj_out); free(scores);
            return;
        }
    }

    /* Project Q, K, V */
    weight_matvec(q, &layer->q_proj, input);
    weight_matvec(k, &layer->k_proj, input);
    weight_matvec(v, &layer->v_proj, input);

    /* Apply RoPE to Q and K */
    for (int h = 0; h < n_heads; h++)
        apply_rope_fast(q + h * head_dim, head_dim, position, rope_theta, scratch);
    for (int h = 0; h < n_kv_heads; h++)
        apply_rope_fast(k + h * head_dim, head_dim, position, rope_theta, scratch);

    /* Store K, V in FP16 cache */
    size_t kv_offset = (size_t)position * (size_t)kv_size;
    size_t layer_offset = (size_t)layer_idx * (size_t)cache->max_seq_len * (size_t)kv_size;
    uint16_t *dst_k = cache->key_cache + layer_offset + kv_offset;
    uint16_t *dst_v = cache->value_cache + layer_offset + kv_offset;
    for (int i = 0; i < kv_size; i++)
    {
        dst_k[i] = fp32_to_fp16(k[i]);
        dst_v[i] = fp32_to_fp16(v[i]);
    }

    /* Attention per head */
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < n_heads; h++)
    {
        int kv_h = h / kv_group;
        const float *q_h = q + h * head_dim;
        float max_score = -1e30f;

        for (int t = 0; t <= position; t++)
        {
            const uint16_t *k_t = cache->key_cache + layer_offset + (size_t)t * kv_size + kv_h * head_dim;
            float score = dot_f32_f16(q_h, k_t, head_dim) * scale;
            scores[t] = score;
            if (score > max_score)
                max_score = score;
        }

        float sum = 0.0f;
        for (int t = 0; t <= position; t++)
        {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        float inv_sum = sum > 0.0f ? (1.0f / sum) : 0.0f;
        for (int t = 0; t <= position; t++)
            scores[t] *= inv_sum;

        float *out_h = attn_out + h * head_dim;
        for (int t = 0; t <= position; t++)
        {
            const uint16_t *v_t = cache->value_cache + layer_offset + (size_t)t * kv_size + kv_h * head_dim;
            accum_f16_weighted(out_h, v_t, scores[t], head_dim);
        }
    }

    /* Output projection */
    weight_matvec(proj_out, &layer->o_proj, attn_out);

    memset(output, 0, (size_t)hidden * sizeof(float));
    int copy = hidden < layer->o_proj.rows ? hidden : layer->o_proj.rows;
    if (copy > 0)
        memcpy(output, proj_out, (size_t)copy * sizeof(float));

    if (own_alloc)
    {
        free(q); free(k); free(v); free(attn_out); free(proj_out); free(scores);
    }
}

void lila_attention(
    float *output,
    const float *input,
    LilaLayer *layer,
    LilaKVCache *cache,
    int layer_idx,
    int position)
{
    lila_attention_ex(output, input, layer, cache, NULL, 10000.0f, layer_idx, position);
}
