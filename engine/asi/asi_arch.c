#include "asi_arch.h"
#include "asi_pool.h"
#include "q4k.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double asi_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

typedef enum {
    QWEN35_LAYER_DELTANET = 0,
    QWEN35_LAYER_FULL_ATTN = 1
} Qwen35LayerKind;

typedef struct {
    Qwen35LayerKind kind;
    const AsiV2TensorEntry *attn_norm;
    const AsiV2TensorEntry *post_attn_norm;
    const AsiV2TensorEntry *ffn_gate;
    const AsiV2TensorEntry *ffn_up;
    const AsiV2TensorEntry *ffn_down;

    /* Gated DeltaNet (SSM) tensors */
    const AsiV2TensorEntry *attn_qkv;
    const AsiV2TensorEntry *attn_gate;
    const AsiV2TensorEntry *ssm_a;
    const AsiV2TensorEntry *ssm_alpha;
    const AsiV2TensorEntry *ssm_beta;
    const AsiV2TensorEntry *ssm_conv1d;
    const AsiV2TensorEntry *ssm_dt_bias;
    const AsiV2TensorEntry *ssm_norm;
    const AsiV2TensorEntry *ssm_out;

    /* Full Gated Attention tensors */
    const AsiV2TensorEntry *attn_q;
    const AsiV2TensorEntry *attn_k;
    const AsiV2TensorEntry *attn_v;
    const AsiV2TensorEntry *attn_q_norm;
    const AsiV2TensorEntry *attn_k_norm;
    const AsiV2TensorEntry *attn_output;
} Qwen35LayerWeights;

typedef struct {
    float *conv_state; /* [conv_kernel - 1, qkv_dim] */
    float *ssm_state;  /* [ssm_group_count, ssm_head_dim, ssm_state_size] */
} Qwen35DeltaNetState;

typedef struct {
    uint16_t *k_cache; /* [max_ctx, n_kv_heads * head_dim] FP16 */
    uint16_t *v_cache; /* [max_ctx, n_kv_heads * head_dim] FP16 */
} Qwen35AttnKVState;

typedef struct {
    float *x;
    float *x_norm;
    float *qkv;
    float *z;
    float *qg;
    float *k;
    float *v;
    float *attn_out;
    float *proj_out;
    float *ffn_gate;
    float *ffn_up;
    float *attn_scores;
    float *alpha;
    float *beta;
    float *v_pred;
    /* Persistent batch scratch buffers (grown up to chunk_max) */
    uint32_t batch_cap;
    float *xb;
    float *xnb;
    float *projb;
    float *qkvb;
    float *zb;
    float *ab;
    float *bb;
    float *qgb;
    float *kb;
    float *vb;
    float *aob;
    float *fgb;
    float *fub;
    LilaQ8KBlock *qx_h;
} Qwen35Scratch;

struct AsiArchSession {
    const AsiV2Cartridge *cart;
    uint32_t max_ctx;
    uint32_t n_threads;
    const AsiV2TensorEntry *token_embd;
    const AsiV2TensorEntry *output_norm;
    const AsiV2TensorEntry *output_weight; /* NULL if tied to token_embd */
    Qwen35LayerWeights *layers;
    Qwen35DeltaNetState *ssm_states;
    Qwen35AttnKVState *kv_states;
    Qwen35Scratch scratch;
    AsiThreadPool *pool;
    uint32_t prefix_len;
    float *prefix_conv;
    float *prefix_ssm;
    uint16_t *prefix_k;
    uint16_t *prefix_v;
    float *rope_inv_freq;
};

#if defined(__AVX2__)
static inline __m256 lila_expf_avx2(__m256 x)
{
    x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0f)), _mm256_set1_ps(-88.0f));
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(x, log2e, half));
    x = _mm256_fnmadd_ps(n, ln2_hi, x);
    x = _mm256_fnmadd_ps(n, ln2_lo, x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i emm0 = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
}
#endif

static inline float fast_silu(float x)
{
    return x / (1.0f + expf(-x));
}

static inline void fast_silu_mul_vec(float *gate, const float *up, uint32_t n)
{
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_set1_ps(-0.0f);
    for (; i + 7 < n; i += 8)
    {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        __m256 e = lila_expf_avx2(_mm256_xor_ps(g, neg));
        __m256 sig = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(gate + i, _mm256_mul_ps(sig, u));
    }
#endif
    for (; i < n; ++i)
    {
        gate[i] = fast_silu(gate[i]) * up[i];
    }
}

static inline void fast_silu_inplace_vec(float *x, uint32_t n)
{
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_set1_ps(-0.0f);
    for (; i + 7 < n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 e = lila_expf_avx2(_mm256_xor_ps(v, neg));
        _mm256_storeu_ps(x + i, _mm256_div_ps(v, _mm256_add_ps(one, e)));
    }
#endif
    for (; i < n; ++i)
    {
        x[i] = fast_silu(x[i]);
    }
}

static inline void fast_silu_mul_out_vec(float *out, const float *gate, uint32_t n)
{
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_set1_ps(-0.0f);
    for (; i + 7 < n; i += 8)
    {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 o = _mm256_loadu_ps(out + i);
        __m256 e = lila_expf_avx2(_mm256_xor_ps(g, neg));
        __m256 sig = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(o, sig));
    }
#endif
    for (; i < n; ++i)
    {
        out[i] *= fast_silu(gate[i]);
    }
}

static inline float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float fast_softplus(float x)
{
    return (x > 20.0f) ? x : log1pf(expf(x));
}

static void rmsnorm(float *out, const float *x, const float *w, uint32_t n, float eps)
{
    float ss = 0.0f;
#if defined(__AVX512F__)
    if ((n & 15u) == 0u)
    {
        __m512 vss = _mm512_setzero_ps();
        for (uint32_t i = 0; i < n; i += 16)
        {
            __m512 vx = _mm512_loadu_ps(x + i);
            vss = _mm512_fmadd_ps(vx, vx, vss);
        }
        ss = _mm512_reduce_add_ps(vss);
        float inv = 1.0f / sqrtf(ss / (float)n + eps);
        __m512 vinv = _mm512_set1_ps(inv);
        for (uint32_t i = 0; i < n; i += 16)
        {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vw = _mm512_loadu_ps(w + i);
            _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(vx, vinv), vw));
        }
        return;
    }
#elif defined(__AVX2__)
    if ((n & 7u) == 0u)
    {
        __m256 vss = _mm256_setzero_ps();
        for (uint32_t i = 0; i < n; i += 8)
        {
            __m256 vx = _mm256_loadu_ps(x + i);
            vss = _mm256_fmadd_ps(vx, vx, vss);
        }
        ss = lila_hsum_f32_avx2(vss);
        float inv = 1.0f / sqrtf(ss / (float)n + eps);
        __m256 vinv = _mm256_set1_ps(inv);
        for (uint32_t i = 0; i < n; i += 8)
        {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(w + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
        }
        return;
    }
#elif defined(__aarch64__)
    if ((n & 3u) == 0u)
    {
        float32x4_t vss = vdupq_n_f32(0.0f);
        for (uint32_t i = 0; i < n; i += 4)
        {
            float32x4_t vx = vld1q_f32(x + i);
            vss = vfmaq_f32(vss, vx, vx);
        }
        ss = vaddvq_f32(vss);
        float inv = 1.0f / sqrtf(ss / (float)n + eps);
        float32x4_t vinv = vdupq_n_f32(inv);
        for (uint32_t i = 0; i < n; i += 4)
        {
            float32x4_t vx = vld1q_f32(x + i);
            float32x4_t vw = vld1q_f32(w + i);
            vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, vinv), vw));
        }
        return;
    }
#endif
    for (uint32_t i = 0; i < n; ++i)
    {
        ss += x[i] * x[i];
    }
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; ++i)
    {
        out[i] = (x[i] * inv) * w[i];
    }
}

static inline const uint8_t *tensor_data(const AsiV2Cartridge *cart, const AsiV2TensorEntry *t)
{
    return cart->weights_data + t->data_offset;
}

typedef struct {
    const uint8_t *w;
    const LilaQ8KBlock *qx;
    float *out;
    int rows;
    int blocks_per_row;
    uint32_t quant_type;
    atomic_int next_row;
    int chunk_rows;
} AsiMatvecCtx;

static void asi_matvec_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiMatvecCtx *ctx = (AsiMatvecCtx *)vctx;
    int bpr = ctx->blocks_per_row;
    int rows = ctx->rows;
    int chunk = ctx->chunk_rows;

    if (n_threads <= 1)
    {
        if (ctx->quant_type == 12)
        {
            size_t rb = (size_t)bpr * Q4_K_BLOCK_SIZE;
            int i = 0;
            for (; i + 1 < rows; i += 2)
            {
                const uint8_t *rp = ctx->w + (size_t)i * rb;
                if (i + 3 < rows) { LILA_PREFETCH(rp + 2 * rb); LILA_PREFETCH(rp + 3 * rb); }
                q4k_dot2_q8k_avx2(rp, rp + rb, ctx->qx, bpr, &ctx->out[i], &ctx->out[i + 1]);
            }
            for (; i < rows; ++i)
                ctx->out[i] = q4k_dot_q8k_avx2(ctx->w + (size_t)i * rb, ctx->qx, bpr);
        }
        else if (ctx->quant_type == 14)
        {
            size_t rb = (size_t)bpr * Q6_K_BLOCK_SIZE;
            int i = 0;
            for (; i + 1 < rows; i += 2)
            {
                const uint8_t *rp = ctx->w + (size_t)i * rb;
                if (i + 3 < rows) { LILA_PREFETCH(rp + 2 * rb); LILA_PREFETCH(rp + 3 * rb); }
                q6k_dot2_q8k_avx2(rp, rp + rb, ctx->qx, bpr, &ctx->out[i], &ctx->out[i + 1]);
            }
            for (; i < rows; ++i)
                ctx->out[i] = q6k_dot_q8k_avx2(ctx->w + (size_t)i * rb, ctx->qx, bpr);
        }
        return;
    }

    for (;;)
    {
        int start = atomic_fetch_add_explicit(&ctx->next_row, chunk, memory_order_relaxed);
        if (start >= rows) break;
        int end = start + chunk;
        if (end > rows) end = rows;

        if (ctx->quant_type == 12)
        {
            size_t rb = (size_t)bpr * Q4_K_BLOCK_SIZE;
            for (int i = start; i < end; ++i)
            {
                const uint8_t *rp = ctx->w + (size_t)i * rb;
                if (i + 2 < end) LILA_PREFETCH(rp + 2 * rb);
                ctx->out[i] = q4k_dot_q8k_simd(rp, ctx->qx, bpr);
            }
        }
        else if (ctx->quant_type == 14)
        {
            size_t rb = (size_t)bpr * Q6_K_BLOCK_SIZE;
            for (int i = start; i < end; ++i)
            {
                const uint8_t *rp = ctx->w + (size_t)i * rb;
                if (i + 2 < end)
                {
                    LILA_PREFETCH(rp + 2 * rb);
                    LILA_PREFETCH(rp + 2 * rb + 64);
                }
                ctx->out[i] = q6k_dot_q8k_simd(rp, ctx->qx, bpr);
            }
        }
    }
}

static int matvec_q8k_pool(AsiThreadPool *pool,
                           const AsiV2Cartridge *cart,
                           const AsiV2TensorEntry *t,
                           float *out,
                           const LilaQ8KBlock *qx,
                           int bpr)
{
    const uint8_t *w = tensor_data(cart, t);
    int rows = (int)t->shape[1];
    int chunk = (rows >= 16384) ? 512 : 64;
    AsiMatvecCtx ctx;
    ctx.w = w;
    ctx.qx = qx;
    ctx.out = out;
    ctx.rows = rows;
    ctx.blocks_per_row = bpr;
    ctx.quant_type = t->quant_type;
    atomic_init(&ctx.next_row, 0);
    ctx.chunk_rows = chunk;
    if (pool && rows >= 64)
        asi_pool_run(pool, asi_matvec_worker, &ctx);
    else
        asi_matvec_worker(0, 1, &ctx);
    return 0;
}

typedef struct {
    const uint8_t *w[4];
    float *out[4];
    int rows[4];
    uint32_t quant_type[4];
    int n_mats;
    const LilaQ8KBlock *qx;
    int blocks_per_row;
    int apply_silu_mul;
} AsiFusedMatvecCtx;

static void asi_matvec_range(const uint8_t *w, const LilaQ8KBlock *qx, float *out,
                             int start, int end, int bpr, uint32_t quant_type)
{
    if (quant_type == 12)
    {
        size_t rb = (size_t)bpr * Q4_K_BLOCK_SIZE;
        for (int i = start; i < end; ++i)
        {
            const uint8_t *rp = w + (size_t)i * rb;
            if (i + 2 < end) LILA_PREFETCH(rp + 2 * rb);
            out[i] = q4k_dot_q8k_simd(rp, qx, bpr);
        }
    }
    else if (quant_type == 14)
    {
        size_t rb = (size_t)bpr * Q6_K_BLOCK_SIZE;
        for (int i = start; i < end; ++i)
        {
            const uint8_t *rp = w + (size_t)i * rb;
            if (i + 2 < end)
            {
                LILA_PREFETCH(rp + 2 * rb);
                LILA_PREFETCH(rp + 2 * rb + 64);
            }
            out[i] = q6k_dot_q8k_simd(rp, qx, bpr);
        }
    }
}

static void asi_fused_matvec_worker(int tid, int n_threads, void *vctx)
{
    const AsiFusedMatvecCtx *ctx = (const AsiFusedMatvecCtx *)vctx;
    if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
    {
        int total = ctx->rows[0];
        int start = (total * tid) / n_threads;
        int end = (total * (tid + 1)) / n_threads;
        asi_matvec_range(ctx->w[0], ctx->qx, ctx->out[0], start, end, ctx->blocks_per_row, ctx->quant_type[0]);
        asi_matvec_range(ctx->w[1], ctx->qx, ctx->out[1], start, end, ctx->blocks_per_row, ctx->quant_type[1]);
        fast_silu_mul_vec(ctx->out[0] + start, ctx->out[1] + start, (uint32_t)(end - start));
        return;
    }
    for (int m = 0; m < ctx->n_mats; ++m)
    {
        int total = ctx->rows[m];
        int start = (total * tid) / n_threads;
        int end = (total * (tid + 1)) / n_threads;
        if (start < end)
            asi_matvec_range(ctx->w[m], ctx->qx, ctx->out[m], start, end, ctx->blocks_per_row, ctx->quant_type[m]);
    }
}

static void matvec_fused_q8k_pool(AsiThreadPool *pool,
                                  const AsiV2Cartridge *cart,
                                  const AsiV2TensorEntry **tensors,
                                  float **outs,
                                  int n_mats,
                                  const LilaQ8KBlock *qx,
                                  int bpr,
                                  int apply_silu_mul)
{
    AsiFusedMatvecCtx ctx;
    ctx.n_mats = n_mats;
    ctx.qx = qx;
    ctx.blocks_per_row = bpr;
    ctx.apply_silu_mul = apply_silu_mul;
    int max_rows = 0;
    for (int m = 0; m < n_mats; ++m)
    {
        ctx.w[m] = tensor_data(cart, tensors[m]);
        ctx.out[m] = outs[m];
        ctx.rows[m] = (int)tensors[m]->shape[1];
        ctx.quant_type[m] = tensors[m]->quant_type;
        if (ctx.rows[m] > max_rows) max_rows = ctx.rows[m];
    }
    if (pool && max_rows >= 64)
        asi_pool_run(pool, asi_fused_matvec_worker, &ctx);
    else
        asi_fused_matvec_worker(0, 1, &ctx);
}

static int matvec_tensor_pool(AsiThreadPool *pool,
                              const AsiV2Cartridge *cart,
                              const AsiV2TensorEntry *t,
                              float *out,
                              const float *in)
{
    const uint8_t *w = tensor_data(cart, t);
    int cols = (int)t->shape[0];
    int rows = (int)t->shape[1];
    if ((t->quant_type == 12 || t->quant_type == 14) && !(t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
    {
        int bpr = (cols + QK_K - 1) / QK_K;
        if (bpr <= 0) return 0;
        LilaQ8KBlock stack_qx[64];
        LilaQ8KBlock *qx = (bpr <= 64) ? stack_qx : (LilaQ8KBlock *)malloc((size_t)bpr * sizeof(LilaQ8KBlock));
        if (qx)
        {
            quantize_vec_q8k(qx, in, cols);
            matvec_q8k_pool(pool, cart, t, out, qx, bpr);
            if (qx != stack_qx) free(qx);
            return 0;
        }
    }
    if (t->quant_type == 12) /* Q4_K */
    {
        q4k_matvec(out, w, in, rows, cols);
        return 0;
    }
    if (t->quant_type == 14) /* Q6_K */
    {
        if (t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK)
            q6k_repacked_matvec(out, w, in, rows, cols);
        else
            q6k_matvec(out, w, in, rows, cols);
        return 0;
    }
    return -1;
}

typedef struct {
    const uint8_t *w;
    const LilaQ8KBlock *qx_all;
    float *out;
    int n_tokens;
    int rows;
    int blocks_per_row;
    uint32_t quant_type;
} AsiMatmulBatchCtx;

static void asi_matmul_batch_worker(int tid, int n_threads, void *vctx)
{
    const AsiMatmulBatchCtx *ctx = (const AsiMatmulBatchCtx *)vctx;
    int start = (ctx->rows * tid) / n_threads;
    int end = (ctx->rows * (tid + 1)) / n_threads;
    if (ctx->quant_type == 12)
    {
        q4k_matmul_batch_q8k_range(ctx->out, ctx->w, ctx->qx_all,
                                   ctx->n_tokens, start, end, ctx->rows, ctx->blocks_per_row);
    }
    else if (ctx->quant_type == 14)
    {
        q6k_matmul_batch_q8k_range(ctx->out, ctx->w, ctx->qx_all,
                                   ctx->n_tokens, start, end, ctx->rows, ctx->blocks_per_row);
    }
}

static int matmul_batch_q8k_pool(AsiThreadPool *pool,
                                 const AsiV2Cartridge *cart,
                                 const AsiV2TensorEntry *t,
                                 float *out,
                                 const LilaQ8KBlock *qx_all,
                                 int n_tokens,
                                 int bpr)
{
    const uint8_t *w = tensor_data(cart, t);
    int rows = (int)t->shape[1];
    AsiMatmulBatchCtx ctx = {w, qx_all, out, n_tokens, rows, bpr, t->quant_type};
    if (pool && rows >= 16)
        asi_pool_run(pool, asi_matmul_batch_worker, &ctx);
    else
        asi_matmul_batch_worker(0, 1, &ctx);
    return 0;
}

static int matmul_batch_tensor(AsiThreadPool *pool,
                               const AsiV2Cartridge *cart,
                               const AsiV2TensorEntry *t,
                               float *out,
                               const float *in,
                               int n_tokens)
{
    const uint8_t *w = tensor_data(cart, t);
    int cols = (int)t->shape[0];
    int rows = (int)t->shape[1];
    if (n_tokens <= 1)
    {
        return matvec_tensor_pool(pool, cart, t, out, in);
    }
    if ((t->quant_type == 12 || t->quant_type == 14) && !(t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
    {
        int bpr = (cols + QK_K - 1) / QK_K;
        size_t need = (size_t)n_tokens * (size_t)bpr;
        LilaQ8KBlock stack_qx[256];
        LilaQ8KBlock *qx_all = (need <= 256) ? stack_qx : (LilaQ8KBlock *)malloc(need * sizeof(LilaQ8KBlock));
        if (qx_all)
        {
            for (int tok = 0; tok < n_tokens; ++tok)
            {
                quantize_vec_q8k_strided(qx_all + tok, n_tokens, in + (size_t)tok * cols, cols);
            }
            matmul_batch_q8k_pool(pool, cart, t, out, qx_all, n_tokens, bpr);
            if (qx_all != stack_qx) free(qx_all);
            return 0;
        }
    }
    if (t->quant_type == 12) /* Q4_K */
    {
        q4k_matmul_batch(out, w, in, n_tokens, rows, cols);
        return 0;
    }
    if (t->quant_type == 14) /* Q6_K */
    {
        if (t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK)
            q6k_repacked_matmul_batch(out, w, in, n_tokens, rows, cols);
        else
            q6k_matmul_batch(out, w, in, n_tokens, rows, cols);
        return 0;
    }
    return -1;
}

typedef struct {
    LilaQ8KBlock *qx_all;
    const float *in;
    int n_tokens;
    int cols;
} AsiQuantBatchCtx;

static void asi_quantize_q8k_worker(int tid, int n_threads, void *vctx)
{
    const AsiQuantBatchCtx *ctx = (const AsiQuantBatchCtx *)vctx;
    int start = (ctx->n_tokens * tid) / n_threads;
    int end = (ctx->n_tokens * (tid + 1)) / n_threads;
    for (int tok = start; tok < end; ++tok)
    {
        quantize_vec_q8k_strided(ctx->qx_all + tok, ctx->n_tokens,
                                 ctx->in + (size_t)tok * ctx->cols, ctx->cols);
    }
}

static void quantize_batch_q8k_pool(AsiThreadPool *pool,
                                    LilaQ8KBlock *qx_all,
                                    const float *in,
                                    int n_tokens,
                                    int cols)
{
    AsiQuantBatchCtx ctx = {qx_all, in, n_tokens, cols};
    if (pool && n_tokens >= 16)
        asi_pool_run(pool, asi_quantize_q8k_worker, &ctx);
    else
        asi_quantize_q8k_worker(0, 1, &ctx);
}

static int matmul_batch_tensor_qx(AsiThreadPool *pool,
                                  const AsiV2Cartridge *cart,
                                  const AsiV2TensorEntry *t,
                                  float *out,
                                  const LilaQ8KBlock *qx_all,
                                  const float *in,
                                  int n_tokens,
                                  int bpr)
{
    if (qx_all && n_tokens > 1 && (t->quant_type == 12 || t->quant_type == 14) &&
        !(t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
    {
        return matmul_batch_q8k_pool(pool, cart, t, out, qx_all, n_tokens, bpr);
    }
    return matmul_batch_tensor(pool, cart, t, out, in, n_tokens);
}

typedef struct {
    const uint8_t *w[4];
    float *out[4];
    int rows[4];
    uint32_t quant_type[4];
    int n_mats;
    const LilaQ8KBlock *qx_all;
    int n_tokens;
    int blocks_per_row;
    int apply_silu_mul;
} AsiFusedMatmulBatchCtx;

static void asi_fused_matmul_batch_worker(int tid, int n_threads, void *vctx)
{
    const AsiFusedMatmulBatchCtx *ctx = (const AsiFusedMatmulBatchCtx *)vctx;
    for (int m = 0; m < ctx->n_mats; ++m)
    {
        int rows = ctx->rows[m];
        int start = (rows * tid) / n_threads;
        int end = (rows * (tid + 1)) / n_threads;
        if (start >= end) continue;
        if (ctx->quant_type[m] == 12)
        {
            q4k_matmul_batch_q8k_range(ctx->out[m], ctx->w[m], ctx->qx_all,
                                       ctx->n_tokens, start, end, rows, ctx->blocks_per_row);
        }
        else if (ctx->quant_type[m] == 14)
        {
            q6k_matmul_batch_q8k_range(ctx->out[m], ctx->w[m], ctx->qx_all,
                                       ctx->n_tokens, start, end, rows, ctx->blocks_per_row);
        }
    }
    if (ctx->apply_silu_mul && ctx->n_mats >= 2)
    {
        int rows = ctx->rows[0];
        int start = (rows * tid) / n_threads;
        int end = (rows * (tid + 1)) / n_threads;
        int count = end - start;
        if (count > 0)
        {
            for (int t = 0; t < ctx->n_tokens; ++t)
            {
                size_t off = (size_t)t * (size_t)rows + (size_t)start;
                fast_silu_mul_vec(ctx->out[0] + off, ctx->out[1] + off, (uint32_t)count);
            }
        }
    }
}

static int matmul_batch_fused_q8k_pool(AsiThreadPool *pool,
                                       const AsiV2Cartridge *cart,
                                       const AsiV2TensorEntry **ts,
                                       float **outs,
                                       int n_mats,
                                       const LilaQ8KBlock *qx_all,
                                       const float *in,
                                       int n_tokens,
                                       int bpr,
                                       int apply_silu_mul)
{
    if (n_tokens <= 1 || !qx_all)
    {
        for (int m = 0; m < n_mats; ++m)
            matmul_batch_tensor(pool, cart, ts[m], outs[m], in, n_tokens);
        if (apply_silu_mul && n_mats >= 2)
            fast_silu_mul_vec(outs[0], outs[1], (uint32_t)((size_t)n_tokens * (size_t)ts[0]->shape[1]));
        return 0;
    }
    for (int m = 0; m < n_mats; ++m)
    {
        if ((ts[m]->quant_type != 12 && ts[m]->quant_type != 14) ||
            (ts[m]->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
        {
            for (int k = 0; k < n_mats; ++k)
                matmul_batch_tensor_qx(pool, cart, ts[k], outs[k], qx_all, in, n_tokens, bpr);
            if (apply_silu_mul && n_mats >= 2)
                fast_silu_mul_vec(outs[0], outs[1], (uint32_t)((size_t)n_tokens * (size_t)ts[0]->shape[1]));
            return 0;
        }
    }
    AsiFusedMatmulBatchCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_mats = n_mats;
    ctx.qx_all = qx_all;
    ctx.n_tokens = n_tokens;
    ctx.blocks_per_row = bpr;
    ctx.apply_silu_mul = apply_silu_mul;
    for (int m = 0; m < n_mats; ++m)
    {
        ctx.w[m] = tensor_data(cart, ts[m]);
        ctx.out[m] = outs[m];
        ctx.rows[m] = (int)ts[m]->shape[1];
        ctx.quant_type[m] = ts[m]->quant_type;
    }
    if (pool)
        asi_pool_run(pool, asi_fused_matmul_batch_worker, &ctx);
    else
        asi_fused_matmul_batch_worker(0, 1, &ctx);
    return 0;
}

static inline float dot_f32_f16_avx2(const float *q, const uint16_t *k_fp16, uint32_t hd)
{
#if defined(__AVX2__) && defined(__F16C__)
    if ((hd & 7u) == 0u)
    {
        __m256 vacc = _mm256_setzero_ps();
        for (uint32_t d = 0; d < hd; d += 8)
        {
            __m128i kh = _mm_loadu_si128((const __m128i *)(k_fp16 + d));
            __m256 kf = _mm256_cvtph_ps(kh);
            __m256 qf = _mm256_loadu_ps(q + d);
            vacc = _mm256_fmadd_ps(qf, kf, vacc);
        }
        return lila_hsum_f32_avx2(vacc);
    }
#endif
    float dot = 0.0f;
    for (uint32_t d = 0; d < hd; ++d) dot += q[d] * fp16_to_fp32(k_fp16[d]);
    return dot;
}

static inline void accum_f16_f32_avx2(float *oh, const uint16_t *v_fp16, float prob, uint32_t hd)
{
#if defined(__AVX2__) && defined(__F16C__)
    if ((hd & 7u) == 0u)
    {
        __m256 vp = _mm256_set1_ps(prob);
        for (uint32_t d = 0; d < hd; d += 8)
        {
            __m128i vh = _mm_loadu_si128((const __m128i *)(v_fp16 + d));
            __m256 vf = _mm256_cvtph_ps(vh);
            __m256 ov = _mm256_loadu_ps(oh + d);
            _mm256_storeu_ps(oh + d, _mm256_fmadd_ps(vp, vf, ov));
        }
        return;
    }
#endif
    for (uint32_t d = 0; d < hd; ++d) oh[d] += prob * fp16_to_fp32(v_fp16[d]);
}

static void apply_rope_neox_partial(float *vec, uint32_t rope_dim, uint32_t pos, float theta, const float *inv_freq)
{
    uint32_t half = rope_dim / 2;
    for (uint32_t i = 0; i < half; ++i)
    {
        float freq = inv_freq ? inv_freq[i] : (1.0f / powf(theta, (float)(2 * i) / (float)rope_dim));
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = vec[i];
        float x1 = vec[i + half];
        vec[i] = x0 * c - x1 * s;
        vec[i + half] = x0 * s + x1 * c;
    }
}

static const AsiV2TensorEntry *req_tensor(const AsiV2Cartridge *cart,
                                          const char *name,
                                          char *err_buf,
                                          size_t err_cap)
{
    const AsiV2TensorEntry *t = asi_v2_find_tensor(cart, name);
    if (!t)
    {
        if (err_buf && err_cap > 0)
        {
            snprintf(err_buf, err_cap, "missing required tensor '%s'", name);
        }
        return NULL;
    }
    return t;
}

static void qwen35_free_session(AsiArchSession *s)
{
    if (!s) return;
    asi_pool_destroy(s->pool);
    free(s->prefix_conv);
    free(s->prefix_ssm);
    free(s->prefix_k);
    free(s->prefix_v);
    free(s->rope_inv_freq);
    if (s->layers)
    {
        for (uint32_t i = 0; i < s->cart->config.n_layers; ++i)
        {
            if (s->ssm_states)
            {
                free(s->ssm_states[i].conv_state);
                free(s->ssm_states[i].ssm_state);
            }
            if (s->kv_states)
            {
                free(s->kv_states[i].k_cache);
                free(s->kv_states[i].v_cache);
            }
        }
    }
    free(s->ssm_states);
    free(s->kv_states);
    free(s->layers);
    free(s->scratch.x);
    free(s->scratch.x_norm);
    free(s->scratch.qkv);
    free(s->scratch.z);
    free(s->scratch.qg);
    free(s->scratch.k);
    free(s->scratch.v);
    free(s->scratch.attn_out);
    free(s->scratch.proj_out);
    free(s->scratch.ffn_gate);
    free(s->scratch.ffn_up);
    free(s->scratch.attn_scores);
    free(s->scratch.alpha);
    free(s->scratch.beta);
    free(s->scratch.v_pred);
    free(s->scratch.xb);
    free(s->scratch.xnb);
    free(s->scratch.projb);
    free(s->scratch.qkvb);
    free(s->scratch.zb);
    free(s->scratch.ab);
    free(s->scratch.bb);
    free(s->scratch.qgb);
    free(s->scratch.kb);
    free(s->scratch.vb);
    free(s->scratch.aob);
    free(s->scratch.fgb);
    free(s->scratch.fub);
    free(s->scratch.qx_h);
    free(s);
}

static void qwen35_reset_state(AsiArchSession *s)
{
    if (!s) return;
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t conv_hist = (c->ssm_conv_kernel > 1) ? (c->ssm_conv_kernel - 1) : 0;
    uint32_t ssm_head_dim = (c->ssm_group_count > 0) ? (c->ssm_inner_size / c->ssm_group_count) : 0;
    size_t kv_elems = (size_t)s->max_ctx * c->n_kv_heads * c->head_dim;

    for (uint32_t i = 0; i < c->n_layers; ++i)
    {
        if (s->layers[i].kind == QWEN35_LAYER_DELTANET)
        {
            if (s->ssm_states[i].conv_state && conv_hist > 0)
            {
                memset(s->ssm_states[i].conv_state, 0, (size_t)conv_hist * qkv_dim * sizeof(float));
            }
            if (s->ssm_states[i].ssm_state)
            {
                memset(s->ssm_states[i].ssm_state, 0,
                       (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim * sizeof(float));
            }
        }
        else
        {
            if (s->kv_states[i].k_cache)
            {
                memset(s->kv_states[i].k_cache, 0, kv_elems * sizeof(uint16_t));
            }
            if (s->kv_states[i].v_cache)
            {
                memset(s->kv_states[i].v_cache, 0, kv_elems * sizeof(uint16_t));
            }
        }
    }
}

static int qwen35_init_session(const AsiV2Cartridge *cart,
                               uint32_t max_ctx,
                               uint32_t n_threads,
                               AsiArchSession **out_session,
                               char *err_buf,
                               size_t err_cap)
{
    if (!cart || !out_session) return -1;
    const AsiV2ModelConfig *c = &cart->config;
    if (c->arch_id != ASI_ARCH_HYBRID_QWEN35 || c->n_layers == 0 || c->full_attention_interval == 0)
    {
        snprintf(err_buf, err_cap, "invalid hybrid_qwen35 config");
        return -1;
    }

    AsiArchSession *s = (AsiArchSession *)calloc(1, sizeof(AsiArchSession));
    if (!s) return -1;
    s->cart = cart;
    s->max_ctx = max_ctx > 0 ? max_ctx : 2048;
    s->n_threads = n_threads > 0 ? n_threads : 1;
    s->pool = asi_pool_create((int)s->n_threads);

    s->token_embd = req_tensor(cart, "token_embd.weight", err_buf, err_cap);
    s->output_norm = req_tensor(cart, "output_norm.weight", err_buf, err_cap);
    if (!s->token_embd || !s->output_norm)
    {
        qwen35_free_session(s);
        return -1;
    }
    s->output_weight = asi_v2_find_tensor(cart, "output.weight");

    s->layers = (Qwen35LayerWeights *)calloc(c->n_layers, sizeof(Qwen35LayerWeights));
    s->ssm_states = (Qwen35DeltaNetState *)calloc(c->n_layers, sizeof(Qwen35DeltaNetState));
    s->kv_states = (Qwen35AttnKVState *)calloc(c->n_layers, sizeof(Qwen35AttnKVState));
    if (!s->layers || !s->ssm_states || !s->kv_states)
    {
        qwen35_free_session(s);
        return -1;
    }

    uint32_t qkv_dim = c->ssm_inner_size * 3;
    s->scratch.x = (float *)calloc(c->hidden_size, sizeof(float));
    s->scratch.x_norm = (float *)calloc(c->hidden_size, sizeof(float));
    s->scratch.qkv = (float *)calloc(qkv_dim, sizeof(float));
    s->scratch.z = (float *)calloc(c->ssm_inner_size, sizeof(float));
    s->scratch.qg = (float *)calloc((size_t)c->n_heads * 2 * c->head_dim, sizeof(float));
    s->scratch.k = (float *)calloc((size_t)c->n_kv_heads * c->head_dim, sizeof(float));
    s->scratch.v = (float *)calloc((size_t)c->n_kv_heads * c->head_dim, sizeof(float));
    s->scratch.attn_out = (float *)calloc((size_t)c->n_heads * c->head_dim, sizeof(float));
    s->scratch.proj_out = (float *)calloc(c->hidden_size, sizeof(float));
    s->scratch.ffn_gate = (float *)calloc(c->intermediate_size, sizeof(float));
    s->scratch.ffn_up = (float *)calloc(c->intermediate_size, sizeof(float));
    s->scratch.attn_scores = (float *)calloc(s->max_ctx, sizeof(float));
    s->scratch.alpha = (float *)calloc(c->ssm_time_step_rank, sizeof(float));
    s->scratch.beta = (float *)calloc(c->ssm_time_step_rank, sizeof(float));
    s->scratch.v_pred = (float *)calloc(c->ssm_state_size, sizeof(float));
    if (c->rope_dim >= 2)
    {
        uint32_t half = c->rope_dim / 2;
        s->rope_inv_freq = (float *)malloc((size_t)half * sizeof(float));
        if (s->rope_inv_freq)
        {
            for (uint32_t i = 0; i < half; ++i)
                s->rope_inv_freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)c->rope_dim);
        }
    }

    if (!s->scratch.x || !s->scratch.x_norm || !s->scratch.qkv || !s->scratch.z ||
        !s->scratch.qg || !s->scratch.k || !s->scratch.v || !s->scratch.attn_out ||
        !s->scratch.proj_out || !s->scratch.ffn_gate || !s->scratch.ffn_up ||
        !s->scratch.attn_scores || !s->scratch.alpha || !s->scratch.beta || !s->scratch.v_pred)
    {
        qwen35_free_session(s);
        return -1;
    }

    char name[64];
    for (uint32_t i = 0; i < c->n_layers; ++i)
    {
        Qwen35LayerWeights *lw = &s->layers[i];
        int is_full = ((i + 1) % c->full_attention_interval == 0);
        lw->kind = is_full ? QWEN35_LAYER_FULL_ATTN : QWEN35_LAYER_DELTANET;

        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", i);
        lw->attn_norm = req_tensor(cart, name, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.post_attention_norm.weight", i);
        lw->post_attn_norm = req_tensor(cart, name, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", i);
        lw->ffn_gate = req_tensor(cart, name, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", i);
        lw->ffn_up = req_tensor(cart, name, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", i);
        lw->ffn_down = req_tensor(cart, name, err_buf, err_cap);

        if (!lw->attn_norm || !lw->post_attn_norm || !lw->ffn_gate || !lw->ffn_up || !lw->ffn_down)
        {
            qwen35_free_session(s);
            return -1;
        }

        if (lw->kind == QWEN35_LAYER_DELTANET)
        {
            snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", i);
            lw->attn_qkv = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_gate.weight", i);
            lw->attn_gate = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_a", i);
            lw->ssm_a = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_alpha.weight", i);
            lw->ssm_alpha = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_beta.weight", i);
            lw->ssm_beta = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", i);
            lw->ssm_conv1d = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", i);
            lw->ssm_dt_bias = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_norm.weight", i);
            lw->ssm_norm = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_out.weight", i);
            lw->ssm_out = req_tensor(cart, name, err_buf, err_cap);

            if (!lw->attn_qkv || !lw->attn_gate || !lw->ssm_a || !lw->ssm_alpha ||
                !lw->ssm_beta || !lw->ssm_conv1d || !lw->ssm_dt_bias || !lw->ssm_norm || !lw->ssm_out)
            {
                qwen35_free_session(s);
                return -1;
            }

            uint32_t conv_hist = (c->ssm_conv_kernel > 1) ? (c->ssm_conv_kernel - 1) : 0;
            uint32_t ssm_head_dim = (c->ssm_group_count > 0) ? (c->ssm_inner_size / c->ssm_group_count) : 0;
            if (conv_hist > 0)
            {
                s->ssm_states[i].conv_state = (float *)calloc((size_t)conv_hist * qkv_dim, sizeof(float));
            }
            s->ssm_states[i].ssm_state = (float *)calloc(
                (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim, sizeof(float));
            if ((conv_hist > 0 && !s->ssm_states[i].conv_state) || !s->ssm_states[i].ssm_state)
            {
                qwen35_free_session(s);
                return -1;
            }
        }
        else
        {
            snprintf(name, sizeof(name), "blk.%u.attn_q.weight", i);
            lw->attn_q = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_k.weight", i);
            lw->attn_k = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_v.weight", i);
            lw->attn_v = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", i);
            lw->attn_q_norm = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", i);
            lw->attn_k_norm = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_output.weight", i);
            lw->attn_output = req_tensor(cart, name, err_buf, err_cap);

            if (!lw->attn_q || !lw->attn_k || !lw->attn_v ||
                !lw->attn_q_norm || !lw->attn_k_norm || !lw->attn_output)
            {
                qwen35_free_session(s);
                return -1;
            }

            size_t kv_elems = (size_t)s->max_ctx * c->n_kv_heads * c->head_dim;
            s->kv_states[i].k_cache = (uint16_t *)calloc(kv_elems, sizeof(uint16_t));
            s->kv_states[i].v_cache = (uint16_t *)calloc(kv_elems, sizeof(uint16_t));
            if (!s->kv_states[i].k_cache || !s->kv_states[i].v_cache)
            {
                qwen35_free_session(s);
                return -1;
            }
        }
    }

    *out_session = s;
    return 0;
}

static inline void deltanet_head_step_avx2(float *S,
                                           float *qh,
                                           float *kh,
                                           const float *vh,
                                           const float *zh,
                                           float *oh,
                                           const float *ssm_norm_w,
                                           float alpha_val,
                                           float dt_bias_val,
                                           float ssm_a_val,
                                           float beta_raw_val,
                                           float q_scale,
                                           uint32_t k_dim,
                                           uint32_t v_dim,
                                           float rms_eps)
{
    /* L2-normalize Q and K */
    float q_ss = 0.0f, k_ss = 0.0f;
#if defined(__AVX512F__)
    if ((k_dim & 15u) == 0u)
    {
        __m512 vq = _mm512_setzero_ps();
        __m512 vk = _mm512_setzero_ps();
        for (uint32_t i = 0; i < k_dim; i += 16)
        {
            __m512 qv = _mm512_loadu_ps(qh + i);
            __m512 kv = _mm512_loadu_ps(kh + i);
            vq = _mm512_fmadd_ps(qv, qv, vq);
            vk = _mm512_fmadd_ps(kv, kv, vk);
        }
        q_ss = _mm512_reduce_add_ps(vq);
        k_ss = _mm512_reduce_add_ps(vk);
        float q_inv = q_scale / sqrtf(q_ss + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_ss + 1e-6f);
        __m512 vqi = _mm512_set1_ps(q_inv);
        __m512 vki = _mm512_set1_ps(k_inv);
        for (uint32_t i = 0; i < k_dim; i += 16)
        {
            _mm512_storeu_ps(qh + i, _mm512_mul_ps(_mm512_loadu_ps(qh + i), vqi));
            _mm512_storeu_ps(kh + i, _mm512_mul_ps(_mm512_loadu_ps(kh + i), vki));
        }
    }
    else
#elif defined(__AVX2__)
    if ((k_dim & 7u) == 0u)
    {
        __m256 vq = _mm256_setzero_ps();
        __m256 vk = _mm256_setzero_ps();
        for (uint32_t i = 0; i < k_dim; i += 8)
        {
            __m256 qv = _mm256_loadu_ps(qh + i);
            __m256 kv = _mm256_loadu_ps(kh + i);
            vq = _mm256_fmadd_ps(qv, qv, vq);
            vk = _mm256_fmadd_ps(kv, kv, vk);
        }
        q_ss = lila_hsum_f32_avx2(vq);
        k_ss = lila_hsum_f32_avx2(vk);
        float q_inv = q_scale / sqrtf(q_ss + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_ss + 1e-6f);
        __m256 vqi = _mm256_set1_ps(q_inv);
        __m256 vki = _mm256_set1_ps(k_inv);
        for (uint32_t i = 0; i < k_dim; i += 8)
        {
            _mm256_storeu_ps(qh + i, _mm256_mul_ps(_mm256_loadu_ps(qh + i), vqi));
            _mm256_storeu_ps(kh + i, _mm256_mul_ps(_mm256_loadu_ps(kh + i), vki));
        }
    }
    else
#elif defined(__aarch64__)
    if ((k_dim & 3u) == 0u)
    {
        float32x4_t vq = vdupq_n_f32(0.0f);
        float32x4_t vk = vdupq_n_f32(0.0f);
        for (uint32_t i = 0; i < k_dim; i += 4)
        {
            float32x4_t qv = vld1q_f32(qh + i);
            float32x4_t kv = vld1q_f32(kh + i);
            vq = vfmaq_f32(vq, qv, qv);
            vk = vfmaq_f32(vk, kv, kv);
        }
        q_ss = vaddvq_f32(vq);
        k_ss = vaddvq_f32(vk);
        float q_inv = q_scale / sqrtf(q_ss + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_ss + 1e-6f);
        float32x4_t vqi = vdupq_n_f32(q_inv);
        float32x4_t vki = vdupq_n_f32(k_inv);
        for (uint32_t i = 0; i < k_dim; i += 4)
        {
            vst1q_f32(qh + i, vmulq_f32(vld1q_f32(qh + i), vqi));
            vst1q_f32(kh + i, vmulq_f32(vld1q_f32(kh + i), vki));
        }
    }
    else
#endif
    {
        for (uint32_t i = 0; i < k_dim; ++i) { q_ss += qh[i] * qh[i]; k_ss += kh[i] * kh[i]; }
        float q_inv = q_scale / sqrtf(q_ss + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_ss + 1e-6f);
        for (uint32_t i = 0; i < k_dim; ++i) { qh[i] *= q_inv; kh[i] *= k_inv; }
    }

    float dt = fast_softplus(alpha_val + dt_bias_val);
    float g = expf(ssm_a_val * dt);
    float beta = fast_sigmoid(beta_raw_val);

#if defined(__AVX512F__)
    if ((k_dim & 15u) == 0u)
    {
        __m512 vg = _mm512_set1_ps(g);
        for (uint32_t j = 0; j < v_dim; ++j)
        {
            float *Sj = S + (size_t)j * k_dim;
            __m512 vpred = _mm512_setzero_ps();
            for (uint32_t i = 0; i < k_dim; i += 16)
            {
                __m512 sd = _mm512_mul_ps(_mm512_loadu_ps(Sj + i), vg);
                _mm512_storeu_ps(Sj + i, sd);
                vpred = _mm512_fmadd_ps(sd, _mm512_loadu_ps(kh + i), vpred);
            }
            float pred = _mm512_reduce_add_ps(vpred);
            float delta = beta * (vh[j] - pred);
            __m512 vdelta = _mm512_set1_ps(delta);
            __m512 vov = _mm512_setzero_ps();
            for (uint32_t i = 0; i < k_dim; i += 16)
            {
                __m512 su = _mm512_fmadd_ps(vdelta, _mm512_loadu_ps(kh + i), _mm512_loadu_ps(Sj + i));
                _mm512_storeu_ps(Sj + i, su);
                vov = _mm512_fmadd_ps(su, _mm512_loadu_ps(qh + i), vov);
            }
            oh[j] = _mm512_reduce_add_ps(vov);
        }
    }
    else
#elif defined(__AVX2__)
    if ((k_dim & 7u) == 0u)
    {
        __m256 vg = _mm256_set1_ps(g);
        for (uint32_t j = 0; j < v_dim; ++j)
        {
            float *Sj = S + (size_t)j * k_dim;
            __m256 vpred = _mm256_setzero_ps();
            for (uint32_t i = 0; i < k_dim; i += 8)
            {
                __m256 sd = _mm256_mul_ps(_mm256_loadu_ps(Sj + i), vg);
                _mm256_storeu_ps(Sj + i, sd);
                vpred = _mm256_fmadd_ps(sd, _mm256_loadu_ps(kh + i), vpred);
            }
            float pred = lila_hsum_f32_avx2(vpred);
            float delta = beta * (vh[j] - pred);
            __m256 vdelta = _mm256_set1_ps(delta);
            __m256 vov = _mm256_setzero_ps();
            for (uint32_t i = 0; i < k_dim; i += 8)
            {
                __m256 su = _mm256_fmadd_ps(vdelta, _mm256_loadu_ps(kh + i), _mm256_loadu_ps(Sj + i));
                _mm256_storeu_ps(Sj + i, su);
                vov = _mm256_fmadd_ps(su, _mm256_loadu_ps(qh + i), vov);
            }
            oh[j] = lila_hsum_f32_avx2(vov);
        }
    }
    else
#elif defined(__aarch64__)
    if ((k_dim & 3u) == 0u)
    {
        float32x4_t vg = vdupq_n_f32(g);
        for (uint32_t j = 0; j < v_dim; ++j)
        {
            float *Sj = S + (size_t)j * k_dim;
            float32x4_t vpred = vdupq_n_f32(0.0f);
            for (uint32_t i = 0; i < k_dim; i += 4)
            {
                float32x4_t sd = vmulq_f32(vld1q_f32(Sj + i), vg);
                vst1q_f32(Sj + i, sd);
                vpred = vfmaq_f32(vpred, sd, vld1q_f32(kh + i));
            }
            float pred = vaddvq_f32(vpred);
            float delta = beta * (vh[j] - pred);
            float32x4_t vdelta = vdupq_n_f32(delta);
            float32x4_t vov = vdupq_n_f32(0.0f);
            for (uint32_t i = 0; i < k_dim; i += 4)
            {
                float32x4_t su = vfmaq_f32(vld1q_f32(Sj + i), vdelta, vld1q_f32(kh + i));
                vst1q_f32(Sj + i, su);
                vov = vfmaq_f32(vov, su, vld1q_f32(qh + i));
            }
            oh[j] = vaddvq_f32(vov);
        }
    }
    else
#endif
    {
        for (uint32_t j = 0; j < v_dim; ++j)
        {
            float *Sj = S + (size_t)j * k_dim;
            float pred = 0.0f;
            for (uint32_t i = 0; i < k_dim; ++i) { float d = Sj[i] * g; Sj[i] = d; pred += d * kh[i]; }
            float delta = beta * (vh[j] - pred);
            float ov = 0.0f;
            for (uint32_t i = 0; i < k_dim; ++i) { float u = Sj[i] + delta * kh[i]; Sj[i] = u; ov += u * qh[i]; }
            oh[j] = ov;
        }
    }
    rmsnorm(oh, oh, ssm_norm_w, v_dim, rms_eps);
    fast_silu_mul_out_vec(oh, zh, v_dim);
}

typedef struct {
    float *ssm_state;
    float *qkvb;
    float *zb;
    float *ab;
    float *bb;
    float *aob;
    const float *ssm_a;
    const float *dt_bias;
    const float *ssm_norm_w;
    uint32_t B;
    uint32_t n_heads;
    uint32_t k_dim;
    uint32_t v_dim;
    uint32_t inner;
    uint32_t qkv_dim;
    uint32_t ssm_r;
    float q_scale;
    float rms_eps;
} AsiDeltaNetCtx;

static void asi_deltanet_worker(int tid, int n_threads, void *vctx)
{
    const AsiDeltaNetCtx *ctx = (const AsiDeltaNetCtx *)vctx;
    uint32_t h_start = (ctx->n_heads * (uint32_t)tid) / (uint32_t)n_threads;
    uint32_t h_end = (ctx->n_heads * (uint32_t)(tid + 1)) / (uint32_t)n_threads;

    for (uint32_t h = h_start; h < h_end; ++h)
    {
        float *S = ctx->ssm_state + (size_t)h * ctx->v_dim * ctx->k_dim;
        for (uint32_t t = 0; t < ctx->B; ++t)
        {
            float *qkv = ctx->qkvb + (size_t)t * ctx->qkv_dim;
            float *qh = qkv + (size_t)h * ctx->k_dim;
            float *kh = qkv + ctx->inner + (size_t)h * ctx->k_dim;
            float *vh = qkv + 2 * ctx->inner + (size_t)h * ctx->v_dim;
            float *zh = ctx->zb + (size_t)t * ctx->inner + (size_t)h * ctx->v_dim;
            float *oh = ctx->aob + (size_t)t * ctx->inner + (size_t)h * ctx->v_dim;
            float a_val = ctx->ab[(size_t)t * ctx->ssm_r + h];
            float b_val = ctx->bb[(size_t)t * ctx->ssm_r + h];

            deltanet_head_step_avx2(S, qh, kh, vh, zh, oh, ctx->ssm_norm_w,
                                    a_val, ctx->dt_bias[h], ctx->ssm_a[h], b_val,
                                    ctx->q_scale, ctx->k_dim, ctx->v_dim, ctx->rms_eps);
        }
    }
}

static void qwen35_step_deltanet(AsiArchSession *s, uint32_t layer_idx)
{
    const AsiV2Cartridge *cart = s->cart;
    const AsiV2ModelConfig *c = &cart->config;
    const Qwen35LayerWeights *lw = &s->layers[layer_idx];
    Qwen35DeltaNetState *st = &s->ssm_states[layer_idx];
    Qwen35Scratch *sc = &s->scratch;

    uint32_t inner = c->ssm_inner_size;
    uint32_t qkv_dim = inner * 3;
    uint32_t n_heads = c->ssm_group_count;
    uint32_t k_dim = c->ssm_state_size;
    uint32_t v_dim = inner / n_heads;
    uint32_t k_conv = c->ssm_conv_kernel;
    uint32_t conv_hist = (k_conv > 1) ? (k_conv - 1) : 0;

    int bpr = ((int)c->hidden_size + QK_K - 1) / QK_K;
    LilaQ8KBlock stack_qx[64];
    LilaQ8KBlock *qx = (bpr <= 64) ? stack_qx : (LilaQ8KBlock *)malloc((size_t)bpr * sizeof(LilaQ8KBlock));
    if (qx)
    {
        quantize_vec_q8k(qx, sc->x_norm, (int)c->hidden_size);
        const AsiV2TensorEntry *ts[4] = {lw->attn_qkv, lw->attn_gate, lw->ssm_alpha, lw->ssm_beta};
        float *outs[4] = {sc->qkv, sc->z, sc->alpha, sc->beta};
        matvec_fused_q8k_pool(s->pool, cart, ts, outs, 4, qx, bpr, 0);
        if (qx != stack_qx) free(qx);
    }
    else
    {
        matvec_tensor_pool(s->pool, cart, lw->attn_qkv, sc->qkv, sc->x_norm);
        matvec_tensor_pool(s->pool, cart, lw->attn_gate, sc->z, sc->x_norm);
        matvec_tensor_pool(s->pool, cart, lw->ssm_alpha, sc->alpha, sc->x_norm);
        matvec_tensor_pool(s->pool, cart, lw->ssm_beta, sc->beta, sc->x_norm);
    }

    const float *conv_w = (const float *)tensor_data(cart, lw->ssm_conv1d);
    for (uint32_t ch = 0; ch < qkv_dim; ++ch)
    {
        const float *w = conv_w + (size_t)ch * k_conv;
        float sum = w[k_conv - 1] * sc->qkv[ch];
        for (uint32_t k = 0; k < conv_hist; ++k)
        {
            sum += w[k] * st->conv_state[(size_t)k * qkv_dim + ch];
        }
        for (uint32_t k = 0; k + 1 < conv_hist; ++k)
        {
            st->conv_state[(size_t)k * qkv_dim + ch] = st->conv_state[(size_t)(k + 1) * qkv_dim + ch];
        }
        if (conv_hist > 0)
        {
            st->conv_state[(size_t)(conv_hist - 1) * qkv_dim + ch] = sc->qkv[ch];
        }
        sc->qkv[ch] = sum;
    }
    fast_silu_inplace_vec(sc->qkv, qkv_dim);

    const float *ssm_a = (const float *)tensor_data(cart, lw->ssm_a);
    const float *dt_bias = (const float *)tensor_data(cart, lw->ssm_dt_bias);
    const float *ssm_norm_w = (const float *)tensor_data(cart, lw->ssm_norm);

    float q_scale = 1.0f / sqrtf((float)k_dim);
    AsiDeltaNetCtx dctx = {
        st->ssm_state, sc->qkv, sc->z, sc->alpha, sc->beta, sc->attn_out,
        ssm_a, dt_bias, ssm_norm_w,
        1, n_heads, k_dim, v_dim, inner, qkv_dim, c->ssm_time_step_rank,
        q_scale, c->rms_norm_eps
    };
    asi_pool_run(s->pool, asi_deltanet_worker, &dctx);

    matvec_tensor_pool(s->pool, cart, lw->ssm_out, sc->proj_out, sc->attn_out);
    for (uint32_t i = 0; i < c->hidden_size; ++i)
    {
        sc->x[i] += sc->proj_out[i];
    }
}

static void qwen35_step_full_attn(AsiArchSession *s, uint32_t layer_idx, uint32_t pos)
{
    const AsiV2Cartridge *cart = s->cart;
    const AsiV2ModelConfig *c = &cart->config;
    const Qwen35LayerWeights *lw = &s->layers[layer_idx];
    Qwen35AttnKVState *kv = &s->kv_states[layer_idx];
    Qwen35Scratch *sc = &s->scratch;

    uint32_t n_heads = c->n_heads;
    uint32_t n_kv = c->n_kv_heads;
    uint32_t hd = c->head_dim;
    uint32_t kv_dim = n_kv * hd;
    uint32_t n_rep = n_heads / n_kv;

    int bpr = ((int)c->hidden_size + QK_K - 1) / QK_K;
    LilaQ8KBlock stack_qx[64];
    LilaQ8KBlock *qx = (bpr <= 64) ? stack_qx : (LilaQ8KBlock *)malloc((size_t)bpr * sizeof(LilaQ8KBlock));
    if (qx)
    {
        quantize_vec_q8k(qx, sc->x_norm, (int)c->hidden_size);
        const AsiV2TensorEntry *ts[3] = {lw->attn_q, lw->attn_k, lw->attn_v};
        float *outs[3] = {sc->qg, sc->k, sc->v};
        matvec_fused_q8k_pool(s->pool, cart, ts, outs, 3, qx, bpr, 0);
        if (qx != stack_qx) free(qx);
    }
    else
    {
        matvec_tensor_pool(s->pool, cart, lw->attn_q, sc->qg, sc->x_norm);
        matvec_tensor_pool(s->pool, cart, lw->attn_k, sc->k, sc->x_norm);
        matvec_tensor_pool(s->pool, cart, lw->attn_v, sc->v, sc->x_norm);
    }

    const float *qn_w = (const float *)tensor_data(cart, lw->attn_q_norm);
    const float *kn_w = (const float *)tensor_data(cart, lw->attn_k_norm);

    for (uint32_t h = 0; h < n_kv; ++h)
    {
        float *kh = sc->k + (size_t)h * hd;
        rmsnorm(kh, kh, kn_w, hd, c->rms_norm_eps);
        apply_rope_neox_partial(kh, c->rope_dim, pos, c->rope_theta, s->rope_inv_freq);
    }

    uint16_t *k_dst = kv->k_cache + (size_t)pos * kv_dim;
    uint16_t *v_dst = kv->v_cache + (size_t)pos * kv_dim;
    for (uint32_t i = 0; i < kv_dim; ++i)
    {
        k_dst[i] = fp32_to_fp16(sc->k[i]);
        v_dst[i] = fp32_to_fp16(sc->v[i]);
    }

    float scale = 1.0f / sqrtf((float)hd);
    for (uint32_t h = 0; h < n_heads; ++h)
    {
        float *qh = sc->qg + (size_t)h * 2 * hd;
        const float *gh = qh + hd;
        rmsnorm(qh, qh, qn_w, hd, c->rms_norm_eps);
        apply_rope_neox_partial(qh, c->rope_dim, pos, c->rope_theta, s->rope_inv_freq);

        uint32_t kvh = h / n_rep;
        float max_score = -1e30f;
        for (uint32_t t = 0; t <= pos; ++t)
        {
            const uint16_t *kt = kv->k_cache + (size_t)t * kv_dim + (size_t)kvh * hd;
            float score = dot_f32_f16_avx2(qh, kt, hd) * scale;
            sc->attn_scores[t] = score;
            if (score > max_score) max_score = score;
        }

        float sum_exp = 0.0f;
        for (uint32_t t = 0; t <= pos; ++t)
        {
            float e = expf(sc->attn_scores[t] - max_score);
            sc->attn_scores[t] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / sum_exp;

        float *oh = sc->attn_out + (size_t)h * hd;
        memset(oh, 0, hd * sizeof(float));
        for (uint32_t t = 0; t <= pos; ++t)
        {
            float prob = sc->attn_scores[t] * inv_sum;
            const uint16_t *vt = kv->v_cache + (size_t)t * kv_dim + (size_t)kvh * hd;
            accum_f16_f32_avx2(oh, vt, prob, hd);
        }
        for (uint32_t d = 0; d < hd; ++d)
        {
            oh[d] *= fast_sigmoid(gh[d]);
        }
    }

    matvec_tensor_pool(s->pool, cart, lw->attn_output, sc->proj_out, sc->attn_out);
    for (uint32_t i = 0; i < c->hidden_size; ++i)
    {
        sc->x[i] += sc->proj_out[i];
    }
}

static int qwen35_forward_token(AsiArchSession *s, int32_t token_id, uint32_t pos, float *out_logits)
{
    if (!s || !out_logits || token_id < 0 || pos >= s->max_ctx) return -1;
    const AsiV2Cartridge *cart = s->cart;
    const AsiV2ModelConfig *c = &cart->config;
    if ((uint32_t)token_id >= c->vocab_size) return -1;

    Qwen35Scratch *sc = &s->scratch;
    int do_prof = (getenv("LILA_ASI_PROFILE") != NULL);
    double t_start = do_prof ? asi_now_ms() : 0.0;
    asi_pool_begin(s->pool);
    const uint8_t *embd_base = tensor_data(cart, s->token_embd);
    size_t blocks_per_row = (size_t)c->hidden_size / QK_K;
    if (s->token_embd->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK)
    {
        size_t row_bytes = blocks_per_row * sizeof(LilaQ6KRepackedBlock);
        dequant_q6k_repacked_row(sc->x, embd_base + (size_t)token_id * row_bytes, (int)c->hidden_size);
    }
    else
    {
        size_t row_bytes = blocks_per_row * Q6_K_BLOCK_SIZE;
        dequant_q6k_row(sc->x, embd_base + (size_t)token_id * row_bytes, (int)c->hidden_size);
    }

    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        const Qwen35LayerWeights *lw = &s->layers[l];
        const float *an_w = (const float *)tensor_data(cart, lw->attn_norm);
        rmsnorm(sc->x_norm, sc->x, an_w, c->hidden_size, c->rms_norm_eps);

        if (lw->kind == QWEN35_LAYER_DELTANET)
        {
            qwen35_step_deltanet(s, l);
        }
        else
        {
            qwen35_step_full_attn(s, l, pos);
        }

        const float *pn_w = (const float *)tensor_data(cart, lw->post_attn_norm);
        rmsnorm(sc->x_norm, sc->x, pn_w, c->hidden_size, c->rms_norm_eps);

        int bpr = ((int)c->hidden_size + QK_K - 1) / QK_K;
        LilaQ8KBlock stack_qx[64];
        LilaQ8KBlock *qx = (bpr <= 64) ? stack_qx : (LilaQ8KBlock *)malloc((size_t)bpr * sizeof(LilaQ8KBlock));
        if (qx)
        {
            quantize_vec_q8k(qx, sc->x_norm, (int)c->hidden_size);
            const AsiV2TensorEntry *ts[2] = {lw->ffn_gate, lw->ffn_up};
            float *outs[2] = {sc->ffn_gate, sc->ffn_up};
            matvec_fused_q8k_pool(s->pool, cart, ts, outs, 2, qx, bpr, 1);
            if (qx != stack_qx) free(qx);
        }
        else
        {
            matvec_tensor_pool(s->pool, cart, lw->ffn_gate, sc->ffn_gate, sc->x_norm);
            matvec_tensor_pool(s->pool, cart, lw->ffn_up, sc->ffn_up, sc->x_norm);
            fast_silu_mul_vec(sc->ffn_gate, sc->ffn_up, c->intermediate_size);
        }
        matvec_tensor_pool(s->pool, cart, lw->ffn_down, sc->proj_out, sc->ffn_gate);
        for (uint32_t i = 0; i < c->hidden_size; ++i)
        {
            sc->x[i] += sc->proj_out[i];
        }
    }

    double t_after_layers = do_prof ? asi_now_ms() : 0.0;
    const float *on_w = (const float *)tensor_data(cart, s->output_norm);
    rmsnorm(sc->x_norm, sc->x, on_w, c->hidden_size, c->rms_norm_eps);

    const AsiV2TensorEntry *lm_head = s->output_weight ? s->output_weight : s->token_embd;
    int rc = matvec_tensor_pool(s->pool, cart, lm_head, out_logits, sc->x_norm);
    asi_pool_end(s->pool);
    if (do_prof)
    {
        double t_end = asi_now_ms();
        fprintf(stderr, "[PROF token pos=%u] layers=%.2f ms | lm_head=%.2f ms | total=%.2f ms\n",
                pos, t_after_layers - t_start, t_end - t_after_layers, t_end - t_start);
    }
    return rc;
}

typedef struct {
    float *qkvb;
    float *conv_state;
    const float *conv_w;
    uint32_t B;
    uint32_t qkv_dim;
    uint32_t k_conv;
    uint32_t conv_hist;
} AsiConv1dCtx;

static void asi_conv1d_worker(int tid, int n_threads, void *vctx)
{
    const AsiConv1dCtx *ctx = (const AsiConv1dCtx *)vctx;
    uint32_t ch_start = (ctx->qkv_dim * (uint32_t)tid) / (uint32_t)n_threads;
    uint32_t ch_end = (ctx->qkv_dim * (uint32_t)(tid + 1)) / (uint32_t)n_threads;
    uint32_t k_conv = ctx->k_conv;
    uint32_t conv_hist = ctx->conv_hist;
    uint32_t qkv_dim = ctx->qkv_dim;

    for (uint32_t t = 0; t < ctx->B; ++t)
    {
        float *qkv = ctx->qkvb + (size_t)t * qkv_dim;
        for (uint32_t ch = ch_start; ch < ch_end; ++ch)
        {
            const float *w = ctx->conv_w + (size_t)ch * k_conv;
            float sum = w[k_conv - 1] * qkv[ch];
            for (uint32_t k = 0; k < conv_hist; ++k)
                sum += w[k] * ctx->conv_state[(size_t)k * qkv_dim + ch];
            for (uint32_t k = 0; k + 1 < conv_hist; ++k)
                ctx->conv_state[(size_t)k * qkv_dim + ch] = ctx->conv_state[(size_t)(k + 1) * qkv_dim + ch];
            if (conv_hist > 0)
                ctx->conv_state[(size_t)(conv_hist - 1) * qkv_dim + ch] = qkv[ch];
            qkv[ch] = sum;
        }
        fast_silu_inplace_vec(qkv + ch_start, ch_end - ch_start);
    }
}

typedef struct {
    float *qgb;
    float *aob;
    const uint16_t *k_cache;
    const uint16_t *v_cache;
    const float *qn_w;
    const float *rope_inv_freq;
    uint32_t B;
    uint32_t pos_start;
    uint32_t n_heads;
    uint32_t n_rep;
    uint32_t hd;
    uint32_t qg_dim;
    uint32_t kv_dim;
    uint32_t attn_dim;
    uint32_t rope_dim;
    float rope_theta;
    float rms_eps;
    float scale;
} AsiFullAttnBatchCtx;

static void asi_full_attn_batch_worker(int tid, int n_threads, void *vctx)
{
    const AsiFullAttnBatchCtx *ctx = (const AsiFullAttnBatchCtx *)vctx;
    uint32_t h_start = (ctx->n_heads * (uint32_t)tid) / (uint32_t)n_threads;
    uint32_t h_end = (ctx->n_heads * (uint32_t)(tid + 1)) / (uint32_t)n_threads;
    uint32_t max_pos = ctx->pos_start + ctx->B;
    float stack_scores[2048];
    float *scores = (max_pos <= 2048) ? stack_scores : (float *)malloc((size_t)max_pos * sizeof(float));
    if (!scores) return;

    uint32_t hd = ctx->hd;
    uint32_t kv_dim = ctx->kv_dim;
    for (uint32_t t = 0; t < ctx->B; ++t)
    {
        uint32_t pos = ctx->pos_start + t;
        float *qg = ctx->qgb + (size_t)t * ctx->qg_dim;
        float *ao = ctx->aob + (size_t)t * ctx->attn_dim;
        for (uint32_t h = h_start; h < h_end; ++h)
        {
            float *qh = qg + (size_t)h * 2 * hd;
            const float *gh = qh + hd;
            rmsnorm(qh, qh, ctx->qn_w, hd, ctx->rms_eps);
            apply_rope_neox_partial(qh, ctx->rope_dim, pos, ctx->rope_theta, ctx->rope_inv_freq);
            uint32_t kvh = h / ctx->n_rep;
            float max_score = -1e30f;
            for (uint32_t p = 0; p <= pos; ++p)
            {
                const uint16_t *kt = ctx->k_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                float sc_val = dot_f32_f16_avx2(qh, kt, hd) * ctx->scale;
                scores[p] = sc_val;
                if (sc_val > max_score) max_score = sc_val;
            }
            float sum_exp = 0.0f;
            for (uint32_t p = 0; p <= pos; ++p)
            {
                float e = expf(scores[p] - max_score);
                scores[p] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / sum_exp;
            float *oh = ao + (size_t)h * hd;
            memset(oh, 0, hd * sizeof(float));
            for (uint32_t p = 0; p <= pos; ++p)
            {
                float prob = scores[p] * inv_sum;
                const uint16_t *vt = ctx->v_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                accum_f16_f32_avx2(oh, vt, prob, hd);
            }
            for (uint32_t d = 0; d < hd; ++d) oh[d] *= fast_sigmoid(gh[d]);
        }
    }
    if (scores != stack_scores) free(scores);
}

static int ensure_batch_scratch(AsiArchSession *s, uint32_t B)
{
    Qwen35Scratch *sc = &s->scratch;
    if (B <= sc->batch_cap) return 0;

    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t H = c->hidden_size;
    uint32_t I = c->intermediate_size;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t z_dim = c->ssm_inner_size;
    uint32_t qg_dim = c->n_heads * 2 * c->head_dim;
    uint32_t kv_dim = c->n_kv_heads * c->head_dim;
    uint32_t attn_dim = c->n_heads * c->head_dim;
    uint32_t ssm_r = c->ssm_time_step_rank;
    uint32_t max_in_dim = H;
    if (I > max_in_dim) max_in_dim = I;
    if (attn_dim > max_in_dim) max_in_dim = attn_dim;
    if (z_dim > max_in_dim) max_in_dim = z_dim;
    int bpr_max = ((int)max_in_dim + QK_K - 1) / QK_K;

    free(sc->xb); free(sc->xnb); free(sc->projb); free(sc->qkvb);
    free(sc->zb); free(sc->ab); free(sc->bb); free(sc->qgb);
    free(sc->kb); free(sc->vb); free(sc->aob); free(sc->fgb);
    free(sc->fub); free(sc->qx_h);

    sc->xb    = (float *)malloc((size_t)B * H * sizeof(float));
    sc->xnb   = (float *)malloc((size_t)B * H * sizeof(float));
    sc->projb = (float *)malloc((size_t)B * H * sizeof(float));
    sc->qkvb  = (float *)malloc((size_t)B * qkv_dim * sizeof(float));
    sc->zb    = (float *)malloc((size_t)B * z_dim * sizeof(float));
    sc->ab    = (float *)malloc((size_t)B * ssm_r * sizeof(float));
    sc->bb    = (float *)malloc((size_t)B * ssm_r * sizeof(float));
    sc->qgb   = (float *)malloc((size_t)B * qg_dim * sizeof(float));
    sc->kb    = (float *)malloc((size_t)B * kv_dim * sizeof(float));
    sc->vb    = (float *)malloc((size_t)B * kv_dim * sizeof(float));
    sc->aob   = (float *)malloc((size_t)B * (attn_dim > z_dim ? attn_dim : z_dim) * sizeof(float));
    sc->fgb   = (float *)malloc((size_t)B * I * sizeof(float));
    sc->fub   = (float *)malloc((size_t)B * I * sizeof(float));
    sc->qx_h  = (LilaQ8KBlock *)malloc((size_t)B * (size_t)bpr_max * sizeof(LilaQ8KBlock));

    if (!sc->xb || !sc->xnb || !sc->projb || !sc->qkvb || !sc->zb || !sc->ab || !sc->bb ||
        !sc->qgb || !sc->kb || !sc->vb || !sc->aob || !sc->fgb || !sc->fub || !sc->qx_h)
    {
        sc->batch_cap = 0;
        return -1;
    }
    sc->batch_cap = B;
    return 0;
}

static int qwen35_forward_chunk(AsiArchSession *s,
                                const int32_t *token_ids,
                                uint32_t B,
                                uint32_t pos_start,
                                int compute_logits,
                                float *out_last_logits)
{
    int do_prof = (getenv("LILA_ASI_PROFILE") != NULL);
    double tc0 = do_prof ? asi_now_ms() : 0.0;
    double t_mm = 0.0, t_ssm = 0.0, t_attn = 0.0, t_conv = 0.0, t_norm = 0.0, t_lm = 0.0;
    const AsiV2Cartridge *cart = s->cart;
    const AsiV2ModelConfig *c = &cart->config;
    uint32_t H = c->hidden_size;
    uint32_t I = c->intermediate_size;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t z_dim = c->ssm_inner_size;
    (void)z_dim;
    uint32_t qg_dim = c->n_heads * 2 * c->head_dim;
    uint32_t kv_dim = c->n_kv_heads * c->head_dim;
    uint32_t attn_dim = c->n_heads * c->head_dim;
    uint32_t ssm_r = c->ssm_time_step_rank;
    int bpr_h = ((int)H + QK_K - 1) / QK_K;

    if (ensure_batch_scratch(s, B) != 0) return -1;
    float *xb    = s->scratch.xb;
    float *xnb   = s->scratch.xnb;
    float *projb = s->scratch.projb;
    float *qkvb  = s->scratch.qkvb;
    float *zb    = s->scratch.zb;
    float *ab    = s->scratch.ab;
    float *bb    = s->scratch.bb;
    float *qgb   = s->scratch.qgb;
    float *kb    = s->scratch.kb;
    float *vb    = s->scratch.vb;
    float *aob   = s->scratch.aob;
    float *fgb   = s->scratch.fgb;
    float *fub   = s->scratch.fub;
    LilaQ8KBlock *qx_h = s->scratch.qx_h;

    const uint8_t *embd_base = tensor_data(cart, s->token_embd);
    if (s->token_embd->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK)
    {
        size_t row_bytes = ((size_t)H / QK_K) * sizeof(LilaQ6KRepackedBlock);
        for (uint32_t t = 0; t < B; ++t)
            dequant_q6k_repacked_row(xb + (size_t)t * H, embd_base + (size_t)token_ids[t] * row_bytes, (int)H);
    }
    else
    {
        size_t row_bytes = ((size_t)H / QK_K) * Q6_K_BLOCK_SIZE;
        for (uint32_t t = 0; t < B; ++t)
            dequant_q6k_row(xb + (size_t)t * H, embd_base + (size_t)token_ids[t] * row_bytes, (int)H);
    }

    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        const Qwen35LayerWeights *lw = &s->layers[l];
        double tp = do_prof ? asi_now_ms() : 0.0;
        const float *an_w = (const float *)tensor_data(cart, lw->attn_norm);
        for (uint32_t t = 0; t < B; ++t)
        {
            rmsnorm(xnb + (size_t)t * H, xb + (size_t)t * H, an_w, H, c->rms_norm_eps);
        }
        if (do_prof) t_norm += asi_now_ms() - tp;

        if (lw->kind == QWEN35_LAYER_DELTANET)
        {
            tp = do_prof ? asi_now_ms() : 0.0;
            quantize_batch_q8k_pool(s->pool, qx_h, xnb, (int)B, (int)H);
            const AsiV2TensorEntry *ts[4] = {lw->attn_qkv, lw->attn_gate, lw->ssm_alpha, lw->ssm_beta};
            float *outs[4] = {qkvb, zb, ab, bb};
            matmul_batch_fused_q8k_pool(s->pool, cart, ts, outs, 4, qx_h, xnb, (int)B, bpr_h, 0);
            if (do_prof) t_mm += asi_now_ms() - tp;

            Qwen35DeltaNetState *st = &s->ssm_states[l];
            const float *conv_w = (const float *)tensor_data(cart, lw->ssm_conv1d);
            const float *ssm_a = (const float *)tensor_data(cart, lw->ssm_a);
            const float *dt_bias = (const float *)tensor_data(cart, lw->ssm_dt_bias);
            const float *ssm_norm_w = (const float *)tensor_data(cart, lw->ssm_norm);
            uint32_t n_heads = c->ssm_group_count;
            uint32_t k_dim = c->ssm_state_size;
            uint32_t v_dim = c->ssm_inner_size / n_heads;
            uint32_t k_conv = c->ssm_conv_kernel;
            uint32_t conv_hist = (k_conv > 1) ? (k_conv - 1) : 0;
            float q_scale = 1.0f / sqrtf((float)k_dim);

            tp = do_prof ? asi_now_ms() : 0.0;
            AsiConv1dCtx cctx = {qkvb, st->conv_state, conv_w, B, qkv_dim, k_conv, conv_hist};
            asi_pool_run(s->pool, asi_conv1d_worker, &cctx);
            if (do_prof) t_conv += asi_now_ms() - tp;
            tp = do_prof ? asi_now_ms() : 0.0;
            AsiDeltaNetCtx dctx = {
                st->ssm_state, qkvb, zb, ab, bb, aob,
                ssm_a, dt_bias, ssm_norm_w,
                B, n_heads, k_dim, v_dim, c->ssm_inner_size, qkv_dim, ssm_r,
                q_scale, c->rms_norm_eps
            };
            asi_pool_run(s->pool, asi_deltanet_worker, &dctx);
            if (do_prof) t_ssm += asi_now_ms() - tp;
            tp = do_prof ? asi_now_ms() : 0.0;
            int ssm_in_cols = (int)lw->ssm_out->shape[0];
            int ssm_in_bpr = (ssm_in_cols + QK_K - 1) / QK_K;
            quantize_batch_q8k_pool(s->pool, qx_h, aob, (int)B, ssm_in_cols);
            matmul_batch_tensor_qx(s->pool, cart, lw->ssm_out, projb, qx_h, aob, (int)B, ssm_in_bpr);
            if (do_prof) t_mm += asi_now_ms() - tp;
        }
        else
        {
            tp = do_prof ? asi_now_ms() : 0.0;
            quantize_batch_q8k_pool(s->pool, qx_h, xnb, (int)B, (int)H);
            const AsiV2TensorEntry *ts[3] = {lw->attn_q, lw->attn_k, lw->attn_v};
            float *outs[3] = {qgb, kb, vb};
            matmul_batch_fused_q8k_pool(s->pool, cart, ts, outs, 3, qx_h, xnb, (int)B, bpr_h, 0);
            if (do_prof) t_mm += asi_now_ms() - tp;

            Qwen35AttnKVState *kv = &s->kv_states[l];
            const float *qn_w = (const float *)tensor_data(cart, lw->attn_q_norm);
            const float *kn_w = (const float *)tensor_data(cart, lw->attn_k_norm);
            uint32_t n_heads = c->n_heads;
            uint32_t n_kv = c->n_kv_heads;
            uint32_t hd = c->head_dim;
            uint32_t n_rep = n_heads / n_kv;
            float scale = 1.0f / sqrtf((float)hd);

            tp = do_prof ? asi_now_ms() : 0.0;
            for (uint32_t t = 0; t < B; ++t)
            {
                uint32_t pos = pos_start + t;
                float *k = kb + (size_t)t * kv_dim;
                float *v = vb + (size_t)t * kv_dim;

                for (uint32_t h = 0; h < n_kv; ++h)
                {
                    float *kh = k + (size_t)h * hd;
                    rmsnorm(kh, kh, kn_w, hd, c->rms_norm_eps);
                    apply_rope_neox_partial(kh, c->rope_dim, pos, c->rope_theta, s->rope_inv_freq);
                }
                uint16_t *k_dst = kv->k_cache + (size_t)pos * kv_dim;
                uint16_t *v_dst = kv->v_cache + (size_t)pos * kv_dim;
                for (uint32_t i = 0; i < kv_dim; ++i)
                {
                    k_dst[i] = fp32_to_fp16(k[i]);
                    v_dst[i] = fp32_to_fp16(v[i]);
                }
            }
            AsiFullAttnBatchCtx fctx = {
                qgb, aob, kv->k_cache, kv->v_cache, qn_w, s->rope_inv_freq,
                B, pos_start, n_heads, n_rep, hd, qg_dim, kv_dim, attn_dim,
                c->rope_dim, c->rope_theta, c->rms_norm_eps, scale
            };
            asi_pool_run(s->pool, asi_full_attn_batch_worker, &fctx);
            if (do_prof) t_attn += asi_now_ms() - tp;
            tp = do_prof ? asi_now_ms() : 0.0;
            int attn_in_cols = (int)lw->attn_output->shape[0];
            int attn_in_bpr = (attn_in_cols + QK_K - 1) / QK_K;
            quantize_batch_q8k_pool(s->pool, qx_h, aob, (int)B, attn_in_cols);
            matmul_batch_tensor_qx(s->pool, cart, lw->attn_output, projb, qx_h, aob, (int)B, attn_in_bpr);
            if (do_prof) t_mm += asi_now_ms() - tp;
        }

        tp = do_prof ? asi_now_ms() : 0.0;
        for (size_t i = 0; i < (size_t)B * H; ++i) xb[i] += projb[i];

        const float *pn_w = (const float *)tensor_data(cart, lw->post_attn_norm);
        for (uint32_t t = 0; t < B; ++t)
        {
            rmsnorm(xnb + (size_t)t * H, xb + (size_t)t * H, pn_w, H, c->rms_norm_eps);
        }
        if (do_prof) t_norm += asi_now_ms() - tp;
        tp = do_prof ? asi_now_ms() : 0.0;
        quantize_batch_q8k_pool(s->pool, qx_h, xnb, (int)B, (int)H);
        const AsiV2TensorEntry *ffn_ts[2] = {lw->ffn_gate, lw->ffn_up};
        float *ffn_outs[2] = {fgb, fub};
        matmul_batch_fused_q8k_pool(s->pool, cart, ffn_ts, ffn_outs, 2, qx_h, xnb, (int)B, bpr_h, 1);
        if (do_prof) t_mm += asi_now_ms() - tp;
        tp = do_prof ? asi_now_ms() : 0.0;
        int ffn_in_bpr = ((int)I + QK_K - 1) / QK_K;
        quantize_batch_q8k_pool(s->pool, qx_h, fgb, (int)B, (int)I);
        matmul_batch_tensor_qx(s->pool, cart, lw->ffn_down, projb, qx_h, fgb, (int)B, ffn_in_bpr);
        if (do_prof) t_mm += asi_now_ms() - tp;
        tp = do_prof ? asi_now_ms() : 0.0;
        for (size_t i = 0; i < (size_t)B * H; ++i) xb[i] += projb[i];
        if (do_prof) t_norm += asi_now_ms() - tp;
    }

    int rc = 0;
    if (compute_logits && out_last_logits)
    {
        double tp = do_prof ? asi_now_ms() : 0.0;
        const float *on_w = (const float *)tensor_data(cart, s->output_norm);
        rmsnorm(s->scratch.x_norm, xb + (size_t)(B - 1) * H, on_w, H, c->rms_norm_eps);
        const AsiV2TensorEntry *lm_head = s->output_weight ? s->output_weight : s->token_embd;
        rc = matvec_tensor_pool(s->pool, cart, lm_head, out_last_logits, s->scratch.x_norm);
        if (do_prof) t_lm += asi_now_ms() - tp;
    }

    if (do_prof)
    {
        double tc1 = asi_now_ms();
        fprintf(stderr, "[PROF chunk B=%u pos=%u] total=%.2f ms (mm=%.2f conv=%.2f ssm=%.2f attn=%.2f norm=%.2f lm=%.2f)\n",
                B, pos_start, tc1 - tc0, t_mm, t_conv, t_ssm, t_attn, t_norm, t_lm);
    }
    return rc;
}

static int qwen35_forward_batch(AsiArchSession *s,
                                const int32_t *token_ids,
                                uint32_t n_tokens,
                                uint32_t pos_start,
                                float *out_last_logits)
{
    if (!s || !token_ids || n_tokens == 0 || !out_last_logits) return -1;
    if (pos_start + n_tokens > s->max_ctx) return -1;
    if (n_tokens == 1)
    {
        return qwen35_forward_token(s, token_ids[0], pos_start, out_last_logits);
    }

    const uint32_t chunk_max = 128;
    uint32_t done = 0;
    asi_pool_begin(s->pool);
    while (done < n_tokens)
    {
        uint32_t B = n_tokens - done;
        if (B > chunk_max) B = chunk_max;
        int is_last = (done + B == n_tokens);
        if (qwen35_forward_chunk(s, token_ids + done, B, pos_start + done,
                                 is_last, out_last_logits) != 0)
        {
            asi_pool_end(s->pool);
            return -1;
        }
        done += B;
    }
    asi_pool_end(s->pool);
    return 0;
}

static int qwen35_save_prefix(AsiArchSession *s, uint32_t n_tokens)
{
    if (!s || n_tokens == 0 || n_tokens > s->max_ctx) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t conv_hist = (c->ssm_conv_kernel > 1) ? (c->ssm_conv_kernel - 1) : 0;
    size_t conv_per_layer = (size_t)conv_hist * qkv_dim;
    uint32_t ssm_head_dim = (c->ssm_group_count > 0) ? (c->ssm_inner_size / c->ssm_group_count) : 0;
    size_t ssm_per_layer = (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim;
    size_t kv_dim = (size_t)c->n_kv_heads * c->head_dim;
    size_t kv_per_layer = (size_t)n_tokens * kv_dim;

    uint32_t n_dn = 0, n_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET) n_dn++; else n_fa++;
    }

    free(s->prefix_conv); free(s->prefix_ssm); free(s->prefix_k); free(s->prefix_v);
    s->prefix_conv = (n_dn && conv_per_layer) ? (float *)malloc((size_t)n_dn * conv_per_layer * sizeof(float)) : NULL;
    s->prefix_ssm  = (n_dn && ssm_per_layer)  ? (float *)malloc((size_t)n_dn * ssm_per_layer * sizeof(float)) : NULL;
    s->prefix_k    = (n_fa && kv_per_layer)   ? (uint16_t *)malloc((size_t)n_fa * kv_per_layer * sizeof(uint16_t)) : NULL;
    s->prefix_v    = (n_fa && kv_per_layer)   ? (uint16_t *)malloc((size_t)n_fa * kv_per_layer * sizeof(uint16_t)) : NULL;
    if ((n_dn && conv_per_layer && !s->prefix_conv) || (n_dn && ssm_per_layer && !s->prefix_ssm) ||
        (n_fa && kv_per_layer && (!s->prefix_k || !s->prefix_v)))
    {
        free(s->prefix_conv); free(s->prefix_ssm); free(s->prefix_k); free(s->prefix_v);
        s->prefix_conv = s->prefix_ssm = NULL; s->prefix_k = s->prefix_v = NULL;
        s->prefix_len = 0;
        return -1;
    }

    uint32_t i_dn = 0, i_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET)
        {
            if (conv_per_layer)
                memcpy(s->prefix_conv + (size_t)i_dn * conv_per_layer, s->ssm_states[l].conv_state, conv_per_layer * sizeof(float));
            if (ssm_per_layer)
                memcpy(s->prefix_ssm + (size_t)i_dn * ssm_per_layer, s->ssm_states[l].ssm_state, ssm_per_layer * sizeof(float));
            i_dn++;
        }
        else
        {
            if (kv_per_layer)
            {
                memcpy(s->prefix_k + (size_t)i_fa * kv_per_layer, s->kv_states[l].k_cache, kv_per_layer * sizeof(uint16_t));
                memcpy(s->prefix_v + (size_t)i_fa * kv_per_layer, s->kv_states[l].v_cache, kv_per_layer * sizeof(uint16_t));
            }
            i_fa++;
        }
    }
    s->prefix_len = n_tokens;
    return 0;
}

static int qwen35_restore_prefix(AsiArchSession *s, uint32_t *out_n_tokens)
{
    if (!s || s->prefix_len == 0) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t conv_hist = (c->ssm_conv_kernel > 1) ? (c->ssm_conv_kernel - 1) : 0;
    size_t conv_per_layer = (size_t)conv_hist * qkv_dim;
    uint32_t ssm_head_dim = (c->ssm_group_count > 0) ? (c->ssm_inner_size / c->ssm_group_count) : 0;
    size_t ssm_per_layer = (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim;
    size_t kv_dim = (size_t)c->n_kv_heads * c->head_dim;
    size_t kv_per_layer = (size_t)s->prefix_len * kv_dim;

    uint32_t i_dn = 0, i_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET)
        {
            if (conv_per_layer && s->prefix_conv)
                memcpy(s->ssm_states[l].conv_state, s->prefix_conv + (size_t)i_dn * conv_per_layer, conv_per_layer * sizeof(float));
            if (ssm_per_layer && s->prefix_ssm)
                memcpy(s->ssm_states[l].ssm_state, s->prefix_ssm + (size_t)i_dn * ssm_per_layer, ssm_per_layer * sizeof(float));
            i_dn++;
        }
        else
        {
            if (kv_per_layer && s->prefix_k && s->prefix_v)
            {
                memcpy(s->kv_states[l].k_cache, s->prefix_k + (size_t)i_fa * kv_per_layer, kv_per_layer * sizeof(uint16_t));
                memcpy(s->kv_states[l].v_cache, s->prefix_v + (size_t)i_fa * kv_per_layer, kv_per_layer * sizeof(uint16_t));
            }
            i_fa++;
        }
    }
    if (out_n_tokens) *out_n_tokens = s->prefix_len;
    return 0;
}

const AsiArchHandler g_asi_arch_hybrid_qwen35 = {
    .arch_id = ASI_ARCH_HYBRID_QWEN35,
    .name = "hybrid_qwen35",
    .init_session = qwen35_init_session,
    .free_session = qwen35_free_session,
    .forward_token = qwen35_forward_token,
    .forward_batch = qwen35_forward_batch,
    .reset_state = qwen35_reset_state,
    .save_prefix = qwen35_save_prefix,
    .restore_prefix = qwen35_restore_prefix,
};

const AsiArchHandler g_asi_arch_dense_transformer = {
    .arch_id = ASI_ARCH_DENSE_LLAMA,
    .name = "dense_transformer",
    .init_session = NULL,
    .free_session = NULL,
    .forward_token = NULL,
    .forward_batch = NULL,
    .reset_state = NULL,
};

const AsiArchHandler *asi_arch_find(uint32_t arch_id)
{
    switch (arch_id)
    {
        case ASI_ARCH_HYBRID_QWEN35:
            return &g_asi_arch_hybrid_qwen35;
        case ASI_ARCH_DENSE_LLAMA:
        case ASI_ARCH_DENSE_GEMMA:
        case ASI_ARCH_DENSE_QWEN2:
        case ASI_ARCH_DENSE_QWEN3:
            return &g_asi_arch_dense_transformer;
        default:
            return NULL;
    }
}
