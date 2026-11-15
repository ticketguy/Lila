/*
 * bench_roofline.c - Memory-Roofline, STREAM/DRAM Bandwidth & Decode Stage Breakdown
 *
 * Measures:
 *   1. Synthetic 256 MiB DRAM Read-Only (AVX2 _mm256_load_si256) & STREAM Copy/Triad
 *      at threads = 1 and 2 via AsiThreadPool.
 *   2. Isolated kernel throughput (GB/s & weight GFLOP/s) on L3-resident (2 MiB) and
 *      DRAM-resident (64 MiB) weight working sets for:
 *        - Q4_K dot1 (q4k_dot_q8k_simd) and dot2 (q4k_dot2_q8k_avx2)
 *        - Q6_K dot1 (q6k_dot_q8k_simd) and dot2 (q6k_dot2_q8k_avx2)
 *   3. Optional .asi cartridge inspection (exact active decode weight bytes broken down
 *      by quant type and tensor role) + full-model decode tok/s, effective GB/s,
 *      DRAM bandwidth utilization, and kernel-predicted vs measured stage breakdown.
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "q4k.h"
#include "asi_v2.h"
#include "asi_pool.h"
#include "lila_asi.h"

#define DEFAULT_ASI_PATH "/tmp/qwen35_0.8b.asi"
#define DEFAULT_N_TOKENS 32
#define DEFAULT_MAX_CTX  256
#define MAX_THREAD_MODES 8

#define DRAM_BUF_BYTES   (256ULL * 1024ULL * 1024ULL) /* 256 MiB >> 3 MiB L3 */
#define STREAM_ARR_BYTES (80ULL  * 1024ULL * 1024ULL) /* 3 x 80 MiB = 240 MiB inside 256 MiB */

#define L3_TARGET_BYTES   (2ULL  * 1024ULL * 1024ULL) /* 2 MiB fits in 3 MiB L3 */
#define DRAM_TARGET_BYTES (64ULL * 1024ULL * 1024ULL) /* 64 MiB >> 3 MiB L3 */

typedef struct {
    double best_gb_s;
    double median_gb_s;
    double mean_gb_s;
    double best_gflops;
    double median_gflops;
    double mean_gflops;
    double elapsed_ms;
    double checksum;
} BenchMetric;

typedef struct {
    int threads;
    BenchMetric dram_ro;
    BenchMetric stream_copy;
    BenchMetric stream_triad;
    BenchMetric l3_q4k_dot1;
    BenchMetric l3_q4k_dot2;
    BenchMetric l3_q6k_dot1;
    BenchMetric l3_q6k_dot2;
    BenchMetric dram_q4k_dot1;
    BenchMetric dram_q4k_dot2;
    BenchMetric dram_q6k_dot1;
    BenchMetric dram_q6k_dot2;
} ThreadBench;

typedef struct {
    bool present;
    bool requested;
    char path[1024];
    char error[256];
    char arch[64];
    uint64_t file_size;
    uint64_t weights_section_size;
    uint32_t n_tensors;
    uint32_t repacked_tensors;
    bool tied_embeddings;

    uint32_t vocab_size;
    uint32_t n_embd;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t n_ff;
    uint32_t full_attention_interval;
    uint32_t ssm_state_size;
    uint32_t ssm_conv_kernel;
    uint32_t ssm_inner_size;

    uint64_t total_model_bytes;
    uint64_t active_decode_bytes;

    uint64_t q6k_total_bytes;
    uint64_t q6k_token_embd_bytes;
    uint64_t q6k_attn_qkv_bytes;
    uint64_t q6k_ffn_down_bytes;
    uint64_t q6k_other_bytes;

    uint64_t q4k_total_bytes;
    uint64_t q4k_attn_qkv_bytes;
    uint64_t q4k_ffn_down_bytes;
    uint64_t q4k_other_bytes;

    uint64_t f32_bytes;
    uint64_t other_qtype_bytes;
} AsiBreakdown;

typedef struct {
    int threads;
    bool ran;
    int n_tokens;
    double elapsed_s;
    double eval_only_s;
    double tok_s;
    double eval_tok_s;
    double ms_per_tok;
    double effective_gb_s;
    double dram_ro_util_pct;
    double stream_copy_util_pct;

    double pred_q6k_ms;
    double pred_q4k_ms;
    double pred_matvec_ms;
    double pred_q6k_time_pct;
    double pred_q4k_time_pct;
    double pred_matvec_gb_s;
    double roofline_dram_tok_s;
    double pred_matvec_tok_s;
    double residual_non_matvec_ms;
    double residual_pct;
} AsiDecodeResult;

static volatile uint64_t g_u64_sink = 0;
static volatile double g_f64_sink = 0.0;

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline size_t align_up(size_t n, size_t a) {
    return (n + a - 1) & ~(a - 1);
}

static void *xaligned_alloc(size_t alignment, size_t size) {
    size_t sz = align_up(size, alignment);
    return aligned_alloc(alignment, sz);
}

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static void compute_stats(const double *samples_s, int count,
                          double total_bytes_per_sample,
                          double total_flops_per_sample,
                          double checksum,
                          BenchMetric *out) {
    double sorted[32];
    int n = count > 32 ? 32 : count;
    double sum_s = 0.0;
    for (int i = 0; i < n; i++) {
        sorted[i] = samples_s[i];
        sum_s += samples_s[i];
    }
    qsort(sorted, (size_t)n, sizeof(double), cmp_double_asc);
    double best_s = sorted[0];
    double med_s = (n % 2 == 1) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    double mean_s = sum_s / (double)n;

    out->best_gb_s = (best_s > 0.0) ? (total_bytes_per_sample / best_s) / 1e9 : 0.0;
    out->median_gb_s = (med_s > 0.0) ? (total_bytes_per_sample / med_s) / 1e9 : 0.0;
    out->mean_gb_s = (mean_s > 0.0) ? (total_bytes_per_sample / mean_s) / 1e9 : 0.0;

    if (total_flops_per_sample > 0.0) {
        out->best_gflops = (best_s > 0.0) ? (total_flops_per_sample / best_s) / 1e9 : 0.0;
        out->median_gflops = (med_s > 0.0) ? (total_flops_per_sample / med_s) / 1e9 : 0.0;
        out->mean_gflops = (mean_s > 0.0) ? (total_flops_per_sample / mean_s) / 1e9 : 0.0;
    } else {
        out->best_gflops = 0.0;
        out->median_gflops = 0.0;
        out->mean_gflops = 0.0;
    }
    out->elapsed_ms = med_s * 1e3;
    out->checksum = checksum;
}

static void fill_nonzero_dram_buffer(uint8_t *buf, size_t bytes) {
    double *dp = (double *)buf;
    size_t n_doubles = bytes / sizeof(double);
    for (size_t i = 0; i < n_doubles; i++) {
        uint64_t h = ((uint64_t)i + 1ULL) * 0x9e3779b97f4a7c15ULL;
        h ^= h >> 30;
        dp[i] = 1.0 + (double)(h & 0xFFFFULL) * (1.0 / 65536.0);
    }
    for (size_t b = n_doubles * sizeof(double); b < bytes; b++) {
        buf[b] = (uint8_t)((b & 0x7Fu) | 0x01u);
    }
}

static void init_q4k_weights(uint8_t *buf, size_t n_blocks) {
    const uint16_t d_fp16 = fp32_to_fp16(0.0125f);
    const uint16_t dmin_fp16 = fp32_to_fp16(0.003125f);
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = buf + b * Q4_K_BLOCK_SIZE;
        memcpy(blk + 0, &d_fp16, sizeof(uint16_t));
        memcpy(blk + 2, &dmin_fp16, sizeof(uint16_t));
        for (int i = 0; i < 12; i++) {
            blk[4 + i] = (uint8_t)(0x11u + (((b + (size_t)i) * 7u) & 0x2Au));
        }
        for (int i = 0; i < 128; i++) {
            blk[16 + i] = (uint8_t)(((b * 13u + (size_t)i * 5u + 1u) & 0x0Fu) |
                                    (((b * 7u + (size_t)i * 3u + 2u) & 0x0Fu) << 4));
        }
    }
}

static void init_q6k_weights(uint8_t *buf, size_t n_blocks) {
    const uint16_t d_fp16 = fp32_to_fp16(0.0125f);
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = buf + b * Q6_K_BLOCK_SIZE;
        for (int i = 0; i < 128; i++) {
            blk[i] = (uint8_t)(((b * 11u + (size_t)i * 5u + 3u) & 0x3Fu));
        }
        for (int i = 0; i < 64; i++) {
            blk[128 + i] = (uint8_t)(((b * 7u + (size_t)i * 9u + 1u) & 0x3Fu));
        }
        for (int i = 0; i < 16; i++) {
            blk[192 + i] = (uint8_t)(1 + ((b + (size_t)i * 3u) & 0x0Fu));
        }
        memcpy(blk + 208, &d_fp16, sizeof(uint16_t));
    }
}

/* ── 256 MiB DRAM Read-Only & STREAM Copy/Triad Workers ── */

typedef struct {
    const uint8_t *buf;
    size_t total_lines; /* 64-byte cachelines */
    uint64_t partial_xor[MAX_THREAD_MODES];
} DramRoCtx;

static void worker_dram_ro(int tid, int n_threads, void *arg) {
    DramRoCtx *ctx = (DramRoCtx *)arg;
    size_t per = ctx->total_lines / (size_t)n_threads;
    size_t l0 = (size_t)tid * per;
    size_t l1 = (tid == n_threads - 1) ? ctx->total_lines : l0 + per;
    const uint8_t *p = ctx->buf + l0 * 64ULL;
    const uint8_t *end = ctx->buf + l1 * 64ULL;

    uint64_t acc = 0;
#if defined(__AVX2__)
    __m256i x0 = _mm256_setzero_si256();
    __m256i x1 = _mm256_setzero_si256();
    __m256i x2 = _mm256_setzero_si256();
    __m256i x3 = _mm256_setzero_si256();
    __m256i x4 = _mm256_setzero_si256();
    __m256i x5 = _mm256_setzero_si256();
    __m256i x6 = _mm256_setzero_si256();
    __m256i x7 = _mm256_setzero_si256();

    for (; p + 256 <= end; p += 256) {
        x0 = _mm256_xor_si256(x0, _mm256_load_si256((const __m256i *)(p + 0)));
        x1 = _mm256_xor_si256(x1, _mm256_load_si256((const __m256i *)(p + 32)));
        x2 = _mm256_xor_si256(x2, _mm256_load_si256((const __m256i *)(p + 64)));
        x3 = _mm256_xor_si256(x3, _mm256_load_si256((const __m256i *)(p + 96)));
        x4 = _mm256_xor_si256(x4, _mm256_load_si256((const __m256i *)(p + 128)));
        x5 = _mm256_xor_si256(x5, _mm256_load_si256((const __m256i *)(p + 160)));
        x6 = _mm256_xor_si256(x6, _mm256_load_si256((const __m256i *)(p + 192)));
        x7 = _mm256_xor_si256(x7, _mm256_load_si256((const __m256i *)(p + 224)));
    }
    for (; p + 64 <= end; p += 64) {
        x0 = _mm256_xor_si256(x0, _mm256_load_si256((const __m256i *)(p + 0)));
        x1 = _mm256_xor_si256(x1, _mm256_load_si256((const __m256i *)(p + 32)));
    }
    __m256i r = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_xor_si256(x0, x1), _mm256_xor_si256(x2, x3)),
        _mm256_xor_si256(_mm256_xor_si256(x4, x5), _mm256_xor_si256(x6, x7)));
    uint64_t lanes[4];
    _mm256_storeu_si256((__m256i *)lanes, r);
    acc = lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3];
#else
    const uint64_t *up = (const uint64_t *)p;
    const uint64_t *uend = (const uint64_t *)end;
    for (; up < uend; up += 8) {
        acc ^= up[0] ^ up[1] ^ up[2] ^ up[3] ^ up[4] ^ up[5] ^ up[6] ^ up[7];
    }
#endif
    ctx->partial_xor[tid] = acc;
}

typedef struct {
    double *a;
    double *b;
    double *c;
    size_t n_elems; /* multiple of 8 for 64-byte slice alignment */
    double scalar;
} StreamCtx;

static void worker_stream_copy(int tid, int n_threads, void *arg) {
    StreamCtx *ctx = (StreamCtx *)arg;
    size_t chunks = ctx->n_elems / 8ULL;
    size_t per = chunks / (size_t)n_threads;
    size_t i0 = ((size_t)tid * per) * 8ULL;
    size_t i1 = (tid == n_threads - 1) ? ctx->n_elems : (((size_t)(tid + 1) * per) * 8ULL);

    const double *a = ctx->a;
    double *c = ctx->c;
#if defined(__AVX2__)
    size_t i = i0;
    for (; i + 8 <= i1; i += 8) {
        __m256d v0 = _mm256_load_pd(a + i);
        __m256d v1 = _mm256_load_pd(a + i + 4);
        _mm256_store_pd(c + i, v0);
        _mm256_store_pd(c + i + 4, v1);
    }
    for (; i < i1; i++) {
        c[i] = a[i];
    }
#else
    for (size_t i = i0; i < i1; i++) {
        c[i] = a[i];
    }
#endif
}

static void worker_stream_triad(int tid, int n_threads, void *arg) {
    StreamCtx *ctx = (StreamCtx *)arg;
    size_t chunks = ctx->n_elems / 8ULL;
    size_t per = chunks / (size_t)n_threads;
    size_t i0 = ((size_t)tid * per) * 8ULL;
    size_t i1 = (tid == n_threads - 1) ? ctx->n_elems : (((size_t)(tid + 1) * per) * 8ULL);

    double *a = ctx->a;
    const double *b = ctx->b;
    const double *c = ctx->c;
    const double s = ctx->scalar;
#if defined(__AVX2__)
    const __m256d vs = _mm256_set1_pd(s);
    size_t i = i0;
    for (; i + 8 <= i1; i += 8) {
        __m256d vb0 = _mm256_load_pd(b + i);
        __m256d vc0 = _mm256_load_pd(c + i);
        __m256d vb1 = _mm256_load_pd(b + i + 4);
        __m256d vc1 = _mm256_load_pd(c + i + 4);
        _mm256_store_pd(a + i, _mm256_fmadd_pd(vs, vc0, vb0));
        _mm256_store_pd(a + i + 4, _mm256_fmadd_pd(vs, vc1, vb1));
    }
    for (; i < i1; i++) {
        a[i] = b[i] + s * c[i];
    }
#else
    for (size_t i = i0; i < i1; i++) {
        a[i] = b[i] + s * c[i];
    }
#endif
}

/* ── Isolated Kernel Benchmarks (Q4_K & Q6_K, dot1 & dot2) ── */

typedef enum {
    KERN_Q4K_DOT1 = 0,
    KERN_Q4K_DOT2 = 1,
    KERN_Q6K_DOT1 = 2,
    KERN_Q6K_DOT2 = 3
} KernelKind;

typedef struct {
    const uint8_t *weights;
    const LilaQ8KBlock *qx;
    int rows;
    int blocks_per_row;
    size_t row_bytes;
    int inner_sweeps;
    KernelKind kind;
    double partial_sum[MAX_THREAD_MODES];
} KernelCtx;

static void worker_kernel(int tid, int n_threads, void *arg) {
    KernelCtx *ctx = (KernelCtx *)arg;
    /* Split rows on even boundary so dot2 operates on exact adjacent pairs */
    int pairs = ctx->rows / 2;
    int p0 = (pairs * tid) / n_threads;
    int p1 = (pairs * (tid + 1)) / n_threads;
    int r0 = p0 * 2;
    int r1 = (tid == n_threads - 1) ? ctx->rows : (p1 * 2);

    const uint8_t *w = ctx->weights;
    const LilaQ8KBlock *qx = ctx->qx;
    const int bpr = ctx->blocks_per_row;
    const size_t rb = ctx->row_bytes;
    double sum = 0.0;

    for (int it = 0; it < ctx->inner_sweeps; it++) {
        switch (ctx->kind) {
            case KERN_Q4K_DOT1:
                for (int r = r0; r < r1; r++) {
                    sum += (double)q4k_dot_q8k_simd(w + (size_t)r * rb, qx, bpr);
                }
                break;
            case KERN_Q4K_DOT2: {
                int r = r0;
                for (; r + 1 < r1; r += 2) {
                    float o0 = 0.0f, o1 = 0.0f;
                    const uint8_t *rp0 = w + (size_t)r * rb;
                    const uint8_t *rp1 = rp0 + rb;
#if defined(__AVX2__)
                    q4k_dot2_q8k_avx2(rp0, rp1, qx, bpr, &o0, &o1);
#else
                    o0 = q4k_dot_q8k_simd(rp0, qx, bpr);
                    o1 = q4k_dot_q8k_simd(rp1, qx, bpr);
#endif
                    sum += (double)o0 + (double)o1;
                }
                if (r < r1) {
                    sum += (double)q4k_dot_q8k_simd(w + (size_t)r * rb, qx, bpr);
                }
                break;
            }
            case KERN_Q6K_DOT1:
                for (int r = r0; r < r1; r++) {
                    sum += (double)q6k_dot_q8k_simd(w + (size_t)r * rb, qx, bpr);
                }
                break;
            case KERN_Q6K_DOT2: {
                int r = r0;
                for (; r + 1 < r1; r += 2) {
                    float o0 = 0.0f, o1 = 0.0f;
                    const uint8_t *rp0 = w + (size_t)r * rb;
                    const uint8_t *rp1 = rp0 + rb;
#if defined(__AVX2__)
                    q6k_dot2_q8k_avx2(rp0, rp1, qx, bpr, &o0, &o1);
#else
                    o0 = q6k_dot_q8k_simd(rp0, qx, bpr);
                    o1 = q6k_dot_q8k_simd(rp1, qx, bpr);
#endif
                    sum += (double)o0 + (double)o1;
                }
                if (r < r1) {
                    sum += (double)q6k_dot_q8k_simd(w + (size_t)r * rb, qx, bpr);
                }
                break;
            }
        }
    }
    ctx->partial_sum[tid] = sum;
}

static void run_dram_and_stream(AsiThreadPool *pool, int n_threads,
                                uint8_t *dram_buf, size_t dram_bytes,
                                ThreadBench *tb) {
    const int n_reps = 7;
    double samples[32];

    /* 1. AVX2 DRAM Read-Only (256 MiB) */
    DramRoCtx ro_ctx;
    memset(&ro_ctx, 0, sizeof(ro_ctx));
    ro_ctx.buf = dram_buf;
    ro_ctx.total_lines = dram_bytes / 64ULL;

    asi_pool_begin(pool);
    asi_pool_run(pool, worker_dram_ro, &ro_ctx); /* warmup */
    uint64_t final_xor = 0;
    for (int rep = 0; rep < n_reps; rep++) {
        double t0 = now_sec();
        asi_pool_run(pool, worker_dram_ro, &ro_ctx);
        double t1 = now_sec();
        samples[rep] = t1 - t0;
        for (int t = 0; t < n_threads; t++) {
            final_xor ^= ro_ctx.partial_xor[t];
        }
    }
    asi_pool_end(pool);
    g_u64_sink ^= final_xor;
    compute_stats(samples, n_reps, (double)dram_bytes, 0.0, (double)(final_xor & 0xFFFFULL), &tb->dram_ro);

    /* 2. STREAM Copy (16 bytes/iter on 80 MiB arrays) */
    StreamCtx sc;
    sc.a = (double *)(dram_buf + 0ULL * STREAM_ARR_BYTES);
    sc.b = (double *)(dram_buf + 1ULL * STREAM_ARR_BYTES);
    sc.c = (double *)(dram_buf + 2ULL * STREAM_ARR_BYTES);
    sc.n_elems = STREAM_ARR_BYTES / sizeof(double);
    sc.scalar = 0.5;

    asi_pool_begin(pool);
    asi_pool_run(pool, worker_stream_copy, &sc); /* warmup */
    for (int rep = 0; rep < n_reps; rep++) {
        double t0 = now_sec();
        asi_pool_run(pool, worker_stream_copy, &sc);
        double t1 = now_sec();
        samples[rep] = t1 - t0;
        g_f64_sink += sc.c[rep];
    }
    asi_pool_end(pool);
    compute_stats(samples, n_reps, (double)sc.n_elems * 16.0, 0.0,
                  sc.c[0] + sc.c[sc.n_elems - 1], &tb->stream_copy);

    /* 3. STREAM Triad (24 bytes/iter on 80 MiB arrays) */
    asi_pool_begin(pool);
    asi_pool_run(pool, worker_stream_triad, &sc); /* warmup */
    for (int rep = 0; rep < n_reps; rep++) {
        double t0 = now_sec();
        asi_pool_run(pool, worker_stream_triad, &sc);
        double t1 = now_sec();
        samples[rep] = t1 - t0;
        g_f64_sink += sc.a[rep];
    }
    asi_pool_end(pool);
    compute_stats(samples, n_reps, (double)sc.n_elems * 24.0,
                  (double)sc.n_elems * 2.0,
                  sc.a[0] + sc.a[sc.n_elems - 1], &tb->stream_triad);
}

static void bench_one_kernel(AsiThreadPool *pool, int n_threads,
                             const uint8_t *weights, const LilaQ8KBlock *qx,
                             int rows, int blocks_per_row, size_t row_bytes,
                             int inner_sweeps, int n_reps, KernelKind kind,
                             BenchMetric *out) {
    KernelCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.weights = weights;
    ctx.qx = qx;
    ctx.rows = rows;
    ctx.blocks_per_row = blocks_per_row;
    ctx.row_bytes = row_bytes;
    ctx.inner_sweeps = inner_sweeps;
    ctx.kind = kind;

    double samples[32];
    double acc_sum = 0.0;

    asi_pool_begin(pool);
    asi_pool_run(pool, worker_kernel, &ctx); /* warmup */
    for (int rep = 0; rep < n_reps; rep++) {
        double t0 = now_sec();
        asi_pool_run(pool, worker_kernel, &ctx);
        double t1 = now_sec();
        samples[rep] = t1 - t0;
        for (int t = 0; t < n_threads; t++) {
            acc_sum += ctx.partial_sum[t];
        }
    }
    asi_pool_end(pool);
    g_f64_sink += acc_sum;

    double bytes_per_sweep = (double)rows * (double)row_bytes;
    double weights_per_sweep = (double)rows * (double)blocks_per_row * (double)QK_K;
    double total_bytes = bytes_per_sweep * (double)inner_sweeps;
    double total_flops = 2.0 * weights_per_sweep * (double)inner_sweeps;
    compute_stats(samples, n_reps, total_bytes, total_flops,
                  ctx.partial_sum[0], out);
}

static void run_isolated_kernels(const int *thread_list, int n_thread_modes,
                                 ThreadBench *results) {
    const int cols = 1024;
    const int blocks_per_row = cols / QK_K; /* 4 */
    const size_t q4_row_bytes = (size_t)blocks_per_row * Q4_K_BLOCK_SIZE; /* 576 B */
    const size_t q6_row_bytes = (size_t)blocks_per_row * Q6_K_BLOCK_SIZE; /* 840 B */

    const int l3_q4_rows = ((int)(L3_TARGET_BYTES / q4_row_bytes)) & ~1;     /* 3640 (~1.999 MiB) */
    const int l3_q6_rows = ((int)(L3_TARGET_BYTES / q6_row_bytes)) & ~1;     /* 2496 (~1.999 MiB) */
    const int dram_q4_rows = ((int)(DRAM_TARGET_BYTES / q4_row_bytes)) & ~1; /* 116508 (~64.00 MiB) */
    const int dram_q6_rows = ((int)(DRAM_TARGET_BYTES / q6_row_bytes)) & ~1; /* 79890 (~64.00 MiB) */

    float *x_fp32 = (float *)xaligned_alloc(64, (size_t)cols * sizeof(float));
    LilaQ8KBlock *qx = (LilaQ8KBlock *)xaligned_alloc(64, (size_t)blocks_per_row * sizeof(LilaQ8KBlock));
    for (int i = 0; i < cols; i++) {
        x_fp32[i] = 0.01f * (float)((i % 37) - 18);
    }
    quantize_vec_q8k(qx, x_fp32, cols);

    uint8_t *w_q4_dram = (uint8_t *)xaligned_alloc(64, (size_t)dram_q4_rows * q4_row_bytes);
    uint8_t *w_q6_dram = (uint8_t *)xaligned_alloc(64, (size_t)dram_q6_rows * q6_row_bytes);
    if (!w_q4_dram || !w_q6_dram) {
        fprintf(stderr, "ERROR: Failed to allocate 64 MiB kernel buffers\n");
        exit(1);
    }
    init_q4k_weights(w_q4_dram, (size_t)dram_q4_rows * (size_t)blocks_per_row);
    init_q6k_weights(w_q6_dram, (size_t)dram_q6_rows * (size_t)blocks_per_row);

    for (int ti = 0; ti < n_thread_modes; ti++) {
        int nt = thread_list[ti];
        AsiThreadPool *pool = asi_pool_create(nt);

        /* L3-resident (~2 MiB = prefix of buffers; 64 inner sweeps = ~128 MiB/sample) */
        bench_one_kernel(pool, nt, w_q4_dram, qx, l3_q4_rows, blocks_per_row,
                         q4_row_bytes, 64, 7, KERN_Q4K_DOT1, &results[ti].l3_q4k_dot1);
        bench_one_kernel(pool, nt, w_q4_dram, qx, l3_q4_rows, blocks_per_row,
                         q4_row_bytes, 64, 7, KERN_Q4K_DOT2, &results[ti].l3_q4k_dot2);
        bench_one_kernel(pool, nt, w_q6_dram, qx, l3_q6_rows, blocks_per_row,
                         q6_row_bytes, 64, 7, KERN_Q6K_DOT1, &results[ti].l3_q6k_dot1);
        bench_one_kernel(pool, nt, w_q6_dram, qx, l3_q6_rows, blocks_per_row,
                         q6_row_bytes, 64, 7, KERN_Q6K_DOT2, &results[ti].l3_q6k_dot2);

        /* DRAM-resident (~64 MiB; 4 inner sweeps = ~256 MiB/sample) */
        bench_one_kernel(pool, nt, w_q4_dram, qx, dram_q4_rows, blocks_per_row,
                         q4_row_bytes, 4, 5, KERN_Q4K_DOT1, &results[ti].dram_q4k_dot1);
        bench_one_kernel(pool, nt, w_q4_dram, qx, dram_q4_rows, blocks_per_row,
                         q4_row_bytes, 4, 5, KERN_Q4K_DOT2, &results[ti].dram_q4k_dot2);
        bench_one_kernel(pool, nt, w_q6_dram, qx, dram_q6_rows, blocks_per_row,
                         q6_row_bytes, 4, 5, KERN_Q6K_DOT1, &results[ti].dram_q6k_dot1);
        bench_one_kernel(pool, nt, w_q6_dram, qx, dram_q6_rows, blocks_per_row,
                         q6_row_bytes, 4, 5, KERN_Q6K_DOT2, &results[ti].dram_q6k_dot2);

        asi_pool_destroy(pool);
    }

    free(w_q4_dram);
    free(w_q6_dram);
    free(qx);
    free(x_fp32);
}

/* ── ASI Model Inspection & Full-Model Decode Benchmark ── */

static void touch_pages(const uint8_t *buf, size_t size) {
    uint64_t sum = 0;
    for (size_t off = 0; off < size; off += 4096ULL) {
        sum += buf[off];
    }
    g_u64_sink ^= sum;
}

static bool inspect_asi_model(const char *path, AsiBreakdown *bd) {
    char err[256] = {0};
    AsiV2Cartridge cart;
    memset(&cart, 0, sizeof(cart));

    snprintf(bd->path, sizeof(bd->path), "%s", path);
    if (asi_v2_open(path, &cart, err, sizeof(err)) != 0) {
        snprintf(bd->error, sizeof(bd->error), "%s", err[0] ? err : "failed to open .asi");
        bd->present = false;
        return false;
    }

    bd->present = true;
    snprintf(bd->arch, sizeof(bd->arch), "%s", asi_v2_arch_name(cart.config.arch_id));
    bd->file_size = cart.file_size;
    bd->weights_section_size = cart.weights_data_size;
    bd->n_tensors = cart.n_tensors;

    bd->vocab_size = cart.config.vocab_size;
    bd->n_embd = cart.config.hidden_size;
    bd->n_layers = cart.config.n_layers;
    bd->n_heads = cart.config.n_heads;
    bd->n_kv_heads = cart.config.n_kv_heads;
    bd->n_ff = cart.config.intermediate_size;
    bd->full_attention_interval = cart.config.full_attention_interval;
    bd->ssm_state_size = cart.config.ssm_state_size;
    bd->ssm_conv_kernel = cart.config.ssm_conv_kernel;
    bd->ssm_inner_size = cart.config.ssm_inner_size;

    const AsiV2TensorEntry *out_w = asi_v2_find_tensor(&cart, "output.weight");
    bd->tied_embeddings = (out_w == NULL);

    for (uint32_t i = 0; i < cart.n_tensors; i++) {
        const AsiV2TensorEntry *t = &cart.tensors[i];
        if (t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) {
            bd->repacked_tensors++;
        }
        bd->total_model_bytes += t->data_size;

        uint64_t active_sz = t->data_size;
        bool is_embd_or_head = false;
        if (strcmp(t->name, "token_embd.weight") == 0) {
            is_embd_or_head = true;
            if (!bd->tied_embeddings && t->shape[0] > 0) {
                /* Untied: embedding lookup touches only 1 row/token */
                active_sz = t->data_size / t->shape[0];
            }
        } else if (strcmp(t->name, "output.weight") == 0) {
            is_embd_or_head = true;
        }

        bd->active_decode_bytes += active_sz;

        if (t->quant_type == 14) { /* Q6_K */
            bd->q6k_total_bytes += active_sz;
            if (is_embd_or_head) {
                bd->q6k_token_embd_bytes += active_sz;
            } else if (strstr(t->name, "attn_qkv") != NULL) {
                bd->q6k_attn_qkv_bytes += active_sz;
            } else if (strstr(t->name, "ffn_down") != NULL) {
                bd->q6k_ffn_down_bytes += active_sz;
            } else {
                bd->q6k_other_bytes += active_sz;
            }
        } else if (t->quant_type == 12) { /* Q4_K */
            bd->q4k_total_bytes += active_sz;
            if (strstr(t->name, "attn_qkv") != NULL) {
                bd->q4k_attn_qkv_bytes += active_sz;
            } else if (strstr(t->name, "ffn_down") != NULL) {
                bd->q4k_ffn_down_bytes += active_sz;
            } else {
                bd->q4k_other_bytes += active_sz;
            }
        } else if (t->quant_type == 0) { /* F32 */
            bd->f32_bytes += active_sz;
        } else {
            bd->other_qtype_bytes += active_sz;
        }
    }

    /* Warm page cache cleanly before benchmarking decode */
    touch_pages(cart.weights_data, cart.weights_data_size);
    asi_v2_close(&cart);
    return true;
}

static void run_asi_decode(const char *asi_path, const AsiBreakdown *bd,
                           const int *thread_list, int n_thread_modes,
                           const ThreadBench *tb,
                           int n_tokens, int max_ctx,
                           AsiDecodeResult *decode_out) {
    char err[256] = {0};
    LilaAsiModel *model = lila_asi_open(asi_path, err, sizeof(err));
    if (!model) {
        fprintf(stderr, "WARN: lila_asi_open(%s) failed: %s\n", asi_path, err);
        return;
    }

    float *logits = (float *)xaligned_alloc(64, (size_t)bd->vocab_size * sizeof(float));
    if (!logits) {
        fprintf(stderr, "ERROR: Failed to allocate logits buffer\n");
        lila_asi_close(model);
        return;
    }

    for (int ti = 0; ti < n_thread_modes; ti++) {
        int nt = thread_list[ti];
        AsiDecodeResult *dr = &decode_out[ti];
        dr->threads = nt;
        dr->n_tokens = n_tokens;

        LilaAsiSession *sess = lila_asi_session_create(model, (uint32_t)max_ctx, (uint32_t)nt, err, sizeof(err));
        if (!sess) {
            fprintf(stderr, "WARN: lila_asi_session_create(threads=%d) failed: %s\n", nt, err);
            continue;
        }

        /* Warmup pass (populates caches, thread pools, SSM/KV scratch) */
        uint64_t rng = 42ULL;
        int32_t tok = 9906;
        if (lila_asi_eval_token(sess, tok, 0, logits) == 0) {
            tok = lila_asi_sample(logits, bd->vocab_size, 0.0f, 0, 1.0f, &rng);
            lila_asi_eval_token(sess, tok, 1, logits);
        }
        lila_asi_session_reset(sess);

        /* Prefill 1 seed token (pos=0) then decode n_tokens greedy tokens */
        tok = 9906;
        if (lila_asi_eval_token(sess, tok, 0, logits) != 0) {
            fprintf(stderr, "WARN: eval_token seed failed at threads=%d\n", nt);
            lila_asi_session_free(sess);
            continue;
        }
        tok = lila_asi_sample(logits, bd->vocab_size, 0.0f, 0, 1.0f, &rng);

        double eval_acc_s = 0.0;
        double t_start = now_sec();
        for (int step = 0; step < n_tokens; step++) {
            double te0 = now_sec();
            if (lila_asi_eval_token(sess, tok, (uint32_t)(1 + step), logits) != 0) {
                fprintf(stderr, "WARN: eval_token step %d failed\n", step);
                break;
            }
            double te1 = now_sec();
            eval_acc_s += (te1 - te0);
            tok = lila_asi_sample(logits, bd->vocab_size, 0.0f, 0, 1.0f, &rng);
        }
        double t_end = now_sec();

        lila_asi_session_free(sess);

        dr->ran = true;
        dr->elapsed_s = t_end - t_start;
        dr->eval_only_s = eval_acc_s;
        dr->tok_s = (dr->elapsed_s > 0.0) ? ((double)n_tokens / dr->elapsed_s) : 0.0;
        dr->eval_tok_s = (dr->eval_only_s > 0.0) ? ((double)n_tokens / dr->eval_only_s) : 0.0;
        dr->ms_per_tok = (dr->tok_s > 0.0) ? (1000.0 / dr->tok_s) : 0.0;
        dr->effective_gb_s = ((double)bd->active_decode_bytes * dr->tok_s) / 1e9;

        double dram_ro_bw = tb[ti].dram_ro.median_gb_s;
        double stream_cp_bw = tb[ti].stream_copy.median_gb_s;
        dr->dram_ro_util_pct = (dram_ro_bw > 0.0) ? (100.0 * dr->effective_gb_s / dram_ro_bw) : 0.0;
        dr->stream_copy_util_pct = (stream_cp_bw > 0.0) ? (100.0 * dr->effective_gb_s / stream_cp_bw) : 0.0;

        /* Roofline & Kernel-Predicted Matvec Time (using DRAM dot2 kernels, matching asi_matvec_range) */
        double b_q6k = tb[ti].dram_q6k_dot2.median_gb_s;
        double b_q4k = tb[ti].dram_q4k_dot2.median_gb_s;
        dr->pred_q6k_ms = (b_q6k > 0.0) ? ((double)bd->q6k_total_bytes / (b_q6k * 1e6)) : 0.0;
        dr->pred_q4k_ms = (b_q4k > 0.0) ? ((double)bd->q4k_total_bytes / (b_q4k * 1e6)) : 0.0;
        dr->pred_matvec_ms = dr->pred_q6k_ms + dr->pred_q4k_ms;
        if (dr->pred_matvec_ms > 0.0) {
            dr->pred_q6k_time_pct = 100.0 * dr->pred_q6k_ms / dr->pred_matvec_ms;
            dr->pred_q4k_time_pct = 100.0 * dr->pred_q4k_ms / dr->pred_matvec_ms;
            dr->pred_matvec_gb_s = (double)(bd->q6k_total_bytes + bd->q4k_total_bytes) / (dr->pred_matvec_ms * 1e6);
            dr->pred_matvec_tok_s = 1000.0 / dr->pred_matvec_ms;
        }
        dr->roofline_dram_tok_s = (dram_ro_bw > 0.0 && bd->active_decode_bytes > 0)
                                  ? ((dram_ro_bw * 1e9) / (double)bd->active_decode_bytes)
                                  : 0.0;
        dr->residual_non_matvec_ms = dr->ms_per_tok - dr->pred_matvec_ms;
        dr->residual_pct = (dr->ms_per_tok > 0.0) ? (100.0 * dr->residual_non_matvec_ms / dr->ms_per_tok) : 0.0;
    }

    free(logits);
    lila_asi_close(model);
}

/* ── Output Formatting (Human-Readable Tables & JSON) ── */

static double pct_u64(uint64_t part, uint64_t whole) {
    return (whole > 0) ? (100.0 * (double)part / (double)whole) : 0.0;
}

static void print_human_report(const ThreadBench *tb, int n_thread_modes,
                               const AsiBreakdown *bd,
                               const AsiDecodeResult *decode_out) {
    printf("==========================================================================================\n");
    printf(" LILA / ASI CPU ROOFLINE, DRAM/STREAM BANDWIDTH & DECODE STAGE BREAKDOWN\n");
    printf("==========================================================================================\n\n");

    printf("[1] 256 MiB DRAM Read-Only & STREAM Copy/Triad Bandwidth\n");
    printf("%-8s | %-22s | %10s | %10s | %10s | %10s\n",
           "Threads", "Benchmark", "Median GB/s", "Best GB/s", "Mean GB/s", "Time (ms)");
    printf("---------+------------------------+------------+------------+------------+-----------\n");
    for (int i = 0; i < n_thread_modes; i++) {
        const ThreadBench *t = &tb[i];
        printf("%-8d | %-22s | %10.2f | %10.2f | %10.2f | %10.2f\n",
               t->threads, "AVX2 DRAM Read-Only",
               t->dram_ro.median_gb_s, t->dram_ro.best_gb_s, t->dram_ro.mean_gb_s, t->dram_ro.elapsed_ms);
        printf("%-8d | %-22s | %10.2f | %10.2f | %10.2f | %10.2f\n",
               t->threads, "STREAM Copy (16 B/it)",
               t->stream_copy.median_gb_s, t->stream_copy.best_gb_s, t->stream_copy.mean_gb_s, t->stream_copy.elapsed_ms);
        printf("%-8d | %-22s | %10.2f | %10.2f | %10.2f | %10.2f\n",
               t->threads, "STREAM Triad (24 B/it)",
               t->stream_triad.median_gb_s, t->stream_triad.best_gb_s, t->stream_triad.mean_gb_s, t->stream_triad.elapsed_ms);
    }
    printf("  (Sinks: u64=0x%016llx, f64=%.6e)\n\n",
           (unsigned long long)g_u64_sink, g_f64_sink);

    printf("[2] Isolated Quantized Kernel Throughput (cols=1024, 2*weights FLOPs)\n");
    printf("%-8s | %-8s | %-18s | %10s | %10s | %12s\n",
           "Threads", "WorkSet", "Kernel", "Median GB/s", "Best GB/s", "Median GFLOP/s");
    printf("---------+----------+--------------------+------------+------------+---------------\n");
    for (int i = 0; i < n_thread_modes; i++) {
        const ThreadBench *t = &tb[i];
        const struct {
            const char *ws;
            const char *name;
            const BenchMetric *m;
        } rows[] = {
            {"L3 2MiB",   "Q4_K dot1 (144B)", &t->l3_q4k_dot1},
            {"L3 2MiB",   "Q4_K dot2 (144B)", &t->l3_q4k_dot2},
            {"L3 2MiB",   "Q6_K dot1 (210B)", &t->l3_q6k_dot1},
            {"L3 2MiB",   "Q6_K dot2 (210B)", &t->l3_q6k_dot2},
            {"DRAM 64M",  "Q4_K dot1 (144B)", &t->dram_q4k_dot1},
            {"DRAM 64M",  "Q4_K dot2 (144B)", &t->dram_q4k_dot2},
            {"DRAM 64M",  "Q6_K dot1 (210B)", &t->dram_q6k_dot1},
            {"DRAM 64M",  "Q6_K dot2 (210B)", &t->dram_q6k_dot2},
        };
        for (size_t k = 0; k < sizeof(rows) / sizeof(rows[0]); k++) {
            printf("%-8d | %-8s | %-18s | %10.2f | %10.2f | %12.2f\n",
                   t->threads, rows[k].ws, rows[k].name,
                   rows[k].m->median_gb_s, rows[k].m->best_gb_s, rows[k].m->median_gflops);
        }
    }
    printf("\n");

    if (!bd->present) {
        printf("[3] ASI Model Inspection & Decode: SKIPPED (%s: %s)\n",
               bd->path, bd->error[0] ? bd->error : "not present");
        return;
    }

    printf("[3] ASI Model Active Decode Weight Breakdown: %s\n", bd->path);
    printf("  Arch=%s | Layers=%u | Emb=%u | Vocab=%u | FFN=%u | Tensors=%u | TiedEmb=%s | AVX2Repack=%u\n",
           bd->arch, bd->n_layers, bd->n_embd, bd->vocab_size, bd->n_ff,
           bd->n_tensors, bd->tied_embeddings ? "yes" : "no", bd->repacked_tensors);
    printf("  Total File=%.2f MiB | Model Tensors=%.2f MB | Active Decode Weights/tok=%llu B (%.2f MB / %.2f MiB)\n\n",
           (double)bd->file_size / (1024.0 * 1024.0),
           (double)bd->total_model_bytes / 1e6,
           (unsigned long long)bd->active_decode_bytes,
           (double)bd->active_decode_bytes / 1e6,
           (double)bd->active_decode_bytes / (1024.0 * 1024.0));

    printf("%-28s | %14s | %8s | %8s\n",
           "Tensor Quant / Role", "Active Bytes", "MB", "% Total");
    printf("-----------------------------+----------------+----------+---------\n");
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "Q6_K TOTAL (210B / 256w)",
           (unsigned long long)bd->q6k_total_bytes, (double)bd->q6k_total_bytes / 1e6,
           pct_u64(bd->q6k_total_bytes, bd->active_decode_bytes));
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "  - token_embd / lm_head",
           (unsigned long long)bd->q6k_token_embd_bytes, (double)bd->q6k_token_embd_bytes / 1e6,
           pct_u64(bd->q6k_token_embd_bytes, bd->active_decode_bytes));
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "  - SSM/Attn attn_qkv",
           (unsigned long long)bd->q6k_attn_qkv_bytes, (double)bd->q6k_attn_qkv_bytes / 1e6,
           pct_u64(bd->q6k_attn_qkv_bytes, bd->active_decode_bytes));
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "  - ffn_down",
           (unsigned long long)bd->q6k_ffn_down_bytes, (double)bd->q6k_ffn_down_bytes / 1e6,
           pct_u64(bd->q6k_ffn_down_bytes, bd->active_decode_bytes));
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "  - other Q6_K (attn_v)",
           (unsigned long long)bd->q6k_other_bytes, (double)bd->q6k_other_bytes / 1e6,
           pct_u64(bd->q6k_other_bytes, bd->active_decode_bytes));
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "Q4_K TOTAL (144B / 256w)",
           (unsigned long long)bd->q4k_total_bytes, (double)bd->q4k_total_bytes / 1e6,
           pct_u64(bd->q4k_total_bytes, bd->active_decode_bytes));
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "  - attn_qkv (full-attn)",
           (unsigned long long)bd->q4k_attn_qkv_bytes, (double)bd->q4k_attn_qkv_bytes / 1e6,
           pct_u64(bd->q4k_attn_qkv_bytes, bd->active_decode_bytes));
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "  - ffn_down",
           (unsigned long long)bd->q4k_ffn_down_bytes, (double)bd->q4k_ffn_down_bytes / 1e6,
           pct_u64(bd->q4k_ffn_down_bytes, bd->active_decode_bytes));
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "  - other Q4_K",
           (unsigned long long)bd->q4k_other_bytes, (double)bd->q4k_other_bytes / 1e6,
           pct_u64(bd->q4k_other_bytes, bd->active_decode_bytes));
    printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "F32 TOTAL (norms/SSM/biases)",
           (unsigned long long)bd->f32_bytes, (double)bd->f32_bytes / 1e6,
           pct_u64(bd->f32_bytes, bd->active_decode_bytes));
    if (bd->other_qtype_bytes > 0) {
        printf("%-28s | %14llu | %8.2f | %7.2f%%\n", "OTHER QTYPES",
               (unsigned long long)bd->other_qtype_bytes, (double)bd->other_qtype_bytes / 1e6,
               pct_u64(bd->other_qtype_bytes, bd->active_decode_bytes));
    }
    printf("\n");

    printf("[4] Full-Model Decode Performance & Roofline Reconciliation\n");
    printf("%-7s | %8s | %8s | %9s | %8s | %10s | %11s | %11s | %10s\n",
           "Threads", "Decode", "ms/tok", "Eff.GB/s", "%DRAM-RO", "PredMV(ms)", "Q6K_t/Q4K_t", "Resid(ms)", "DRAM Ceilt");
    printf("--------+----------+----------+-----------+----------+------------+-------------+-------------+-----------\n");
    for (int i = 0; i < n_thread_modes; i++) {
        const AsiDecodeResult *d = &decode_out[i];
        if (!d->ran) continue;
        printf("%-7d | %6.2f/s | %8.2f | %9.2f | %7.1f%% | %10.2f | %4.1f%%/%4.1f%% | %11.2f | %9.2f/s\n",
               d->threads, d->tok_s, d->ms_per_tok, d->effective_gb_s, d->dram_ro_util_pct,
               d->pred_matvec_ms, d->pred_q6k_time_pct, d->pred_q4k_time_pct,
               d->residual_non_matvec_ms, d->roofline_dram_tok_s);
    }
    printf("\n[5] Key Reconciliation Notes:\n");
    printf("  * Working set: bench_dot (1T Q4_K-only, ~2 MiB) fits inside the 3 MiB L3 cache,\n"
           "    whereas full-model decode streams %llu B (%.2f MB) from DRAM each token.\n",
           (unsigned long long)bd->active_decode_bytes, (double)bd->active_decode_bytes / 1e6);
    printf("  * Block footprint & mix: Q4_K is 144 B/256w (0.5625 B/w) vs Q6_K 210 B/256w (0.8203 B/w, +45.8%%).\n"
           "    Active decode bytes = %.2f%% Q6_K + %.2f%% Q4_K + %.2f%% F32 (token_embd/lm_head alone = %.2f%%).\n",
           pct_u64(bd->q6k_total_bytes, bd->active_decode_bytes),
           pct_u64(bd->q4k_total_bytes, bd->active_decode_bytes),
           pct_u64(bd->f32_bytes, bd->active_decode_bytes),
           pct_u64(bd->q6k_token_embd_bytes, bd->active_decode_bytes));
    for (int i = 0; i < n_thread_modes; i++) {
        const AsiDecodeResult *d = &decode_out[i];
        if (!d->ran) continue;
        printf("  * threads=%d: Isolated DRAM Q6_K dot2=%.2f GB/s vs Q4_K dot2=%.2f GB/s -> Q6_K takes %.1f%% of\n"
               "    predicted matvec time (%.2f ms Q6_K + %.2f ms Q4_K = %.2f ms, %.2f GB/s harmonic BW).\n"
               "    Measured decode=%.2f tok/s (%.2f ms/tok, %.2f GB/s = %.1f%% of %.2f GB/s DRAM-RO),\n"
               "    leaving %.2f ms (%.1f%%) for quantize_vec_q8k, SSM/Attn ops, lm_head softmax, and sync barriers.\n",
               d->threads, tb[i].dram_q6k_dot2.median_gb_s, tb[i].dram_q4k_dot2.median_gb_s,
               d->pred_q6k_time_pct, d->pred_q6k_ms, d->pred_q4k_ms, d->pred_matvec_ms,
               d->pred_matvec_gb_s, d->tok_s, d->ms_per_tok, d->effective_gb_s,
               d->dram_ro_util_pct, tb[i].dram_ro.median_gb_s,
               d->residual_non_matvec_ms, d->residual_pct);
    }
}

static void write_json_metric(FILE *fp, const char *name, const BenchMetric *m, bool comma) {
    fprintf(fp,
            "      \"%s\": {\"median_gb_s\": %.4f, \"best_gb_s\": %.4f, \"mean_gb_s\": %.4f, "
            "\"median_gflops\": %.4f, \"best_gflops\": %.4f, \"elapsed_ms\": %.4f}%s\n",
            name, m->median_gb_s, m->best_gb_s, m->mean_gb_s,
            m->median_gflops, m->best_gflops, m->elapsed_ms,
            comma ? "," : "");
}

static void write_json_escaped(FILE *fp, const char *s) {
    fputc('"', fp);
    for (const char *p = s; *p; p++) {
        if (*p == '"' || *p == '\\') fputc('\\', fp);
        else if (*p == '\n') { fputs("\\n", fp); continue; }
        fputc(*p, fp);
    }
    fputc('"', fp);
}

static int write_json_report(const char *path, const ThreadBench *tb, int n_thread_modes,
                             const AsiBreakdown *bd, const AsiDecodeResult *decode_out) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Failed to open json output path %s: %s\n", path, strerror(errno));
        return -1;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"config\": {\n");
    fprintf(fp, "    \"dram_buffer_bytes\": %llu,\n", (unsigned long long)DRAM_BUF_BYTES);
    fprintf(fp, "    \"stream_array_bytes\": %llu,\n", (unsigned long long)STREAM_ARR_BYTES);
    fprintf(fp, "    \"l3_target_bytes\": %llu,\n", (unsigned long long)L3_TARGET_BYTES);
    fprintf(fp, "    \"dram_target_bytes\": %llu\n", (unsigned long long)DRAM_TARGET_BYTES);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"synthetic_benchmarks\": [\n");
    for (int i = 0; i < n_thread_modes; i++) {
        const ThreadBench *t = &tb[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"threads\": %d,\n", t->threads);
        write_json_metric(fp, "avx2_dram_read_only", &t->dram_ro, true);
        write_json_metric(fp, "stream_copy", &t->stream_copy, true);
        write_json_metric(fp, "stream_triad", &t->stream_triad, true);
        write_json_metric(fp, "l3_q4k_dot1", &t->l3_q4k_dot1, true);
        write_json_metric(fp, "l3_q4k_dot2", &t->l3_q4k_dot2, true);
        write_json_metric(fp, "l3_q6k_dot1", &t->l3_q6k_dot1, true);
        write_json_metric(fp, "l3_q6k_dot2", &t->l3_q6k_dot2, true);
        write_json_metric(fp, "dram_q4k_dot1", &t->dram_q4k_dot1, true);
        write_json_metric(fp, "dram_q4k_dot2", &t->dram_q4k_dot2, true);
        write_json_metric(fp, "dram_q6k_dot1", &t->dram_q6k_dot1, true);
        write_json_metric(fp, "dram_q6k_dot2", &t->dram_q6k_dot2, false);
        fprintf(fp, "    }%s\n", (i + 1 < n_thread_modes) ? "," : "");
    }
    fprintf(fp, "  ],\n");

    fprintf(fp, "  \"asi_model\": {\n");
    fprintf(fp, "    \"present\": %s,\n", bd->present ? "true" : "false");
    fprintf(fp, "    \"path\": ");
    write_json_escaped(fp, bd->path);
    fprintf(fp, ",\n");
    if (!bd->present) {
        fprintf(fp, "    \"error\": ");
        write_json_escaped(fp, bd->error);
        fprintf(fp, "\n");
    } else {
        fprintf(fp, "    \"arch\": ");
        write_json_escaped(fp, bd->arch);
        fprintf(fp, ",\n");
        fprintf(fp, "    \"file_size_bytes\": %llu,\n", (unsigned long long)bd->file_size);
        fprintf(fp, "    \"weights_section_bytes\": %llu,\n", (unsigned long long)bd->weights_section_size);
        fprintf(fp, "    \"n_tensors\": %u,\n", bd->n_tensors);
        fprintf(fp, "    \"repacked_tensors\": %u,\n", bd->repacked_tensors);
        fprintf(fp, "    \"tied_embeddings\": %s,\n", bd->tied_embeddings ? "true" : "false");
        fprintf(fp, "    \"vocab_size\": %u,\n", bd->vocab_size);
        fprintf(fp, "    \"n_embd\": %u,\n", bd->n_embd);
        fprintf(fp, "    \"n_layers\": %u,\n", bd->n_layers);
        fprintf(fp, "    \"total_model_tensor_bytes\": %llu,\n", (unsigned long long)bd->total_model_bytes);
        fprintf(fp, "    \"active_decode_weight_bytes\": %llu,\n", (unsigned long long)bd->active_decode_bytes);
        fprintf(fp, "    \"byte_breakdown\": {\n");
        fprintf(fp, "      \"q6k_total_bytes\": %llu,\n", (unsigned long long)bd->q6k_total_bytes);
        fprintf(fp, "      \"q6k_pct\": %.4f,\n", pct_u64(bd->q6k_total_bytes, bd->active_decode_bytes));
        fprintf(fp, "      \"q6k_token_embd_lm_head_bytes\": %llu,\n", (unsigned long long)bd->q6k_token_embd_bytes);
        fprintf(fp, "      \"q6k_attn_qkv_bytes\": %llu,\n", (unsigned long long)bd->q6k_attn_qkv_bytes);
        fprintf(fp, "      \"q6k_ffn_down_bytes\": %llu,\n", (unsigned long long)bd->q6k_ffn_down_bytes);
        fprintf(fp, "      \"q6k_other_bytes\": %llu,\n", (unsigned long long)bd->q6k_other_bytes);
        fprintf(fp, "      \"q6k_ffn_down_and_other_bytes\": %llu,\n",
                (unsigned long long)(bd->q6k_ffn_down_bytes + bd->q6k_other_bytes));
        fprintf(fp, "      \"q4k_total_bytes\": %llu,\n", (unsigned long long)bd->q4k_total_bytes);
        fprintf(fp, "      \"q4k_pct\": %.4f,\n", pct_u64(bd->q4k_total_bytes, bd->active_decode_bytes));
        fprintf(fp, "      \"q4k_attn_qkv_bytes\": %llu,\n", (unsigned long long)bd->q4k_attn_qkv_bytes);
        fprintf(fp, "      \"q4k_ffn_down_bytes\": %llu,\n", (unsigned long long)bd->q4k_ffn_down_bytes);
        fprintf(fp, "      \"q4k_other_bytes\": %llu,\n", (unsigned long long)bd->q4k_other_bytes);
        fprintf(fp, "      \"f32_bytes\": %llu,\n", (unsigned long long)bd->f32_bytes);
        fprintf(fp, "      \"f32_pct\": %.4f,\n", pct_u64(bd->f32_bytes, bd->active_decode_bytes));
        fprintf(fp, "      \"other_qtype_bytes\": %llu\n", (unsigned long long)bd->other_qtype_bytes);
        fprintf(fp, "    },\n");
        fprintf(fp, "    \"decode_runs\": [\n");
        bool first_ran = true;
        for (int i = 0; i < n_thread_modes; i++) {
            const AsiDecodeResult *d = &decode_out[i];
            if (!d->ran) continue;
            if (!first_ran) fprintf(fp, ",\n");
            first_ran = false;
            fprintf(fp, "      {\n");
            fprintf(fp, "        \"threads\": %d,\n", d->threads);
            fprintf(fp, "        \"n_tokens\": %d,\n", d->n_tokens);
            fprintf(fp, "        \"elapsed_s\": %.6f,\n", d->elapsed_s);
            fprintf(fp, "        \"eval_only_s\": %.6f,\n", d->eval_only_s);
            fprintf(fp, "        \"tok_s\": %.4f,\n", d->tok_s);
            fprintf(fp, "        \"eval_tok_s\": %.4f,\n", d->eval_tok_s);
            fprintf(fp, "        \"ms_per_tok\": %.4f,\n", d->ms_per_tok);
            fprintf(fp, "        \"effective_gb_s\": %.4f,\n", d->effective_gb_s);
            fprintf(fp, "        \"dram_ro_utilization_pct\": %.4f,\n", d->dram_ro_util_pct);
            fprintf(fp, "        \"stream_copy_utilization_pct\": %.4f,\n", d->stream_copy_util_pct);
            fprintf(fp, "        \"pred_q6k_ms\": %.4f,\n", d->pred_q6k_ms);
            fprintf(fp, "        \"pred_q4k_ms\": %.4f,\n", d->pred_q4k_ms);
            fprintf(fp, "        \"pred_matvec_ms\": %.4f,\n", d->pred_matvec_ms);
            fprintf(fp, "        \"pred_q6k_time_pct\": %.4f,\n", d->pred_q6k_time_pct);
            fprintf(fp, "        \"pred_q4k_time_pct\": %.4f,\n", d->pred_q4k_time_pct);
            fprintf(fp, "        \"pred_matvec_gb_s\": %.4f,\n", d->pred_matvec_gb_s);
            fprintf(fp, "        \"roofline_dram_tok_s\": %.4f,\n", d->roofline_dram_tok_s);
            fprintf(fp, "        \"pred_matvec_tok_s\": %.4f,\n", d->pred_matvec_tok_s);
            fprintf(fp, "        \"residual_non_matvec_ms\": %.4f,\n", d->residual_non_matvec_ms);
            fprintf(fp, "        \"residual_non_matvec_pct\": %.4f\n", d->residual_pct);
            fprintf(fp, "      }");
        }
        fprintf(fp, "\n    ]\n");
    }
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");
    fclose(fp);
    return 0;
}

static int parse_threads_arg(const char *arg, int *out_list, int max_n) {
    char tmp[128];
    snprintf(tmp, sizeof(tmp), "%s", arg);
    int count = 0;
    char *save = NULL;
    char *tok = strtok_r(tmp, ",", &save);
    while (tok && count < max_n) {
        int val = atoi(tok);
        if (val >= 1 && val <= MAX_THREAD_MODES) {
            bool dup = false;
            for (int i = 0; i < count; i++) {
                if (out_list[i] == val) dup = true;
            }
            if (!dup) out_list[count++] = val;
        }
        tok = strtok_r(NULL, ",", &save);
    }
    return count;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --asi <path>       Path to .asi model file (defaults to %s if readable)\n", DEFAULT_ASI_PATH);
    printf("  --threads <list>   Comma-separated thread counts (default: 1,2)\n");
    printf("  --n-tokens <N>     Number of decode tokens to benchmark (default: %d)\n", DEFAULT_N_TOKENS);
    printf("  --max-ctx <N>      Session context capacity (default: %d)\n", DEFAULT_MAX_CTX);
    printf("  --json-out <path>  Write JSON summary report to <path>\n");
    printf("  --help             Show this help\n");
}

int main(int argc, char **argv) {
    const char *asi_path = NULL;
    bool asi_explicit = false;
    const char *json_out = NULL;
    int n_tokens = DEFAULT_N_TOKENS;
    int max_ctx = DEFAULT_MAX_CTX;
    int thread_list[MAX_THREAD_MODES] = {1, 2};
    int n_thread_modes = 2;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--asi") == 0 && i + 1 < argc) {
            asi_path = argv[++i];
            asi_explicit = true;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            int n = parse_threads_arg(argv[++i], thread_list, MAX_THREAD_MODES);
            if (n <= 0) {
                fprintf(stderr, "ERROR: Invalid --threads list\n");
                return 1;
            }
            n_thread_modes = n;
        } else if (strcmp(argv[i], "--n-tokens") == 0 && i + 1 < argc) {
            n_tokens = atoi(argv[++i]);
            if (n_tokens < 1) n_tokens = 1;
        } else if (strcmp(argv[i], "--max-ctx") == 0 && i + 1 < argc) {
            max_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json-out") == 0 && i + 1 < argc) {
            json_out = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "ERROR: Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (max_ctx < n_tokens + 16) {
        max_ctx = n_tokens + 16;
    }

    ThreadBench tb[MAX_THREAD_MODES];
    memset(tb, 0, sizeof(tb));
    for (int i = 0; i < n_thread_modes; i++) {
        tb[i].threads = thread_list[i];
    }

    /* Phase 1: 256 MiB DRAM Read-Only & STREAM Copy/Triad */
    uint8_t *dram_buf = (uint8_t *)xaligned_alloc(64, DRAM_BUF_BYTES);
    if (!dram_buf) {
        fprintf(stderr, "ERROR: Failed to allocate 256 MiB aligned buffer\n");
        return 1;
    }
    fill_nonzero_dram_buffer(dram_buf, DRAM_BUF_BYTES);

    for (int i = 0; i < n_thread_modes; i++) {
        AsiThreadPool *pool = asi_pool_create(thread_list[i]);
        run_dram_and_stream(pool, thread_list[i], dram_buf, DRAM_BUF_BYTES, &tb[i]);
        asi_pool_destroy(pool);
    }
    free(dram_buf);

    /* Phase 2: Isolated Q4_K / Q6_K Kernel Throughput (L3 2 MiB & DRAM 64 MiB) */
    run_isolated_kernels(thread_list, n_thread_modes, tb);

    /* Phase 3: Optional .asi Model Inspection & Full-Model Decode */
    AsiBreakdown bd;
    memset(&bd, 0, sizeof(bd));
    AsiDecodeResult decode_out[MAX_THREAD_MODES];
    memset(decode_out, 0, sizeof(decode_out));

    if (asi_explicit) {
        bd.requested = true;
        if (inspect_asi_model(asi_path, &bd)) {
            run_asi_decode(asi_path, &bd, thread_list, n_thread_modes, tb, n_tokens, max_ctx, decode_out);
        }
    } else if (access(DEFAULT_ASI_PATH, R_OK) == 0) {
        bd.requested = false;
        if (inspect_asi_model(DEFAULT_ASI_PATH, &bd)) {
            run_asi_decode(DEFAULT_ASI_PATH, &bd, thread_list, n_thread_modes, tb, n_tokens, max_ctx, decode_out);
        }
    } else {
        snprintf(bd.path, sizeof(bd.path), "%s", DEFAULT_ASI_PATH);
        snprintf(bd.error, sizeof(bd.error), "default file not found");
    }

    print_human_report(tb, n_thread_modes, &bd, decode_out);

    if (json_out) {
        if (write_json_report(json_out, tb, n_thread_modes, &bd, decode_out) != 0) {
            return 1;
        }
        printf("\nJSON report written to: %s\n", json_out);
    }

    return 0;
}
