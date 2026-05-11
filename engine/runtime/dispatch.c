#include "model.h"
#include "detect.h"
#include <string.h>
#include <math.h>

/*
 * Kernel dispatch — routes compute calls to the best available kernel
 * based on detected CPU features.
 *
 * On Windows: no native .S kernels are linked. All compute goes through
 * the C scalar fallback (or LilaVM bytecode when the ASI runtime is active).
 *
 * On Linux/macOS x86_64: native AVX2 assembly kernels are used.
 * On ARM64: NEON kernels (when available).
 */

/* Assembly kernel declarations — only on platforms where we link .S files */
#if defined(__x86_64__) && !defined(_WIN32)
extern void lila_matvec_avx2(float *out, const float *mat, const float *vec, int rows, int cols);
extern void lila_rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps);
extern void lila_dequant_int4_avx2(float *out, const uint8_t *indices, const float *codebook,
                                    const float *scales, int n_elements, int group_size);
#elif defined(__aarch64__) && !defined(_WIN32)
extern void lila_dequant_int4_neon(float *out, const uint8_t *indices, const float *codebook,
                                    const float *scales, int n_elements, int group_size);
#endif

/* C scalar fallbacks — work everywhere including Windows */
static void matvec_scalar(float *out, const float *mat, const float *vec, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) sum += mat[i * cols + j] * vec[j];
        out[i] = sum;
    }
}

static void rmsnorm_scalar(float *out, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
}

/* Function pointers — set at init time */
typedef void (*matvec_fn)(float*, const float*, const float*, int, int);
typedef void (*rmsnorm_fn)(float*, const float*, const float*, int, float);

static matvec_fn  _matvec = matvec_scalar;
static rmsnorm_fn _rmsnorm = rmsnorm_scalar;

/* Initialize dispatch — call once at startup */
void lila_init_dispatch(void) {
#if defined(__x86_64__) && !defined(_WIN32)
    /* Linux/macOS: use native AVX2 assembly kernels */
    _matvec = lila_matvec_avx2;
    _rmsnorm = lila_rmsnorm_avx2;
#elif defined(__aarch64__) && !defined(_WIN32)
    /* ARM: NEON is always available */
    /* TODO: wire NEON matvec when written */
#else
    /* Windows or unknown: use C scalar fallbacks.
     * When running from a .asi file, the ASI runtime will use
     * LilaVM bytecode kernels instead of these scalar fallbacks
     * for better performance through vectorized bytecode ops. */
#endif
    lila_print_cpu_features();
}

/* Public dispatch functions — called by transformer.c / attention.c */
void lila_dispatch_matvec(float *out, const float *mat, const float *vec, int rows, int cols) {
    _matvec(out, mat, vec, rows, cols);
}

/* Also expose rmsnorm dispatch for use by transformer.c */
void lila_dispatch_rmsnorm(float *out, const float *x, const float *weight, int size, float eps) {
    _rmsnorm(out, x, weight, size, eps);
}

/*
 * On Windows, transformer.c references lila_rmsnorm_avx2 directly.
 * Provide a shim that routes to the dispatch function pointer.
 */
#ifdef _WIN32
#include <math.h>
void lila_rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps) {
    rmsnorm_scalar(out, x, weight, size, eps);
}
#endif
