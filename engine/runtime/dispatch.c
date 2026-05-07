#include "model.h"
#include "detect.h"
#include <string.h>

/*
 * Kernel dispatch — routes compute calls to the best available kernel
 * based on detected CPU features.
 *
 * At startup, detect_cpu() is called once. Based on the result,
 * function pointers are set to the fastest available implementation.
 */

/* Assembly kernel declarations (extern from .S files) */
#ifdef __x86_64__
extern void lila_matvec_avx2(float *out, const float *mat, const float *vec, int rows, int cols);
extern void lila_rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps);
extern void lila_dequant_int4_avx2(float *out, const uint8_t *indices, const float *codebook,
                                    const float *scales, int n_elements, int group_size);
#elif defined(__aarch64__)
extern void lila_dequant_int4_neon(float *out, const uint8_t *indices, const float *codebook,
                                    const float *scales, int n_elements, int group_size);
#endif

/* C scalar fallbacks (defined in inference.c) */
static void matvec_scalar(float *out, const float *mat, const float *vec, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) sum += mat[i * cols + j] * vec[j];
        out[i] = sum;
    }
}

/* Function pointers — set at init time */
typedef void (*matvec_fn)(float*, const float*, const float*, int, int);
typedef void (*rmsnorm_fn)(float*, const float*, const float*, int, float);

static matvec_fn  _matvec = matvec_scalar;
static rmsnorm_fn _rmsnorm = NULL;  /* Set in init */

/* Initialize dispatch — call once at startup */
void lila_init_dispatch(void) {
#ifdef __x86_64__
    /* Always use AVX2 on x86_64 (all modern CPUs have it) */
    _matvec = lila_matvec_avx2;
    _rmsnorm = lila_rmsnorm_avx2;
    /* TODO: detect AVX-512 and use faster kernels if available */
#elif defined(__aarch64__)
    /* ARM: NEON is always available */
    /* TODO: wire NEON matvec when written */
#endif
    lila_print_cpu_features();
}

/* Public dispatch functions — called by transformer.c / attention.c */
void lila_dispatch_matvec(float *out, const float *mat, const float *vec, int rows, int cols) {
    _matvec(out, mat, vec, rows, cols);
}
