#ifndef LILA_DISPATCH_H
#define LILA_DISPATCH_H

void lila_init_dispatch(void);
void lila_dispatch_matvec(float *out, const float *mat, const float *vec, int rows, int cols);
void lila_dispatch_rmsnorm(float *out, const float *x, const float *weight, int size, float eps);

/* On Windows, this is a shim that routes to scalar fallback.
 * On Linux/macOS x86, this links to the AVX2 assembly kernel. */
void lila_rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps);

#endif
