/*
 * Stub implementations of assembly kernels for testing on systems without NASM.
 * These are scalar C fallbacks that match the assembly function signatures.
 */
#include <string.h>
#include <math.h>
#include <stdint.h>

void lila_matvec_avx2(float *out, const float *mat, const float *vec, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += mat[i * cols + j] * vec[j];
        }
        out[i] = sum;
    }
}

void lila_rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
}

void lila_dequant_int4_avx2(float *out, const uint8_t *indices, const float *codebook,
                             const float *scales, int n_elements, int group_size) {
    for (int i = 0; i < n_elements; i++) {
        int byte_idx = i / 2;
        int nibble = (i % 2 == 0) ? (indices[byte_idx] & 0x0F) : ((indices[byte_idx] >> 4) & 0x0F);
        int group_idx = i / group_size;
        out[i] = codebook[nibble] * scales[group_idx];
    }
}

void lila_softmax_avx2(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}
