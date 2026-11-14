/*
 * Verify AVX2 Q4_K x Q8_K and Q6_K x Q8_K matvec kernels against scalar reference.
 */
#include "q4k.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void q4k_matvec_ref(float *out, const uint8_t *w, const float *v, int rows, int cols) {
    int bpr = (cols + QK_K - 1) / QK_K;
    float tmp[QK_K];
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const uint8_t *rb = w + (size_t)i * bpr * Q4_K_BLOCK_SIZE;
        for (int b = 0; b < bpr; b++) {
            dequant_q4k_block(tmp, rb + (size_t)b * Q4_K_BLOCK_SIZE);
            int cs = b * QK_K;
            int ce = cs + QK_K;
            if (ce > cols) ce = cols;
            for (int j = cs; j < ce; j++) sum += tmp[j - cs] * v[j];
        }
        out[i] = sum;
    }
}

static void q6k_matvec_ref(float *out, const uint8_t *w, const float *v, int rows, int cols) {
    int bpr = (cols + QK_K - 1) / QK_K;
    float tmp[QK_K];
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const uint8_t *rb = w + (size_t)i * bpr * Q6_K_BLOCK_SIZE;
        for (int b = 0; b < bpr; b++) {
            dequant_q6k_block(tmp, rb + (size_t)b * Q6_K_BLOCK_SIZE);
            int cs = b * QK_K;
            int ce = cs + QK_K;
            if (ce > cols) ce = cols;
            for (int j = cs; j < ce; j++) sum += tmp[j - cs] * v[j];
        }
        out[i] = sum;
    }
}

int main(void) {
    srand(42);
    const int rows = 128;
    const int cols = 512;
    const int bpr = (cols + QK_K - 1) / QK_K;

    uint8_t *w4 = (uint8_t *)malloc((size_t)rows * bpr * Q4_K_BLOCK_SIZE);
    uint8_t *w6 = (uint8_t *)malloc((size_t)rows * bpr * Q6_K_BLOCK_SIZE);
    float *vec = (float *)malloc((size_t)cols * sizeof(float));
    float *out_fast = (float *)malloc((size_t)rows * sizeof(float));
    float *out_ref = (float *)malloc((size_t)rows * sizeof(float));

    for (size_t i = 0; i < (size_t)rows * bpr * Q4_K_BLOCK_SIZE; i++) w4[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < (size_t)rows * bpr * Q6_K_BLOCK_SIZE; i++) w6[i] = (uint8_t)(rand() & 0xFF);
    /* Set clean FP16 scales (0x3800 = 0.5) so random blocks aren't NaN/Inf */
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < bpr; b++) {
            uint8_t *b4 = w4 + ((size_t)r * bpr + b) * Q4_K_BLOCK_SIZE;
            b4[0] = 0x00; b4[1] = 0x38; b4[2] = 0x00; b4[3] = 0x34;
            uint8_t *b6 = w6 + ((size_t)r * bpr + b) * Q6_K_BLOCK_SIZE;
            b6[208] = 0x00; b6[209] = 0x38;
        }
    }
    for (int i = 0; i < cols; i++) vec[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;

    q4k_matvec_ref(out_ref, w4, vec, rows, cols);
    q4k_matvec(out_fast, w4, vec, rows, cols);
    double err4 = 0.0, norm4 = 0.0;
    for (int i = 0; i < rows; i++) {
        double d = (double)out_fast[i] - (double)out_ref[i];
        err4 += d * d;
        norm4 += (double)out_ref[i] * (double)out_ref[i];
    }
    double rel_l2_4 = sqrt(err4 / (norm4 + 1e-12));
    printf("Q4_K relative L2 err: %.6f\n", rel_l2_4);

    q6k_matvec_ref(out_ref, w6, vec, rows, cols);
    q6k_matvec(out_fast, w6, vec, rows, cols);
    double err6 = 0.0, norm6 = 0.0;
    for (int i = 0; i < rows; i++) {
        double d = (double)out_fast[i] - (double)out_ref[i];
        err6 += d * d;
        norm6 += (double)out_ref[i] * (double)out_ref[i];
    }
    double rel_l2_6 = sqrt(err6 / (norm6 + 1e-12));
    printf("Q6_K relative L2 err: %.6f\n", rel_l2_6);

    free(w4); free(w6); free(vec); free(out_fast); free(out_ref);
    if (rel_l2_4 > 0.01 || rel_l2_6 > 0.01) {
        fprintf(stderr, "FAIL: quant kernel error too high\n");
        return 1;
    }
    printf("PASS: Q4_K and Q6_K AVX2 kernels match scalar reference\n");
    return 0;
}
