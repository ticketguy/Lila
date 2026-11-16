/*
 * Verify AVX2 Q4_K x Q8_K and Q6_K x Q8_K matvec kernels against scalar reference.
 */
#include "q4k.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

    /* Bit-for-bit Fix 1 verification: dot2_q8k_avx2 vs dot_q8k_simd */
    LilaQ8KBlock *qx = lila_q8k_alloc((size_t)bpr);
    quantize_vec_q8k(qx, vec, cols);
    for (int r = 0; r + 1 < rows; r += 2) {
        float ref0 = q4k_dot_q8k_simd(w4 + (size_t)r * bpr * Q4_K_BLOCK_SIZE, qx, bpr);
        float ref1 = q4k_dot_q8k_simd(w4 + (size_t)(r + 1) * bpr * Q4_K_BLOCK_SIZE, qx, bpr);
        float d0 = 0.0f, d1 = 0.0f;
        q4k_dot2_q8k_avx2(w4 + (size_t)r * bpr * Q4_K_BLOCK_SIZE,
                          w4 + (size_t)(r + 1) * bpr * Q4_K_BLOCK_SIZE,
                          qx, bpr, &d0, &d1);
        if (memcmp(&d0, &ref0, sizeof(float)) != 0 || memcmp(&d1, &ref1, sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: q4k_dot2_q8k_avx2 row %d mismatch (got %.9g/%.9g vs %.9g/%.9g)\n",
                    r, d0, d1, ref0, ref1);
            return 1;
        }

        float q6ref0 = q6k_dot_q8k_simd(w6 + (size_t)r * bpr * Q6_K_BLOCK_SIZE, qx, bpr);
        float q6ref1 = q6k_dot_q8k_simd(w6 + (size_t)(r + 1) * bpr * Q6_K_BLOCK_SIZE, qx, bpr);
        float q6d0 = 0.0f, q6d1 = 0.0f;
        q6k_dot2_q8k_avx2(w6 + (size_t)r * bpr * Q6_K_BLOCK_SIZE,
                          w6 + (size_t)(r + 1) * bpr * Q6_K_BLOCK_SIZE,
                          qx, bpr, &q6d0, &q6d1);
        if (memcmp(&q6d0, &q6ref0, sizeof(float)) != 0 || memcmp(&q6d1, &q6ref1, sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: q6k_dot2_q8k_avx2 row %d mismatch (got %.9g/%.9g vs %.9g/%.9g)\n",
                    r, q6d0, q6d1, q6ref0, q6ref1);
            return 1;
        }
    }
    free(qx);
    printf("PASS: Fix 1 q4k_dot2_q8k_avx2 and q6k_dot2_q8k_avx2 match dot_q8k_simd bit-for-bit (memcmp=0)\n");

    /* Fix 3 verification: small-batch (B=2,3,4,5) & prefill (B=16,17) vs per-token matvec */
    const int test_batches[] = {2, 3, 4, 5, 16, 17};
    const int test_rows[] = {17, 128};
    float *vecs = (float *)malloc((size_t)17 * cols * sizeof(float));
    float *bat_out = (float *)malloc((size_t)17 * rows * sizeof(float));
    float *matvec_ref = (float *)malloc((size_t)17 * rows * sizeof(float));
    float *split_rest = (float *)malloc((size_t)3 * rows * sizeof(float));
    LilaQ8KBlock *qx_all = lila_q8k_alloc((size_t)4 * bpr);
    for (int t = 0; t < 17; t++) {
        for (int i = 0; i < cols; i++) {
            vecs[(size_t)t * cols + i] = ((float)((rand() % 400) - 200)) / 127.0f;
        }
    }
    for (size_t ri = 0; ri < 2; ri++) {
        int tr = test_rows[ri];
        for (int t = 0; t < 17; t++) {
            q4k_matvec(matvec_ref + (size_t)t * tr, w4, vecs + (size_t)t * cols, tr, cols);
        }
        for (size_t bi = 0; bi < sizeof(test_batches) / sizeof(test_batches[0]); bi++) {
            int B = test_batches[bi];
            q4k_matmul_batch(bat_out, w4, vecs, B, tr, cols);
            for (int e = 0; e < B * tr; e++) {
                if (fabsf(bat_out[e] - matvec_ref[e]) > 1e-4f) {
                    fprintf(stderr, "FAIL: q4k_matmul_batch B=%d rows=%d idx=%d diff=%.9g\n",
                            B, tr, e, fabsf(bat_out[e] - matvec_ref[e]));
                    return 1;
                }
            }
        }
        for (int t = 0; t < 17; t++) {
            q6k_matvec(matvec_ref + (size_t)t * tr, w6, vecs + (size_t)t * cols, tr, cols);
        }
        for (size_t bi = 0; bi < sizeof(test_batches) / sizeof(test_batches[0]); bi++) {
            int B = test_batches[bi];
            q6k_matmul_batch(bat_out, w6, vecs, B, tr, cols);
            for (int e = 0; e < B * tr; e++) {
                if (fabsf(bat_out[e] - matvec_ref[e]) > 1e-4f) {
                    fprintf(stderr, "FAIL: q6k_matmul_batch B=%d rows=%d idx=%d diff=%.9g\n",
                            B, tr, e, fabsf(bat_out[e] - matvec_ref[e]));
                    return 1;
                }
            }
        }
        for (int B = 1; B <= 4; B++) {
            for (int t = 0; t < B; t++) {
                quantize_vec_q8k_strided(qx_all + t, B, vecs + (size_t)t * cols, cols);
            }
            q6k_matmul_batch_q8k_split_range(bat_out, (B > 1) ? split_rest : NULL, w6, qx_all, B, 0, tr, tr, bpr);
            for (int i = 0; i < tr; i++) {
                if (fabsf(bat_out[i] - matvec_ref[i]) > 1e-4f) {
                    fprintf(stderr, "FAIL: q6k_split out0 B=%d rows=%d i=%d diff=%.9g\n",
                            B, tr, i, fabsf(bat_out[i] - matvec_ref[i]));
                    return 1;
                }
            }
            for (int t = 1; t < B; t++) {
                for (int i = 0; i < tr; i++) {
                    if (fabsf(split_rest[(size_t)(t - 1) * tr + i] - matvec_ref[(size_t)t * tr + i]) > 1e-4f) {
                        fprintf(stderr, "FAIL: q6k_split rest B=%d rows=%d t=%d i=%d diff=%.9g\n",
                                B, tr, t, i, fabsf(split_rest[(size_t)(t - 1) * tr + i] - matvec_ref[(size_t)t * tr + i]));
                        return 1;
                    }
                }
            }
        }
    }
    free(vecs); free(bat_out); free(matvec_ref); free(split_rest); free(qx_all);
    printf("PASS: Fix 3 q4k/q6k batch (B=2..17) & q6k_split_range match matvec cleanly\n");

    free(w4); free(w6); free(vec); free(out_fast); free(out_ref);
    if (rel_l2_4 > 0.01 || rel_l2_6 > 0.01) {
        fprintf(stderr, "FAIL: quant kernel error too high\n");
        return 1;
    }
    printf("PASS: Q4_K and Q6_K AVX2 kernels match scalar reference\n");
    return 0;
}
