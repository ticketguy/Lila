/*
 * Q4_K Dequantization — Native GGUF format support
 * 
 * Instead of re-quantizing GGUF weights to FigQuant (lossy),
 * we store Q4_K blocks directly in the .asi and dequantize here.
 * Same quality as llama.cpp — no double quantization.
 *
 * Q4_K block layout (144 bytes → 256 float values):
 *   uint16_t d;        // FP16 super-block scale
 *   uint16_t dmin;     // FP16 super-block min
 *   uint8_t scales[12]; // Sub-block scales/mins (packed)
 *   uint8_t qs[128];   // 4-bit quantized values
 */

#ifndef LILA_Q4K_H
#define LILA_Q4K_H

#include <stdint.h>
#include <math.h>
#include <string.h>

#define QK_K 256
#define Q4_K_BLOCK_SIZE 144

/* FP16 → FP32 conversion */
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    if (exp == 0) {
        if (mant == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, 4);
            return f;
        }
        /* Denormalized */
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        uint32_t result = sign | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    }
    
    uint32_t result = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
}

/*
 * Dequantize a Q4_K block (256 values).
 * Matches llama.cpp's dequantize_row_q4_K exactly.
 *
 * Layout: 4 iterations of 64 weights each.
 * Per iteration: 32 bytes of qs encode 64 nibbles (low=sub-block A, high=sub-block B).
 * Each pair of sub-blocks uses TWO scale/min values decoded via get_scale_min_k4.
 */

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static inline void dequant_q4k_block(float *out, const uint8_t *block) {
    uint16_t d_raw, dmin_raw;
    memcpy(&d_raw, block, 2);
    memcpy(&dmin_raw, block + 2, 2);
    
    float d = fp16_to_fp32(d_raw);
    float min = fp16_to_fp32(dmin_raw);
    
    const uint8_t *scales_packed = block + 4;   /* 12 bytes */
    const uint8_t *qs = block + 16;             /* 128 bytes */
    
    int is = 0;
    float *y = out;
    
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        
        get_scale_min_k4(is + 0, scales_packed, &sc, &m);
        float d1 = d * sc;
        float m1 = min * m;
        
        get_scale_min_k4(is + 1, scales_packed, &sc, &m);
        float d2 = d * sc;
        float m2 = min * m;
        
        /* Low nibbles → 32 elements with scale d1, min m1 */
        for (int l = 0; l < 32; ++l) {
            *y++ = d1 * (qs[l] & 0xF) - m1;
        }
        /* High nibbles → 32 elements with scale d2, min m2 */
        for (int l = 0; l < 32; ++l) {
            *y++ = d2 * (qs[l] >> 4) - m2;
        }
        
        qs += 32;
        is += 2;
    }
}

/*
 * Dequantize + matrix-vector multiply for Q4_K weights.
 * 
 * out[i] = sum_j( dequant(weight[i,j]) * vec[j] )
 * 
 * Processes one row at a time, dequantizing one block (256 values) at a time.
 * Never materializes the full dequantized matrix in memory.
 */
static void q4k_matvec(float *out, const uint8_t *weight_data, const float *vec,
                        int rows, int cols) {
    /* Number of Q4_K blocks per row */
    int blocks_per_row = (cols + QK_K - 1) / QK_K;
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const uint8_t *row_blocks = weight_data + (size_t)i * blocks_per_row * Q4_K_BLOCK_SIZE;
        
        for (int b = 0; b < blocks_per_row; b++) {
            /* Dequantize this block of 256 values */
            float block_vals[QK_K];
            dequant_q4k_block(block_vals, row_blocks + b * Q4_K_BLOCK_SIZE);
            
            /* Dot product with corresponding vec elements */
            int col_start = b * QK_K;
            int col_end = col_start + QK_K;
            if (col_end > cols) col_end = cols;
            
            for (int j = col_start; j < col_end; j++) {
                sum += block_vals[j - col_start] * vec[j];
            }
        }
        out[i] = sum;
    }
}

#endif /* LILA_Q4K_H */
