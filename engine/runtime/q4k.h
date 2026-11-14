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
#include <stdlib.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__SSE2__)
#define LILA_PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define LILA_PREFETCH(p) __builtin_prefetch((p), 0, 3)
#else
#define LILA_PREFETCH(p) ((void)0)
#endif

#define QK_K 256
#define Q4_K_BLOCK_SIZE 144

/* FP16 -> FP32 conversion */
static inline float fp16_to_fp32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
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
#endif
}

/* FP32 -> FP16 conversion */
static inline uint16_t fp32_to_fp16(float f) {
#if defined(__F16C__)
    return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
#endif
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
 * Fast Q4_K x Q8_K matrix-vector multiply.
 *
 * Quantizes the input vector `vec` once into 256-element Q8 blocks (8 sub-blocks of 32),
 * then uses AVX2 integer SIMD (_mm256_maddubs_epi16 + _mm256_madd_epi16) to compute
 * 32 multiply-adds per instruction directly on packed 4-bit nibbles.
 */
typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[16];
} LilaQ8KBlock;

static inline void quantize_vec_q8k_strided(LilaQ8KBlock *dst, int dst_stride, const float *vec, int cols) {
    int blocks = (cols + QK_K - 1) / QK_K;
    for (int b = 0; b < blocks; b++) {
        LilaQ8KBlock *out_b = dst + (size_t)b * dst_stride;
        int base = b * QK_K;
        int avail_blk = cols - base;
        if (avail_blk > QK_K) avail_blk = QK_K;
#if defined(__AVX512F__) && defined(__AVX512BW__)
        if (avail_blk == QK_K) {
            const float *vp = vec + base;
            __m512 vmax = _mm512_setzero_ps();
            for (int sb = 0; sb < 4; sb++) {
                const float *sp = vp + sb * 64;
                __m512 m0 = _mm512_abs_ps(_mm512_loadu_ps(sp + 0));
                __m512 m1 = _mm512_abs_ps(_mm512_loadu_ps(sp + 16));
                __m512 m2 = _mm512_abs_ps(_mm512_loadu_ps(sp + 32));
                __m512 m3 = _mm512_abs_ps(_mm512_loadu_ps(sp + 48));
                vmax = _mm512_max_ps(vmax, _mm512_max_ps(_mm512_max_ps(m0, m1), _mm512_max_ps(m2, m3)));
            }
            float amax = _mm512_reduce_max_ps(vmax);
            float d = amax / 127.0f;
            float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            out_b->d = d;

            __m512 vid = _mm512_set1_ps(id);
            const __m128i ones8 = _mm_set1_epi8(1);
            const __m128i ones16_128 = _mm_set1_epi16(1);
            for (int sb = 0; sb < 4; sb++) {
                const float *sp = vp + sb * 64;
                for (int q = 0; q < 4; q++) {
                    __m512i i32 = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(sp + q * 16), vid));
                    __m128i q16 = _mm512_cvtsepi32_epi8(i32);
                    _mm_storeu_si128((__m128i *)(out_b->qs + sb * 64 + q * 16), q16);
                    __m128i h = _mm_madd_epi16(_mm_maddubs_epi16(ones8, q16), ones16_128);
                    h = _mm_hadd_epi32(h, h);
                    h = _mm_hadd_epi32(h, h);
                    out_b->bsums[sb * 4 + q] = (int16_t)_mm_cvtsi128_si32(h);
                }
            }
            continue;
        }
#elif defined(__AVX2__)
        if (avail_blk == QK_K) {
            const float *vp = vec + base;
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            __m256 vmax = _mm256_setzero_ps();
            for (int sb = 0; sb < 8; sb++) {
                const float *sp = vp + sb * 32;
                __m256 m0 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(sp + 0));
                __m256 m1 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(sp + 8));
                __m256 m2 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(sp + 16));
                __m256 m3 = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(sp + 24));
                __m256 m01 = _mm256_max_ps(m0, m1);
                __m256 m23 = _mm256_max_ps(m2, m3);
                vmax = _mm256_max_ps(vmax, _mm256_max_ps(m01, m23));
            }
            __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
            m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
            float amax = _mm_cvtss_f32(m128);
            float d = amax / 127.0f;
            float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            out_b->d = d;

            __m256 vid = _mm256_set1_ps(id);
            const __m128i ones8 = _mm_set1_epi8(1);
            const __m128i ones16_128 = _mm_set1_epi16(1);
            for (int sb = 0; sb < 8; sb++) {
                const float *sp = vp + sb * 32;
                __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(sp + 0), vid));
                __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(sp + 8), vid));
                __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(sp + 16), vid));
                __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(sp + 24), vid));

                __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
                __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
                __m256i q8  = _mm256_permute4x64_epi64(_mm256_packs_epi16(p01, p23), 0xD8);
                _mm256_storeu_si256((__m256i *)(out_b->qs + sb * 32), q8);

                __m128i qlo = _mm256_castsi256_si128(q8);
                __m128i qhi = _mm256_extracti128_si256(q8, 1);
                __m128i h0 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, qlo), ones16_128);
                __m128i h1 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, qhi), ones16_128);
                __m128i ht = _mm_hadd_epi32(h0, h1);
                ht = _mm_hadd_epi32(ht, ht);
                out_b->bsums[sb * 2 + 0] = (int16_t)_mm_cvtsi128_si32(ht);
                out_b->bsums[sb * 2 + 1] = (int16_t)_mm_extract_epi32(ht, 1);
            }
            continue;
        }
#elif defined(__aarch64__)
        if (avail_blk == QK_K) {
            const float *vp = vec + base;
            float32x4_t vmax = vdupq_n_f32(0.0f);
            for (int l = 0; l < QK_K; l += 16) {
                float32x4_t m0 = vabsq_f32(vld1q_f32(vp + l + 0));
                float32x4_t m1 = vabsq_f32(vld1q_f32(vp + l + 4));
                float32x4_t m2 = vabsq_f32(vld1q_f32(vp + l + 8));
                float32x4_t m3 = vabsq_f32(vld1q_f32(vp + l + 12));
                vmax = vmaxq_f32(vmax, vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
            }
            float amax = vmaxvq_f32(vmax);
            float d = amax / 127.0f;
            float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            out_b->d = d;
            float32x4_t vid = vdupq_n_f32(id);
            for (int sb = 0; sb < 16; sb++) {
                const float *sp = vp + sb * 16;
                int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(sp + 0), vid));
                int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(sp + 4), vid));
                int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(sp + 8), vid));
                int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(sp + 12), vid));
                int16x8_t p01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                int16x8_t p23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
                int8x16_t q8 = vcombine_s8(vqmovn_s16(p01), vqmovn_s16(p23));
                vst1q_s8(out_b->qs + sb * 16, q8);
                out_b->bsums[sb] = (int16_t)vaddvq_s16(vpaddlq_s8(q8));
            }
            continue;
        }
#endif
        float amax = 0.0f;
        for (int l = 0; l < avail_blk; l++) {
            float av = fabsf(vec[base + l]);
            if (av > amax) amax = av;
        }
        float d = amax / 127.0f;
        float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        out_b->d = d;
        for (int k = 0; k < 16; k++) out_b->bsums[k] = 0;
        for (int l = 0; l < avail_blk; l++) {
            int q = (int)lrintf(vec[base + l] * id);
            if (q > 127) q = 127;
            else if (q < -127) q = -127;
            out_b->qs[l] = (int8_t)q;
            out_b->bsums[l >> 4] += (int16_t)q;
        }
        for (int l = avail_blk; l < QK_K; l++) {
            out_b->qs[l] = 0;
        }
    }
}

static inline void quantize_vec_q8k(LilaQ8KBlock *dst, const float *vec, int cols) {
    quantize_vec_q8k_strided(dst, 1, vec, cols);
}

#if defined(__AVX2__) || defined(__AVX512F__)
static inline int32_t lila_hsum_i32_avx2(__m256i v) {
    __m128i vlow  = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(vlow, vhigh);
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64 = _mm_add_epi32(sum128, hi64);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(1, 1, 1, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

static inline float lila_hsum_f32_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sum);
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}
#endif

static inline float q4k_dot_q8k_simd(const uint8_t *row_blocks, const LilaQ8KBlock *qx, int blocks_per_row) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512i mask_0f = _mm512_set1_epi8(0x0F);
    float total_sum = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *block = row_blocks + (size_t)b * Q4_K_BLOCK_SIZE;
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, block, 2);
        memcpy(&dmin_raw, block + 2, 2);
        float d = fp16_to_fp32(d_raw);
        float min = fp16_to_fp32(dmin_raw);

        const uint8_t *scales_packed = block + 4;
        const uint8_t *qs = block + 16;
        const LilaQ8KBlock *xb = &qx[b];

        __m512i sumi = _mm512_setzero_si512();
        int32_t sum_mins = 0;

        for (int half = 0; half < 2; half++) {
            int is = half * 4;
            int sb = half * 4;
            uint8_t sc0, m0, sc1, m1, sc2, m2, sc3, m3;
            get_scale_min_k4(is + 0, scales_packed, &sc0, &m0);
            get_scale_min_k4(is + 1, scales_packed, &sc1, &m1);
            get_scale_min_k4(is + 2, scales_packed, &sc2, &m2);
            get_scale_min_k4(is + 3, scales_packed, &sc3, &m3);

            __m512i qbits = _mm512_loadu_si512((const void *)(qs + half * 64));
            __m512i q_lo = _mm512_and_si512(qbits, mask_0f);
            __m512i q_hi = _mm512_and_si512(_mm512_srli_epi16(qbits, 4), mask_0f);

            __m512i x_lo = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(xb->qs + half * 128 + 0))),
                _mm256_loadu_si256((const __m256i *)(xb->qs + half * 128 + 64)), 1);
            __m512i x_hi = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(xb->qs + half * 128 + 32))),
                _mm256_loadu_si256((const __m256i *)(xb->qs + half * 128 + 96)), 1);

#if defined(__AVX512VNNI__)
            __m512i d_lo = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q_lo, x_lo);
            __m512i d_hi = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q_hi, x_hi);
            __m512i vsc_lo = _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32((int32_t)sc0), _mm512_set1_epi32((int32_t)sc2));
            __m512i vsc_hi = _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32((int32_t)sc1), _mm512_set1_epi32((int32_t)sc3));
            sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(_mm512_mullo_epi32(d_lo, vsc_lo), _mm512_mullo_epi32(d_hi, vsc_hi)));
#else
            __m512i vsc_lo = _mm512_mask_blend_epi16(0xFFFF0000u, _mm512_set1_epi16((int16_t)sc0), _mm512_set1_epi16((int16_t)sc2));
            __m512i vsc_hi = _mm512_mask_blend_epi16(0xFFFF0000u, _mm512_set1_epi16((int16_t)sc1), _mm512_set1_epi16((int16_t)sc3));
            __m512i p_lo = _mm512_madd_epi16(_mm512_maddubs_epi16(q_lo, x_lo), vsc_lo);
            __m512i p_hi = _mm512_madd_epi16(_mm512_maddubs_epi16(q_hi, x_hi), vsc_hi);
            sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(p_lo, p_hi));
#endif

            sum_mins += (int32_t)m0 * ((int32_t)xb->bsums[(sb + 0) * 2 + 0] + (int32_t)xb->bsums[(sb + 0) * 2 + 1])
                      + (int32_t)m1 * ((int32_t)xb->bsums[(sb + 1) * 2 + 0] + (int32_t)xb->bsums[(sb + 1) * 2 + 1])
                      + (int32_t)m2 * ((int32_t)xb->bsums[(sb + 2) * 2 + 0] + (int32_t)xb->bsums[(sb + 2) * 2 + 1])
                      + (int32_t)m3 * ((int32_t)xb->bsums[(sb + 3) * 2 + 0] + (int32_t)xb->bsums[(sb + 3) * 2 + 1]);
        }
        float xd = xb->d;
        total_sum += (d * xd) * (float)_mm512_reduce_add_epi32(sumi) - (min * xd) * (float)sum_mins;
    }
    return total_sum;
#elif defined(__AVX2__)
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    __m256 sum_vec = _mm256_setzero_ps();
    float bias = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *block = row_blocks + (size_t)b * Q4_K_BLOCK_SIZE;
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, block, 2);
        memcpy(&dmin_raw, block + 2, 2);

        float d = fp16_to_fp32(d_raw);
        float min = fp16_to_fp32(dmin_raw);

        const uint8_t *scales_packed = block + 4;
        const uint8_t *qs = block + 16;
        const LilaQ8KBlock *xb = &qx[b];

        __m256i sumi = _mm256_setzero_si256();
        int32_t sum_mins = 0;

        int is = 0;
        for (int sb = 0; sb < 8; sb += 2) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, scales_packed, &sc1, &m1);
            get_scale_min_k4(is + 1, scales_packed, &sc2, &m2);

            __m256i qbits = _mm256_loadu_si256((const __m256i *)qs);
            __m256i q_lo = _mm256_and_si256(qbits, mask_0f);
            __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(qbits, 4), mask_0f);

            __m256i x_lo = _mm256_loadu_si256((const __m256i *)(xb->qs + sb * 32));
            __m256i x_hi = _mm256_loadu_si256((const __m256i *)(xb->qs + (sb + 1) * 32));

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
            __m256i d_lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q_lo, x_lo);
            __m256i d_hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), q_hi, x_hi);
            __m256i p_lo = _mm256_mullo_epi32(d_lo, _mm256_set1_epi32((int32_t)sc1));
            __m256i p_hi = _mm256_mullo_epi32(d_hi, _mm256_set1_epi32((int32_t)sc2));
#else
            __m256i p_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(q_lo, x_lo), _mm256_set1_epi16((int16_t)sc1));
            __m256i p_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(q_hi, x_hi), _mm256_set1_epi16((int16_t)sc2));
#endif
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p_lo, p_hi));

            sum_mins += (int32_t)m1 * ((int32_t)xb->bsums[sb * 2 + 0] + (int32_t)xb->bsums[sb * 2 + 1])
                      + (int32_t)m2 * ((int32_t)xb->bsums[(sb + 1) * 2 + 0] + (int32_t)xb->bsums[(sb + 1) * 2 + 1]);

            qs += 32;
            is += 2;
        }
        float xd = xb->d;
        sum_vec = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi), _mm256_set1_ps(d * xd), sum_vec);
        bias += (min * xd) * (float)sum_mins;
    }
    return lila_hsum_f32_avx2(sum_vec) - bias;
#elif defined(__aarch64__)
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    float total_sum = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *block = row_blocks + (size_t)b * Q4_K_BLOCK_SIZE;
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, block, 2);
        memcpy(&dmin_raw, block + 2, 2);
        float d = fp16_to_fp32(d_raw);
        float min = fp16_to_fp32(dmin_raw);

        const uint8_t *scales_packed = block + 4;
        const uint8_t *qs = block + 16;
        const LilaQ8KBlock *xb = &qx[b];

        int32_t sumi = 0;
        int32_t sum_mins = 0;
        int is = 0;
        for (int sb = 0; sb < 8; sb += 2) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, scales_packed, &sc1, &m1);
            get_scale_min_k4(is + 1, scales_packed, &sc2, &m2);

            int32x4_t acc_lo = vdupq_n_s32(0);
            int32x4_t acc_hi = vdupq_n_s32(0);
            for (int c = 0; c < 2; c++) {
                uint8x16_t qbits = vld1q_u8(qs + c * 16);
                int8x16_t q_lo = vreinterpretq_s8_u8(vandq_u8(qbits, mask_0f));
                int8x16_t q_hi = vreinterpretq_s8_u8(vshrq_n_u8(qbits, 4));
                int8x16_t x_lo = vld1q_s8(xb->qs + sb * 32 + c * 16);
                int8x16_t x_hi = vld1q_s8(xb->qs + (sb + 1) * 32 + c * 16);
#if defined(__ARM_FEATURE_DOTPROD)
                acc_lo = vdotq_s32(acc_lo, q_lo, x_lo);
                acc_hi = vdotq_s32(acc_hi, q_hi, x_hi);
#else
                int16x8_t p_lo0 = vmull_s8(vget_low_s8(q_lo), vget_low_s8(x_lo));
                int16x8_t p_lo1 = vmull_s8(vget_high_s8(q_lo), vget_high_s8(x_lo));
                acc_lo = vpadalq_s16(acc_lo, p_lo0);
                acc_lo = vpadalq_s16(acc_lo, p_lo1);
                int16x8_t p_hi0 = vmull_s8(vget_low_s8(q_hi), vget_low_s8(x_hi));
                int16x8_t p_hi1 = vmull_s8(vget_high_s8(q_hi), vget_high_s8(x_hi));
                acc_hi = vpadalq_s16(acc_hi, p_hi0);
                acc_hi = vpadalq_s16(acc_hi, p_hi1);
#endif
            }
            sumi += vaddvq_s32(acc_lo) * (int32_t)sc1 + vaddvq_s32(acc_hi) * (int32_t)sc2;
            sum_mins += (int32_t)m1 * ((int32_t)xb->bsums[sb * 2 + 0] + (int32_t)xb->bsums[sb * 2 + 1])
                      + (int32_t)m2 * ((int32_t)xb->bsums[(sb + 1) * 2 + 0] + (int32_t)xb->bsums[(sb + 1) * 2 + 1]);
            qs += 32;
            is += 2;
        }
        float xd = xb->d;
        total_sum += (d * xd) * (float)sumi - (min * xd) * (float)sum_mins;
    }
    return total_sum;
#else
    float total_sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *block = row_blocks + (size_t)b * Q4_K_BLOCK_SIZE;
        uint16_t d_raw, dmin_raw;
        memcpy(&d_raw, block, 2);
        memcpy(&dmin_raw, block + 2, 2);
        float d = fp16_to_fp32(d_raw);
        float min = fp16_to_fp32(dmin_raw);
        const uint8_t *scales_packed = block + 4;
        const uint8_t *qs = block + 16;
        const LilaQ8KBlock *xb = &qx[b];
        int32_t sumi = 0, sum_mins = 0;
        int is = 0;
        for (int sb = 0; sb < 8; sb += 2) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, scales_packed, &sc1, &m1);
            get_scale_min_k4(is + 1, scales_packed, &sc2, &m2);
            int32_t s1 = 0, s2 = 0;
            for (int l = 0; l < 32; l++) {
                s1 += (int32_t)(qs[l] & 0x0F) * (int32_t)xb->qs[sb * 32 + l];
                s2 += (int32_t)(qs[l] >> 4) * (int32_t)xb->qs[(sb + 1) * 32 + l];
            }
            sumi += s1 * (int32_t)sc1 + s2 * (int32_t)sc2;
            sum_mins += (int32_t)m1 * ((int32_t)xb->bsums[sb * 2 + 0] + (int32_t)xb->bsums[sb * 2 + 1])
                      + (int32_t)m2 * ((int32_t)xb->bsums[(sb + 1) * 2 + 0] + (int32_t)xb->bsums[(sb + 1) * 2 + 1]);
            qs += 32;
            is += 2;
        }
        float xd = xb->d;
        total_sum += (d * xd) * (float)sumi - (min * xd) * (float)sum_mins;
    }
    return total_sum;
#endif
}

static inline float q4k_dot_q8k_avx2(const uint8_t *row_blocks, const LilaQ8KBlock *qx, int blocks_per_row) {
    return q4k_dot_q8k_simd(row_blocks, qx, blocks_per_row);
}

#if defined(__AVX2__)

static inline void q4k_dot2_q8k_avx2(const uint8_t *r0, const uint8_t *r1,
                                     const LilaQ8KBlock *qx, int blocks_per_row,
                                     float *out0, float *out1) {
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    __m256 sum_vec0 = _mm256_setzero_ps();
    __m256 sum_vec1 = _mm256_setzero_ps();
    float bias0 = 0.0f, bias1 = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *b0 = r0 + (size_t)b * Q4_K_BLOCK_SIZE;
        const uint8_t *b1 = r1 + (size_t)b * Q4_K_BLOCK_SIZE;
        uint16_t d0_r, m0_r, d1_r, m1_r;
        memcpy(&d0_r, b0, 2); memcpy(&m0_r, b0 + 2, 2);
        memcpy(&d1_r, b1, 2); memcpy(&m1_r, b1 + 2, 2);
        float d0 = fp16_to_fp32(d0_r), min0 = fp16_to_fp32(m0_r);
        float d1 = fp16_to_fp32(d1_r), min1 = fp16_to_fp32(m1_r);
        const uint8_t *sp0 = b0 + 4, *qs0 = b0 + 16;
        const uint8_t *sp1 = b1 + 4, *qs1 = b1 + 16;
        const LilaQ8KBlock *xb = &qx[b];

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        int32_t mins0 = 0, mins1 = 0;

        int is = 0;
        for (int sb = 0; sb < 8; sb += 2) {
            __m256i x_lo = _mm256_loadu_si256((const __m256i *)(xb->qs + sb * 32));
            __m256i x_hi = _mm256_loadu_si256((const __m256i *)(xb->qs + (sb + 1) * 32));
            int32_t bs_lo = (int32_t)xb->bsums[sb * 2 + 0] + (int32_t)xb->bsums[sb * 2 + 1];
            int32_t bs_hi = (int32_t)xb->bsums[(sb + 1) * 2 + 0] + (int32_t)xb->bsums[(sb + 1) * 2 + 1];

            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, sp0, &sc1, &m1);
            get_scale_min_k4(is + 1, sp0, &sc2, &m2);
            __m256i qb0 = _mm256_loadu_si256((const __m256i *)qs0);
            sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(qb0, mask_0f), x_lo), _mm256_set1_epi16((int16_t)sc1)));
            sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(qb0, 4), mask_0f), x_hi), _mm256_set1_epi16((int16_t)sc2)));
            mins0 += (int32_t)m1 * bs_lo + (int32_t)m2 * bs_hi;

            get_scale_min_k4(is + 0, sp1, &sc1, &m1);
            get_scale_min_k4(is + 1, sp1, &sc2, &m2);
            __m256i qb1 = _mm256_loadu_si256((const __m256i *)qs1);
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(qb1, mask_0f), x_lo), _mm256_set1_epi16((int16_t)sc1)));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(qb1, 4), mask_0f), x_hi), _mm256_set1_epi16((int16_t)sc2)));
            mins1 += (int32_t)m1 * bs_lo + (int32_t)m2 * bs_hi;

            qs0 += 32; qs1 += 32;
            is += 2;
        }
        float xd = xb->d;
        sum_vec0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi0), _mm256_set1_ps(d0 * xd), sum_vec0);
        sum_vec1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi1), _mm256_set1_ps(d1 * xd), sum_vec1);
        bias0 += (min0 * xd) * (float)mins0;
        bias1 += (min1 * xd) * (float)mins1;
    }
    *out0 = lila_hsum_f32_avx2(sum_vec0) - bias0;
    *out1 = lila_hsum_f32_avx2(sum_vec1) - bias1;
}
#endif

static inline void q4k_matvec(float *out, const uint8_t *weight_data, const float *vec,
                              int rows, int cols) {
    int blocks_per_row = (cols + QK_K - 1) / QK_K;

#if defined(__AVX2__)
    LilaQ8KBlock stack_qx[64];
    LilaQ8KBlock *qx = (blocks_per_row <= 64)
        ? stack_qx
        : (LilaQ8KBlock *)malloc((size_t)blocks_per_row * sizeof(LilaQ8KBlock));
    if (qx) {
        quantize_vec_q8k(qx, vec, cols);
        const size_t row_bytes = (size_t)blocks_per_row * Q4_K_BLOCK_SIZE;
        #pragma omp parallel for schedule(static) if(rows >= 64)
        for (int i = 0; i < rows; i++) {
            const uint8_t *rp = weight_data + (size_t)i * row_bytes;
            if (i + 2 < rows) {
                _mm_prefetch((const char *)(rp + 2 * row_bytes), _MM_HINT_T0);
            }
            out[i] = q4k_dot_q8k_avx2(rp, qx, blocks_per_row);
        }
        if (qx != stack_qx) free(qx);
        return;
    }
#endif

    #pragma omp parallel for schedule(static) if(rows >= 64)
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const uint8_t *row_blocks = weight_data + (size_t)i * blocks_per_row * Q4_K_BLOCK_SIZE;
        float block_vals[QK_K];
        for (int b = 0; b < blocks_per_row; b++) {
            dequant_q4k_block(block_vals, row_blocks + (size_t)b * Q4_K_BLOCK_SIZE);
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

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  Q6_K Dequantization (for embedding lookup)                               */
/*  Block: 210 bytes → 256 float values                                      */
/*  Layout: ql[128] + qh[64] + scales[16] + d(fp16)                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define Q6_K_BLOCK_SIZE 210

static inline void dequant_q6k_block(float *out, const uint8_t *block) {
    const uint8_t *ql = block;           /* 128 bytes: low 4 bits */
    const uint8_t *qh = block + 128;     /* 64 bytes: high 2 bits */
    const int8_t *sc = (const int8_t *)(block + 192); /* 16 bytes: int8 scales */
    uint16_t d_raw;
    memcpy(&d_raw, block + 208, 2);      /* fp16 super-block scale */
    float d = fp16_to_fp32(d_raw);

    float *y = out;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l >> 4;
            int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

/*
 * Dequantize a single embedding row from Q6_K format.
 * Each row = hidden_size/256 blocks × 210 bytes/block.
 */
static inline void dequant_q6k_row(float *out, const uint8_t *row_data, int hidden_size) {
    int n_blocks = (hidden_size + 255) / 256;
    for (int b = 0; b < n_blocks; b++) {
        if (b * 256 + 256 <= hidden_size) {
            dequant_q6k_block(out + b * 256, row_data + (size_t)b * Q6_K_BLOCK_SIZE);
        } else {
            float tmp[256];
            dequant_q6k_block(tmp, row_data + (size_t)b * Q6_K_BLOCK_SIZE);
            memcpy(out + b * 256, tmp, (size_t)(hidden_size - b * 256) * sizeof(float));
        }
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
static inline __m512i lila_scale4_epi32(int8_t s0, int8_t s1, int8_t s2, int8_t s3) {
    return _mm512_set_epi32(
        s3, s3, s3, s3, s2, s2, s2, s2, s1, s1, s1, s1, s0, s0, s0, s0);
}
static inline __m512i lila_scale4_epi16(int8_t s0, int8_t s1, int8_t s2, int8_t s3) {
    uint32_t p0 = ((uint32_t)(uint16_t)(int16_t)s0 << 16) | (uint32_t)(uint16_t)(int16_t)s0;
    uint32_t p1 = ((uint32_t)(uint16_t)(int16_t)s1 << 16) | (uint32_t)(uint16_t)(int16_t)s1;
    uint32_t p2 = ((uint32_t)(uint16_t)(int16_t)s2 << 16) | (uint32_t)(uint16_t)(int16_t)s2;
    uint32_t p3 = ((uint32_t)(uint16_t)(int16_t)s3 << 16) | (uint32_t)(uint16_t)(int16_t)s3;
    return _mm512_set_epi32(
        (int)p3, (int)p3, (int)p3, (int)p3, (int)p2, (int)p2, (int)p2, (int)p2,
        (int)p1, (int)p1, (int)p1, (int)p1, (int)p0, (int)p0, (int)p0, (int)p0);
}
#endif

static inline float q6k_dot_q8k_simd(const uint8_t *row_ptr, const LilaQ8KBlock *qx, int blocks_per_row) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512i mask_0f = _mm512_set1_epi8(0x0F);
    const __m512i mask_03 = _mm512_set1_epi8(0x03);
    float total_sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *block = row_ptr + (size_t)b * Q6_K_BLOCK_SIZE;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *scales = (const int8_t *)(block + 192);
        uint16_t d_raw;
        memcpy(&d_raw, block + 208, 2);
        float d = fp16_to_fp32(d_raw);
        const LilaQ8KBlock *xb = &qx[b];

        __m512i sumi = _mm512_setzero_si512();
        int32_t sum_bias = 0;

        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const int8_t  *sc_h = scales + half * 8;
            int sb_base = half * 4;

            __m512i ql64 = _mm512_loadu_si512((const void *)ql_h);
            __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh_h);
            __m512i qh512 = _mm512_inserti64x4(_mm512_castsi256_si512(qh0), qh0, 1);

            __m512i qh_01 = _mm512_slli_epi16(_mm512_and_si512(_mm512_mask_srli_epi16(qh512, 0xFFFF0000u, qh512, 2), mask_03), 4);
            __m512i qh_23 = _mm512_slli_epi16(_mm512_and_si512(_mm512_mask_srli_epi16(_mm512_srli_epi16(qh512, 4), 0xFFFF0000u, qh512, 6), mask_03), 4);
            __m512i u01 = _mm512_or_si512(_mm512_and_si512(ql64, mask_0f), qh_01);
            __m512i u23 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(ql64, 4), mask_0f), qh_23);

            __m512i x01 = _mm512_loadu_si512((const void *)(xb->qs + sb_base * 32));
            __m512i x23 = _mm512_loadu_si512((const void *)(xb->qs + (sb_base + 2) * 32));

#if defined(__AVX512VNNI__)
            __m512i d01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u01, x01);
            __m512i d23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u23, x23);
            sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(d01, lila_scale4_epi32(sc_h[0], sc_h[1], sc_h[2], sc_h[3])));
            sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(d23, lila_scale4_epi32(sc_h[4], sc_h[5], sc_h[6], sc_h[7])));
#else
            __m512i p01 = _mm512_maddubs_epi16(u01, x01);
            __m512i p23 = _mm512_maddubs_epi16(u23, x23);
            sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(p01, lila_scale4_epi16(sc_h[0], sc_h[1], sc_h[2], sc_h[3])));
            sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(p23, lila_scale4_epi16(sc_h[4], sc_h[5], sc_h[6], sc_h[7])));
#endif

            __m128i sc8 = _mm_loadl_epi64((const __m128i *)sc_h);
            __m128i sc16 = _mm_cvtepi8_epi16(sc8);
            __m128i bsums128 = _mm_loadu_si128((const __m128i *)(xb->bsums + sb_base * 2));
            __m128i bprod32 = _mm_madd_epi16(sc16, bsums128);
            __m128i bsum_hi = _mm_unpackhi_epi64(bprod32, bprod32);
            __m128i bsum64 = _mm_add_epi32(bprod32, bsum_hi);
            __m128i bsum32 = _mm_add_epi32(bsum64, _mm_shuffle_epi32(bsum64, _MM_SHUFFLE(1, 1, 1, 1)));
            sum_bias += _mm_cvtsi128_si32(bsum32);
        }
        float xd = d * xb->d;
        total_sum += xd * (float)(_mm512_reduce_add_epi32(sumi) - 32 * sum_bias);
    }
    return total_sum;
#elif defined(__AVX2__)
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_03 = _mm256_set1_epi8(0x03);
    __m256 sum_vec = _mm256_setzero_ps();
    float bias = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *block = row_ptr + (size_t)b * Q6_K_BLOCK_SIZE;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *scales = (const int8_t *)(block + 192);
        uint16_t d_raw;
        memcpy(&d_raw, block + 208, 2);
        float d = fp16_to_fp32(d_raw);
        const LilaQ8KBlock *xb = &qx[b];

        __m256i sumi = _mm256_setzero_si256();
        int32_t sum_bias = 0;

        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const int8_t  *sc_h = scales + half * 8;
            int sb_base = half * 4;

            __m256i ql0 = _mm256_loadu_si256((const __m256i *)(ql_h + 0));
            __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql_h + 32));
            __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh_h);

            __m256i u_vecs[4];
            u_vecs[0] = _mm256_or_si256(
                _mm256_and_si256(ql0, mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(qh0, mask_03), 4));
            u_vecs[1] = _mm256_or_si256(
                _mm256_and_si256(ql1, mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 2), mask_03), 4));
            u_vecs[2] = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 4), mask_03), 4));
            u_vecs[3] = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_0f),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 6), mask_03), 4));

            __m128i sc8 = _mm_loadl_epi64((const __m128i *)sc_h);
            __m128i sc16 = _mm_cvtepi8_epi16(sc8);
            __m128i bsums128 = _mm_loadu_si128((const __m128i *)(xb->bsums + sb_base * 2));
            __m128i bprod32 = _mm_madd_epi16(sc16, bsums128);
            __m128i bsum_hi = _mm_unpackhi_epi64(bprod32, bprod32);
            __m128i bsum64 = _mm_add_epi32(bprod32, bsum_hi);
            __m128i bsum32 = _mm_add_epi32(bsum64, _mm_shuffle_epi32(bsum64, _MM_SHUFFLE(1, 1, 1, 1)));
            sum_bias += _mm_cvtsi128_si32(bsum32);

            for (int q = 0; q < 4; q++) {
                int sb = sb_base + q;
                __m256i x_vec = _mm256_loadu_si256((const __m256i *)(xb->qs + sb * 32));
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                __m256i dp32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u_vecs[q], x_vec);
                __m256i sc_vec32 = _mm256_set_m128i(_mm_set1_epi32((int32_t)sc_h[q * 2 + 1]),
                                                    _mm_set1_epi32((int32_t)sc_h[q * 2 + 0]));
                sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(dp32, sc_vec32));
#else
                __m256i prod16 = _mm256_maddubs_epi16(u_vecs[q], x_vec);
                __m128i vs0 = _mm_set1_epi16((int16_t)sc_h[q * 2 + 0]);
                __m128i vs1 = _mm_set1_epi16((int16_t)sc_h[q * 2 + 1]);
                __m256i sc_vec16 = _mm256_set_m128i(vs1, vs0);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(prod16, sc_vec16));
#endif
            }
        }
        float xd = d * xb->d;
        sum_vec = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi), _mm256_set1_ps(xd), sum_vec);
        bias += 32.0f * xd * (float)sum_bias;
    }
    return lila_hsum_f32_avx2(sum_vec) - bias;
#elif defined(__aarch64__)
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    const uint8x16_t mask_03 = vdupq_n_u8(0x03);
    float total_sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *block = row_ptr + (size_t)b * Q6_K_BLOCK_SIZE;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *scales = (const int8_t *)(block + 192);
        uint16_t d_raw;
        memcpy(&d_raw, block + 208, 2);
        float d = fp16_to_fp32(d_raw);
        const LilaQ8KBlock *xb = &qx[b];

        int32_t sumi = 0;
        int32_t sum_bias = 0;
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const int8_t  *sc_h = scales + half * 8;
            int sb_base = half * 4;
            for (int c = 0; c < 2; c++) {
                uint8x16_t ql0 = vld1q_u8(ql_h + c * 16);
                uint8x16_t ql1 = vld1q_u8(ql_h + 32 + c * 16);
                uint8x16_t qh0 = vld1q_u8(qh_h + c * 16);

                int8x16_t u0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, mask_0f), vshlq_n_u8(vandq_u8(qh0, mask_03), 4)));
                int8x16_t u1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, mask_0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_03), 4)));
                int8x16_t u2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_03), 4)));
                int8x16_t u3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), vshlq_n_u8(vshrq_n_u8(qh0, 6), 4)));

                int8x16_t us[4] = {u0, u1, u2, u3};
                for (int q = 0; q < 4; q++) {
                    int sb = sb_base + q;
                    int8x16_t xv = vld1q_s8(xb->qs + sb * 32 + c * 16);
                    int32x4_t acc = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
                    acc = vdotq_s32(acc, us[q], xv);
#else
                    int16x8_t p0 = vmull_s8(vget_low_s8(us[q]), vget_low_s8(xv));
                    int16x8_t p1 = vmull_s8(vget_high_s8(us[q]), vget_high_s8(xv));
                    acc = vpadalq_s16(acc, p0);
                    acc = vpadalq_s16(acc, p1);
#endif
                    int sc_val = (int)sc_h[q * 2 + c];
                    sumi += vaddvq_s32(acc) * sc_val;
                    sum_bias += sc_val * (int32_t)xb->bsums[sb * 2 + c];
                }
            }
        }
        float xd = d * xb->d;
        total_sum += xd * (float)(sumi - 32 * sum_bias);
    }
    return total_sum;
#else
    float total_sum = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *block = row_ptr + (size_t)b * Q6_K_BLOCK_SIZE;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *scales = (const int8_t *)(block + 192);
        uint16_t d_raw;
        memcpy(&d_raw, block + 208, 2);
        float d = fp16_to_fp32(d_raw);
        const LilaQ8KBlock *xb = &qx[b];
        int32_t sumi = 0, sum_bias = 0;
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const int8_t  *sc_h = scales + half * 8;
            int sb_base = half * 4;
            for (int l = 0; l < 32; l++) {
                int is = l >> 4;
                int8_t u0 = (int8_t)((ql_h[l] & 0xF) | (((qh_h[l] >> 0) & 3) << 4));
                int8_t u1 = (int8_t)((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4));
                int8_t u2 = (int8_t)((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4));
                int8_t u3 = (int8_t)((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4));
                sumi += (int32_t)sc_h[is + 0] * u0 * xb->qs[(sb_base + 0) * 32 + l]
                      + (int32_t)sc_h[is + 2] * u1 * xb->qs[(sb_base + 1) * 32 + l]
                      + (int32_t)sc_h[is + 4] * u2 * xb->qs[(sb_base + 2) * 32 + l]
                      + (int32_t)sc_h[is + 6] * u3 * xb->qs[(sb_base + 3) * 32 + l];
            }
            for (int q = 0; q < 8; q++) {
                sum_bias += (int32_t)sc_h[q] * (int32_t)xb->bsums[sb_base * 2 + q];
            }
        }
        float xd = d * xb->d;
        total_sum += xd * (float)(sumi - 32 * sum_bias);
    }
    return total_sum;
#endif
}

static inline float q6k_dot_q8k_avx2(const uint8_t *row_ptr, const LilaQ8KBlock *qx, int blocks_per_row) {
    return q6k_dot_q8k_simd(row_ptr, qx, blocks_per_row);
}

#if defined(__AVX2__)

static inline void q6k_dot2_q8k_avx2(const uint8_t *r0, const uint8_t *r1,
                                     const LilaQ8KBlock *qx, int blocks_per_row,
                                     float *out0, float *out1) {
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_03 = _mm256_set1_epi8(0x03);
    __m256 sum_vec0 = _mm256_setzero_ps();
    __m256 sum_vec1 = _mm256_setzero_ps();
    float bias0 = 0.0f, bias1 = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const uint8_t *b0 = r0 + (size_t)b * Q6_K_BLOCK_SIZE;
        const uint8_t *b1 = r1 + (size_t)b * Q6_K_BLOCK_SIZE;
        uint16_t d0_r, d1_r;
        memcpy(&d0_r, b0 + 208, 2);
        memcpy(&d1_r, b1 + 208, 2);
        float d0 = fp16_to_fp32(d0_r), d1 = fp16_to_fp32(d1_r);
        const LilaQ8KBlock *xb = &qx[b];

        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        int32_t sbias0 = 0, sbias1 = 0;

        for (int half = 0; half < 2; half++) {
            const uint8_t *ql0_h = b0 + half * 64, *qh0_h = b0 + 128 + half * 32;
            const int8_t  *sc0_h = (const int8_t *)(b0 + 192) + half * 8;
            const uint8_t *ql1_h = b1 + half * 64, *qh1_h = b1 + 128 + half * 32;
            const int8_t  *sc1_h = (const int8_t *)(b1 + 192) + half * 8;
            int sb_base = half * 4;

            __m256i ql00 = _mm256_loadu_si256((const __m256i *)(ql0_h + 0));
            __m256i ql01 = _mm256_loadu_si256((const __m256i *)(ql0_h + 32));
            __m256i qh00 = _mm256_loadu_si256((const __m256i *)qh0_h);

            __m256i ql10 = _mm256_loadu_si256((const __m256i *)(ql1_h + 0));
            __m256i ql11 = _mm256_loadu_si256((const __m256i *)(ql1_h + 32));
            __m256i qh10 = _mm256_loadu_si256((const __m256i *)qh1_h);

            __m256i u0[4], u1[4];
            u0[0] = _mm256_or_si256(_mm256_and_si256(ql00, mask_0f), _mm256_slli_epi16(_mm256_and_si256(qh00, mask_03), 4));
            u0[1] = _mm256_or_si256(_mm256_and_si256(ql01, mask_0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh00, 2), mask_03), 4));
            u0[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql00, 4), mask_0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh00, 4), mask_03), 4));
            u0[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql01, 4), mask_0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh00, 6), mask_03), 4));

            u1[0] = _mm256_or_si256(_mm256_and_si256(ql10, mask_0f), _mm256_slli_epi16(_mm256_and_si256(qh10, mask_03), 4));
            u1[1] = _mm256_or_si256(_mm256_and_si256(ql11, mask_0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh10, 2), mask_03), 4));
            u1[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql10, 4), mask_0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh10, 4), mask_03), 4));
            u1[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql11, 4), mask_0f), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh10, 6), mask_03), 4));

            for (int q = 0; q < 4; q++) {
                int sb = sb_base + q;
                __m256i xv = _mm256_loadu_si256((const __m256i *)(xb->qs + sb * 32));
                int16_t b0_s = xb->bsums[sb * 2 + 0], b1_s = xb->bsums[sb * 2 + 1];

                int16_t s00 = (int16_t)sc0_h[q * 2 + 0], s01 = (int16_t)sc0_h[q * 2 + 1];
                __m256i sv0 = _mm256_set_epi16(s01,s01,s01,s01,s01,s01,s01,s01, s00,s00,s00,s00,s00,s00,s00,s00);
                sumi0 = _mm256_add_epi32(sumi0, _mm256_madd_epi16(_mm256_maddubs_epi16(u0[q], xv), sv0));
                sbias0 += (int32_t)s00 * b0_s + (int32_t)s01 * b1_s;

                int16_t s10 = (int16_t)sc1_h[q * 2 + 0], s11 = (int16_t)sc1_h[q * 2 + 1];
                __m256i sv1 = _mm256_set_epi16(s11,s11,s11,s11,s11,s11,s11,s11, s10,s10,s10,s10,s10,s10,s10,s10);
                sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(_mm256_maddubs_epi16(u1[q], xv), sv1));
                sbias1 += (int32_t)s10 * b0_s + (int32_t)s11 * b1_s;
            }
        }
        float xd0 = d0 * xb->d, xd1 = d1 * xb->d;
        sum_vec0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi0), _mm256_set1_ps(xd0), sum_vec0);
        sum_vec1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi1), _mm256_set1_ps(xd1), sum_vec1);
        bias0 += 32.0f * xd0 * (float)sbias0;
        bias1 += 32.0f * xd1 * (float)sbias1;
    }
    *out0 = lila_hsum_f32_avx2(sum_vec0) - bias0;
    *out1 = lila_hsum_f32_avx2(sum_vec1) - bias1;
}
#endif

/*
 * Fast Q6_K x Q8_K matrix-vector multiply (for LM head & Q6_K projections).
 * Uses (q6_signed) = u6 - 32 so u6 is unsigned [0..63].
 */
static inline void q6k_matvec(float *out, const uint8_t *weight_data, const float *vec,
                              int rows, int cols) {
    int blocks_per_row = (cols + QK_K - 1) / QK_K;

#if defined(__AVX2__)
    LilaQ8KBlock stack_qx[64];
    LilaQ8KBlock *qx = (blocks_per_row <= 64)
        ? stack_qx
        : (LilaQ8KBlock *)malloc((size_t)blocks_per_row * sizeof(LilaQ8KBlock));
    if (qx) {
        quantize_vec_q8k(qx, vec, cols);
        const size_t row_bytes = (size_t)blocks_per_row * Q6_K_BLOCK_SIZE;
        #pragma omp parallel for schedule(static) if(rows >= 64)
        for (int i = 0; i < rows; i++) {
            const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
            if (i + 2 < rows) {
                _mm_prefetch((const char *)(row_ptr + 2 * row_bytes), _MM_HINT_T0);
                _mm_prefetch((const char *)(row_ptr + 2 * row_bytes + 64), _MM_HINT_T0);
            }
            out[i] = q6k_dot_q8k_avx2(row_ptr, qx, blocks_per_row);
        }
        if (qx != stack_qx) free(qx);
        return;
    }
#endif

    #pragma omp parallel for schedule(static) if(rows >= 64)
    for (int i = 0; i < rows; i++) {
        const uint8_t *row_ptr = weight_data + (size_t)i * blocks_per_row * Q6_K_BLOCK_SIZE;
        float sum = 0.0f;
        float tmp[QK_K];
        for (int b = 0; b < blocks_per_row; b++) {
            dequant_q6k_block(tmp, row_ptr + (size_t)b * Q6_K_BLOCK_SIZE);
            int col_start = b * QK_K;
            int col_end = col_start + QK_K;
            if (col_end > cols) col_end = cols;
            for (int j = col_start; j < col_end; j++) {
                sum += tmp[j - col_start] * vec[j];
            }
        }
        out[i] = sum;
    }
}

/*
 * Batched Q4_K x Q8_K matrix-matrix multiply for fast prefill.
 * vecs: [n_tokens, cols], out: [n_tokens, rows]
 * Unpacks each Q4_K block ONCE and accumulates 4 tokens in YMM registers.
 */
static inline void q4k_matmul_batch_q8k_range(float *out, const uint8_t *weight_data,
                                              const LilaQ8KBlock *qx_all,
                                              int n_tokens, int row_start, int row_end,
                                              int total_rows, int blocks_per_row) {
    const size_t row_bytes = (size_t)blocks_per_row * Q4_K_BLOCK_SIZE;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512i mask_0f = _mm512_set1_epi8(0x0F);

    for (int i = row_start; i < row_end; i++) {
        const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
        if (i + 2 < row_end) {
            LILA_PREFETCH(row_ptr + 2 * row_bytes);
        }
        for (int t0 = 0; t0 < n_tokens; t0 += 32) {
            int nt = n_tokens - t0;
            if (nt > 32) nt = 32;
            float acc[32];
            for (int t = 0; t < nt; t++) acc[t] = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const uint8_t *block = row_ptr + (size_t)b * Q4_K_BLOCK_SIZE;
                uint16_t d_raw, dmin_raw;
                memcpy(&d_raw, block, 2);
                memcpy(&dmin_raw, block + 2, 2);
                float d = fp16_to_fp32(d_raw);
                float min = fp16_to_fp32(dmin_raw);
                const uint8_t *scales_packed = block + 4;
                const uint8_t *qs = block + 16;

                __m512i sumi[32];
                int32_t mins[32];
                for (int t = 0; t < nt; t++) {
                    sumi[t] = _mm512_setzero_si512();
                    mins[t] = 0;
                }

                for (int half = 0; half < 2; half++) {
                    int is = half * 4;
                    int sb = half * 4;
                    uint8_t sc0, m0, sc1, m1, sc2, m2, sc3, m3;
                    get_scale_min_k4(is + 0, scales_packed, &sc0, &m0);
                    get_scale_min_k4(is + 1, scales_packed, &sc1, &m1);
                    get_scale_min_k4(is + 2, scales_packed, &sc2, &m2);
                    get_scale_min_k4(is + 3, scales_packed, &sc3, &m3);

                    __m512i qbits = _mm512_loadu_si512((const void *)(qs + half * 64));
                    __m512i q_lo = _mm512_and_si512(qbits, mask_0f);
                    __m512i q_hi = _mm512_and_si512(_mm512_srli_epi16(qbits, 4), mask_0f);

#if defined(__AVX512VNNI__)
                    __m512i vsc_lo = _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32((int32_t)sc0), _mm512_set1_epi32((int32_t)sc2));
                    __m512i vsc_hi = _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32((int32_t)sc1), _mm512_set1_epi32((int32_t)sc3));
#else
                    __m512i vsc_lo = _mm512_mask_blend_epi16(0xFFFF0000u, _mm512_set1_epi16((int16_t)sc0), _mm512_set1_epi16((int16_t)sc2));
                    __m512i vsc_hi = _mm512_mask_blend_epi16(0xFFFF0000u, _mm512_set1_epi16((int16_t)sc1), _mm512_set1_epi16((int16_t)sc3));
#endif

                    for (int t = 0; t < nt; t++) {
                        const LilaQ8KBlock *xb = qx_all + (size_t)b * n_tokens + (t0 + t);
                        __m512i x_lo = _mm512_inserti64x4(
                            _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(xb->qs + half * 128 + 0))),
                            _mm256_loadu_si256((const __m256i *)(xb->qs + half * 128 + 64)), 1);
                        __m512i x_hi = _mm512_inserti64x4(
                            _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(xb->qs + half * 128 + 32))),
                            _mm256_loadu_si256((const __m256i *)(xb->qs + half * 128 + 96)), 1);
#if defined(__AVX512VNNI__)
                        __m512i d_lo = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q_lo, x_lo);
                        __m512i d_hi = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q_hi, x_hi);
                        sumi[t] = _mm512_add_epi32(sumi[t], _mm512_add_epi32(_mm512_mullo_epi32(d_lo, vsc_lo), _mm512_mullo_epi32(d_hi, vsc_hi)));
#else
                        __m512i pl = _mm512_madd_epi16(_mm512_maddubs_epi16(q_lo, x_lo), vsc_lo);
                        __m512i ph = _mm512_madd_epi16(_mm512_maddubs_epi16(q_hi, x_hi), vsc_hi);
                        sumi[t] = _mm512_add_epi32(sumi[t], _mm512_add_epi32(pl, ph));
#endif
                        mins[t] += (int32_t)m0 * ((int32_t)xb->bsums[(sb + 0) * 2 + 0] + (int32_t)xb->bsums[(sb + 0) * 2 + 1])
                                 + (int32_t)m1 * ((int32_t)xb->bsums[(sb + 1) * 2 + 0] + (int32_t)xb->bsums[(sb + 1) * 2 + 1])
                                 + (int32_t)m2 * ((int32_t)xb->bsums[(sb + 2) * 2 + 0] + (int32_t)xb->bsums[(sb + 2) * 2 + 1])
                                 + (int32_t)m3 * ((int32_t)xb->bsums[(sb + 3) * 2 + 0] + (int32_t)xb->bsums[(sb + 3) * 2 + 1]);
                    }
                }
                for (int t = 0; t < nt; t++) {
                    float xd = qx_all[(size_t)b * n_tokens + (t0 + t)].d;
                    acc[t] += d * xd * (float)_mm512_reduce_add_epi32(sumi[t]) - min * xd * (float)mins[t];
                }
            }
            for (int t = 0; t < nt; t++) {
                out[(size_t)(t0 + t) * total_rows + i] = acc[t];
            }
        }
    }
#elif defined(__AVX2__)
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);

    for (int i = row_start; i < row_end; i++) {
        const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
        if (i + 2 < row_end) {
            LILA_PREFETCH(row_ptr + 2 * row_bytes);
        }
        for (int t0 = 0; t0 < n_tokens; t0 += 32) {
            int nt = n_tokens - t0;
            if (nt > 32) nt = 32;
            float acc[32];
            for (int t = 0; t < nt; t++) acc[t] = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const uint8_t *block = row_ptr + (size_t)b * Q4_K_BLOCK_SIZE;
                uint16_t d_raw, dmin_raw;
                memcpy(&d_raw, block, 2);
                memcpy(&dmin_raw, block + 2, 2);
                float d = fp16_to_fp32(d_raw);
                float min = fp16_to_fp32(dmin_raw);
                const uint8_t *scales_packed = block + 4;
                const uint8_t *qs = block + 16;

                __m256i sumi[32];
                int32_t mins[32];
                for (int t = 0; t < nt; t++) {
                    sumi[t] = _mm256_setzero_si256();
                    mins[t] = 0;
                }

                int is = 0;
                for (int sb = 0; sb < 8; sb += 2) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, scales_packed, &sc1, &m1);
                    get_scale_min_k4(is + 1, scales_packed, &sc2, &m2);
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                    __m256i vsc1 = _mm256_set1_epi32((int32_t)sc1);
                    __m256i vsc2 = _mm256_set1_epi32((int32_t)sc2);
#else
                    __m256i vsc1 = _mm256_set1_epi16((int16_t)sc1);
                    __m256i vsc2 = _mm256_set1_epi16((int16_t)sc2);
#endif

                    __m256i qbits = _mm256_loadu_si256((const __m256i *)qs);
                    __m256i q_lo = _mm256_and_si256(qbits, mask_0f);
                    __m256i q_hi = _mm256_and_si256(_mm256_srli_epi16(qbits, 4), mask_0f);

                    for (int t = 0; t < nt; t++) {
                        const LilaQ8KBlock *xb = qx_all + (size_t)b * n_tokens + (t0 + t);
                        __m256i xl = _mm256_loadu_si256((const __m256i *)(xb->qs + sb * 32));
                        __m256i xh = _mm256_loadu_si256((const __m256i *)(xb->qs + (sb + 1) * 32));
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                        __m256i pl = _mm256_mullo_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), q_lo, xl), vsc1);
                        __m256i ph = _mm256_mullo_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), q_hi, xh), vsc2);
#else
                        __m256i pl = _mm256_madd_epi16(_mm256_maddubs_epi16(q_lo, xl), vsc1);
                        __m256i ph = _mm256_madd_epi16(_mm256_maddubs_epi16(q_hi, xh), vsc2);
#endif
                        sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(pl, ph));
                        mins[t] += (int32_t)m1 * ((int32_t)xb->bsums[sb * 2 + 0] + (int32_t)xb->bsums[sb * 2 + 1])
                                 + (int32_t)m2 * ((int32_t)xb->bsums[(sb + 1) * 2 + 0] + (int32_t)xb->bsums[(sb + 1) * 2 + 1]);
                    }
                    qs += 32;
                    is += 2;
                }
                for (int t = 0; t < nt; t++) {
                    float xd = qx_all[(size_t)b * n_tokens + (t0 + t)].d;
                    acc[t] += d * xd * (float)lila_hsum_i32_avx2(sumi[t]) - min * xd * (float)mins[t];
                }
            }
            for (int t = 0; t < nt; t++) {
                out[(size_t)(t0 + t) * total_rows + i] = acc[t];
            }
        }
    }
#elif defined(__aarch64__)
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);

    for (int i = row_start; i < row_end; i++) {
        const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
        if (i + 2 < row_end) {
            LILA_PREFETCH(row_ptr + 2 * row_bytes);
        }
        for (int t0 = 0; t0 < n_tokens; t0 += 32) {
            int nt = n_tokens - t0;
            if (nt > 32) nt = 32;
            float acc[32];
            for (int t = 0; t < nt; t++) acc[t] = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const uint8_t *block = row_ptr + (size_t)b * Q4_K_BLOCK_SIZE;
                uint16_t d_raw, dmin_raw;
                memcpy(&d_raw, block, 2);
                memcpy(&dmin_raw, block + 2, 2);
                float d = fp16_to_fp32(d_raw);
                float min = fp16_to_fp32(dmin_raw);
                const uint8_t *scales_packed = block + 4;
                const uint8_t *qs = block + 16;

                int32_t sumi[32];
                int32_t mins[32];
                for (int t = 0; t < nt; t++) { sumi[t] = 0; mins[t] = 0; }

                int is = 0;
                for (int sb = 0; sb < 8; sb += 2) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, scales_packed, &sc1, &m1);
                    get_scale_min_k4(is + 1, scales_packed, &sc2, &m2);
                    for (int c = 0; c < 2; c++) {
                        uint8x16_t qbits = vld1q_u8(qs + c * 16);
                        int8x16_t q_lo = vreinterpretq_s8_u8(vandq_u8(qbits, mask_0f));
                        int8x16_t q_hi = vreinterpretq_s8_u8(vshrq_n_u8(qbits, 4));
                        for (int t = 0; t < nt; t++) {
                            const LilaQ8KBlock *xb = qx_all + (size_t)b * n_tokens + (t0 + t);
                            int8x16_t xl = vld1q_s8(xb->qs + sb * 32 + c * 16);
                            int8x16_t xh = vld1q_s8(xb->qs + (sb + 1) * 32 + c * 16);
                            int32x4_t al = vdupq_n_s32(0), ah = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
                            al = vdotq_s32(al, q_lo, xl);
                            ah = vdotq_s32(ah, q_hi, xh);
#else
                            al = vpadalq_s16(al, vmull_s8(vget_low_s8(q_lo), vget_low_s8(xl)));
                            al = vpadalq_s16(al, vmull_s8(vget_high_s8(q_lo), vget_high_s8(xl)));
                            ah = vpadalq_s16(ah, vmull_s8(vget_low_s8(q_hi), vget_low_s8(xh)));
                            ah = vpadalq_s16(ah, vmull_s8(vget_high_s8(q_hi), vget_high_s8(xh)));
#endif
                            sumi[t] += vaddvq_s32(al) * (int32_t)sc1 + vaddvq_s32(ah) * (int32_t)sc2;
                        }
                    }
                    for (int t = 0; t < nt; t++) {
                        const LilaQ8KBlock *xb = qx_all + (size_t)b * n_tokens + (t0 + t);
                        mins[t] += (int32_t)m1 * ((int32_t)xb->bsums[sb * 2 + 0] + (int32_t)xb->bsums[sb * 2 + 1])
                                 + (int32_t)m2 * ((int32_t)xb->bsums[(sb + 1) * 2 + 0] + (int32_t)xb->bsums[(sb + 1) * 2 + 1]);
                    }
                    qs += 32;
                    is += 2;
                }
                for (int t = 0; t < nt; t++) {
                    float xd = qx_all[(size_t)b * n_tokens + (t0 + t)].d;
                    acc[t] += d * xd * (float)sumi[t] - min * xd * (float)mins[t];
                }
            }
            for (int t = 0; t < nt; t++) {
                out[(size_t)(t0 + t) * total_rows + i] = acc[t];
            }
        }
    }
#else
    for (int i = row_start; i < row_end; i++) {
        const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
        for (int t = 0; t < n_tokens; t++) {
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; b++) {
                const uint8_t *block = row_ptr + (size_t)b * Q4_K_BLOCK_SIZE;
                uint16_t d_raw, dmin_raw;
                memcpy(&d_raw, block, 2);
                memcpy(&dmin_raw, block + 2, 2);
                float d = fp16_to_fp32(d_raw);
                float min = fp16_to_fp32(dmin_raw);
                const uint8_t *scales_packed = block + 4;
                const uint8_t *qs = block + 16;
                const LilaQ8KBlock *xb = qx_all + (size_t)b * n_tokens + t;
                int32_t sumi = 0, mins = 0;
                int is = 0;
                for (int sb = 0; sb < 8; sb += 2) {
                    uint8_t sc1, m1, sc2, m2;
                    get_scale_min_k4(is + 0, scales_packed, &sc1, &m1);
                    get_scale_min_k4(is + 1, scales_packed, &sc2, &m2);
                    int32_t s1 = 0, s2 = 0;
                    for (int l = 0; l < 32; l++) {
                        s1 += (int32_t)(qs[l] & 0x0F) * (int32_t)xb->qs[sb * 32 + l];
                        s2 += (int32_t)(qs[l] >> 4) * (int32_t)xb->qs[(sb + 1) * 32 + l];
                    }
                    sumi += s1 * (int32_t)sc1 + s2 * (int32_t)sc2;
                    mins += (int32_t)m1 * ((int32_t)xb->bsums[sb * 2 + 0] + (int32_t)xb->bsums[sb * 2 + 1])
                          + (int32_t)m2 * ((int32_t)xb->bsums[(sb + 1) * 2 + 0] + (int32_t)xb->bsums[(sb + 1) * 2 + 1]);
                    qs += 32;
                    is += 2;
                }
                sum += d * xb->d * (float)sumi - min * xb->d * (float)mins;
            }
            out[(size_t)t * total_rows + i] = sum;
        }
    }
#endif
}

static inline void q4k_matmul_batch(float *out, const uint8_t *weight_data, const float *vecs,
                                    int n_tokens, int rows, int cols) {
    if (n_tokens <= 1) {
        if (n_tokens == 1) q4k_matvec(out, weight_data, vecs, rows, cols);
        return;
    }
    int blocks_per_row = (cols + QK_K - 1) / QK_K;
    LilaQ8KBlock *qx_all = (LilaQ8KBlock *)malloc((size_t)n_tokens * blocks_per_row * sizeof(LilaQ8KBlock));
    if (qx_all) {
        for (int t = 0; t < n_tokens; t++) {
            quantize_vec_q8k_strided(qx_all + t, n_tokens, vecs + (size_t)t * cols, cols);
        }
        #pragma omp parallel for schedule(static) if(rows >= 16)
        for (int i = 0; i < rows; i++) {
            q4k_matmul_batch_q8k_range(out, weight_data, qx_all, n_tokens, i, i + 1, rows, blocks_per_row);
        }
        free(qx_all);
        return;
    }
    for (int t = 0; t < n_tokens; t++) {
        q4k_matvec(out + (size_t)t * rows, weight_data, vecs + (size_t)t * cols, rows, cols);
    }
}

/*
 * Batched Q6_K x Q8_K matrix-matrix multiply for fast prefill.
 * vecs: [n_tokens, cols], out: [n_tokens, rows]
 * Unpacks each Q6_K block ONCE and accumulates 4 tokens in YMM registers.
 */
static inline void q6k_matmul_batch_q8k_range(float *out, const uint8_t *weight_data,
                                              const LilaQ8KBlock *qx_all,
                                              int n_tokens, int row_start, int row_end,
                                              int total_rows, int blocks_per_row) {
    const size_t row_bytes = (size_t)blocks_per_row * Q6_K_BLOCK_SIZE;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512i mask_0f = _mm512_set1_epi8(0x0F);
    const __m512i mask_03 = _mm512_set1_epi8(0x03);

    for (int i = row_start; i < row_end; i++) {
        const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
        if (i + 2 < row_end) {
            LILA_PREFETCH(row_ptr + 2 * row_bytes);
        }
        for (int t0 = 0; t0 < n_tokens; t0 += 32) {
            int nt = n_tokens - t0;
            if (nt > 32) nt = 32;
            float acc[32];
            for (int t = 0; t < nt; t++) acc[t] = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const uint8_t *block = row_ptr + (size_t)b * Q6_K_BLOCK_SIZE;
                const uint8_t *ql = block;
                const uint8_t *qh = block + 128;
                const int8_t *scales = (const int8_t *)(block + 192);
                uint16_t d_raw;
                memcpy(&d_raw, block + 208, 2);
                float d = fp16_to_fp32(d_raw);

                __m512i sumi[32];
                int32_t sbias[32];
                for (int t = 0; t < nt; t++) {
                    sumi[t] = _mm512_setzero_si512();
                    sbias[t] = 0;
                }

                for (int half = 0; half < 2; half++) {
                    const uint8_t *ql_h = ql + half * 64;
                    const uint8_t *qh_h = qh + half * 32;
                    const int8_t  *sc_h = scales + half * 8;
                    int sb_base = half * 4;

                    __m512i ql64 = _mm512_loadu_si512((const void *)ql_h);
                    __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh_h);
                    __m512i qh512 = _mm512_inserti64x4(_mm512_castsi256_si512(qh0), qh0, 1);

                    __m512i qh_01 = _mm512_slli_epi16(_mm512_and_si512(_mm512_mask_srli_epi16(qh512, 0xFFFF0000u, qh512, 2), mask_03), 4);
                    __m512i qh_23 = _mm512_slli_epi16(_mm512_and_si512(_mm512_mask_srli_epi16(_mm512_srli_epi16(qh512, 4), 0xFFFF0000u, qh512, 6), mask_03), 4);
                    __m512i u01 = _mm512_or_si512(_mm512_and_si512(ql64, mask_0f), qh_01);
                    __m512i u23 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(ql64, 4), mask_0f), qh_23);

#if defined(__AVX512VNNI__)
                    __m512i vsc01 = lila_scale4_epi32(sc_h[0], sc_h[1], sc_h[2], sc_h[3]);
                    __m512i vsc23 = lila_scale4_epi32(sc_h[4], sc_h[5], sc_h[6], sc_h[7]);
#else
                    __m512i vsc01 = lila_scale4_epi16(sc_h[0], sc_h[1], sc_h[2], sc_h[3]);
                    __m512i vsc23 = lila_scale4_epi16(sc_h[4], sc_h[5], sc_h[6], sc_h[7]);
#endif
                    __m128i sc8 = _mm_loadl_epi64((const __m128i *)sc_h);
                    __m128i sc16 = _mm_cvtepi8_epi16(sc8);

                    for (int t = 0; t < nt; t++) {
                        const LilaQ8KBlock *xb = qx_all + (size_t)b * n_tokens + (t0 + t);
                        __m512i x01 = _mm512_loadu_si512((const void *)(xb->qs + sb_base * 32));
                        __m512i x23 = _mm512_loadu_si512((const void *)(xb->qs + (sb_base + 2) * 32));
#if defined(__AVX512VNNI__)
                        __m512i d01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u01, x01);
                        __m512i d23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u23, x23);
                        sumi[t] = _mm512_add_epi32(sumi[t], _mm512_add_epi32(_mm512_mullo_epi32(d01, vsc01), _mm512_mullo_epi32(d23, vsc23)));
#else
                        __m512i p01 = _mm512_madd_epi16(_mm512_maddubs_epi16(u01, x01), vsc01);
                        __m512i p23 = _mm512_madd_epi16(_mm512_maddubs_epi16(u23, x23), vsc23);
                        sumi[t] = _mm512_add_epi32(sumi[t], _mm512_add_epi32(p01, p23));
#endif
                        __m128i bsums128 = _mm_loadu_si128((const __m128i *)(xb->bsums + sb_base * 2));
                        __m128i bprod32 = _mm_madd_epi16(sc16, bsums128);
                        __m128i bsum64 = _mm_add_epi32(bprod32, _mm_unpackhi_epi64(bprod32, bprod32));
                        __m128i bsum32 = _mm_add_epi32(bsum64, _mm_shuffle_epi32(bsum64, _MM_SHUFFLE(1, 1, 1, 1)));
                        sbias[t] += _mm_cvtsi128_si32(bsum32);
                    }
                }
                for (int t = 0; t < nt; t++) {
                    float xd = d * qx_all[(size_t)b * n_tokens + (t0 + t)].d;
                    acc[t] += xd * (float)(_mm512_reduce_add_epi32(sumi[t]) - 32 * sbias[t]);
                }
            }
            for (int t = 0; t < nt; t++) {
                out[(size_t)(t0 + t) * total_rows + i] = acc[t];
            }
        }
    }
#elif defined(__AVX2__)
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const __m256i mask_03 = _mm256_set1_epi8(0x03);

    for (int i = row_start; i < row_end; i++) {
        const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
        if (i + 2 < row_end) {
            LILA_PREFETCH(row_ptr + 2 * row_bytes);
        }
        for (int t0 = 0; t0 < n_tokens; t0 += 32) {
            int nt = n_tokens - t0;
            if (nt > 32) nt = 32;
            float acc[32];
            for (int t = 0; t < nt; t++) acc[t] = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const uint8_t *block = row_ptr + (size_t)b * Q6_K_BLOCK_SIZE;
                const uint8_t *ql = block;
                const uint8_t *qh = block + 128;
                const int8_t *scales = (const int8_t *)(block + 192);
                uint16_t d_raw;
                memcpy(&d_raw, block + 208, 2);
                float d = fp16_to_fp32(d_raw);

                __m256i sumi[32];
                int32_t sbias[32];
                for (int t = 0; t < nt; t++) {
                    sumi[t] = _mm256_setzero_si256();
                    sbias[t] = 0;
                }

                for (int half = 0; half < 2; half++) {
                    const uint8_t *ql_h = ql + half * 64;
                    const uint8_t *qh_h = qh + half * 32;
                    const int8_t  *sc_h = scales + half * 8;
                    int sb_base = half * 4;

                    __m256i ql0 = _mm256_loadu_si256((const __m256i *)(ql_h + 0));
                    __m256i ql1 = _mm256_loadu_si256((const __m256i *)(ql_h + 32));
                    __m256i qh0 = _mm256_loadu_si256((const __m256i *)qh_h);

                    __m256i u_vecs[4];
                    u_vecs[0] = _mm256_or_si256(
                        _mm256_and_si256(ql0, mask_0f),
                        _mm256_slli_epi16(_mm256_and_si256(qh0, mask_03), 4));
                    u_vecs[1] = _mm256_or_si256(
                        _mm256_and_si256(ql1, mask_0f),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 2), mask_03), 4));
                    u_vecs[2] = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask_0f),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 4), mask_03), 4));
                    u_vecs[3] = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask_0f),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh0, 6), mask_03), 4));

                    for (int q = 0; q < 4; q++) {
                        int sb = sb_base + q;
                        int16_t s0 = (int16_t)sc_h[q * 2 + 0];
                        int16_t s1 = (int16_t)sc_h[q * 2 + 1];
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                        __m256i sc_vec32 = _mm256_set_m128i(_mm_set1_epi32((int32_t)s1), _mm_set1_epi32((int32_t)s0));
#else
                        __m256i sc_vec16 = _mm256_set_m128i(_mm_set1_epi16(s1), _mm_set1_epi16(s0));
#endif
                        for (int t = 0; t < nt; t++) {
                            const LilaQ8KBlock *xb = qx_all + (size_t)b * n_tokens + (t0 + t);
                            __m256i xv = _mm256_loadu_si256((const __m256i *)(xb->qs + sb * 32));
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                            __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u_vecs[q], xv);
                            sumi[t] = _mm256_add_epi32(sumi[t], _mm256_mullo_epi32(dp, sc_vec32));
#else
                            sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(_mm256_maddubs_epi16(u_vecs[q], xv), sc_vec16));
#endif
                            sbias[t] += (int32_t)s0 * xb->bsums[sb * 2 + 0] + (int32_t)s1 * xb->bsums[sb * 2 + 1];
                        }
                    }
                }
                for (int t = 0; t < nt; t++) {
                    float xd = d * qx_all[(size_t)b * n_tokens + (t0 + t)].d;
                    acc[t] += xd * (float)(lila_hsum_i32_avx2(sumi[t]) - 32 * sbias[t]);
                }
            }
            for (int t = 0; t < nt; t++) {
                out[(size_t)(t0 + t) * total_rows + i] = acc[t];
            }
        }
    }
#elif defined(__aarch64__)
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    const uint8x16_t mask_03 = vdupq_n_u8(0x03);

    for (int i = row_start; i < row_end; i++) {
        const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
        if (i + 2 < row_end) {
            LILA_PREFETCH(row_ptr + 2 * row_bytes);
        }
        for (int t0 = 0; t0 < n_tokens; t0 += 32) {
            int nt = n_tokens - t0;
            if (nt > 32) nt = 32;
            float acc[32];
            for (int t = 0; t < nt; t++) acc[t] = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                const uint8_t *block = row_ptr + (size_t)b * Q6_K_BLOCK_SIZE;
                const uint8_t *ql = block;
                const uint8_t *qh = block + 128;
                const int8_t *scales = (const int8_t *)(block + 192);
                uint16_t d_raw;
                memcpy(&d_raw, block + 208, 2);
                float d = fp16_to_fp32(d_raw);

                int32_t sumi[32];
                int32_t sbias[32];
                for (int t = 0; t < nt; t++) { sumi[t] = 0; sbias[t] = 0; }

                for (int half = 0; half < 2; half++) {
                    const uint8_t *ql_h = ql + half * 64;
                    const uint8_t *qh_h = qh + half * 32;
                    const int8_t  *sc_h = scales + half * 8;
                    int sb_base = half * 4;
                    for (int c = 0; c < 2; c++) {
                        uint8x16_t ql0 = vld1q_u8(ql_h + c * 16);
                        uint8x16_t ql1 = vld1q_u8(ql_h + 32 + c * 16);
                        uint8x16_t qh0 = vld1q_u8(qh_h + c * 16);
                        int8x16_t us[4];
                        us[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, mask_0f), vshlq_n_u8(vandq_u8(qh0, mask_03), 4)));
                        us[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, mask_0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), mask_03), 4)));
                        us[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), mask_03), 4)));
                        us[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), vshlq_n_u8(vshrq_n_u8(qh0, 6), 4)));
                        for (int q = 0; q < 4; q++) {
                            int sb = sb_base + q;
                            int sc_val = (int)sc_h[q * 2 + c];
                            for (int t = 0; t < nt; t++) {
                                const LilaQ8KBlock *xb = qx_all + (size_t)b * n_tokens + (t0 + t);
                                int8x16_t xv = vld1q_s8(xb->qs + sb * 32 + c * 16);
                                int32x4_t acc4 = vdupq_n_s32(0);
#if defined(__ARM_FEATURE_DOTPROD)
                                acc4 = vdotq_s32(acc4, us[q], xv);
#else
                                acc4 = vpadalq_s16(acc4, vmull_s8(vget_low_s8(us[q]), vget_low_s8(xv)));
                                acc4 = vpadalq_s16(acc4, vmull_s8(vget_high_s8(us[q]), vget_high_s8(xv)));
#endif
                                sumi[t] += vaddvq_s32(acc4) * sc_val;
                                sbias[t] += sc_val * (int32_t)xb->bsums[sb * 2 + c];
                            }
                        }
                    }
                }
                for (int t = 0; t < nt; t++) {
                    float xd = d * qx_all[(size_t)b * n_tokens + (t0 + t)].d;
                    acc[t] += xd * (float)(sumi[t] - 32 * sbias[t]);
                }
            }
            for (int t = 0; t < nt; t++) {
                out[(size_t)(t0 + t) * total_rows + i] = acc[t];
            }
        }
    }
#else
    for (int i = row_start; i < row_end; i++) {
        const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
        for (int t = 0; t < n_tokens; t++) {
            out[(size_t)t * total_rows + i] = q6k_dot_q8k_simd(row_ptr, qx_all + t, blocks_per_row);
            /* Note: qx_all is strided by n_tokens across blocks in batch mode */
            float sum = 0.0f;
            for (int b = 0; b < blocks_per_row; b++) {
                sum += q6k_dot_q8k_simd(row_ptr + (size_t)b * Q6_K_BLOCK_SIZE, qx_all + (size_t)b * n_tokens + t, 1);
            }
            out[(size_t)t * total_rows + i] = sum;
        }
    }
#endif
}

static inline void q6k_matmul_batch(float *out, const uint8_t *weight_data, const float *vecs,
                                    int n_tokens, int rows, int cols) {
    if (n_tokens <= 1) {
        if (n_tokens == 1) q6k_matvec(out, weight_data, vecs, rows, cols);
        return;
    }
    int blocks_per_row = (cols + QK_K - 1) / QK_K;
    LilaQ8KBlock *qx_all = (LilaQ8KBlock *)malloc((size_t)n_tokens * blocks_per_row * sizeof(LilaQ8KBlock));
    if (qx_all) {
        for (int t = 0; t < n_tokens; t++) {
            quantize_vec_q8k_strided(qx_all + t, n_tokens, vecs + (size_t)t * cols, cols);
        }
        #pragma omp parallel for schedule(static) if(rows >= 16)
        for (int i = 0; i < rows; i++) {
            q6k_matmul_batch_q8k_range(out, weight_data, qx_all, n_tokens, i, i + 1, rows, blocks_per_row);
        }
        free(qx_all);
        return;
    }
    for (int t = 0; t < n_tokens; t++) {
        q6k_matvec(out + (size_t)t * rows, weight_data, vecs + (size_t)t * cols, rows, cols);
    }
}

/*
 * Pre-unpacked Q6_K block for ASI_TENSOR_LAYOUT_AVX2_REPACK (288 bytes per 256 weights).
 * Eliminates runtime ql/qh nibble & 2-bit unpacking on the 248k-row LM head and projections.
 */
typedef struct {
    uint8_t u6[256];
    int8_t  scales[16];
    float   d;
    uint8_t pad[12];
} LilaQ6KRepackedBlock;

static inline void dequant_q6k_repacked_row(float *out, const uint8_t *row_data, int hidden_size) {
    int n_blocks = (hidden_size + 255) / 256;
    for (int b = 0; b < n_blocks; b++) {
        const LilaQ6KRepackedBlock *blk = (const LilaQ6KRepackedBlock *)(row_data + (size_t)b * sizeof(LilaQ6KRepackedBlock));
        float d = blk->d;
        int base = b * 256;
        int avail = hidden_size - base;
        if (avail > 256) avail = 256;
        for (int j = 0; j < avail; j++) {
            out[base + j] = d * (float)blk->scales[j >> 4] * (float)((int)blk->u6[j] - 32);
        }
    }
}

static inline void q6k_repacked_matvec(float *out, const uint8_t *weight_data, const float *vec,
                                       int rows, int cols) {
    int blocks_per_row = (cols + QK_K - 1) / QK_K;
#if defined(__AVX2__)
    LilaQ8KBlock stack_qx[64];
    LilaQ8KBlock *qx = (blocks_per_row <= 64)
        ? stack_qx
        : (LilaQ8KBlock *)malloc((size_t)blocks_per_row * sizeof(LilaQ8KBlock));
    if (qx) {
        quantize_vec_q8k(qx, vec, cols);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const size_t row_bytes = (size_t)blocks_per_row * sizeof(LilaQ6KRepackedBlock);
        #pragma omp parallel for schedule(static) if(rows >= 64)
        for (int i = 0; i < rows; i++) {
            const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
            if (i + 2 < rows) {
                _mm_prefetch((const char *)(row_ptr + 2 * row_bytes), _MM_HINT_T0);
                _mm_prefetch((const char *)(row_ptr + 2 * row_bytes + 128), _MM_HINT_T0);
            }
            __m256 sum_vec = _mm256_setzero_ps();
            float bias = 0.0f;
            for (int b = 0; b < blocks_per_row; b++) {
                const LilaQ6KRepackedBlock *blk = (const LilaQ6KRepackedBlock *)(row_ptr + (size_t)b * sizeof(LilaQ6KRepackedBlock));
                const LilaQ8KBlock *xb = &qx[b];
                float d = blk->d;
                for (int sb = 0; sb < 8; sb++) {
                    __m256i uv = _mm256_loadu_si256((const __m256i *)(blk->u6 + sb * 32));
                    __m256i xv = _mm256_loadu_si256((const __m256i *)(xb->qs + sb * 32));
                    __m256i prod32 = _mm256_madd_epi16(_mm256_maddubs_epi16(uv, xv), ones16);
                    float xd = xb->d;
                    float sc0 = d * (float)blk->scales[sb * 2 + 0] * xd;
                    float sc1 = d * (float)blk->scales[sb * 2 + 1] * xd;
                    sum_vec = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(prod32),
                        _mm256_set_ps(sc1, sc1, sc1, sc1, sc0, sc0, sc0, sc0),
                        sum_vec);
                    bias += 32.0f * (sc0 * (float)xb->bsums[sb * 2 + 0] + sc1 * (float)xb->bsums[sb * 2 + 1]);
                }
            }
            out[i] = lila_hsum_f32_avx2(sum_vec) - bias;
        }
        if (qx != stack_qx) free(qx);
        return;
    }
#endif
    (void)out; (void)weight_data; (void)vec; (void)rows; (void)cols;
}

static inline void q6k_repacked_matmul_batch(float *out, const uint8_t *weight_data, const float *vecs,
                                             int n_tokens, int rows, int cols) {
    if (n_tokens <= 1) {
        if (n_tokens == 1) q6k_repacked_matvec(out, weight_data, vecs, rows, cols);
        return;
    }
#if defined(__AVX2__)
    int blocks_per_row = (cols + QK_K - 1) / QK_K;
    LilaQ8KBlock *qx_all = (LilaQ8KBlock *)malloc((size_t)n_tokens * blocks_per_row * sizeof(LilaQ8KBlock));
    if (qx_all) {
        for (int t = 0; t < n_tokens; t++) {
            quantize_vec_q8k_strided(qx_all + t, n_tokens, vecs + (size_t)t * cols, cols);
        }
        const __m256i ones16 = _mm256_set1_epi16(1);
        const size_t row_bytes = (size_t)blocks_per_row * sizeof(LilaQ6KRepackedBlock);
        #pragma omp parallel for schedule(static) if(rows >= 16)
        for (int i = 0; i < rows; i++) {
            const uint8_t *row_ptr = weight_data + (size_t)i * row_bytes;
            int t = 0;
            for (; t + 3 < n_tokens; t += 4) {
                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
                float bias0 = 0.0f, bias1 = 0.0f, bias2 = 0.0f, bias3 = 0.0f;
                for (int b = 0; b < blocks_per_row; b++) {
                    const LilaQ6KRepackedBlock *blk = (const LilaQ6KRepackedBlock *)(row_ptr + (size_t)b * sizeof(LilaQ6KRepackedBlock));
                    float d = blk->d;
                    const LilaQ8KBlock *xb0 = qx_all + (size_t)b * n_tokens + (t + 0);
                    const LilaQ8KBlock *xb1 = qx_all + (size_t)b * n_tokens + (t + 1);
                    const LilaQ8KBlock *xb2 = qx_all + (size_t)b * n_tokens + (t + 2);
                    const LilaQ8KBlock *xb3 = qx_all + (size_t)b * n_tokens + (t + 3);
                    for (int sb = 0; sb < 8; sb++) {
                        __m256i uv = _mm256_loadu_si256((const __m256i *)(blk->u6 + sb * 32));
                        float dsc0 = d * (float)blk->scales[sb * 2 + 0];
                        float dsc1 = d * (float)blk->scales[sb * 2 + 1];
                        #define LILA_Q6R_TSTEP(ACC, BIAS, XB) do { \
                            __m256i xv = _mm256_loadu_si256((const __m256i *)((XB)->qs + sb * 32)); \
                            __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(uv, xv), ones16); \
                            float xd = (XB)->d, s0 = dsc0 * xd, s1 = dsc1 * xd; \
                            (ACC) = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set_ps(s1, s1, s1, s1, s0, s0, s0, s0), (ACC)); \
                            (BIAS) += 32.0f * (s0 * (float)(XB)->bsums[sb * 2 + 0] + s1 * (float)(XB)->bsums[sb * 2 + 1]); \
                        } while (0)
                        LILA_Q6R_TSTEP(acc0, bias0, xb0);
                        LILA_Q6R_TSTEP(acc1, bias1, xb1);
                        LILA_Q6R_TSTEP(acc2, bias2, xb2);
                        LILA_Q6R_TSTEP(acc3, bias3, xb3);
                        #undef LILA_Q6R_TSTEP
                    }
                }
                out[(size_t)(t + 0) * rows + i] = lila_hsum_f32_avx2(acc0) - bias0;
                out[(size_t)(t + 1) * rows + i] = lila_hsum_f32_avx2(acc1) - bias1;
                out[(size_t)(t + 2) * rows + i] = lila_hsum_f32_avx2(acc2) - bias2;
                out[(size_t)(t + 3) * rows + i] = lila_hsum_f32_avx2(acc3) - bias3;
            }
            for (; t < n_tokens; t++) {
                __m256 acc = _mm256_setzero_ps();
                float bias = 0.0f;
                for (int b = 0; b < blocks_per_row; b++) {
                    const LilaQ6KRepackedBlock *blk = (const LilaQ6KRepackedBlock *)(row_ptr + (size_t)b * sizeof(LilaQ6KRepackedBlock));
                    const LilaQ8KBlock *xb = qx_all + (size_t)b * n_tokens + t;
                    float d = blk->d;
                    for (int sb = 0; sb < 8; sb++) {
                        __m256i uv = _mm256_loadu_si256((const __m256i *)(blk->u6 + sb * 32));
                        __m256i xv = _mm256_loadu_si256((const __m256i *)(xb->qs + sb * 32));
                        __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(uv, xv), ones16);
                        float xd = xb->d, s0 = d * (float)blk->scales[sb * 2 + 0] * xd, s1 = d * (float)blk->scales[sb * 2 + 1] * xd;
                        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set_ps(s1, s1, s1, s1, s0, s0, s0, s0), acc);
                        bias += 32.0f * (s0 * (float)xb->bsums[sb * 2 + 0] + s1 * (float)xb->bsums[sb * 2 + 1]);
                    }
                }
                out[(size_t)t * rows + i] = lila_hsum_f32_avx2(acc) - bias;
            }
        }
        free(qx_all);
        return;
    }
#endif
    (void)out; (void)weight_data; (void)vecs; (void)n_tokens; (void)rows; (void)cols;
}

#endif /* LILA_Q4K_H */


