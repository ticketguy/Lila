#include <stdio.h>
#include <string.h>
#include "detect.h"

/*
 * CPU Feature Detection — Platform Independent
 *
 * On Windows: uses MSVC intrinsics or skips to LilaVM fallback
 * On Linux/macOS x86: uses GCC __cpuid
 * On ARM64: NEON always available
 */

#if defined(_WIN32) && defined(_MSC_VER)
/* ── Windows (MSVC) ── */
#include <intrin.h>

typedef struct {
    int has_avx2;
    int has_fma;
    int has_avx512f;
    int has_avx512bw;
    int has_avx512vnni;
} LilaCPUFeatures;

LilaCPUFeatures lila_detect_cpu(void) {
    LilaCPUFeatures f = {0};
    int cpuInfo[4];

    /* Function 7, sub 0 for AVX2 */
    __cpuidex(cpuInfo, 7, 0);
    f.has_avx2 = (cpuInfo[1] >> 5) & 1;
    f.has_avx512f = (cpuInfo[1] >> 16) & 1;
    f.has_avx512bw = (cpuInfo[1] >> 30) & 1;
    f.has_avx512vnni = (cpuInfo[2] >> 11) & 1;

    /* Function 1 for FMA */
    __cpuid(cpuInfo, 1);
    f.has_fma = (cpuInfo[2] >> 12) & 1;

    return f;
}

void lila_print_cpu_features(void) {
    LilaCPUFeatures f = lila_detect_cpu();
    printf("CPU Features (Windows):\n");
    printf("  AVX2:       %s\n", f.has_avx2 ? "YES" : "no");
    printf("  FMA:        %s\n", f.has_fma ? "YES" : "no");
    printf("  AVX-512F:   %s\n", f.has_avx512f ? "YES" : "no");
    printf("  AVX-512BW:  %s\n", f.has_avx512bw ? "YES" : "no");
    printf("  AVX-512VNNI:%s\n", f.has_avx512vnni ? "YES" : "no");

    if (f.has_avx2) {
        printf("  >> Using LilaVM bytecode (native ASM kernels not linked on Windows)\n");
    } else {
        printf("  >> Using scalar fallback via LilaVM\n");
    }
}

#elif defined(_WIN32) && !defined(_MSC_VER)
/* ── Windows (MinGW/GCC) ── */
#include <cpuid.h>

typedef struct {
    int has_avx2;
    int has_fma;
    int has_avx512f;
    int has_avx512bw;
    int has_avx512vnni;
} LilaCPUFeatures;

LilaCPUFeatures lila_detect_cpu(void) {
    LilaCPUFeatures f = {0};
    unsigned int eax, ebx, ecx, edx;

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    f.has_avx2 = (ebx >> 5) & 1;
    f.has_avx512f = (ebx >> 16) & 1;
    f.has_avx512bw = (ebx >> 30) & 1;
    f.has_avx512vnni = (ecx >> 11) & 1;

    __cpuid(1, eax, ebx, ecx, edx);
    f.has_fma = (ecx >> 12) & 1;

    return f;
}

void lila_print_cpu_features(void) {
    LilaCPUFeatures f = lila_detect_cpu();
    printf("CPU Features (Windows/MinGW):\n");
    printf("  AVX2:       %s\n", f.has_avx2 ? "YES" : "no");
    printf("  FMA:        %s\n", f.has_fma ? "YES" : "no");
    printf("  AVX-512F:   %s\n", f.has_avx512f ? "YES" : "no");
    printf("  >> Using LilaVM bytecode kernels\n");
}

#elif defined(__x86_64__) && !defined(_WIN32)
/* ── Linux/macOS x86_64 ── */
#include <cpuid.h>

typedef struct {
    int has_avx2;
    int has_fma;
    int has_avx512f;
    int has_avx512bw;
    int has_avx512vnni;
} LilaCPUFeatures;

LilaCPUFeatures lila_detect_cpu(void) {
    LilaCPUFeatures f = {0};
    unsigned int eax, ebx, ecx, edx;
    
    /* Check AVX2 + FMA (function 7, sub 0) */
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    f.has_avx2 = (ebx >> 5) & 1;
    
    /* FMA (function 1) */
    __cpuid(1, eax, ebx, ecx, edx);
    f.has_fma = (ecx >> 12) & 1;
    
    /* AVX-512 (function 7, sub 0) */
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    f.has_avx512f = (ebx >> 16) & 1;
    f.has_avx512bw = (ebx >> 30) & 1;
    f.has_avx512vnni = (ecx >> 11) & 1;
    
    return f;
}

void lila_print_cpu_features(void) {
    LilaCPUFeatures f = lila_detect_cpu();
    printf("CPU Features:\n");
    printf("  AVX2:       %s\n", f.has_avx2 ? "YES" : "no");
    printf("  FMA:        %s\n", f.has_fma ? "YES" : "no");
    printf("  AVX-512F:   %s\n", f.has_avx512f ? "YES" : "no");
    printf("  AVX-512BW:  %s\n", f.has_avx512bw ? "YES" : "no");
    printf("  AVX-512VNNI:%s\n", f.has_avx512vnni ? "YES" : "no");
    
    if (f.has_avx512f) {
        printf("  >> Using AVX-512 kernels\n");
    } else if (f.has_avx2 && f.has_fma) {
        printf("  >> Using AVX2+FMA kernels\n");
    } else {
        printf("  >> Using scalar fallback\n");
    }
}

#elif defined(__aarch64__)
/* ── ARM64 (Linux, macOS M-series) ── */

typedef struct {
    int has_neon;       /* Always on ARM64 */
    int has_sve;
    int has_dotprod;
    int has_fp16;
} LilaCPUFeatures;

LilaCPUFeatures lila_detect_cpu(void) {
    LilaCPUFeatures f = {0};
    f.has_neon = 1;  /* Always available on aarch64 */
    
    /* SVE detection via /proc/cpuinfo or hwcap */
    /* TODO: proper detection */
    
    return f;
}

void lila_print_cpu_features(void) {
    LilaCPUFeatures f = lila_detect_cpu();
    printf("CPU Features (ARM64):\n");
    printf("  NEON:    %s\n", f.has_neon ? "YES" : "no");
    printf("  SVE:     %s\n", f.has_sve ? "YES" : "no");
    printf("  DotProd: %s\n", f.has_dotprod ? "YES" : "no");
    printf("  FP16:    %s\n", f.has_fp16 ? "YES" : "no");
}

#else
/* ── Unknown architecture — use LilaVM bytecode ── */
void lila_print_cpu_features(void) {
    printf("CPU Features: Unknown architecture\n");
    printf("  >> Using LilaVM bytecode kernels (universal fallback)\n");
}
#endif
