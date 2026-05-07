#include <stdio.h>
#include <string.h>

#ifdef __x86_64__
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
void lila_print_cpu_features(void) {
    printf("Unknown architecture\n");
}
#endif
