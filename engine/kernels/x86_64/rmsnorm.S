; ═══════════════════════════════════════════════════════════════════════════════
; Lila Engine — RMS Normalization (x86_64 AVX2)
;
; Computes: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
; Two passes: 1) compute variance, 2) normalize + scale
;
; void lila_rmsnorm_avx2(
;     float *out,          ; rdi
;     const float *x,      ; rsi — input [hidden_size]
;     const float *weight,  ; rdx — learned scale [hidden_size]
;     int size,            ; ecx — hidden_size
;     float eps            ; xmm0 — epsilon (usually 1e-6)
; );
; ═══════════════════════════════════════════════════════════════════════════════

    section .text
    global lila_rmsnorm_avx2

lila_rmsnorm_avx2:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    
    mov r12, rdi        ; out
    mov rbx, rsi        ; x
    ; rdx = weight, ecx = size, xmm0 = eps
    
    ; Save eps
    movss [rsp-4], xmm0
    
    ; ── Pass 1: Compute sum of squares ──
    vxorps ymm1, ymm1, ymm1    ; sum_sq = 0
    mov eax, ecx
    and eax, ~7                  ; aligned count
    xor r8d, r8d                 ; counter
    
.sum_loop:
    cmp r8d, eax
    jge .sum_remainder
    vmovups ymm2, [rbx + r8*4]
    vfmadd231ps ymm1, ymm2, ymm2   ; sum_sq += x[i]^2
    add r8d, 8
    jmp .sum_loop
    
.sum_remainder:
    ; Horizontal sum ymm1
    vextractf128 xmm2, ymm1, 1
    vaddps xmm1, xmm1, xmm2
    vhaddps xmm1, xmm1, xmm1
    vhaddps xmm1, xmm1, xmm1
    ; xmm1[0] = sum of squares (partial — add scalar remainder)
    
    ; Scalar remainder for sum
.sum_scalar:
    cmp r8d, ecx
    jge .compute_scale
    movss xmm2, [rbx + r8*4]
    mulss xmm2, xmm2
    addss xmm1, xmm2
    inc r8d
    jmp .sum_scalar
    
.compute_scale:
    ; mean = sum_sq / size
    cvtsi2ss xmm3, ecx
    divss xmm1, xmm3           ; mean(x^2)
    
    ; Add eps
    movss xmm0, [rsp-4]        ; reload eps
    addss xmm1, xmm0           ; mean + eps
    
    ; rsqrt
    rsqrtss xmm1, xmm1         ; inv_rms = 1/sqrt(mean + eps)
    
    ; Broadcast inv_rms to ymm1
    vbroadcastss ymm1, xmm1
    
    ; ── Pass 2: Normalize and scale ──
    xor r8d, r8d
    mov eax, ecx
    and eax, ~7
    
.norm_loop:
    cmp r8d, eax
    jge .norm_remainder
    vmovups ymm2, [rbx + r8*4]     ; x[i]
    vmulps ymm2, ymm2, ymm1        ; x[i] * inv_rms
    vmovups ymm3, [rdx + r8*4]     ; weight[i]
    vmulps ymm2, ymm2, ymm3        ; * weight[i]
    vmovups [r12 + r8*4], ymm2     ; store
    add r8d, 8
    jmp .norm_loop
    
.norm_remainder:
    ; Scalar remainder
.norm_scalar:
    cmp r8d, ecx
    jge .norm_done
    movss xmm2, [rbx + r8*4]
    mulss xmm2, xmm1
    movss xmm3, [rdx + r8*4]
    mulss xmm2, xmm3
    movss [r12 + r8*4], xmm2
    inc r8d
    jmp .norm_scalar
    
.norm_done:
    vzeroupper
    pop r12
    pop rbx
    pop rbp
    ret
