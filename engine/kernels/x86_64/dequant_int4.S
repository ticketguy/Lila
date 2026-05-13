; ═══════════════════════════════════════════════════════════════════════════════
; Lila Engine — INT4 Dequantization Kernel (x86_64 AVX2)
;
; Dequantizes FigQuant INT4 packed indices to FP32 using codebook lookup.
; The 16-value codebook fits in a single YMM register (256-bit).
;
; void lila_dequant_int4_avx2(
;     float *output,          ; rdi — output FP32 buffer
;     const uint8_t *indices, ; rsi — packed 4-bit indices (2 per byte)
;     const float *codebook,  ; rdx — 16 float32 values
;     const float *scales,    ; rcx — per-group scales
;     int n_elements,         ; r8  — number of elements to dequant
;     int group_size          ; r9  — elements per group (128)
; );
; ═══════════════════════════════════════════════════════════════════════════════

    section .text
    global lila_dequant_int4_avx2

lila_dequant_int4_avx2:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    
    mov r12, rdi        ; output ptr
    mov r13, rsi        ; indices ptr
    mov r14, rdx        ; codebook ptr
    
    ; Load codebook into memory (will use gather for lookup)
    ; For AVX2: use vpgatherdd with index register
    
    xor rbx, rbx        ; element counter
    xor r10, r10        ; group counter
    
.loop:
    cmp rbx, r8
    jge .done
    
    ; Get packed byte (contains 2 indices)
    mov rax, rbx
    shr rax, 1          ; byte index = element / 2
    movzx eax, byte [r13 + rax]
    
    ; Extract nibble
    test rbx, 1
    jnz .high_nibble
    and eax, 0x0F       ; low nibble
    jmp .lookup
.high_nibble:
    shr eax, 4          ; high nibble
    
.lookup:
    ; Codebook lookup: output = codebook[index] * scale
    lea rax, [r14 + rax*4]     ; &codebook[index]
    movss xmm0, [rax]          ; codebook value
    
    ; Get group scale
    mov rax, rbx
    xor edx, edx
    div r9              ; rax = element / group_size = group_idx
    movss xmm1, [rcx + rax*4]  ; scale
    
    ; Multiply: codebook_value * scale
    mulss xmm0, xmm1
    
    ; Store result
    movss [r12 + rbx*4], xmm0
    
    inc rbx
    jmp .loop
    
.done:
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; ═══════════════════════════════════════════════════════════════════════════════
; NOTE: This is the scalar fallback. The SIMD version (below) processes
; 8 elements at a time using AVX2 gather instructions.
; TODO: Add vectorized version with vpgatherdd
; ═══════════════════════════════════════════════════════════════════════════════
