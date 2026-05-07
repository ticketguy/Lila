; ═══════════════════════════════════════════════════════════════════════════════
; Lila Engine — Matrix-Vector Multiply (x86_64 AVX2 + FMA)
;
; Computes: out[i] = dot(matrix[i,:], vector[:])  for all rows
; Processes 8 floats per cycle using 256-bit YMM registers.
;
; void lila_matvec_avx2(
;     float *out,         ; rdi — output [rows]
;     const float *mat,   ; rsi — matrix [rows × cols], row-major
;     const float *vec,   ; rdx — vector [cols]
;     int rows,           ; ecx
;     int cols            ; r8d
; );
;
; Performance: ~8 FLOPs/cycle (FMA: multiply + add in one instruction)
; ═══════════════════════════════════════════════════════════════════════════════

    section .text
    global lila_matvec_avx2

lila_matvec_avx2:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    mov r12, rdi        ; out
    mov r13, rsi        ; mat
    mov r14, rdx        ; vec
    mov r15d, ecx       ; rows
    mov ebx, r8d        ; cols
    
    ; cols_aligned = cols & ~7 (multiple of 8 for SIMD)
    mov r10d, ebx
    and r10d, ~7        ; r10 = cols rounded down to 8
    
    xor ecx, ecx        ; row counter
    
.row_loop:
    cmp ecx, r15d
    jge .done
    
    ; Compute row offset: mat_row = mat + row * cols * 4
    mov rax, rcx
    imul rax, rbx       ; row * cols
    lea rsi, [r13 + rax*4]  ; mat_row ptr
    
    ; Zero accumulator
    vxorps ymm0, ymm0, ymm0    ; sum = 0 (8 floats)
    
    ; SIMD loop: process 8 elements at a time
    xor edx, edx        ; col counter
.col_loop:
    cmp edx, r10d
    jge .col_remainder
    
    ; Load 8 floats from matrix row and vector
    vmovups ymm1, [rsi + rdx*4]     ; mat[row, col:col+8]
    vmovups ymm2, [r14 + rdx*4]     ; vec[col:col+8]
    
    ; Fused multiply-add: sum += mat * vec
    vfmadd231ps ymm0, ymm1, ymm2
    
    add edx, 8
    jmp .col_loop
    
.col_remainder:
    ; Horizontal sum of ymm0 (8 floats → 1 float)
    vextractf128 xmm1, ymm0, 1     ; high 128 bits
    vaddps xmm0, xmm0, xmm1        ; add high to low
    vhaddps xmm0, xmm0, xmm0       ; horizontal add
    vhaddps xmm0, xmm0, xmm0       ; horizontal add again
    
    ; Handle remaining columns (cols % 8) with scalar
    cmp edx, ebx
    jge .store_result
    
.scalar_loop:
    cmp edx, ebx
    jge .store_result
    movss xmm1, [rsi + rdx*4]
    movss xmm2, [r14 + rdx*4]
    mulss xmm1, xmm2
    addss xmm0, xmm1
    inc edx
    jmp .scalar_loop
    
.store_result:
    ; Store result for this row
    movss [r12 + rcx*4], xmm0
    
    inc ecx
    jmp .row_loop
    
.done:
    vzeroupper           ; Clear upper YMM to avoid SSE/AVX transition penalty
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
