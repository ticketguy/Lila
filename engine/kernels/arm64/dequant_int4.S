// ═══════════════════════════════════════════════════════════════════════════════
// Lila Engine — INT4 Dequantization Kernel (ARM64 NEON)
//
// Same operation as x86 version but using ARM NEON intrinsics pattern.
// Processes 4 elements at a time using 128-bit NEON registers.
//
// void lila_dequant_int4_neon(
//     float *output,          // x0
//     const uint8_t *indices, // x1
//     const float *codebook,  // x2
//     const float *scales,    // x3
//     int n_elements,         // x4 (w4)
//     int group_size          // x5 (w5)
// );
// ═══════════════════════════════════════════════════════════════════════════════

    .text
    .global lila_dequant_int4_neon
    .type lila_dequant_int4_neon, %function

lila_dequant_int4_neon:
    // Save callee-saved registers
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    
    mov x19, x0        // output
    mov x20, x1        // indices
    mov x21, x2        // codebook
    mov x22, xzr       // counter
    
.Lloop:
    cmp w22, w4
    bge .Ldone
    
    // Get packed byte
    lsr x6, x22, #1        // byte_idx = element / 2
    ldrb w7, [x20, x6]     // load packed byte
    
    // Extract nibble
    tst x22, #1
    bne .Lhigh
    and w7, w7, #0x0F      // low nibble
    b .Llookup
.Lhigh:
    lsr w7, w7, #4         // high nibble
    
.Llookup:
    // Codebook lookup
    ldr s0, [x21, x7, lsl #2]  // codebook[index]
    
    // Get group scale
    udiv w8, w22, w5       // group_idx = element / group_size
    ldr s1, [x3, x8, lsl #2]   // scale
    
    // Multiply
    fmul s0, s0, s1
    
    // Store
    str s0, [x19, x22, lsl #2]
    
    add w22, w22, #1
    b .Lloop
    
.Ldone:
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ret

// ═══════════════════════════════════════════════════════════════════════════════
// NOTE: Scalar fallback. NEON vectorized version TODO.
// ═══════════════════════════════════════════════════════════════════════════════
