#ifndef LILA_VM_H
#define LILA_VM_H

/*
 * LilaVM — Portable Bytecode Virtual Machine
 *
 * A register-based VM that executes compute kernels on ANY CPU.
 * The bytecode ISA is designed specifically for tensor math:
 *   - Vector registers (256 floats each)
 *   - Fused multiply-add
 *   - Reduction operations
 *   - Memory access patterns optimized for neural network inference
 *
 * On load, the VM can either:
 *   1. Interpret bytecode directly (works everywhere, slower)
 *   2. JIT compile to native code (fast, requires writable+executable memory)
 *
 * The ISA targets the operations Lila actually needs:
 *   matmul, matvec, rmsnorm, softmax, silu, rope, dequant
 *
 * This is NOT a general-purpose VM. It's a tensor compute VM.
 * Think of it as a portable GPU shader language for CPUs.
 */

#include <stdint.h>
#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  VM CONFIGURATION                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define VM_N_REGS       32      /* 32 scalar registers (float) */
#define VM_N_VREGS      16      /* 16 vector registers (256 floats each) */
#define VM_VREG_WIDTH   256     /* Floats per vector register */
#define VM_MAX_STACK    4096    /* Stack depth (floats) */
#define VM_MAX_CODE     65536   /* Max bytecode per kernel */

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  INSTRUCTION SET (LilaVM ISA v1)                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Instruction encoding: 32-bit fixed width
 *
 * Format A (reg-reg):     [opcode:8][dst:5][src1:5][src2:5][flags:9]
 * Format B (reg-imm):     [opcode:8][dst:5][imm:19]
 * Format C (memory):      [opcode:8][reg:5][base:5][offset:14]
 * Format D (vector):      [opcode:8][vdst:4][vsrc1:4][vsrc2:4][count:12]
 * Format E (control):     [opcode:8][cond:4][target:20]
 */

typedef enum {
    /* ── Scalar arithmetic ── */
    OP_NOP      = 0x00,
    OP_MOV      = 0x01,     /* dst = src1 */
    OP_ADD      = 0x02,     /* dst = src1 + src2 */
    OP_SUB      = 0x03,     /* dst = src1 - src2 */
    OP_MUL      = 0x04,     /* dst = src1 * src2 */
    OP_DIV      = 0x05,     /* dst = src1 / src2 */
    OP_FMA      = 0x06,     /* dst = src1 * src2 + dst (fused) */
    OP_SQRT     = 0x07,     /* dst = sqrt(src1) */
    OP_RSQRT    = 0x08,     /* dst = 1/sqrt(src1) */
    OP_EXP      = 0x09,     /* dst = exp(src1) */
    OP_NEG      = 0x0A,     /* dst = -src1 */
    OP_ABS      = 0x0B,     /* dst = |src1| */
    OP_MAX      = 0x0C,     /* dst = max(src1, src2) */
    OP_MIN      = 0x0D,     /* dst = min(src1, src2) */
    
    /* ── Immediate loads ── */
    OP_LOADI    = 0x10,     /* dst = immediate (19-bit float16 subset) */
    OP_LOADZERO = 0x11,     /* dst = 0.0 */
    OP_LOADONE  = 0x12,     /* dst = 1.0 */
    
    /* ── Memory (scalar) ── */
    OP_LOAD     = 0x20,     /* reg = mem[base + offset] */
    OP_STORE    = 0x21,     /* mem[base + offset] = reg */
    OP_LOAD_IDX = 0x22,     /* reg = mem[base + src1*4] (indexed) */
    OP_STORE_IDX= 0x23,     /* mem[base + src1*4] = reg */
    
    /* ── Vector operations (THE hot path) ── */
    OP_VLOAD    = 0x30,     /* vreg = mem[base:base+count] */
    OP_VSTORE   = 0x31,     /* mem[base:base+count] = vreg */
    OP_VADD     = 0x32,     /* vdst = vsrc1 + vsrc2 (element-wise) */
    OP_VMUL     = 0x33,     /* vdst = vsrc1 * vsrc2 (element-wise) */
    OP_VFMA     = 0x34,     /* vdst += vsrc1 * vsrc2 (fused multiply-add) */
    OP_VSCALE   = 0x35,     /* vdst = vsrc1 * scalar (broadcast) */
    OP_VREDUCE  = 0x36,     /* scalar = sum(vreg) (horizontal reduction) */
    OP_VMAX     = 0x37,     /* vdst = max(vsrc1, vsrc2) */
    OP_VSQRT    = 0x38,     /* vdst = sqrt(vsrc1) */
    OP_VRSQRT   = 0x39,     /* vdst = 1/sqrt(vsrc1) */
    OP_VEXP     = 0x3A,     /* vdst = exp(vsrc1) */
    OP_VSILU    = 0x3B,     /* vdst = vsrc1 / (1 + exp(-vsrc1)) */
    OP_VZERO    = 0x3C,     /* vdst = 0 */
    OP_VBCAST   = 0x3D,     /* vdst = broadcast(scalar) */
    OP_VREDUCE_MAX = 0x3E,  /* scalar = max(vreg) */
    
    /* ── Quantization ops ── */
    OP_DEQUANT4 = 0x40,     /* vdst = dequant_int4(indices, codebook, scale) */
    OP_DEQUANT_FMA = 0x41,  /* vdst += dequant_int4(indices, codebook, scale) * vsrc */
    
    /* ── Control flow ── */
    OP_JMP      = 0x50,     /* unconditional jump */
    OP_JLT      = 0x51,     /* jump if reg < 0 */
    OP_JGE      = 0x52,     /* jump if reg >= 0 */
    OP_JEQ      = 0x53,     /* jump if reg == 0 */
    OP_LOOP     = 0x54,     /* decrement counter, jump if > 0 */
    OP_CALL     = 0x55,     /* call kernel subroutine */
    OP_RET      = 0x56,     /* return from kernel */
    
    /* ── Special ── */
    OP_SETARG   = 0x60,     /* Load kernel argument (from caller) */
    OP_GETDIM   = 0x61,     /* Get dimension argument (rows, cols, etc.) */
    OP_SYNC     = 0x62,     /* Memory barrier / sync point */
    OP_HALT     = 0xFF,     /* End of kernel */
} LilaVMOpcode;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  INSTRUCTION ENCODING HELPERS                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Pack Format A instruction */
static inline uint32_t vm_encode_A(uint8_t op, uint8_t dst, uint8_t src1, uint8_t src2, uint16_t flags) {
    return ((uint32_t)op << 24) | ((uint32_t)(dst & 0x1F) << 19) |
           ((uint32_t)(src1 & 0x1F) << 14) | ((uint32_t)(src2 & 0x1F) << 9) |
           (flags & 0x1FF);
}

/* Pack Format B instruction */
static inline uint32_t vm_encode_B(uint8_t op, uint8_t dst, uint32_t imm) {
    return ((uint32_t)op << 24) | ((uint32_t)(dst & 0x1F) << 19) | (imm & 0x7FFFF);
}

/* Pack Format D (vector) instruction */
static inline uint32_t vm_encode_D(uint8_t op, uint8_t vdst, uint8_t vsrc1, uint8_t vsrc2, uint16_t count) {
    return ((uint32_t)op << 24) | ((uint32_t)(vdst & 0xF) << 20) |
           ((uint32_t)(vsrc1 & 0xF) << 16) | ((uint32_t)(vsrc2 & 0xF) << 12) |
           (count & 0xFFF);
}

/* Pack Format E (control) instruction */
static inline uint32_t vm_encode_E(uint8_t op, uint8_t cond, uint32_t target) {
    return ((uint32_t)op << 24) | ((uint32_t)(cond & 0xF) << 20) | (target & 0xFFFFF);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  VM STATE                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Scalar registers */
    float regs[VM_N_REGS];
    
    /* Vector registers — THE workhorse */
    float vregs[VM_N_VREGS][VM_VREG_WIDTH];
    
    /* Program counter */
    uint32_t pc;
    
    /* Stack (for CALL/RET) */
    uint32_t call_stack[64];
    int sp;
    
    /* Kernel arguments (set by caller before execution) */
    float *args[8];         /* Pointer arguments (buffers) */
    int    dims[8];         /* Dimension arguments (sizes) */
    
    /* Status */
    int halted;
    int error;
    
    /* Statistics */
    uint64_t instructions_executed;
    uint64_t flops;
} LilaVM;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  VM API                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Initialize VM state */
void vm_init(LilaVM *vm);

/* Execute bytecode kernel */
int vm_execute(LilaVM *vm, const uint32_t *code, int code_len);

/* Set kernel arguments before execution */
void vm_set_arg(LilaVM *vm, int idx, float *ptr);
void vm_set_dim(LilaVM *vm, int idx, int value);

/* Reset VM state between kernel calls */
void vm_reset(LilaVM *vm);

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PRE-BUILT KERNEL BYTECODE                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * These functions return bytecode for standard kernels.
 * Used by the .asi packer to embed portable kernels.
 */

/* Get bytecode for matrix-vector multiply */
int vm_kernel_matvec(uint32_t *code_out, int max_len);

/* Get bytecode for RMSNorm */
int vm_kernel_rmsnorm(uint32_t *code_out, int max_len);

/* Get bytecode for softmax */
int vm_kernel_softmax(uint32_t *code_out, int max_len);

/* Get bytecode for SiLU activation */
int vm_kernel_silu(uint32_t *code_out, int max_len);

/* Get bytecode for RoPE */
int vm_kernel_rope(uint32_t *code_out, int max_len);

/* Get bytecode for INT4 dequant + matvec (fused) */
int vm_kernel_dequant_matvec(uint32_t *code_out, int max_len);

#endif /* LILA_VM_H */
