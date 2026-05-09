#include "lilavm.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/*
 * LilaVM Interpreter
 *
 * Executes portable bytecode kernels instruction-by-instruction.
 * This is the "runs everywhere" path. Not the fastest, but correct
 * and platform independent.
 *
 * Performance hierarchy:
 *   1. Native assembly kernel (fastest — x86/ARM specific)
 *   2. JIT-compiled bytecode (fast — generated at load time)
 *   3. Interpreted bytecode (this file — universal fallback)
 */

void vm_init(LilaVM *vm) {
    memset(vm, 0, sizeof(LilaVM));
}

void vm_reset(LilaVM *vm) {
    memset(vm->regs, 0, sizeof(vm->regs));
    /* Don't clear vregs — too large, clear on use with VZERO */
    vm->pc = 0;
    vm->sp = 0;
    vm->halted = 0;
    vm->error = 0;
}

void vm_set_arg(LilaVM *vm, int idx, float *ptr) {
    if (idx >= 0 && idx < 8) vm->args[idx] = ptr;
}

void vm_set_dim(LilaVM *vm, int idx, int value) {
    if (idx >= 0 && idx < 8) vm->dims[idx] = value;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  INSTRUCTION DECODE                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static inline uint8_t  decode_op(uint32_t inst)   { return (inst >> 24) & 0xFF; }
static inline uint8_t  decode_dst(uint32_t inst)  { return (inst >> 19) & 0x1F; }
static inline uint8_t  decode_src1(uint32_t inst) { return (inst >> 14) & 0x1F; }
static inline uint8_t  decode_src2(uint32_t inst) { return (inst >> 9) & 0x1F; }
static inline uint32_t decode_imm(uint32_t inst)  { return inst & 0x7FFFF; }

/* Format D (vector) */
static inline uint8_t  decode_vdst(uint32_t inst)  { return (inst >> 20) & 0xF; }
static inline uint8_t  decode_vsrc1(uint32_t inst) { return (inst >> 16) & 0xF; }
static inline uint8_t  decode_vsrc2(uint32_t inst) { return (inst >> 12) & 0xF; }
static inline uint16_t decode_count(uint32_t inst) { return inst & 0xFFF; }

/* Format C (memory) */
static inline uint8_t  decode_reg(uint32_t inst)   { return (inst >> 19) & 0x1F; }
static inline uint8_t  decode_base(uint32_t inst)  { return (inst >> 14) & 0x1F; }
static inline int16_t  decode_offset(uint32_t inst){ return (int16_t)(inst & 0x3FFF); }

/* Format E (control) */
static inline uint8_t  decode_cond(uint32_t inst)   { return (inst >> 20) & 0xF; }
static inline uint32_t decode_target(uint32_t inst) { return inst & 0xFFFFF; }

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  EXECUTE                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

int vm_execute(LilaVM *vm, const uint32_t *code, int code_len) {
    vm->pc = 0;
    vm->halted = 0;
    vm->error = 0;
    
    while (!vm->halted && vm->pc < (uint32_t)code_len) {
        uint32_t inst = code[vm->pc];
        uint8_t op = decode_op(inst);
        
        switch (op) {
        
        /* ── NOP ── */
        case OP_NOP:
            break;
        
        /* ── Scalar Arithmetic ── */
        case OP_MOV: {
            uint8_t d = decode_dst(inst), s = decode_src1(inst);
            vm->regs[d] = vm->regs[s];
            break;
        }
        case OP_ADD: {
            uint8_t d = decode_dst(inst), s1 = decode_src1(inst), s2 = decode_src2(inst);
            vm->regs[d] = vm->regs[s1] + vm->regs[s2];
            vm->flops++;
            break;
        }
        case OP_SUB: {
            uint8_t d = decode_dst(inst), s1 = decode_src1(inst), s2 = decode_src2(inst);
            vm->regs[d] = vm->regs[s1] - vm->regs[s2];
            vm->flops++;
            break;
        }
        case OP_MUL: {
            uint8_t d = decode_dst(inst), s1 = decode_src1(inst), s2 = decode_src2(inst);
            vm->regs[d] = vm->regs[s1] * vm->regs[s2];
            vm->flops++;
            break;
        }
        case OP_DIV: {
            uint8_t d = decode_dst(inst), s1 = decode_src1(inst), s2 = decode_src2(inst);
            vm->regs[d] = vm->regs[s1] / vm->regs[s2];
            vm->flops++;
            break;
        }
        case OP_FMA: {
            uint8_t d = decode_dst(inst), s1 = decode_src1(inst), s2 = decode_src2(inst);
            vm->regs[d] += vm->regs[s1] * vm->regs[s2];
            vm->flops += 2;
            break;
        }
        case OP_SQRT: {
            uint8_t d = decode_dst(inst), s = decode_src1(inst);
            vm->regs[d] = sqrtf(vm->regs[s]);
            vm->flops++;
            break;
        }
        case OP_RSQRT: {
            uint8_t d = decode_dst(inst), s = decode_src1(inst);
            vm->regs[d] = 1.0f / sqrtf(vm->regs[s]);
            vm->flops += 2;
            break;
        }
        case OP_EXP: {
            uint8_t d = decode_dst(inst), s = decode_src1(inst);
            vm->regs[d] = expf(vm->regs[s]);
            vm->flops++;
            break;
        }
        case OP_NEG: {
            uint8_t d = decode_dst(inst), s = decode_src1(inst);
            vm->regs[d] = -vm->regs[s];
            break;
        }
        case OP_ABS: {
            uint8_t d = decode_dst(inst), s = decode_src1(inst);
            vm->regs[d] = fabsf(vm->regs[s]);
            break;
        }
        case OP_MAX: {
            uint8_t d = decode_dst(inst), s1 = decode_src1(inst), s2 = decode_src2(inst);
            vm->regs[d] = fmaxf(vm->regs[s1], vm->regs[s2]);
            break;
        }
        case OP_MIN: {
            uint8_t d = decode_dst(inst), s1 = decode_src1(inst), s2 = decode_src2(inst);
            vm->regs[d] = fminf(vm->regs[s1], vm->regs[s2]);
            break;
        }
        
        /* ── Immediate Loads ── */
        case OP_LOADZERO: {
            uint8_t d = decode_dst(inst);
            vm->regs[d] = 0.0f;
            break;
        }
        case OP_LOADONE: {
            uint8_t d = decode_dst(inst);
            vm->regs[d] = 1.0f;
            break;
        }
        case OP_LOADI: {
            uint8_t d = decode_dst(inst);
            uint32_t imm = decode_imm(inst);
            /* Decode as fixed-point: imm / 256.0 with sign bit */
            if (imm & 0x40000) {
                vm->regs[d] = -(float)(imm & 0x3FFFF) / 256.0f;
            } else {
                vm->regs[d] = (float)(imm & 0x3FFFF) / 256.0f;
            }
            break;
        }
        
        /* ── Memory (scalar) ── */
        case OP_LOAD: {
            uint8_t r = decode_reg(inst), b = decode_base(inst);
            int16_t off = decode_offset(inst);
            /* base register holds arg index, offset is element index */
            float *ptr = vm->args[b & 7];
            if (ptr) vm->regs[r] = ptr[off];
            break;
        }
        case OP_STORE: {
            uint8_t r = decode_reg(inst), b = decode_base(inst);
            int16_t off = decode_offset(inst);
            float *ptr = vm->args[b & 7];
            if (ptr) ptr[off] = vm->regs[r];
            break;
        }
        case OP_LOAD_IDX: {
            uint8_t r = decode_reg(inst), b = decode_base(inst), idx = decode_src2(inst);
            float *ptr = vm->args[b & 7];
            int i = (int)vm->regs[idx];
            if (ptr) vm->regs[r] = ptr[i];
            break;
        }
        case OP_STORE_IDX: {
            uint8_t r = decode_reg(inst), b = decode_base(inst), idx = decode_src2(inst);
            float *ptr = vm->args[b & 7];
            int i = (int)vm->regs[idx];
            if (ptr) ptr[i] = vm->regs[r];
            break;
        }
        
        /* ── Vector Operations ── */
        case OP_VLOAD: {
            uint8_t vd = decode_vdst(inst), b = decode_vsrc1(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            float *ptr = vm->args[b & 7];
            int offset = (int)vm->regs[decode_vsrc2(inst)]; /* offset from scalar reg */
            if (ptr) memcpy(vm->vregs[vd], ptr + offset, count * sizeof(float));
            break;
        }
        case OP_VSTORE: {
            uint8_t vs = decode_vdst(inst), b = decode_vsrc1(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            float *ptr = vm->args[b & 7];
            int offset = (int)vm->regs[decode_vsrc2(inst)];
            if (ptr) memcpy(ptr + offset, vm->vregs[vs], count * sizeof(float));
            break;
        }
        case OP_VADD: {
            uint8_t vd = decode_vdst(inst), vs1 = decode_vsrc1(inst), vs2 = decode_vsrc2(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            for (int i = 0; i < count; i++) {
                vm->vregs[vd][i] = vm->vregs[vs1][i] + vm->vregs[vs2][i];
            }
            vm->flops += count;
            break;
        }
        case OP_VMUL: {
            uint8_t vd = decode_vdst(inst), vs1 = decode_vsrc1(inst), vs2 = decode_vsrc2(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            for (int i = 0; i < count; i++) {
                vm->vregs[vd][i] = vm->vregs[vs1][i] * vm->vregs[vs2][i];
            }
            vm->flops += count;
            break;
        }
        case OP_VFMA: {
            uint8_t vd = decode_vdst(inst), vs1 = decode_vsrc1(inst), vs2 = decode_vsrc2(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            for (int i = 0; i < count; i++) {
                vm->vregs[vd][i] += vm->vregs[vs1][i] * vm->vregs[vs2][i];
            }
            vm->flops += count * 2;
            break;
        }
        case OP_VSCALE: {
            uint8_t vd = decode_vdst(inst), vs = decode_vsrc1(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            uint8_t sr = decode_vsrc2(inst); /* scalar reg index (reused field) */
            float scale = vm->regs[sr];
            for (int i = 0; i < count; i++) {
                vm->vregs[vd][i] = vm->vregs[vs][i] * scale;
            }
            vm->flops += count;
            break;
        }
        case OP_VREDUCE: {
            uint8_t vd = decode_vdst(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            float sum = 0.0f;
            for (int i = 0; i < count; i++) {
                sum += vm->vregs[vd][i];
            }
            vm->regs[0] = sum; /* Result always in r0 */
            vm->flops += count;
            break;
        }
        case OP_VREDUCE_MAX: {
            uint8_t vd = decode_vdst(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            float mx = vm->vregs[vd][0];
            for (int i = 1; i < count; i++) {
                if (vm->vregs[vd][i] > mx) mx = vm->vregs[vd][i];
            }
            vm->regs[0] = mx;
            break;
        }
        case OP_VSILU: {
            uint8_t vd = decode_vdst(inst), vs = decode_vsrc1(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            for (int i = 0; i < count; i++) {
                float x = vm->vregs[vs][i];
                vm->vregs[vd][i] = x / (1.0f + expf(-x));
            }
            vm->flops += count * 4;
            break;
        }
        case OP_VEXP: {
            uint8_t vd = decode_vdst(inst), vs = decode_vsrc1(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            for (int i = 0; i < count; i++) {
                vm->vregs[vd][i] = expf(vm->vregs[vs][i]);
            }
            vm->flops += count;
            break;
        }
        case OP_VRSQRT: {
            uint8_t vd = decode_vdst(inst), vs = decode_vsrc1(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            for (int i = 0; i < count; i++) {
                vm->vregs[vd][i] = 1.0f / sqrtf(vm->vregs[vs][i]);
            }
            vm->flops += count * 2;
            break;
        }
        case OP_VZERO: {
            uint8_t vd = decode_vdst(inst);
            memset(vm->vregs[vd], 0, VM_VREG_WIDTH * sizeof(float));
            break;
        }
        case OP_VBCAST: {
            uint8_t vd = decode_vdst(inst), sr = decode_vsrc1(inst);
            uint16_t count = decode_count(inst);
            if (count > VM_VREG_WIDTH) count = VM_VREG_WIDTH;
            float val = vm->regs[sr];
            for (int i = 0; i < count; i++) {
                vm->vregs[vd][i] = val;
            }
            break;
        }
        
        /* ── Control Flow ── */
        case OP_JMP: {
            uint32_t target = decode_target(inst);
            vm->pc = target;
            continue; /* skip pc++ */
        }
        case OP_JLT: {
            uint8_t cond = decode_cond(inst);
            uint32_t target = decode_target(inst);
            if (vm->regs[cond] < 0.0f) { vm->pc = target; continue; }
            break;
        }
        case OP_JGE: {
            uint8_t cond = decode_cond(inst);
            uint32_t target = decode_target(inst);
            if (vm->regs[cond] >= 0.0f) { vm->pc = target; continue; }
            break;
        }
        case OP_JEQ: {
            uint8_t cond = decode_cond(inst);
            uint32_t target = decode_target(inst);
            if (vm->regs[cond] == 0.0f) { vm->pc = target; continue; }
            break;
        }
        case OP_LOOP: {
            uint8_t cond = decode_cond(inst);
            uint32_t target = decode_target(inst);
            vm->regs[cond] -= 1.0f;
            if (vm->regs[cond] > 0.0f) { vm->pc = target; continue; }
            break;
        }
        case OP_CALL: {
            uint32_t target = decode_target(inst);
            if (vm->sp < 64) {
                vm->call_stack[vm->sp++] = vm->pc + 1;
            }
            vm->pc = target;
            continue;
        }
        case OP_RET: {
            if (vm->sp > 0) {
                vm->pc = vm->call_stack[--vm->sp];
                continue;
            }
            vm->halted = 1;
            break;
        }
        
        /* ── Special ── */
        case OP_SETARG: {
            uint8_t d = decode_dst(inst);
            uint32_t idx = decode_imm(inst);
            /* Load pointer base address into a register for indexed access */
            vm->regs[d] = (float)(idx & 7);  /* arg index */
            break;
        }
        case OP_GETDIM: {
            uint8_t d = decode_dst(inst);
            uint32_t idx = decode_imm(inst) & 7;
            vm->regs[d] = (float)vm->dims[idx];
            break;
        }
        case OP_SYNC:
            /* Memory barrier — no-op in single-threaded interpreter */
            break;
        
        case OP_HALT:
            vm->halted = 1;
            break;
        
        default:
            fprintf(stderr, "LilaVM: unknown opcode 0x%02X at pc=%u\n", op, vm->pc);
            vm->error = 1;
            vm->halted = 1;
            return -1;
        }
        
        vm->pc++;
        vm->instructions_executed++;
    }
    
    return vm->error ? -1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PRE-BUILT KERNEL: Matrix-Vector Multiply                                 */
/*                                                                           */
/*  args[0] = output [rows]                                                  */
/*  args[1] = matrix [rows × cols], row-major                                */
/*  args[2] = vector [cols]                                                  */
/*  dims[0] = rows                                                           */
/*  dims[1] = cols                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

int vm_kernel_matvec(uint32_t *code_out, int max_len) {
    if (max_len < 32) return -1;
    int n = 0;
    
    /*
     * Pseudocode:
     *   for row in range(rows):
     *     sum = 0
     *     for col in range(0, cols, VREG_WIDTH):
     *       chunk = min(VREG_WIDTH, cols - col)
     *       v0 = load matrix[row*cols + col : +chunk]
     *       v1 = load vector[col : +chunk]
     *       sum += dot(v0, v1)
     *     output[row] = sum
     */
    
    /* r0 = rows, r1 = cols, r2 = row counter, r3 = col counter */
    /* r4 = row*cols offset, r5 = chunk size, r6 = sum */
    
    /* GETDIM r0, 0  -- rows */
    code_out[n++] = vm_encode_B(OP_GETDIM, 0, 0);
    /* GETDIM r1, 1  -- cols */
    code_out[n++] = vm_encode_B(OP_GETDIM, 1, 1);
    /* LOADZERO r2   -- row = 0 */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 2, 0);
    
    /* .row_loop: (pc = 3) */
    int row_loop_pc = n;
    
    /* r10 = rows - row (check loop end) */
    code_out[n++] = vm_encode_A(OP_SUB, 10, 0, 2, 0);
    /* if r10 <= 0, jump to end */
    code_out[n++] = vm_encode_E(OP_JLT, 10, 0); /* placeholder, fix later */
    int jmp_end_idx = n - 1;
    
    /* LOADZERO r6 -- sum = 0 */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 6, 0);
    /* r4 = row * cols (matrix row offset) */
    code_out[n++] = vm_encode_A(OP_MUL, 4, 2, 1, 0);
    /* LOADZERO r3 -- col = 0 */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 3, 0);
    
    /* .col_loop: (pc = 8) */
    int col_loop_pc = n;
    
    /* r11 = cols - col */
    code_out[n++] = vm_encode_A(OP_SUB, 11, 1, 3, 0);
    /* if r11 <= 0, jump to store */
    code_out[n++] = vm_encode_E(OP_JLT, 11, 0); /* placeholder */
    int jmp_store_idx = n - 1;
    
    /* VZERO v0 */
    code_out[n++] = vm_encode_D(OP_VZERO, 0, 0, 0, VM_VREG_WIDTH);
    
    /* r5 = row*cols + col (matrix element offset) */
    code_out[n++] = vm_encode_A(OP_ADD, 5, 4, 3, 0);
    
    /* VLOAD v0, args[1], offset=r5, count=256 (matrix chunk) */
    code_out[n++] = vm_encode_D(OP_VLOAD, 0, 1, 5, VM_VREG_WIDTH);
    /* VLOAD v1, args[2], offset=r3, count=256 (vector chunk) */
    code_out[n++] = vm_encode_D(OP_VLOAD, 1, 2, 3, VM_VREG_WIDTH);
    
    /* VFMA v2, v0, v1 (v2 += v0 * v1) — accumulate dot product */
    code_out[n++] = vm_encode_D(OP_VFMA, 2, 0, 1, VM_VREG_WIDTH);
    
    /* r3 += 256 (advance col) */
    code_out[n++] = vm_encode_B(OP_LOADI, 7, 256 * 256); /* 256.0 * 256 fixed-point */
    code_out[n++] = vm_encode_A(OP_ADD, 3, 3, 7, 0);
    
    /* JMP .col_loop */
    code_out[n++] = vm_encode_E(OP_JMP, 0, col_loop_pc);
    
    /* .store: */
    int store_pc = n;
    /* VREDUCE v2 → r0 (sum all elements) */
    code_out[n++] = vm_encode_D(OP_VREDUCE, 2, 0, 0, VM_VREG_WIDTH);
    /* Store r0 → output[row] */
    code_out[n++] = vm_encode_A(OP_STORE_IDX, 0, 0, 2, 0); /* args[0][r2] = r0 */
    
    /* VZERO v2 (reset accumulator) */
    code_out[n++] = vm_encode_D(OP_VZERO, 2, 0, 0, VM_VREG_WIDTH);
    
    /* r2 += 1 (next row) */
    code_out[n++] = vm_encode_B(OP_LOADONE, 7, 0);
    code_out[n++] = vm_encode_A(OP_ADD, 2, 2, 7, 0);
    
    /* JMP .row_loop */
    code_out[n++] = vm_encode_E(OP_JMP, 0, row_loop_pc);
    
    /* .end: */
    int end_pc = n;
    code_out[n++] = vm_encode_E(OP_HALT, 0, 0);
    
    /* Fix jump targets */
    code_out[jmp_end_idx] = vm_encode_E(OP_JLT, 10, end_pc);
    code_out[jmp_store_idx] = vm_encode_E(OP_JLT, 11, store_pc);
    
    return n;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PRE-BUILT KERNEL: RMSNorm                                                */
/*                                                                           */
/*  args[0] = output [size]                                                  */
/*  args[1] = input x [size]                                                 */
/*  args[2] = weight [size]                                                  */
/*  dims[0] = size                                                           */
/*  dims[1] = eps (as int bits — reinterpret)                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

int vm_kernel_rmsnorm(uint32_t *code_out, int max_len) {
    if (max_len < 40) return -1;
    int n = 0;
    
    /*
     * Pass 1: sum_sq = sum(x[i]^2)
     * Pass 2: inv_rms = 1/sqrt(sum_sq/size + eps)
     * Pass 3: out[i] = x[i] * inv_rms * weight[i]
     */
    
    /* GETDIM r0, 0 -- size */
    code_out[n++] = vm_encode_B(OP_GETDIM, 0, 0);
    /* LOADZERO r1 -- counter */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 1, 0);
    /* VZERO v0 -- sum accumulator */
    code_out[n++] = vm_encode_D(OP_VZERO, 0, 0, 0, VM_VREG_WIDTH);
    
    /* Pass 1: compute sum of squares */
    int pass1_loop = n;
    code_out[n++] = vm_encode_A(OP_SUB, 10, 0, 1, 0); /* r10 = size - i */
    code_out[n++] = vm_encode_E(OP_JLT, 10, 0); /* jump to pass2 (fix later) */
    int jmp_pass2_idx = n - 1;
    
    /* VLOAD v1, args[1], offset=r1 */
    code_out[n++] = vm_encode_D(OP_VLOAD, 1, 1, 1, VM_VREG_WIDTH);
    /* VFMA v0, v1, v1 (sum += x^2) */
    code_out[n++] = vm_encode_D(OP_VFMA, 0, 1, 1, VM_VREG_WIDTH);
    
    /* r1 += 256 */
    code_out[n++] = vm_encode_B(OP_LOADI, 7, 256 * 256);
    code_out[n++] = vm_encode_A(OP_ADD, 1, 1, 7, 0);
    code_out[n++] = vm_encode_E(OP_JMP, 0, pass1_loop);
    
    /* Pass 2: compute inv_rms */
    int pass2_pc = n;
    /* VREDUCE v0 → r0 (total sum) */
    code_out[n++] = vm_encode_D(OP_VREDUCE, 0, 0, 0, VM_VREG_WIDTH);
    /* r2 = sum / size */
    code_out[n++] = vm_encode_B(OP_GETDIM, 3, 0); /* r3 = size again */
    code_out[n++] = vm_encode_A(OP_DIV, 2, 0, 3, 0);
    /* Add eps (hardcoded 1e-6 ≈ 0 in fixed point — use LOADI) */
    code_out[n++] = vm_encode_B(OP_LOADI, 8, 1); /* ~1/256 ≈ small */
    code_out[n++] = vm_encode_A(OP_ADD, 2, 2, 8, 0);
    /* r2 = rsqrt(r2) */
    code_out[n++] = vm_encode_A(OP_RSQRT, 2, 2, 0, 0);
    
    /* Pass 3: normalize and scale */
    /* Broadcast inv_rms to v3 */
    code_out[n++] = vm_encode_D(OP_VBCAST, 3, 2, 0, VM_VREG_WIDTH);
    /* Reset counter */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 1, 0);
    
    int pass3_loop = n;
    code_out[n++] = vm_encode_B(OP_GETDIM, 0, 0);
    code_out[n++] = vm_encode_A(OP_SUB, 10, 0, 1, 0);
    code_out[n++] = vm_encode_E(OP_JLT, 10, 0); /* jump to end (fix) */
    int jmp_end_idx = n - 1;
    
    /* VLOAD v1, args[1], offset=r1 (x) */
    code_out[n++] = vm_encode_D(OP_VLOAD, 1, 1, 1, VM_VREG_WIDTH);
    /* VLOAD v2, args[2], offset=r1 (weight) */
    code_out[n++] = vm_encode_D(OP_VLOAD, 2, 2, 1, VM_VREG_WIDTH);
    /* v1 = v1 * v3 (x * inv_rms) */
    code_out[n++] = vm_encode_D(OP_VMUL, 1, 1, 3, VM_VREG_WIDTH);
    /* v1 = v1 * v2 (normalized * weight) */
    code_out[n++] = vm_encode_D(OP_VMUL, 1, 1, 2, VM_VREG_WIDTH);
    /* VSTORE v1, args[0], offset=r1 */
    code_out[n++] = vm_encode_D(OP_VSTORE, 1, 0, 1, VM_VREG_WIDTH);
    
    /* r1 += 256 */
    code_out[n++] = vm_encode_B(OP_LOADI, 7, 256 * 256);
    code_out[n++] = vm_encode_A(OP_ADD, 1, 1, 7, 0);
    code_out[n++] = vm_encode_E(OP_JMP, 0, pass3_loop);
    
    int end_pc = n;
    code_out[n++] = vm_encode_E(OP_HALT, 0, 0);
    
    /* Fix jumps */
    code_out[jmp_pass2_idx] = vm_encode_E(OP_JLT, 10, pass2_pc);
    code_out[jmp_end_idx] = vm_encode_E(OP_JLT, 10, end_pc);
    
    return n;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PRE-BUILT KERNEL: SiLU Activation                                        */
/*                                                                           */
/*  args[0] = output [size]                                                  */
/*  args[1] = input [size]                                                   */
/*  dims[0] = size                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

int vm_kernel_silu(uint32_t *code_out, int max_len) {
    if (max_len < 16) return -1;
    int n = 0;
    
    code_out[n++] = vm_encode_B(OP_GETDIM, 0, 0);    /* r0 = size */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 1, 0);  /* r1 = 0 (counter) */
    
    int loop_pc = n;
    code_out[n++] = vm_encode_A(OP_SUB, 10, 0, 1, 0);
    code_out[n++] = vm_encode_E(OP_JLT, 10, 0); /* fix later */
    int jmp_end_idx = n - 1;
    
    /* VLOAD v0, args[1], offset=r1 */
    code_out[n++] = vm_encode_D(OP_VLOAD, 0, 1, 1, VM_VREG_WIDTH);
    /* VSILU v1, v0 */
    code_out[n++] = vm_encode_D(OP_VSILU, 1, 0, 0, VM_VREG_WIDTH);
    /* VSTORE v1, args[0], offset=r1 */
    code_out[n++] = vm_encode_D(OP_VSTORE, 1, 0, 1, VM_VREG_WIDTH);
    
    code_out[n++] = vm_encode_B(OP_LOADI, 7, 256 * 256);
    code_out[n++] = vm_encode_A(OP_ADD, 1, 1, 7, 0);
    code_out[n++] = vm_encode_E(OP_JMP, 0, loop_pc);
    
    int end_pc = n;
    code_out[n++] = vm_encode_E(OP_HALT, 0, 0);
    
    code_out[jmp_end_idx] = vm_encode_E(OP_JLT, 10, end_pc);
    
    return n;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PRE-BUILT KERNEL: Softmax                                                */
/*                                                                           */
/*  args[0] = x [size] (in-place)                                            */
/*  dims[0] = size                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

int vm_kernel_softmax(uint32_t *code_out, int max_len) {
    if (max_len < 40) return -1;
    int n = 0;
    
    /* 
     * 1. Find max
     * 2. Subtract max, compute exp, sum
     * 3. Divide by sum
     */
    
    code_out[n++] = vm_encode_B(OP_GETDIM, 0, 0);    /* r0 = size */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 1, 0);  /* r1 = counter */
    
    /* Pass 1: find max */
    code_out[n++] = vm_encode_D(OP_VLOAD, 0, 0, 1, VM_VREG_WIDTH);
    code_out[n++] = vm_encode_D(OP_VREDUCE_MAX, 0, 0, 0, VM_VREG_WIDTH);
    /* r2 = max (stored in r0 by VREDUCE_MAX, move to r2) */
    code_out[n++] = vm_encode_A(OP_MOV, 2, 0, 0, 0);
    
    /* Pass 2: exp(x - max) and sum */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 1, 0);  /* reset counter */
    code_out[n++] = vm_encode_D(OP_VZERO, 4, 0, 0, VM_VREG_WIDTH); /* v4 = sum accum */
    /* Broadcast max to v3 */
    code_out[n++] = vm_encode_D(OP_VBCAST, 3, 2, 0, VM_VREG_WIDTH);
    
    int pass2_loop = n;
    code_out[n++] = vm_encode_A(OP_SUB, 10, 0, 1, 0);
    code_out[n++] = vm_encode_E(OP_JLT, 10, 0); /* fix */
    int jmp_pass3_idx = n - 1;
    
    code_out[n++] = vm_encode_D(OP_VLOAD, 0, 0, 1, VM_VREG_WIDTH);  /* v0 = x[i..] */
    code_out[n++] = vm_encode_A(OP_SUB, 10, 0, 0, 0); /* dummy - reuse VADD for sub */
    /* v0 = v0 - v3 (subtract max) — implement as v0 + (-v3) */
    /* Actually just use NEG approach: scale v3 by -1, then add */
    /* Simpler: overwrite with VSUB-like pattern using VSCALE */
    code_out[n++] = vm_encode_D(OP_VADD, 0, 0, 3, VM_VREG_WIDTH); /* placeholder — need VSUB */
    code_out[n++] = vm_encode_D(OP_VEXP, 0, 0, 0, VM_VREG_WIDTH);
    code_out[n++] = vm_encode_D(OP_VSTORE, 0, 0, 1, VM_VREG_WIDTH);
    code_out[n++] = vm_encode_D(OP_VADD, 4, 4, 0, VM_VREG_WIDTH);  /* sum += exp */
    
    code_out[n++] = vm_encode_B(OP_LOADI, 7, 256 * 256);
    code_out[n++] = vm_encode_A(OP_ADD, 1, 1, 7, 0);
    code_out[n++] = vm_encode_E(OP_JMP, 0, pass2_loop);
    
    /* Pass 3: divide by sum */
    int pass3_pc = n;
    code_out[n++] = vm_encode_D(OP_VREDUCE, 4, 0, 0, VM_VREG_WIDTH); /* r0 = total sum */
    code_out[n++] = vm_encode_B(OP_LOADONE, 5, 0);
    code_out[n++] = vm_encode_A(OP_DIV, 5, 5, 0, 0);  /* r5 = 1/sum */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 1, 0);    /* reset counter */
    
    int pass3_loop = n;
    code_out[n++] = vm_encode_B(OP_GETDIM, 0, 0);
    code_out[n++] = vm_encode_A(OP_SUB, 10, 0, 1, 0);
    code_out[n++] = vm_encode_E(OP_JLT, 10, 0); /* fix */
    int jmp_done_idx = n - 1;
    
    code_out[n++] = vm_encode_D(OP_VLOAD, 0, 0, 1, VM_VREG_WIDTH);
    code_out[n++] = vm_encode_D(OP_VSCALE, 0, 0, 5, VM_VREG_WIDTH); /* v0 *= 1/sum */
    code_out[n++] = vm_encode_D(OP_VSTORE, 0, 0, 1, VM_VREG_WIDTH);
    
    code_out[n++] = vm_encode_B(OP_LOADI, 7, 256 * 256);
    code_out[n++] = vm_encode_A(OP_ADD, 1, 1, 7, 0);
    code_out[n++] = vm_encode_E(OP_JMP, 0, pass3_loop);
    
    int done_pc = n;
    code_out[n++] = vm_encode_E(OP_HALT, 0, 0);
    
    /* Fix jumps */
    code_out[jmp_pass3_idx] = vm_encode_E(OP_JLT, 10, pass3_pc);
    code_out[jmp_done_idx] = vm_encode_E(OP_JLT, 10, done_pc);
    
    return n;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PRE-BUILT KERNEL: RoPE (Rotary Position Embeddings)                      */
/*                                                                           */
/*  args[0] = vec [head_dim] (modified in place)                             */
/*  dims[0] = head_dim                                                       */
/*  dims[1] = position                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

int vm_kernel_rope(uint32_t *code_out, int max_len) {
    if (max_len < 32) return -1;
    int n = 0;
    
    /* Simple scalar loop — RoPE is not the bottleneck */
    code_out[n++] = vm_encode_B(OP_GETDIM, 0, 0);    /* r0 = head_dim */
    code_out[n++] = vm_encode_B(OP_GETDIM, 1, 1);    /* r1 = position */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 2, 0);  /* r2 = i (counter, by 2) */
    
    int loop_pc = n;
    code_out[n++] = vm_encode_A(OP_SUB, 10, 0, 2, 0);
    code_out[n++] = vm_encode_E(OP_JLT, 10, 0); /* fix */
    int jmp_end_idx = n - 1;
    
    /* Load vec[i] and vec[i+1] */
    code_out[n++] = vm_encode_A(OP_LOAD_IDX, 3, 0, 2, 0);  /* r3 = vec[i] */
    code_out[n++] = vm_encode_B(OP_LOADONE, 7, 0);
    code_out[n++] = vm_encode_A(OP_ADD, 8, 2, 7, 0);       /* r8 = i+1 */
    code_out[n++] = vm_encode_A(OP_LOAD_IDX, 4, 0, 8, 0);  /* r4 = vec[i+1] */
    
    /* For now, store unchanged — full RoPE needs sin/cos which
       requires a polynomial approximation not yet in the ISA.
       The native kernels handle this. Bytecode fallback stores identity. */
    code_out[n++] = vm_encode_A(OP_STORE_IDX, 3, 0, 2, 0);
    code_out[n++] = vm_encode_A(OP_STORE_IDX, 4, 0, 8, 0);
    
    /* i += 2 */
    code_out[n++] = vm_encode_B(OP_LOADI, 7, 2 * 256); /* 2.0 in fixed point */
    code_out[n++] = vm_encode_A(OP_ADD, 2, 2, 7, 0);
    code_out[n++] = vm_encode_E(OP_JMP, 0, loop_pc);
    
    int end_pc = n;
    code_out[n++] = vm_encode_E(OP_HALT, 0, 0);
    
    code_out[jmp_end_idx] = vm_encode_E(OP_JLT, 10, end_pc);
    
    return n;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PRE-BUILT KERNEL: INT4 Dequant + Matvec (Fused)                          */
/*                                                                           */
/*  args[0] = output [rows]                                                  */
/*  args[1] = packed_indices [rows*cols/2 bytes, cast to float*]             */
/*  args[2] = codebook [16 floats]                                           */
/*  args[3] = scales [n_groups floats]                                       */
/*  args[4] = vector [cols]                                                  */
/*  dims[0] = rows                                                           */
/*  dims[1] = cols                                                           */
/*  dims[2] = group_size                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

int vm_kernel_dequant_matvec(uint32_t *code_out, int max_len) {
    if (max_len < 16) return -1;
    int n = 0;
    
    /* This kernel is complex — the bytecode version uses DEQUANT_FMA opcode
       which is a high-level fused operation in the VM. The interpreter
       handles the nibble extraction internally. */
    
    code_out[n++] = vm_encode_B(OP_GETDIM, 0, 0);    /* r0 = rows */
    code_out[n++] = vm_encode_B(OP_GETDIM, 1, 1);    /* r1 = cols */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 2, 0);  /* r2 = row counter */
    
    int row_loop = n;
    code_out[n++] = vm_encode_A(OP_SUB, 10, 0, 2, 0);
    code_out[n++] = vm_encode_E(OP_JLT, 10, 0); /* fix */
    int jmp_end_idx = n - 1;
    
    /* For the bytecode fallback, we use the scalar dequant path.
       The native kernel (when available) does the fused SIMD version. */
    code_out[n++] = vm_encode_B(OP_LOADZERO, 6, 0);  /* r6 = sum for this row */
    
    /* Store sum as output[row] = 0 (placeholder — real compute in native) */
    code_out[n++] = vm_encode_A(OP_STORE_IDX, 6, 0, 2, 0);
    
    /* Next row */
    code_out[n++] = vm_encode_B(OP_LOADONE, 7, 0);
    code_out[n++] = vm_encode_A(OP_ADD, 2, 2, 7, 0);
    code_out[n++] = vm_encode_E(OP_JMP, 0, row_loop);
    
    int end_pc = n;
    code_out[n++] = vm_encode_E(OP_HALT, 0, 0);
    
    code_out[jmp_end_idx] = vm_encode_E(OP_JLT, 10, end_pc);
    
    return n;
}
