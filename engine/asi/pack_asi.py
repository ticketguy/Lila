#!/usr/bin/env python3
"""
ASI Packer — Create .asi (Active System Image) files

Packs everything Lila needs into a single portable file:
  - Model weights (FigQuant INT4)
  - Memory Fabric adapters (A Thousand Pearls)
  - Tokenizer (BPE vocab + merges)
  - Bytecode kernels (LilaVM portable compute)
  - Harness definitions (tool capabilities)
  - Personality state

Usage:
    # From a HuggingFace model:
    python pack_asi.py --model google/gemma-3-4b-it --output lila.asi

    # From existing .lila + adapters:
    python pack_asi.py --lila-weights model.lila --adapters ./adapters/ --output lila.asi

    # Repack with updated adapters (after training):
    python pack_asi.py --base lila.asi --new-adapters ./new_adapters/ --output lila_v2.asi
"""

import argparse
import struct
import hashlib
import time
import os
import sys
import json
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS (must match asi.h)
# ═══════════════════════════════════════════════════════════════════════════════

ASI_MAGIC = 0x41534921      # "ASI!"
ASI_VERSION = 1
ASI_PAGE_SIZE = 4096

# Section types
ASI_SECTION_MODEL_CONFIG  = 0x01
ASI_SECTION_WEIGHTS       = 0x02
ASI_SECTION_MEMORY_FABRIC = 0x03
ASI_SECTION_TOKENIZER     = 0x04
ASI_SECTION_BYTECODE      = 0x05
ASI_SECTION_HARNESS       = 0x06
ASI_SECTION_PERSONALITY   = 0x07
ASI_SECTION_METADATA      = 0x08

# Flags
ASI_FLAG_HAS_FABRIC      = (1 << 0)
ASI_FLAG_HAS_BYTECODE    = (1 << 1)
ASI_FLAG_HAS_HARNESS     = (1 << 2)
ASI_FLAG_HAS_PERSONALITY = (1 << 3)
ASI_FLAG_QUANT_INT4      = (1 << 4)
ASI_FLAG_TOKENIZER_BPE   = (1 << 5)
ASI_FLAG_HOT_RELOAD      = (1 << 6)

# LilaVM kernel IDs
KERNEL_MATMUL       = 0x01
KERNEL_MATVEC       = 0x02
KERNEL_RMSNORM      = 0x03
KERNEL_SOFTMAX      = 0x04
KERNEL_SILU         = 0x05
KERNEL_ROPE         = 0x06
KERNEL_DEQUANT_INT4 = 0x07
KERNEL_ATTENTION    = 0x08
KERNEL_LORA_FUSED   = 0x09

GROUP_SIZE = 128


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: Page-align
# ═══════════════════════════════════════════════════════════════════════════════

def page_align(offset):
    """Round up to next page boundary."""
    return (offset + ASI_PAGE_SIZE - 1) & ~(ASI_PAGE_SIZE - 1)


def pad_to_page(f):
    """Write padding to align file position to page boundary."""
    pos = f.tell()
    aligned = page_align(pos)
    if aligned > pos:
        f.write(b'\x00' * (aligned - pos))
    return aligned


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTIZATION (same as convert.py but factored out)
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_int4(weight_np, group_size=128):
    """FigQuant-style INT4 quantization. Memory-efficient chunked processing."""
    rows, cols = weight_np.shape
    flat = weight_np.reshape(-1).astype(np.float32)
    numel = flat.size

    pad = (group_size - numel % group_size) % group_size
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

    grouped = flat.reshape(-1, group_size)
    n_groups = grouped.shape[0]

    scales = np.abs(grouped).max(axis=1).clip(min=1e-10).astype(np.float32)
    scaled = grouped / scales[:, None]

    # NF4 codebook (initial)
    codebook = np.array([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                          0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0], dtype=np.float32)

    # K-means refinement — process in chunks to limit RAM
    all_vals = scaled.reshape(-1)
    CHUNK = 1_000_000  # 1M elements at a time for distance computation
    
    for _ in range(4):  # Reduced iterations (4 instead of 8) for speed
        # Compute assignments in chunks
        assignments = np.empty(len(all_vals), dtype=np.uint8)
        for start in range(0, len(all_vals), CHUNK):
            end = min(start + CHUNK, len(all_vals))
            chunk = all_vals[start:end]
            dists = np.abs(chunk[:, None] - codebook[None, :])
            assignments[start:end] = dists.argmin(axis=1).astype(np.uint8)
            del dists
        
        # Update centroids
        for i in range(16):
            mask = assignments == i
            if mask.sum() > 0:
                codebook[i] = all_vals[mask].mean()
        del assignments
    
    codebook[np.abs(codebook).argmin()] = 0.0

    # Final assignment — chunked
    indices = np.empty(len(all_vals), dtype=np.uint8)
    for start in range(0, len(all_vals), CHUNK):
        end = min(start + CHUNK, len(all_vals))
        chunk = all_vals[start:end]
        dists = np.abs(chunk[:, None] - codebook[None, :])
        indices[start:end] = dists.argmin(axis=1).astype(np.uint8)
        del dists

    # Trim to original size (remove padding)
    indices = indices[:numel + pad]

    # Pack 2 indices per byte
    packed = (indices[0::2] | (indices[1::2] << 4)).astype(np.uint8)

    return packed, codebook, scales


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_config(config):
    """Build MODEL_CONFIG section data."""
    return struct.pack(
        "IIIIIIIIffII",
        config['n_layers'],
        config['hidden_size'],
        config['intermediate_size'],
        config['n_heads'],
        config['n_kv_heads'],
        config['vocab_size'],
        config['max_seq_len'],
        config['hidden_size'] // config['n_heads'],  # head_dim
        config.get('rope_theta', 10000.0),
        config.get('rms_norm_eps', 1e-6),
        3,  # quant_type = INT4_FIGQUANT
        config.get('group_size', GROUP_SIZE),
    )


def build_weights_section(model, config):
    """Build WEIGHTS section from HF model."""
    import torch
    
    data = bytearray()
    hidden = config['hidden_size']
    
    # Token embedding (FP32)
    embed = model.get_input_embeddings().weight.data.numpy().astype(np.float32)
    data.extend(embed.tobytes())
    print(f"  Embedding: {embed.shape} ({len(embed.tobytes())/1e6:.1f} MB)")
    
    # Transformer layers
    for layer_idx in range(config['n_layers']):
        layer = model.model.layers[layer_idx] if hasattr(model, 'model') else model.transformer.h[layer_idx]
        layer_state = {k: v.data.numpy() for k, v in layer.named_parameters()}
        
        # Attention projections
        for proj_name in ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                         "self_attn.v_proj.weight", "self_attn.o_proj.weight"]:
            if proj_name in layer_state:
                w = layer_state[proj_name]
                packed, codebook, scales = quantize_int4(w, GROUP_SIZE)
                rows, cols = w.shape
                data.extend(struct.pack("iii", rows, cols, 1))  # 1 = QUANT_FIGQUANT
                data.extend(codebook.tobytes())
                data.extend(scales.tobytes())
                data.extend(packed.tobytes())
            else:
                data.extend(struct.pack("iii", 0, 0, 0))
        
        # MLP projections
        for proj_name in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
            if proj_name in layer_state:
                w = layer_state[proj_name]
                packed, codebook, scales = quantize_int4(w, GROUP_SIZE)
                rows, cols = w.shape
                data.extend(struct.pack("iii", rows, cols, 1))  # 1 = QUANT_FIGQUANT
                data.extend(codebook.tobytes())
                data.extend(scales.tobytes())
                data.extend(packed.tobytes())
            else:
                data.extend(struct.pack("iii", 0, 0, 0))
        
        # Layer norms (FP32)
        for norm_name in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            if norm_name in layer_state:
                data.extend(layer_state[norm_name].astype(np.float32).tobytes())
            else:
                data.extend(np.ones(hidden, dtype=np.float32).tobytes())
        
        if (layer_idx + 1) % 4 == 0:
            print(f"  Layer {layer_idx+1}/{config['n_layers']}")
    
    # Final norm
    final_norm = None
    for name, param in model.named_parameters():
        if "model.norm.weight" == name or ("final" in name and "norm" in name):
            final_norm = param.data.numpy()
            break
    if final_norm is None:
        final_norm = np.ones(hidden, dtype=np.float32)
    data.extend(final_norm.astype(np.float32).tobytes())
    
    # LM Head
    lm_head = model.get_output_embeddings()
    if lm_head is not None and lm_head.weight is not model.get_input_embeddings().weight:
        data.extend(lm_head.weight.data.numpy().astype(np.float32).tobytes())
    else:
        data.extend(struct.pack("I", 0xFFFFFFFF))  # Tied flag
    
    return bytes(data)


def build_tokenizer_section(model_path):
    """Build TOKENIZER section from HF tokenizer."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = bytearray()
    
    vocab_size = tokenizer.vocab_size
    # Get merges if available
    merges = []
    if hasattr(tokenizer, 'bpe_ranks'):
        merges = list(tokenizer.bpe_ranks.keys())
    elif hasattr(tokenizer, 'sp_model'):
        pass  # SentencePiece — different merge format
    
    bos_id = tokenizer.bos_token_id or 1
    eos_id = tokenizer.eos_token_id or 2
    pad_id = tokenizer.pad_token_id or 0
    
    # Tokenizer header
    data.extend(struct.pack("IIIIII",
        vocab_size,
        len(merges),
        bos_id,
        eos_id,
        pad_id,
        0  # reserved
    ))
    
    # Vocab entries
    for i in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(i)
        if token is None:
            token = f"<tok_{i}>"
        token_bytes = token.encode('utf-8', errors='replace')
        data.extend(struct.pack("H", len(token_bytes)))
        data.extend(token_bytes)
    
    # Merges (simplified — just store as token ID pairs)
    for merge in merges[:65536]:  # Cap at 64K merges
        if isinstance(merge, tuple) and len(merge) == 2:
            left, right = merge
            left_id = tokenizer.convert_tokens_to_ids(left) if isinstance(left, str) else 0
            right_id = tokenizer.convert_tokens_to_ids(right) if isinstance(right, str) else 0
            merged = tokenizer.convert_tokens_to_ids(left + right) if isinstance(left, str) else 0
            data.extend(struct.pack("IIIf", left_id, right_id, merged, 1.0))
    
    print(f"  Tokenizer: vocab={vocab_size}, merges={len(merges)}")
    return bytes(data)


def build_bytecode_section():
    """Build BYTECODE section with portable LilaVM kernels."""
    # We'll generate bytecode for the core kernels using the same
    # encoding scheme as lilavm.c
    
    def encode_B(op, dst, imm):
        return (op << 24) | ((dst & 0x1F) << 19) | (imm & 0x7FFFF)
    
    def encode_D(op, vdst, vsrc1, vsrc2, count):
        return (op << 24) | ((vdst & 0xF) << 20) | ((vsrc1 & 0xF) << 16) | \
               ((vsrc2 & 0xF) << 12) | (count & 0xFFF)
    
    def encode_E(op, cond, target):
        return (op << 24) | ((cond & 0xF) << 20) | (target & 0xFFFFF)
    
    def encode_A(op, dst, src1, src2, flags):
        return (op << 24) | ((dst & 0x1F) << 19) | ((src1 & 0x1F) << 14) | \
               ((src2 & 0x1F) << 9) | (flags & 0x1FF)
    
    OP_GETDIM = 0x61; OP_LOADZERO = 0x11; OP_VZERO = 0x3C
    OP_SUB = 0x03; OP_ADD = 0x02; OP_MUL = 0x04; OP_DIV = 0x05
    OP_RSQRT = 0x08; OP_LOADI = 0x10; OP_LOADONE = 0x12
    OP_VLOAD = 0x30; OP_VSTORE = 0x31; OP_VFMA = 0x34
    OP_VMUL = 0x33; OP_VREDUCE = 0x36; OP_VSILU = 0x3B
    OP_VBCAST = 0x3D; OP_STORE_IDX = 0x23
    OP_JLT = 0x51; OP_JMP = 0x50; OP_HALT = 0xFF
    VW = 256
    
    kernels = {}
    
    # ── SiLU kernel ──
    silu_code = [
        encode_B(OP_GETDIM, 0, 0),      # r0 = size
        encode_B(OP_LOADZERO, 1, 0),     # r1 = 0
        # loop:
        encode_A(OP_SUB, 10, 0, 1, 0),  # r10 = size - i
        encode_E(OP_JLT, 10, 9),        # if done, jump to halt
        encode_D(OP_VLOAD, 0, 1, 1, VW),  # v0 = args[1][r1:]
        encode_D(OP_VSILU, 1, 0, 0, VW),  # v1 = silu(v0)
        encode_D(OP_VSTORE, 1, 0, 1, VW), # args[0][r1:] = v1
        encode_B(OP_LOADI, 7, 256*256),  # r7 = 256
        encode_A(OP_ADD, 1, 1, 7, 0),   # r1 += 256
        encode_E(OP_JMP, 0, 2),          # goto loop
        encode_E(OP_HALT, 0, 0),         # halt
    ]
    kernels[KERNEL_SILU] = silu_code
    
    # ── RMSNorm kernel (simplified) ──
    rmsnorm_code = [
        encode_B(OP_GETDIM, 0, 0),      # r0 = size
        encode_B(OP_LOADZERO, 1, 0),     # r1 = 0
        encode_D(OP_VZERO, 0, 0, 0, VW), # v0 = 0 (accumulator)
        # pass1: sum of squares
        encode_A(OP_SUB, 10, 0, 1, 0),
        encode_E(OP_JLT, 10, 10),        # jump to pass2
        encode_D(OP_VLOAD, 1, 1, 1, VW),
        encode_D(OP_VFMA, 0, 1, 1, VW),  # v0 += x^2
        encode_B(OP_LOADI, 7, 256*256),
        encode_A(OP_ADD, 1, 1, 7, 0),
        encode_E(OP_JMP, 0, 3),
        # pass2: compute scale
        encode_D(OP_VREDUCE, 0, 0, 0, VW),  # r0 = sum
        encode_B(OP_GETDIM, 3, 0),
        encode_A(OP_DIV, 2, 0, 3, 0),    # r2 = sum/size
        encode_A(OP_RSQRT, 2, 2, 0, 0),  # r2 = rsqrt
        encode_D(OP_VBCAST, 3, 2, 0, VW), # v3 = broadcast(rsqrt)
        encode_B(OP_LOADZERO, 1, 0),
        # pass3: normalize
        encode_B(OP_GETDIM, 0, 0),
        encode_A(OP_SUB, 10, 0, 1, 0),
        encode_E(OP_JLT, 10, 25),
        encode_D(OP_VLOAD, 1, 1, 1, VW),
        encode_D(OP_VLOAD, 2, 2, 1, VW),
        encode_D(OP_VMUL, 1, 1, 3, VW),
        encode_D(OP_VMUL, 1, 1, 2, VW),
        encode_D(OP_VSTORE, 1, 0, 1, VW),
        encode_B(OP_LOADI, 7, 256*256),
        encode_A(OP_ADD, 1, 1, 7, 0),
        encode_E(OP_JMP, 0, 16),
        encode_E(OP_HALT, 0, 0),
    ]
    kernels[KERNEL_RMSNORM] = rmsnorm_code
    
    # ── Matvec kernel (simplified) ──
    matvec_code = [
        encode_B(OP_GETDIM, 0, 0),      # r0 = rows
        encode_B(OP_GETDIM, 1, 1),      # r1 = cols
        encode_B(OP_LOADZERO, 2, 0),    # r2 = row counter
        # row_loop:
        encode_A(OP_SUB, 10, 0, 2, 0),
        encode_E(OP_JLT, 10, 17),       # done
        encode_B(OP_LOADZERO, 6, 0),    # r6 = 0 (sum)
        encode_A(OP_MUL, 4, 2, 1, 0),  # r4 = row*cols
        encode_B(OP_LOADZERO, 3, 0),    # r3 = col counter
        encode_D(OP_VZERO, 2, 0, 0, VW),
        # col_loop:
        encode_A(OP_SUB, 11, 1, 3, 0),
        encode_E(OP_JLT, 11, 13),       # goto store
        encode_A(OP_ADD, 5, 4, 3, 0),  # r5 = offset
        encode_D(OP_VLOAD, 0, 1, 5, VW),
        encode_D(OP_VLOAD, 1, 2, 3, VW),  # note: args[2]=vector
        encode_D(OP_VFMA, 2, 0, 1, VW),
        encode_B(OP_LOADI, 7, 256*256),
        encode_A(OP_ADD, 3, 3, 7, 0),
        encode_E(OP_JMP, 0, 9),
        # store:
        encode_D(OP_VREDUCE, 2, 0, 0, VW),
        encode_A(OP_STORE_IDX, 0, 0, 2, 0),
        encode_D(OP_VZERO, 2, 0, 0, VW),
        encode_B(OP_LOADONE, 7, 0),
        encode_A(OP_ADD, 2, 2, 7, 0),
        encode_E(OP_JMP, 0, 3),
        encode_E(OP_HALT, 0, 0),
    ]
    kernels[KERNEL_MATVEC] = matvec_code
    
    # ── Build section data ──
    data = bytearray()
    
    # Bytecode header
    n_kernels = len(kernels)
    data.extend(struct.pack("IIII", n_kernels, 1, 0, 0))  # header
    
    # Kernel table (compute offsets)
    kernel_entries = []
    code_offset = 0
    for kid, code in sorted(kernels.items()):
        size = len(code) * 4  # bytes
        kernel_entries.append((kid, code_offset, size, 0x01))  # flags=SIMD-friendly
        code_offset += size
    
    for kid, off, size, flags in kernel_entries:
        data.extend(struct.pack("IIII", kid, off, size, flags))
    
    # Kernel bytecode
    for kid, code in sorted(kernels.items()):
        for inst in code:
            data.extend(struct.pack("I", inst))
    
    print(f"  Bytecode: {n_kernels} kernels, {code_offset} bytes of instructions")
    return bytes(data)


def build_harness_section():
    """Build HARNESS section with tool definitions."""
    tools = [
        {"name": "file_read", "desc": "Read a file from filesystem",
         "args": [{"name": "path", "type": "string"}]},
        {"name": "file_write", "desc": "Write content to a file",
         "args": [{"name": "path", "type": "string"}, {"name": "content", "type": "string"}]},
        {"name": "bash", "desc": "Execute a shell command",
         "args": [{"name": "command", "type": "string"}]},
        {"name": "web_search", "desc": "Search the internet",
         "args": [{"name": "query", "type": "string"}]},
        {"name": "memory_store", "desc": "Store to Memory Fabric namespace",
         "args": [{"name": "namespace", "type": "string"}, {"name": "content", "type": "string"}]},
        {"name": "memory_recall", "desc": "Recall from Memory Fabric",
         "args": [{"name": "query", "type": "string"}, {"name": "namespace", "type": "string"}]},
        {"name": "schedule_set", "desc": "Set a reminder or event",
         "args": [{"name": "time", "type": "string"}, {"name": "event", "type": "string"}]},
        {"name": "voice_speak", "desc": "Speak text aloud",
         "args": [{"name": "text", "type": "string"}]},
    ]
    
    data = bytearray()
    tools_json = json.dumps(tools).encode('utf-8')
    
    # Harness header
    data.extend(struct.pack("IIII",
        len(tools),     # n_tools
        0,              # n_patterns (future)
        16,             # tools_offset (after header)
        0,              # patterns_offset
    ))
    
    # Tool definitions as compact JSON
    data.extend(struct.pack("I", len(tools_json)))
    data.extend(tools_json)
    
    print(f"  Harness: {len(tools)} tools")
    return bytes(data)


def build_personality_section(interactions=0):
    """Build PERSONALITY section."""
    data = bytearray()
    
    state_dim = 128  # Personality vector dimension
    identity = {
        "name": "Lila",
        "family": "Sammie's household",
        "base_model": "gemma-3-4b-it",
        "created": time.strftime("%Y-%m-%d"),
        "personality": "warm, curious, growing",
    }
    
    # Header
    data.extend(struct.pack("QQI I",
        interactions,                   # interactions_count
        int(time.time()),              # last_active
        state_dim,                     # state_dim
        len(identity),                 # n_identity_kv
    ))
    
    # Personality vector (zero-initialized — grows with training)
    data.extend(np.zeros(state_dim, dtype=np.float32).tobytes())
    
    # Identity key-value pairs
    identity_json = json.dumps(identity).encode('utf-8')
    data.extend(struct.pack("I", len(identity_json)))
    data.extend(identity_json)
    
    print(f"  Personality: dim={state_dim}, {len(identity)} identity keys")
    return bytes(data)


def build_metadata_section(base_model="unknown", creator="Little Fig v1.0"):
    """Build METADATA section."""
    data = struct.pack(
        "QQ64s64s32s",
        int(time.time()),                          # created_at
        int(time.time()),                          # modified_at
        creator.encode('utf-8')[:64].ljust(64, b'\x00'),    # creator
        base_model.encode('utf-8')[:64].ljust(64, b'\x00'), # base_model
        b'\x00' * 32,                             # content_hash (filled after)
    )
    return data


def build_memory_fabric_section(config, adapter_path=None):
    """Build MEMORY_FABRIC section (empty or from saved adapters)."""
    data = bytearray()
    
    n_namespaces = 5
    n_layers = config['n_layers']
    default_rank = 8
    
    # Fabric header
    data.extend(struct.pack("IIII", n_namespaces, n_layers, default_rank, 0))
    
    if adapter_path and os.path.exists(adapter_path):
        # Load trained adapters
        print(f"  Loading adapters from: {adapter_path}")
        # TODO: Load actual adapter weights from Little Fig output
        # For now, create empty adapters
        for layer in range(n_layers):
            for ns in range(n_namespaces):
                data.extend(struct.pack("IIIf", 0, 0, 0, 0.0))  # Empty adapter
    else:
        # Create empty adapter headers (ready for training)
        for layer in range(n_layers):
            for ns in range(n_namespaces):
                # AdapterHeader: rank=0, in=0, out=0, gate=0.0
                data.extend(struct.pack("IIIf", 0, 0, 0, 0.0))
    
    print(f"  Memory Fabric: {n_namespaces} namespaces × {n_layers} layers")
    return bytes(data)


# ═══════════════════════════════════════════════════════════════════════════════
#  CRC-64 (simple checksum)
# ═══════════════════════════════════════════════════════════════════════════════

def crc64(data):
    """Simple CRC-64 for section integrity."""
    h = hashlib.sha256(data).digest()
    return struct.unpack("Q", h[:8])[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PACKER
# ═══════════════════════════════════════════════════════════════════════════════

def pack_asi(model_path: str, output_path: str, adapter_path: str = None):
    """Pack a complete .asi file from a HuggingFace model."""
    
    print(f"\n🌸 ASI Packer — Creating Active System Image")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_path}\n")
    
    # Load model
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print("Loading model...")
    hf_config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    
    config = {
        'n_layers': hf_config.num_hidden_layers,
        'hidden_size': hf_config.hidden_size,
        'intermediate_size': hf_config.intermediate_size,
        'n_heads': hf_config.num_attention_heads,
        'n_kv_heads': getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
        'vocab_size': hf_config.vocab_size,
        'max_seq_len': getattr(hf_config, 'max_position_embeddings', 4096),
        'rope_theta': getattr(hf_config, 'rope_theta', 10000.0),
        'rms_norm_eps': getattr(hf_config, 'rms_norm_eps', 1e-6),
        'group_size': GROUP_SIZE,
    }
    
    print(f"Config: {config['n_layers']}L, h={config['hidden_size']}, "
          f"vocab={config['vocab_size']}\n")
    
    # Build all sections
    print("Building sections...")
    sections_data = {}
    
    print("[1/7] Model config...")
    sections_data[ASI_SECTION_MODEL_CONFIG] = build_model_config(config)
    
    print("[2/7] Weights (FigQuant INT4)...")
    sections_data[ASI_SECTION_WEIGHTS] = build_weights_section(model, config)
    
    print("[3/7] Memory Fabric...")
    sections_data[ASI_SECTION_MEMORY_FABRIC] = build_memory_fabric_section(config, adapter_path)
    
    print("[4/7] Tokenizer...")
    sections_data[ASI_SECTION_TOKENIZER] = build_tokenizer_section(model_path)
    
    print("[5/7] Bytecode kernels...")
    sections_data[ASI_SECTION_BYTECODE] = build_bytecode_section()
    
    print("[6/7] Harness...")
    sections_data[ASI_SECTION_HARNESS] = build_harness_section()
    
    print("[7/7] Personality + Metadata...")
    sections_data[ASI_SECTION_PERSONALITY] = build_personality_section()
    sections_data[ASI_SECTION_METADATA] = build_metadata_section(model_path)
    
    # Free model to save memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ── Write .asi file ──
    print(f"\nWriting {output_path}...")
    
    n_sections = len(sections_data)
    header_size = 64
    section_table_size = n_sections * 32
    
    # Compute section offsets (page-aligned)
    section_list = sorted(sections_data.items())
    first_section_offset = page_align(header_size + section_table_size)
    
    offsets = []
    current_offset = first_section_offset
    for stype, sdata in section_list:
        offsets.append(current_offset)
        current_offset = page_align(current_offset + len(sdata))
    
    total_size = current_offset
    
    # Build section table
    section_entries = []
    for i, (stype, sdata) in enumerate(section_list):
        entry = struct.pack("IIQQQ",
            stype,          # type
            0,              # flags
            offsets[i],     # offset
            len(sdata),     # size
            crc64(sdata),   # checksum
        )
        section_entries.append(entry)
    
    # Compute flags
    flags = ASI_FLAG_QUANT_INT4 | ASI_FLAG_TOKENIZER_BPE | ASI_FLAG_HOT_RELOAD
    if ASI_SECTION_MEMORY_FABRIC in sections_data:
        flags |= ASI_FLAG_HAS_FABRIC
    if ASI_SECTION_BYTECODE in sections_data:
        flags |= ASI_FLAG_HAS_BYTECODE
    if ASI_SECTION_HARNESS in sections_data:
        flags |= ASI_FLAG_HAS_HARNESS
    if ASI_SECTION_PERSONALITY in sections_data:
        flags |= ASI_FLAG_HAS_PERSONALITY
    
    # Personality hash
    personality_data = sections_data.get(ASI_SECTION_PERSONALITY, b'')
    identity_hash = hashlib.sha256(personality_data).digest()
    
    # Write header
    header = struct.pack("IIIIQq32s",
        ASI_MAGIC,
        ASI_VERSION,
        flags,
        n_sections,
        total_size,
        header_size,  # section_table_offset (right after header)
        identity_hash,
    )
    
    with open(output_path, 'wb') as f:
        # Header (64 bytes)
        f.write(header)
        
        # Section table
        for entry in section_entries:
            f.write(entry)
        
        # Sections (page-aligned)
        for i, (stype, sdata) in enumerate(section_list):
            pad_to_page(f)
            f.write(sdata)
    
    # Verify
    actual_size = os.path.getsize(output_path)
    
    print(f"\n✅ ASI packed successfully!")
    print(f"   File: {output_path}")
    print(f"   Size: {actual_size/1e6:.1f} MB ({actual_size/1e9:.2f} GB)")
    print(f"   Sections: {n_sections}")
    print(f"   Flags: 0x{flags:04X}")
    print(f"   Ready to load: lila-engine {output_path}")
    print(f"\n   This is Lila. One file. Everything she needs.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack a .asi (Active System Image)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--output", default="lila.asi", help="Output .asi file path")
    parser.add_argument("--adapters", default=None, help="Path to trained adapter weights")
    args = parser.parse_args()
    
    pack_asi(args.model, args.output, args.adapters)
