#!/usr/bin/env python3
"""
GGUF → ASI Converter (v2) — Stores Q4_K blocks RAW

No re-quantization. No precision loss. No OOM.
Just reformats the GGUF data into .asi sections.

The C engine reads Q4_K blocks natively (same algo as llama.cpp).

Usage:
    python gguf_to_asi_v2.py --gguf model.gguf --output lila.asi
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

sys.path.insert(0, str(Path(__file__).parent))
from pack_asi import (ASI_MAGIC, ASI_VERSION, ASI_PAGE_SIZE,
    ASI_SECTION_MODEL_CONFIG, ASI_SECTION_WEIGHTS, ASI_SECTION_MEMORY_FABRIC,
    ASI_SECTION_TOKENIZER, ASI_SECTION_BYTECODE, ASI_SECTION_HARNESS,
    ASI_SECTION_PERSONALITY, ASI_SECTION_METADATA,
    ASI_FLAG_HAS_FABRIC, ASI_FLAG_HAS_BYTECODE, ASI_FLAG_HAS_HARNESS,
    ASI_FLAG_HAS_PERSONALITY, ASI_FLAG_QUANT_INT4, ASI_FLAG_TOKENIZER_BPE,
    ASI_FLAG_HOT_RELOAD,
    page_align, build_bytecode_section, build_harness_section,
    build_personality_section, build_metadata_section)

# Quant type constants (must match model.h)
QUANT_NONE = 0
QUANT_FIGQUANT = 1
QUANT_Q4_K = 2
QUANT_Q6_K = 3
QUANT_F16 = 4

# GGUF quant type IDs
GGUF_F32 = 0
GGUF_F16 = 1
GGUF_Q4_K = 12
GGUF_Q6_K = 14
GGUF_Q5_K = 13
GGUF_Q8_0 = 8

# Block sizes per quant type
BLOCK_SIZES = {
    GGUF_Q4_K: 144,   # 144 bytes → 256 values
    GGUF_Q6_K: 210,   # 210 bytes → 256 values
    GGUF_Q5_K: 176,   # 176 bytes → 256 values
    GGUF_Q8_0: 136,   # 136 bytes → 256 values (2 + 32*4 + 2 pad)
}

QK_K = 256  # Values per block


def get_asi_quant_type(gguf_type):
    """Map GGUF quant type to ASI quant type."""
    if gguf_type == GGUF_Q4_K:
        return QUANT_Q4_K
    elif gguf_type == GGUF_Q6_K:
        return QUANT_Q6_K
    elif gguf_type in (GGUF_F32, GGUF_F16):
        return QUANT_NONE
    else:
        return QUANT_Q4_K  # Treat unknown as Q4_K (most common)


def gguf_to_asi(gguf_path: str, output_path: str):
    """Convert GGUF directly to .asi — stores Q4_K blocks raw."""
    from gguf import GGUFReader
    
    print(f"\n🌸 GGUF → ASI Converter (v2 — native Q4_K, no re-quantization)")
    print(f"   Input:  {gguf_path}")
    print(f"   Output: {output_path}\n")
    
    file_size = os.path.getsize(gguf_path)
    print(f"Reading GGUF ({file_size/1e9:.2f} GB)...")
    reader = GGUFReader(gguf_path)
    
    # ── Extract config from GGUF metadata ──
    def get_meta(key, default=None):
        for field in reader.fields.values():
            if field.name == key:
                if field.types and field.types[0] == 4:  # uint32
                    return int(field.parts[-1][0])
                elif field.types and field.types[0] == 5:  # int32
                    return int(field.parts[-1][0])
                elif field.types and field.types[0] == 6:  # float32
                    return float(field.parts[-1][0])
                elif field.types and field.types[0] == 8:  # string
                    return bytes(field.parts[-1]).decode('utf-8')
                else:
                    try:
                        return int(field.parts[-1][0])
                    except:
                        return default
        return default
    
    # Try various metadata key patterns
    n_layers = get_meta('llama.block_count') or get_meta('gemma.block_count') or \
               get_meta('qwen2.block_count') or get_meta('phi3.block_count') or 26
    hidden = get_meta('llama.embedding_length') or get_meta('gemma.embedding_length') or \
             get_meta('qwen2.embedding_length') or 2560
    n_heads = get_meta('llama.attention.head_count') or get_meta('gemma.attention.head_count') or \
              get_meta('qwen2.attention.head_count') or 16
    n_kv_heads = get_meta('llama.attention.head_count_kv') or get_meta('gemma.attention.head_count_kv') or \
                 get_meta('qwen2.attention.head_count_kv') or n_heads
    vocab_size = get_meta('llama.vocab_size') or get_meta('gemma.vocab_size') or \
                 get_meta('qwen2.vocab_size') or 262144
    intermediate = get_meta('llama.feed_forward_length') or get_meta('gemma.feed_forward_length') or \
                   get_meta('qwen2.feed_forward_length') or hidden * 4
    max_seq = get_meta('llama.context_length') or get_meta('gemma.context_length') or 4096
    rope_theta = get_meta('llama.rope.freq_base') or get_meta('gemma.rope.freq_base') or 10000.0
    rms_eps = get_meta('llama.attention.layer_norm_rms_epsilon') or 1e-6
    
    print(f"Config: {n_layers}L, hidden={hidden}, vocab={vocab_size}, heads={n_heads}/{n_kv_heads}")
    print(f"  Intermediate: {intermediate}, Seq: {max_seq}")
    
    # Build tensor lookup
    tensor_map = {}
    for tensor in reader.tensors:
        tensor_map[tensor.name] = tensor
    print(f"  Tensors: {len(tensor_map)}")
    
    # Detect true hidden size from embedding tensor
    embed_tensor = tensor_map.get('token_embd.weight')
    if embed_tensor is not None:
        if len(embed_tensor.shape) >= 2:
            true_hidden = int(embed_tensor.shape[1]) if embed_tensor.shape[1] < embed_tensor.shape[0] else int(embed_tensor.shape[0])
            true_vocab = int(embed_tensor.shape[0]) if embed_tensor.shape[0] > embed_tensor.shape[1] else int(embed_tensor.shape[1])
            if true_hidden != hidden:
                print(f"  Correcting hidden: {hidden} → {true_hidden}")
                hidden = true_hidden
            if true_vocab != vocab_size:
                print(f"  Correcting vocab: {vocab_size} → {true_vocab}")
                vocab_size = true_vocab
    
    # ── Write .asi ──
    print(f"\nBuilding .asi...")
    
    # We'll write the weights section directly to a temp file (too large for RAM)
    weights_tmp = output_path + '.weights.tmp'
    bytes_written = 0
    
    with open(weights_tmp, 'wb') as wf:
        # Token embedding — dequant to FP32 (must be FP32 for lookup)
        print(f"  Embedding [{vocab_size} x {hidden}]...")
        if embed_tensor is not None:
            from gguf import dequantize
            # Process embedding in chunks to avoid OOM
            raw = embed_tensor.data
            total_elements = vocab_size * hidden
            
            if embed_tensor.tensor_type.value == GGUF_F32:
                # Already FP32, just copy
                wf.write(raw[:total_elements * 4].tobytes())
                bytes_written += total_elements * 4
            else:
                # Dequant in chunks
                bytes_per_row = len(raw) // vocab_size
                CHUNK = 4096
                for start in range(0, vocab_size, CHUNK):
                    end = min(start + CHUNK, vocab_size)
                    chunk_raw = raw[start * bytes_per_row : end * bytes_per_row]
                    try:
                        chunk_f32 = dequantize(chunk_raw, embed_tensor.tensor_type)
                        chunk_f32 = chunk_f32.reshape(end - start, hidden).astype(np.float32)
                    except:
                        chunk_f32 = np.zeros((end - start, hidden), dtype=np.float32)
                    wf.write(chunk_f32.tobytes())
                    bytes_written += chunk_f32.nbytes
                    if start % 32768 == 0 and start > 0:
                        print(f"    Embedding: {start}/{vocab_size}")
        else:
            # No embedding found — write zeros
            for start in range(0, vocab_size, 4096):
                end = min(start + 4096, vocab_size)
                wf.write(np.zeros((end - start, hidden), dtype=np.float32).tobytes())
                bytes_written += (end - start) * hidden * 4
        
        print(f"  Embedding done ({bytes_written/1e6:.0f} MB)")
        
        # Per-layer weights — store Q4_K blocks RAW
        layer_patterns = {
            'q_proj': ['blk.{}.attn_q.weight', 'blk.{}.self_attn.q_proj.weight'],
            'k_proj': ['blk.{}.attn_k.weight', 'blk.{}.self_attn.k_proj.weight'],
            'v_proj': ['blk.{}.attn_v.weight', 'blk.{}.self_attn.v_proj.weight'],
            'o_proj': ['blk.{}.attn_output.weight', 'blk.{}.self_attn.o_proj.weight'],
            'gate_proj': ['blk.{}.ffn_gate.weight', 'blk.{}.mlp.gate_proj.weight'],
            'up_proj': ['blk.{}.ffn_up.weight', 'blk.{}.mlp.up_proj.weight'],
            'down_proj': ['blk.{}.ffn_down.weight', 'blk.{}.mlp.down_proj.weight'],
        }
        
        for layer_idx in range(n_layers):
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                patterns = layer_patterns[proj_name]
                found = False
                
                for pattern in patterns:
                    tname = pattern.format(layer_idx)
                    if tname in tensor_map:
                        tensor = tensor_map[tname]
                        gguf_type = tensor.tensor_type.value
                        asi_quant = get_asi_quant_type(gguf_type)
                        
                        # Get shape
                        shape = list(reversed(tensor.shape.tolist())) if len(tensor.shape) >= 2 else [0, 0]
                        rows, cols = int(shape[0]), int(shape[1])
                        
                        # Write header: rows, cols, quant_type
                        wf.write(struct.pack('iii', rows, cols, asi_quant))
                        bytes_written += 12
                        
                        # Write raw quantized data (NO re-quantization!)
                        raw_data = tensor.data.tobytes()
                        wf.write(raw_data)
                        bytes_written += len(raw_data)
                        
                        found = True
                        break
                
                if not found:
                    # Write empty tensor header
                    wf.write(struct.pack('iii', 0, 0, 0))
                    bytes_written += 12
            
            # Layer norms (always FP32 in GGUF)
            norm_patterns = [
                ['blk.{}.attn_norm.weight', 'blk.{}.input_layernorm.weight'],
                ['blk.{}.ffn_norm.weight', 'blk.{}.post_attention_layernorm.weight'],
            ]
            for norm_pats in norm_patterns:
                found = False
                for pattern in norm_pats:
                    tname = pattern.format(layer_idx)
                    if tname in tensor_map:
                        tensor = tensor_map[tname]
                        # Norms are F32 — just copy
                        data = tensor.data
                        if tensor.tensor_type.value == GGUF_F32:
                            norm_bytes = data[:hidden * 4].tobytes()
                        else:
                            from gguf import dequantize
                            norm_bytes = dequantize(data, tensor.tensor_type).astype(np.float32)[:hidden].tobytes()
                        wf.write(norm_bytes)
                        bytes_written += len(norm_bytes)
                        found = True
                        break
                if not found:
                    wf.write(np.ones(hidden, dtype=np.float32).tobytes())
                    bytes_written += hidden * 4
            
            if (layer_idx + 1) % 4 == 0:
                print(f"  Layer {layer_idx + 1}/{n_layers} ({bytes_written/1e6:.0f} MB)")
        
        # Final norm
        final_norm_names = ['output_norm.weight', 'model.norm.weight']
        found = False
        for name in final_norm_names:
            if name in tensor_map:
                tensor = tensor_map[name]
                if tensor.tensor_type.value == GGUF_F32:
                    wf.write(tensor.data[:hidden * 4].tobytes())
                else:
                    from gguf import dequantize
                    wf.write(dequantize(tensor.data, tensor.tensor_type).astype(np.float32)[:hidden].tobytes())
                bytes_written += hidden * 4
                found = True
                break
        if not found:
            wf.write(np.ones(hidden, dtype=np.float32).tobytes())
            bytes_written += hidden * 4
        
        # LM head
        lm_head_names = ['output.weight', 'lm_head.weight']
        found = False
        for name in lm_head_names:
            if name in tensor_map:
                # LM head is often tied — write tied flag
                wf.write(struct.pack('I', 0xFFFFFFFF))
                bytes_written += 4
                found = True
                break
        if not found:
            wf.write(struct.pack('I', 0xFFFFFFFF))  # Tied
            bytes_written += 4
    
    print(f"  Weights total: {bytes_written/1e9:.2f} GB")
    
    # ── Assemble the .asi ──
    print(f"\nAssembling .asi...")
    
    # Read weights back
    weights_data_size = os.path.getsize(weights_tmp)
    
    # Build other sections
    config_data = struct.pack('IIIIIIIIffII',
        n_layers, hidden, intermediate, n_heads, n_kv_heads, vocab_size,
        max_seq, hidden // n_heads, float(rope_theta), float(rms_eps),
        QUANT_Q4_K, 128)
    
    fabric = bytearray(struct.pack('IIII', 5, n_layers, 8, 0))
    for _ in range(n_layers * 5):
        fabric.extend(struct.pack('IIIf', 0, 0, 0, 0.0))
    
    # Tokenizer from GGUF
    tok_data = bytearray()
    bos = get_meta('tokenizer.ggml.bos_token_id') or 1
    eos = get_meta('tokenizer.ggml.eos_token_id') or 2
    pad = get_meta('tokenizer.ggml.padding_token_id') or 0
    
    # Extract vocab tokens
    vocab_tokens = []
    for field in reader.fields.values():
        if field.name == 'tokenizer.ggml.tokens':
            # Array of strings
            for part in field.parts:
                if hasattr(part, 'tobytes'):
                    tok = bytes(part)
                    vocab_tokens.append(tok)
            break
    
    actual_vocab = len(vocab_tokens) if vocab_tokens else vocab_size
    n_merges = 0  # TODO: extract merges
    
    tok_data.extend(struct.pack('IIIIII', actual_vocab, n_merges, bos, eos, pad, 0))
    if vocab_tokens:
        for tok in vocab_tokens:
            tok_data.extend(struct.pack('H', len(tok)))
            tok_data.extend(tok)
    else:
        for i in range(actual_vocab):
            t = f'<{i}>'.encode()
            tok_data.extend(struct.pack('H', len(t)))
            tok_data.extend(t)
    
    bytecode_data = build_bytecode_section()
    harness_data = build_harness_section()
    personality_data = build_personality_section()
    metadata_data = build_metadata_section(gguf_path.split('/')[-1].split('\\')[-1], 'gguf_to_asi v2')
    
    # Write final .asi with streaming weights
    sections_small = {
        ASI_SECTION_MODEL_CONFIG: config_data,
        ASI_SECTION_MEMORY_FABRIC: bytes(fabric),
        ASI_SECTION_TOKENIZER: bytes(tok_data),
        ASI_SECTION_BYTECODE: bytecode_data,
        ASI_SECTION_HARNESS: harness_data,
        ASI_SECTION_PERSONALITY: personality_data,
        ASI_SECTION_METADATA: metadata_data,
    }
    
    # Calculate offsets — weights section is the big one
    n_sections = len(sections_small) + 1  # +1 for weights
    header_size = 64
    section_table_size = n_sections * 32
    first_offset = page_align(header_size + section_table_size)
    
    # Order: model_config, weights, fabric, tokenizer, bytecode, harness, personality, metadata
    all_sections = [(ASI_SECTION_MODEL_CONFIG, config_data)]
    all_sections.append((ASI_SECTION_WEIGHTS, None))  # placeholder — streamed from tmp file
    all_sections.append((ASI_SECTION_MEMORY_FABRIC, bytes(fabric)))
    all_sections.append((ASI_SECTION_TOKENIZER, bytes(tok_data)))
    all_sections.append((ASI_SECTION_BYTECODE, bytecode_data))
    all_sections.append((ASI_SECTION_HARNESS, harness_data))
    all_sections.append((ASI_SECTION_PERSONALITY, personality_data))
    all_sections.append((ASI_SECTION_METADATA, metadata_data))
    
    offsets = []
    current = first_offset
    for stype, sdata in all_sections:
        offsets.append(current)
        if sdata is None:
            current = page_align(current + weights_data_size)
        else:
            current = page_align(current + len(sdata))
    total_size = current
    
    flags = ASI_FLAG_QUANT_INT4 | ASI_FLAG_TOKENIZER_BPE | ASI_FLAG_HOT_RELOAD | \
            ASI_FLAG_HAS_FABRIC | ASI_FLAG_HAS_BYTECODE | ASI_FLAG_HAS_HARNESS | ASI_FLAG_HAS_PERSONALITY
    identity_hash = hashlib.sha256(personality_data).digest()
    
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('IIIIQq32s', ASI_MAGIC, ASI_VERSION, flags, n_sections, total_size, header_size, identity_hash))
        
        # Section table
        for i, (stype, sdata) in enumerate(all_sections):
            size = weights_data_size if sdata is None else len(sdata)
            checksum = 0  # Skip checksum for speed
            f.write(struct.pack('IIQQQ', stype, 0, offsets[i], size, checksum))
        
        # Section data
        for i, (stype, sdata) in enumerate(all_sections):
            # Pad to page alignment
            pos = f.tell()
            aligned = page_align(pos)
            if aligned > pos:
                f.write(b'\x00' * (aligned - pos))
            
            if sdata is None:
                # Stream weights from tmp file
                with open(weights_tmp, 'rb') as wf:
                    while True:
                        chunk = wf.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                f.write(sdata)
    
    # Cleanup
    os.remove(weights_tmp)
    
    actual_size = os.path.getsize(output_path)
    print(f"\n✅ Done!")
    print(f"   Output: {output_path}")
    print(f"   Size: {actual_size/1e9:.2f} GB")
    print(f"   Format: Q4_K native (same quality as llama.cpp)")
    print(f"   Run: lila-asi.exe {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGUF → ASI (v2, native Q4_K)")
    parser.add_argument("--gguf", required=True, help="Input GGUF file")
    parser.add_argument("--output", default="lila.asi", help="Output .asi file")
    args = parser.parse_args()
    gguf_to_asi(args.gguf, args.output)
