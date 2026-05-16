#!/usr/bin/env python3
"""
GGUF → ASI Converter v3 (FIXED)

Correctly handles Gemma 3 4B:
- 34 layers, 8 heads, 4 KV heads, 2560 hidden, 10240 intermediate
- Embedding stored as Q6_K (engine dequants per-token on lookup)
- Layer weights stored as raw Q4_K/Q6_K blocks
- Proper data offsets read directly from GGUF file (not the broken gguf library .data)

Usage:
    python gguf_to_asi_v3.py --gguf path/to/gemma.gguf --output lila.asi
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_asi import (
    ASI_MAGIC, ASI_VERSION, ASI_PAGE_SIZE,
    ASI_SECTION_MODEL_CONFIG, ASI_SECTION_WEIGHTS, ASI_SECTION_MEMORY_FABRIC,
    ASI_SECTION_TOKENIZER, ASI_SECTION_BYTECODE, ASI_SECTION_HARNESS,
    ASI_SECTION_PERSONALITY, ASI_SECTION_METADATA,
    ASI_FLAG_HAS_FABRIC, ASI_FLAG_HAS_BYTECODE, ASI_FLAG_HAS_HARNESS,
    ASI_FLAG_HAS_PERSONALITY, ASI_FLAG_QUANT_INT4, ASI_FLAG_TOKENIZER_BPE,
    ASI_FLAG_HOT_RELOAD,
    page_align, crc64,
    build_bytecode_section, build_harness_section,
    build_personality_section, build_metadata_section,
)

try:
    from gguf import GGUFReader, dequantize
except ImportError:
    print("ERROR: pip install gguf")
    sys.exit(1)

# Quant types (must match model.h)
QUANT_NONE = 0
QUANT_FIGQUANT = 1
QUANT_Q4_K = 2
QUANT_Q6_K = 3
QUANT_F16 = 4
QUANT_Q6_K_EMBED = 5  # Special: Q6_K stored for embedding (dequant on lookup)

# GGUF type IDs
GGUF_F32 = 0
GGUF_F16 = 1
GGUF_Q4_K = 12
GGUF_Q6_K = 14

# Block sizes
Q4_K_BLOCK = 144  # bytes per 256 elements
Q6_K_BLOCK = 210  # bytes per 256 elements


def get_tensor_raw(gguf_path, tensor):
    """Read raw tensor data directly from the GGUF file at correct offset."""
    with open(gguf_path, 'rb') as f:
        f.seek(tensor.data_offset)
        # Calculate expected size based on type
        n_elements = int(tensor.n_elements)
        t = tensor.tensor_type.value if hasattr(tensor.tensor_type, 'value') else int(tensor.tensor_type)
        
        if t == GGUF_Q4_K:
            n_blocks = (n_elements + 255) // 256
            size = n_blocks * Q4_K_BLOCK
        elif t == GGUF_Q6_K:
            n_blocks = (n_elements + 255) // 256
            size = n_blocks * Q6_K_BLOCK
        elif t == GGUF_F32:
            size = n_elements * 4
        elif t == GGUF_F16:
            size = n_elements * 2
        else:
            # Fallback estimate
            n_blocks = (n_elements + 255) // 256
            size = n_blocks * Q4_K_BLOCK  # assume Q4_K
        
        data = f.read(size)
        return data


def convert(gguf_path: str, output_path: str):
    """Convert GGUF to .asi with correct data."""
    
    print(f"\n🌸 GGUF → ASI v3 (correct)")
    print(f"   Input:  {gguf_path}")
    print(f"   Output: {output_path}\n")
    
    reader = GGUFReader(gguf_path)
    
    # Build tensor map
    tensor_map = {}
    for t in reader.tensors:
        tensor_map[t.name] = t
    
    # Extract config
    fields = reader.fields
    def get_field(name):
        if name in fields:
            f = fields[name]
            if hasattr(f, 'parts') and len(f.parts) > 1:
                return int(f.parts[-1][0])
        return None
    
    # Detect architecture prefix
    arch = 'gemma3'
    for prefix in ['gemma3', 'gemma2', 'llama', 'phi']:
        if f'{prefix}.block_count' in fields:
            arch = prefix
            break
    
    n_layers = get_field(f'{arch}.block_count') or 34
    hidden = get_field(f'{arch}.embedding_length') or 2560
    intermediate = get_field(f'{arch}.feed_forward_length') or 10240
    n_heads = get_field(f'{arch}.attention.head_count') or 8
    n_kv_heads = get_field(f'{arch}.attention.head_count_kv') or 4
    head_dim = hidden // n_heads
    
    # Correct head_dim from actual Q projection shape if available
    q_tensor = tensor_map.get('blk.0.attn_q.weight')
    if q_tensor is not None:
        q_out = int(q_tensor.shape[1])  # output dim of Q proj
        actual_head_dim = q_out // n_heads
        if actual_head_dim != head_dim:
            print(f"  Correcting head_dim: {head_dim} → {actual_head_dim} (from Q proj shape)")
            head_dim = actual_head_dim
    vocab_size = get_field(f'{arch}.vocab_size') or 262144
    max_seq = min(get_field(f'{arch}.context_length') or 8192, 8192)
    rope_theta = 10000.0
    rms_eps = 1e-6
    
    # Correct vocab from embedding shape
    embed_t = tensor_map.get('token_embd.weight')
    if embed_t is not None:
        v = max(int(embed_t.shape[0]), int(embed_t.shape[1]))
        if v > 1000:
            vocab_size = v
    
    print(f"Config:")
    print(f"  Layers: {n_layers}, Hidden: {hidden}, Intermediate: {intermediate}")
    print(f"  Heads: {n_heads}, KV Heads: {n_kv_heads}, Head dim: {head_dim}")
    print(f"  Vocab: {vocab_size}, Max seq: {max_seq}")
    print()
    
    # ── Write weights to temp file (streaming, too large for RAM) ──
    weights_tmp = output_path + '.weights.tmp'
    
    print("Writing weights...")
    with open(weights_tmp, 'wb') as wf:
        bytes_written = 0
        
        # 1. Token embedding — store as Q6_K raw (engine will dequant per-token)
        print(f"  Embedding [{vocab_size} x {hidden}] as Q6_K...")
        embed_data = get_tensor_raw(gguf_path, embed_t)
        
        # Write embedding header: vocab_size, hidden, quant_type, then raw data
        wf.write(struct.pack('III', vocab_size, hidden, QUANT_Q6_K_EMBED))
        bytes_written += 12
        wf.write(embed_data)
        bytes_written += len(embed_data)
        print(f"    Wrote {len(embed_data)/1e6:.1f} MB embedding data")
        
        # 2. Per-layer weights
        # GGUF tensor names for Gemma3:
        #   blk.{i}.attn_q.weight      → q_proj  [hidden, n_heads*head_dim]
        #   blk.{i}.attn_k.weight      → k_proj  [hidden, n_kv_heads*head_dim]
        #   blk.{i}.attn_v.weight      → v_proj  [hidden, n_kv_heads*head_dim]
        #   blk.{i}.attn_output.weight  → o_proj  [n_heads*head_dim, hidden]
        #   blk.{i}.ffn_gate.weight     → gate    [hidden, intermediate]
        #   blk.{i}.ffn_up.weight       → up      [hidden, intermediate]
        #   blk.{i}.ffn_down.weight     → down    [intermediate, hidden]
        
        proj_names = [
            'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
            'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight'
        ]
        
        for layer_idx in range(n_layers):
            for proj_name in proj_names:
                full_name = f'blk.{layer_idx}.{proj_name}'
                t = tensor_map.get(full_name)
                
                if t is None:
                    # Write empty marker
                    wf.write(struct.pack('III', 0, 0, 0))
                    bytes_written += 12
                    continue
                
                # GGUF stores [cols, rows] (transposed)
                # For matvec: out = W @ x, W is [out_features, in_features]
                rows = int(t.shape[1])  # out_features
                cols = int(t.shape[0])  # in_features
                
                # Determine quant type
                t_type = t.tensor_type.value if hasattr(t.tensor_type, 'value') else int(t.tensor_type)
                if t_type == GGUF_Q4_K:
                    asi_quant = QUANT_Q4_K
                elif t_type == GGUF_Q6_K:
                    asi_quant = QUANT_Q6_K
                else:
                    asi_quant = QUANT_Q4_K  # fallback
                
                # Write header: rows, cols, quant_type
                wf.write(struct.pack('III', rows, cols, asi_quant))
                bytes_written += 12
                
                # Write raw quantized data
                raw_data = get_tensor_raw(gguf_path, t)
                wf.write(raw_data)
                bytes_written += len(raw_data)
            
            # Layer norms (FP32)
            for norm_name in ['attn_norm.weight', 'post_ffw_norm.weight']:
                full_name = f'blk.{layer_idx}.{norm_name}'
                t = tensor_map.get(full_name)
                if t is not None:
                    norm_data = get_tensor_raw(gguf_path, t)
                    wf.write(norm_data)
                    bytes_written += len(norm_data)
                else:
                    wf.write(np.ones(hidden, dtype=np.float32).tobytes())
                    bytes_written += hidden * 4
            
            if (layer_idx + 1) % 5 == 0:
                print(f"    Layer {layer_idx+1}/{n_layers} ({bytes_written/1e9:.2f} GB)")
        
        # 3. Final norm
        final_norm_t = tensor_map.get('output_norm.weight')
        if final_norm_t:
            norm_data = get_tensor_raw(gguf_path, final_norm_t)
            wf.write(norm_data)
            bytes_written += len(norm_data)
        else:
            wf.write(np.ones(hidden, dtype=np.float32).tobytes())
            bytes_written += hidden * 4
        
        # 4. LM Head — tied to embedding (flag)
        wf.write(struct.pack('I', 0xFFFFFFFF))
        bytes_written += 4
        
        print(f"  Total weights: {bytes_written/1e9:.2f} GB")
    
    # ── Build other sections ──
    print("\nBuilding sections...")
    
    config_data = struct.pack('IIIIIIIIffII',
        n_layers, hidden, intermediate, n_heads, n_kv_heads, vocab_size,
        max_seq, head_dim, rope_theta, rms_eps,
        QUANT_Q4_K, 128)
    
    # Memory Fabric (empty — ready for training)
    fabric = bytearray(struct.pack('IIII', 5, n_layers, 8, 0))
    for _ in range(n_layers * 5):
        fabric.extend(struct.pack('IIIf', 0, 0, 0, 0.0))
    
    # Tokenizer from GGUF
    print("  Extracting tokenizer...")
    tok_data = bytearray()
    
    # Get vocab from GGUF tokens field
    tokens_field = fields.get('tokenizer.ggml.tokens')
    n_merges = 0
    bos_id = 2
    eos_id = 1
    
    if tokens_field and hasattr(tokens_field, 'parts'):
        # parts[0] = array of string lengths, parts[1..] = string data
        actual_vocab = len(tokens_field.parts) - 1  # rough estimate
        tok_data.extend(struct.pack('IIIIII', vocab_size, n_merges, bos_id, eos_id, 0, 0))
        
        # Write token strings
        for i in range(min(vocab_size, 512000)):
            try:
                if i + 1 < len(tokens_field.parts):
                    token_bytes = bytes(tokens_field.parts[i + 1])
                else:
                    token_bytes = f"<tok_{i}>".encode()
            except:
                token_bytes = f"<tok_{i}>".encode()
            tok_data.extend(struct.pack('H', len(token_bytes)))
            tok_data.extend(token_bytes)
    else:
        # Fallback: byte-level tokenizer
        tok_data.extend(struct.pack('IIIIII', vocab_size, 0, 2, 1, 0, 0))
        for i in range(min(vocab_size, 256)):
            token = bytes([i]) if 32 <= i < 127 else f"<{i:02X}>".encode()
            tok_data.extend(struct.pack('H', len(token)))
            tok_data.extend(token)
    
    bytecode_data = build_bytecode_section()
    harness_data = build_harness_section()
    personality_data = build_personality_section()
    metadata_data = build_metadata_section(gguf_path, 'gguf_to_asi v3')
    
    # ── Assemble final .asi ──
    print("\nAssembling .asi...")
    
    weights_size = os.path.getsize(weights_tmp)
    
    sections = {
        ASI_SECTION_MODEL_CONFIG: config_data,
        # WEIGHTS will be streamed from temp file
        ASI_SECTION_MEMORY_FABRIC: bytes(fabric),
        ASI_SECTION_TOKENIZER: bytes(tok_data),
        ASI_SECTION_BYTECODE: bytecode_data,
        ASI_SECTION_HARNESS: harness_data,
        ASI_SECTION_PERSONALITY: personality_data,
        ASI_SECTION_METADATA: metadata_data,
    }
    
    # Calculate layout
    n_sections = len(sections) + 1  # +1 for weights
    header_size = 64
    table_size = n_sections * 32
    first_offset = page_align(header_size + table_size)
    
    # Weights go first (largest section)
    weights_offset = first_offset
    current = page_align(weights_offset + weights_size)
    
    section_list = sorted(sections.items())
    offsets = {}
    offsets[ASI_SECTION_WEIGHTS] = weights_offset
    for stype, sdata in section_list:
        offsets[stype] = current
        current = page_align(current + len(sdata))
    
    total_size = current
    
    # Write final file
    flags = ASI_FLAG_QUANT_INT4 | ASI_FLAG_TOKENIZER_BPE | ASI_FLAG_HOT_RELOAD | \
            ASI_FLAG_HAS_FABRIC | ASI_FLAG_HAS_BYTECODE | ASI_FLAG_HAS_HARNESS | ASI_FLAG_HAS_PERSONALITY
    identity_hash = hashlib.sha256(personality_data).digest()
    
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('IIIIQq32s',
            ASI_MAGIC, ASI_VERSION, flags, n_sections,
            total_size, header_size, identity_hash))
        
        # Section table
        # Weights entry
        wchk = 0  # Skip checksum for huge section
        f.write(struct.pack('IIQQQ', ASI_SECTION_WEIGHTS, 0, weights_offset, weights_size, wchk))
        # Other sections
        for stype, sdata in section_list:
            chk = struct.unpack('Q', hashlib.sha256(sdata).digest()[:8])[0]
            f.write(struct.pack('IIQQQ', stype, 0, offsets[stype], len(sdata), chk))
        
        # Pad to weights offset
        pos = f.tell()
        f.write(b'\x00' * (weights_offset - pos))
        
        # Stream weights from temp file
        print(f"  Streaming weights ({weights_size/1e9:.2f} GB)...")
        with open(weights_tmp, 'rb') as wf:
            while True:
                chunk = wf.read(64 * 1024 * 1024)  # 64MB chunks
                if not chunk:
                    break
                f.write(chunk)
        
        # Other sections
        for stype, sdata in section_list:
            pos = f.tell()
            aligned = page_align(pos)
            if aligned > pos:
                f.write(b'\x00' * (aligned - pos))
            f.write(sdata)
    
    # Cleanup
    os.unlink(weights_tmp)
    
    final_size = os.path.getsize(output_path)
    print(f"\n✅ Done!")
    print(f"   Output: {output_path}")
    print(f"   Size: {final_size/1e9:.2f} GB")
    print(f"   Config: {n_layers}L, h={hidden}, heads={n_heads}/{n_kv_heads}")
    print(f"   Repack with: lila-asi.exe {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGUF → ASI v3")
    parser.add_argument("--gguf", required=True, help="Input GGUF file")
    parser.add_argument("--output", default="lila.asi", help="Output .asi file")
    args = parser.parse_args()
    convert(args.gguf, args.output)
