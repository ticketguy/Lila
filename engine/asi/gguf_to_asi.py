#!/usr/bin/env python3
"""
GGUF → ASI Converter

Convert an existing .gguf file directly to .asi format.
No need to download anything — use the model you already have.

Usage:
    python gguf_to_asi.py --gguf ~/models/gemma-4b-q4.gguf --output lila.asi

This reads the GGUF file (weights + tokenizer + config), repackages everything
into the .asi format with bytecode kernels, harness, and personality.
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

# Add parent for imports
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
    build_memory_fabric_section,
)

try:
    from gguf import GGUFReader
except ImportError:
    print("ERROR: 'gguf' package required. Install with: pip install gguf")
    sys.exit(1)


def read_gguf_metadata(reader):
    """Extract model config from GGUF metadata fields."""
    meta = {}
    for field in reader.fields.values():
        # Extract key metadata
        name = field.name if hasattr(field, 'name') else str(field)
        try:
            if field.types and len(field.data) > 0:
                # Get the actual value
                if hasattr(field, 'parts'):
                    val = field.parts[-1][0] if len(field.parts) > 0 else None
                else:
                    val = None
                meta[name] = val
        except (IndexError, TypeError):
            pass
    return meta


def extract_config_from_gguf(reader):
    """Parse GGUF fields into model config dict."""
    fields = reader.fields
    
    def get_field_value(name):
        """Get a scalar value from a GGUF field."""
        if name not in fields:
            return None
        field = fields[name]
        if hasattr(field, 'parts') and len(field.parts) > 1:
            data = field.parts[-1]
            if len(data) > 0:
                return int(data[0])
        return None
    
    def get_field_float(name):
        """Get a float value from a GGUF field."""
        if name not in fields:
            return None
        field = fields[name]
        if hasattr(field, 'parts') and len(field.parts) > 1:
            data = field.parts[-1]
            if len(data) > 0:
                return float(data[0])
        return None
    
    def get_field_string(name):
        """Get a string value from a GGUF field."""
        if name not in fields:
            return None
        field = fields[name]
        if hasattr(field, 'parts') and len(field.parts) > 1:
            data = field.parts[-1]
            if hasattr(data, 'tobytes'):
                return data.tobytes().decode('utf-8', errors='replace')
        return None
    
    # Standard GGUF metadata keys
    n_layers = get_field_value('llama.block_count') or \
               get_field_value('gemma.block_count') or \
               get_field_value('qwen2.block_count') or \
               get_field_value('phi3.block_count') or 26
    
    hidden = get_field_value('llama.embedding_length') or \
             get_field_value('gemma.embedding_length') or \
             get_field_value('qwen2.embedding_length') or \
             get_field_value('phi3.embedding_length') or 2048
    
    intermediate = get_field_value('llama.feed_forward_length') or \
                   get_field_value('gemma.feed_forward_length') or \
                   get_field_value('qwen2.feed_forward_length') or \
                   int(hidden * 2.67)
    
    n_heads = get_field_value('llama.attention.head_count') or \
              get_field_value('gemma.attention.head_count') or \
              get_field_value('qwen2.attention.head_count') or 16
    
    n_kv_heads = get_field_value('llama.attention.head_count_kv') or \
                 get_field_value('gemma.attention.head_count_kv') or \
                 get_field_value('qwen2.attention.head_count_kv') or n_heads
    
    vocab_size = get_field_value('llama.vocab_size') or \
                 get_field_value('gemma.vocab_size') or 32000
    # Also check from tokenizer tokens list
    if 'tokenizer.ggml.tokens' in fields:
        field = fields['tokenizer.ggml.tokens']
        if hasattr(field, 'parts') and len(field.parts) > 1:
            vocab_size = max(vocab_size or 0, len(field.parts) - 1)
    
    max_seq = get_field_value('llama.context_length') or \
              get_field_value('gemma.context_length') or \
              get_field_value('qwen2.context_length') or 4096
    
    rope_theta = get_field_float('llama.rope.freq_base') or \
                 get_field_float('gemma.rope.freq_base') or 10000.0
    
    rms_eps = get_field_float('llama.attention.layer_norm_rms_epsilon') or \
              get_field_float('gemma.attention.layer_norm_rms_epsilon') or 1e-6
    
    arch = get_field_string('general.architecture') or "unknown"
    name = get_field_string('general.name') or "unknown"
    
    config = {
        'n_layers': n_layers,
        'hidden_size': hidden,
        'intermediate_size': intermediate,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'vocab_size': vocab_size,
        'max_seq_len': max_seq,
        'rope_theta': rope_theta,
        'rms_norm_eps': rms_eps,
        'group_size': 128,
        'arch': arch,
        'name': name,
    }
    
    return config


def extract_tokenizer_from_gguf(reader, config):
    """Extract tokenizer vocab from GGUF into ASI tokenizer section format."""
    fields = reader.fields
    
    data = bytearray()
    
    vocab_size = config['vocab_size']
    
    # Get special token IDs
    def get_id(name, default):
        if name in fields and hasattr(fields[name], 'parts') and len(fields[name].parts) > 1:
            try:
                return int(fields[name].parts[-1][0])
            except (IndexError, TypeError, ValueError):
                pass
        return default
    
    bos_id = get_id('tokenizer.ggml.bos_token_id', 1)
    eos_id = get_id('tokenizer.ggml.eos_token_id', 2)
    pad_id = get_id('tokenizer.ggml.padding_token_id', 0)
    
    # Get token list
    tokens = []
    if 'tokenizer.ggml.tokens' in fields:
        field = fields['tokenizer.ggml.tokens']
        if hasattr(field, 'parts'):
            # GGUF stores tokens as array of strings
            # parts[0] = length array, parts[1+] = string data
            for i in range(1, len(field.parts)):
                tok_bytes = field.parts[i].tobytes() if hasattr(field.parts[i], 'tobytes') else b''
                tokens.append(tok_bytes)
    
    if not tokens:
        # Fallback: generate placeholder tokens
        tokens = [f"<tok_{i}>".encode('utf-8') for i in range(vocab_size)]
    
    actual_vocab = len(tokens)
    
    # Get merges
    merges = []
    if 'tokenizer.ggml.merges' in fields:
        field = fields['tokenizer.ggml.merges']
        if hasattr(field, 'parts'):
            for i in range(1, len(field.parts)):
                merge_bytes = field.parts[i].tobytes() if hasattr(field.parts[i], 'tobytes') else b''
                merges.append(merge_bytes)
    
    # Tokenizer header
    data.extend(struct.pack("IIIIII",
        actual_vocab,
        len(merges),
        bos_id,
        eos_id,
        pad_id,
        0,  # reserved
    ))
    
    # Vocab entries
    for tok_bytes in tokens:
        data.extend(struct.pack("H", len(tok_bytes)))
        data.extend(tok_bytes)
    
    # Merges (store as raw bytes for now)
    # In GGUF, merges are stored as "token1 token2" strings
    for merge_bytes in merges[:65536]:
        # Store as: left_id=0, right_id=0, merged_id=0, priority (simplified)
        data.extend(struct.pack("IIIf", 0, 0, 0, 1.0))
    
    print(f"  Tokenizer: vocab={actual_vocab}, merges={len(merges)}, "
          f"bos={bos_id}, eos={eos_id}")
    
    return bytes(data)


def extract_weights_from_gguf(reader, config):
    """
    Extract weight tensors from GGUF and repackage for ASI.
    
    GGUF already stores quantized weights. We can either:
    1. Re-dequantize and re-quantize with FigQuant (lossy but consistent)
    2. Store the raw GGUF quant data and dequantize at runtime (faster to pack)
    
    For now, approach 2: we store weights in a format compatible with
    the ASI runtime, preserving the GGUF quantization where possible.
    """
    from gguf import dequantize
    
    data = bytearray()
    hidden = config['hidden_size']
    vocab_size = config['vocab_size']
    n_layers = config['n_layers']
    
    # Find tensors by name
    tensor_map = {}
    for tensor in reader.tensors:
        tensor_map[tensor.name] = tensor
    
    print(f"  Found {len(tensor_map)} tensors in GGUF")
    
    # Token embedding
    embed_name = None
    for name in ['token_embd.weight', 'model.embed_tokens.weight']:
        if name in tensor_map:
            embed_name = name
            break
    
    if embed_name:
        tensor = tensor_map[embed_name]
        embed = dequantize(tensor.data, tensor.tensor_type)
        # Infer shape from the tensor itself — don't trust metadata vocab_size
        # GGUF stores shape reversed
        if hasattr(tensor, 'shape') and len(tensor.shape) >= 2:
            shape = list(reversed(tensor.shape.tolist()))
        else:
            # Fallback: infer from total elements
            n_elements = embed.size
            embed_vocab = n_elements // hidden
            shape = [embed_vocab, hidden]
        embed = embed.reshape(shape[0], shape[1]).astype(np.float32)
        # Update vocab_size to match actual embedding dimension
        actual_embed_vocab = shape[0]
        if actual_embed_vocab != vocab_size:
            print(f"  NOTE: Embedding vocab ({actual_embed_vocab}) differs from metadata ({vocab_size})")
            print(f"        Using embedding dimension for weight layout")
        vocab_size = actual_embed_vocab
        config['vocab_size'] = vocab_size
        data.extend(embed.tobytes())
        print(f"  Embedding: {embed.shape} ({embed.nbytes/1e6:.1f} MB)")
    else:
        # Zero embedding as placeholder
        embed = np.zeros((vocab_size, hidden), dtype=np.float32)
        data.extend(embed.tobytes())
        print(f"  Embedding: not found, using zeros")
    
    # Layer weights
    layer_patterns = {
        'q_proj': ['blk.{}.attn_q.weight', 'blk.{}.self_attn.q_proj.weight'],
        'k_proj': ['blk.{}.attn_k.weight', 'blk.{}.self_attn.k_proj.weight'],
        'v_proj': ['blk.{}.attn_v.weight', 'blk.{}.self_attn.v_proj.weight'],
        'o_proj': ['blk.{}.attn_output.weight', 'blk.{}.self_attn.o_proj.weight'],
        'gate_proj': ['blk.{}.ffn_gate.weight', 'blk.{}.mlp.gate_proj.weight'],
        'up_proj': ['blk.{}.ffn_up.weight', 'blk.{}.mlp.up_proj.weight'],
        'down_proj': ['blk.{}.ffn_down.weight', 'blk.{}.mlp.down_proj.weight'],
    }
    
    norm_patterns = {
        'input_layernorm': ['blk.{}.attn_norm.weight', 'blk.{}.input_layernorm.weight'],
        'post_attn_norm': ['blk.{}.ffn_norm.weight', 'blk.{}.post_attention_layernorm.weight'],
    }
    
    GROUP_SIZE = 128
    
    for layer_idx in range(n_layers):
        # Projection weights (quantize with FigQuant)
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            patterns = layer_patterns[proj_name]
            found = False
            
            for pattern in patterns:
                tname = pattern.format(layer_idx)
                if tname in tensor_map:
                    tensor = tensor_map[tname]
                    # Dequantize from GGUF format
                    w = dequantize(tensor.data, tensor.tensor_type)
                    # Infer shape from tensor metadata
                    if hasattr(tensor, 'shape') and len(tensor.shape) >= 2:
                        shape = list(reversed(tensor.shape.tolist()))
                    else:
                        # 2D weight: guess square-ish or hidden×hidden
                        n_el = w.size
                        if n_el % hidden == 0:
                            shape = [n_el // hidden, hidden]
                        else:
                            shape = [int(n_el**0.5), int(n_el**0.5)]
                    w = w.reshape(shape[0], shape[1]).astype(np.float32)
                    rows, cols = w.shape
                    
                    # Re-quantize with FigQuant INT4
                    from pack_asi import quantize_int4
                    packed, codebook, scales = quantize_int4(w, GROUP_SIZE)
                    
                    data.extend(struct.pack("ii", rows, cols))
                    data.extend(codebook.tobytes())
                    data.extend(scales.tobytes())
                    data.extend(packed.tobytes())
                    found = True
                    break
            
            if not found:
                data.extend(struct.pack("ii", 0, 0))
        
        # Layer norms (FP32)
        for norm_name, patterns in norm_patterns.items():
            found = False
            for pattern in patterns:
                tname = pattern.format(layer_idx)
                if tname in tensor_map:
                    tensor = tensor_map[tname]
                    w = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
                    data.extend(w.tobytes())
                    found = True
                    break
            if not found:
                data.extend(np.ones(hidden, dtype=np.float32).tobytes())
        
        if (layer_idx + 1) % 4 == 0:
            print(f"  Layer {layer_idx+1}/{n_layers} done")
    
    # Final norm
    final_norm_names = ['output_norm.weight', 'model.norm.weight']
    found = False
    for name in final_norm_names:
        if name in tensor_map:
            tensor = tensor_map[name]
            w = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
            data.extend(w.tobytes())
            found = True
            break
    if not found:
        data.extend(np.ones(hidden, dtype=np.float32).tobytes())
    
    # LM head
    lm_head_names = ['output.weight', 'lm_head.weight']
    found = False
    for name in lm_head_names:
        if name in tensor_map:
            tensor = tensor_map[name]
            w = dequantize(tensor.data, tensor.tensor_type)
            # Infer shape from tensor metadata
            if hasattr(tensor, 'shape') and len(tensor.shape) >= 2:
                shape = list(reversed(tensor.shape.tolist()))
            else:
                # Fallback: assume [vocab_size, hidden]
                lm_vocab = w.size // hidden
                shape = [lm_vocab, hidden]
            w = w.reshape(shape[0], shape[1]).astype(np.float32)
            data.extend(w.tobytes())
            found = True
            print(f"  LM Head: {shape}")
            break
    if not found:
        data.extend(struct.pack("I", 0xFFFFFFFF))  # Tied with embedding
        print(f"  LM Head: tied with embedding")
    
    return bytes(data)


def gguf_to_asi(gguf_path: str, output_path: str):
    """Convert a GGUF file to ASI format."""
    
    print(f"\n🌸 GGUF → ASI Converter")
    print(f"   Input:  {gguf_path}")
    print(f"   Output: {output_path}\n")
    
    if not os.path.exists(gguf_path):
        print(f"ERROR: File not found: {gguf_path}")
        return
    
    file_size = os.path.getsize(gguf_path)
    print(f"Reading GGUF ({file_size/1e9:.2f} GB)...")
    
    reader = GGUFReader(gguf_path)
    
    # Extract config
    print("\nExtracting config...")
    config = extract_config_from_gguf(reader)
    print(f"  Architecture: {config['arch']}")
    print(f"  Name: {config['name']}")
    print(f"  Layers: {config['n_layers']}, Hidden: {config['hidden_size']}, "
          f"Vocab: {config['vocab_size']}")
    print(f"  Heads: {config['n_heads']}, KV Heads: {config['n_kv_heads']}")
    
    # Build sections
    print("\nBuilding ASI sections...")
    
    print("[1/7] Weights (dequant GGUF → FigQuant INT4)...")
    weights_section = extract_weights_from_gguf(reader, config)
    # NOTE: config['vocab_size'] may have been corrected by extract_weights_from_gguf
    
    print("[2/7] Model config...")
    config_section = struct.pack("IIIIIIIIffII",
        config['n_layers'], config['hidden_size'], config['intermediate_size'],
        config['n_heads'], config['n_kv_heads'], config['vocab_size'],
        config['max_seq_len'], config['hidden_size'] // config['n_heads'],
        config['rope_theta'], config['rms_norm_eps'],
        3, config['group_size'],
    )
    
    print("[3/7] Memory Fabric (empty, ready for training)...")
    fabric_section = build_memory_fabric_section(config)
    
    print("[4/7] Tokenizer...")
    tokenizer_section = extract_tokenizer_from_gguf(reader, config)
    
    print("[5/7] Bytecode kernels...")
    bytecode_section = build_bytecode_section()
    
    print("[6/7] Harness...")
    harness_section = build_harness_section()
    
    print("[7/7] Personality + Metadata...")
    personality_section = build_personality_section()
    base_name = config.get('name', os.path.basename(gguf_path))
    metadata_section = build_metadata_section(base_name, "GGUF→ASI Converter v1.0")
    
    # Assemble
    sections = {
        ASI_SECTION_MODEL_CONFIG: config_section,
        ASI_SECTION_WEIGHTS: weights_section,
        ASI_SECTION_MEMORY_FABRIC: fabric_section,
        ASI_SECTION_TOKENIZER: tokenizer_section,
        ASI_SECTION_BYTECODE: bytecode_section,
        ASI_SECTION_HARNESS: harness_section,
        ASI_SECTION_PERSONALITY: personality_section,
        ASI_SECTION_METADATA: metadata_section,
    }
    
    n_sections = len(sections)
    header_size = 64
    section_table_size = n_sections * 32
    
    section_list = sorted(sections.items())
    first_offset = page_align(header_size + section_table_size)
    
    offsets = []
    current = first_offset
    for stype, sdata in section_list:
        offsets.append(current)
        current = page_align(current + len(sdata))
    total_size = current
    
    flags = (ASI_FLAG_HAS_FABRIC | ASI_FLAG_HAS_BYTECODE | ASI_FLAG_HAS_HARNESS |
             ASI_FLAG_HAS_PERSONALITY | ASI_FLAG_QUANT_INT4 | ASI_FLAG_TOKENIZER_BPE |
             ASI_FLAG_HOT_RELOAD)
    
    identity_hash = hashlib.sha256(personality_section).digest()
    
    print(f"\nWriting {output_path}...")
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack("IIIIQq32s",
            ASI_MAGIC, ASI_VERSION, flags, n_sections,
            total_size, header_size, identity_hash
        ))
        
        # Section table
        for i, (stype, sdata) in enumerate(section_list):
            f.write(struct.pack("IIQQQ",
                stype, 0, offsets[i], len(sdata), crc64(sdata)
            ))
        
        # Section data
        for i, (stype, sdata) in enumerate(section_list):
            pos = f.tell()
            aligned = page_align(pos)
            if aligned > pos:
                f.write(b'\x00' * (aligned - pos))
            f.write(sdata)
    
    actual_size = os.path.getsize(output_path)
    print(f"\n✅ Conversion complete!")
    print(f"   Output: {output_path}")
    print(f"   Size: {actual_size/1e6:.1f} MB ({actual_size/1e9:.2f} GB)")
    print(f"   GGUF: {file_size/1e9:.2f} GB → ASI: {actual_size/1e9:.2f} GB")
    print(f"\n   Run: ./lila-asi {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GGUF to ASI format")
    parser.add_argument("--gguf", required=True, help="Path to .gguf file")
    parser.add_argument("--output", default="lila.asi", help="Output .asi path")
    args = parser.parse_args()
    
    gguf_to_asi(args.gguf, args.output)
