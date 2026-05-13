#!/usr/bin/env python3
"""
Test: Create a minimal .asi file WITHOUT a real model.
Uses synthetic/tiny weights to validate the packing logic end-to-end.
"""
import sys
sys.path.insert(0, '/app/lila/engine/asi')

import struct
import numpy as np
import hashlib
import json
import time
import os

# Import constants from pack_asi
ASI_MAGIC = 0x41534921
ASI_VERSION = 1
ASI_PAGE_SIZE = 4096

ASI_SECTION_MODEL_CONFIG  = 0x01
ASI_SECTION_WEIGHTS       = 0x02
ASI_SECTION_MEMORY_FABRIC = 0x03
ASI_SECTION_TOKENIZER     = 0x04
ASI_SECTION_BYTECODE      = 0x05
ASI_SECTION_HARNESS       = 0x06
ASI_SECTION_PERSONALITY   = 0x07
ASI_SECTION_METADATA      = 0x08

ASI_FLAG_HAS_FABRIC      = (1 << 0)
ASI_FLAG_HAS_BYTECODE    = (1 << 1)
ASI_FLAG_HAS_HARNESS     = (1 << 2)
ASI_FLAG_HAS_PERSONALITY = (1 << 3)
ASI_FLAG_QUANT_INT4      = (1 << 4)
ASI_FLAG_TOKENIZER_BPE   = (1 << 5)
ASI_FLAG_HOT_RELOAD      = (1 << 6)


def page_align(x):
    return (x + ASI_PAGE_SIZE - 1) & ~(ASI_PAGE_SIZE - 1)

def crc64(data):
    return struct.unpack("Q", hashlib.sha256(data).digest()[:8])[0]


def create_test_asi(output_path):
    """Create a tiny .asi file with synthetic data for testing."""
    
    # Tiny model config (2 layers, hidden=64, vocab=32)
    n_layers = 2
    hidden = 64
    intermediate = 128
    n_heads = 4
    n_kv_heads = 2
    vocab_size = 32
    max_seq = 128
    head_dim = hidden // n_heads
    
    print("Creating minimal test .asi file...")
    print(f"  Config: {n_layers}L, h={hidden}, vocab={vocab_size}")
    
    # ── Section 1: MODEL_CONFIG ──
    config_data = struct.pack("IIIIIIIIffII",
        n_layers, hidden, intermediate, n_heads, n_kv_heads,
        vocab_size, max_seq, head_dim,
        10000.0,  # rope_theta
        1e-6,     # rms_norm_eps
        3,        # quant_type = INT4
        128,      # group_size
    )
    print(f"  MODEL_CONFIG: {len(config_data)} bytes")
    
    # ── Section 2: WEIGHTS (synthetic tiny weights) ──
    weights_data = bytearray()
    
    # Token embedding (FP32): vocab_size × hidden
    embed = np.random.randn(vocab_size, hidden).astype(np.float32) * 0.02
    weights_data.extend(embed.tobytes())
    
    # Per-layer weights (simplified — just store raw FP32 for test)
    for layer in range(n_layers):
        # For each projection, store rows/cols + dummy data
        for proj in range(7):  # q,k,v,o,gate,up,down
            if proj < 4:
                rows, cols = hidden, hidden
            else:
                rows, cols = intermediate, hidden
            # Store as "quantized" — just rows/cols + codebook + scales + packed
            weights_data.extend(struct.pack("ii", rows, cols))
            # Codebook (16 floats)
            cb = np.linspace(-1, 1, 16).astype(np.float32)
            weights_data.extend(cb.tobytes())
            # Scales (n_groups)
            n_elements = rows * cols
            n_groups = (n_elements + 127) // 128
            scales = np.ones(n_groups, dtype=np.float32) * 0.01
            weights_data.extend(scales.tobytes())
            # Packed indices (n_elements / 2 bytes)
            packed = np.zeros((n_elements + 1) // 2, dtype=np.uint8)
            weights_data.extend(packed.tobytes())
        
        # Layer norms (2 × hidden FP32)
        weights_data.extend(np.ones(hidden, dtype=np.float32).tobytes())
        weights_data.extend(np.ones(hidden, dtype=np.float32).tobytes())
    
    # Final norm
    weights_data.extend(np.ones(hidden, dtype=np.float32).tobytes())
    # LM head tied flag
    weights_data.extend(struct.pack("I", 0xFFFFFFFF))
    
    weights_data = bytes(weights_data)
    print(f"  WEIGHTS: {len(weights_data)} bytes")
    
    # ── Section 3: MEMORY_FABRIC ──
    fabric_data = bytearray()
    fabric_data.extend(struct.pack("IIII", 5, n_layers, 8, 0))  # header
    for layer in range(n_layers):
        for ns in range(5):
            # Empty adapters (rank=0)
            fabric_data.extend(struct.pack("IIIf", 0, 0, 0, 0.0))
    fabric_data = bytes(fabric_data)
    print(f"  MEMORY_FABRIC: {len(fabric_data)} bytes")
    
    # ── Section 4: TOKENIZER ──
    tok_data = bytearray()
    tok_data.extend(struct.pack("IIIIII", vocab_size, 0, 1, 2, 0, 0))  # header
    # Vocab entries
    for i in range(vocab_size):
        token = f"tok_{i}".encode('utf-8')
        tok_data.extend(struct.pack("H", len(token)))
        tok_data.extend(token)
    tok_data = bytes(tok_data)
    print(f"  TOKENIZER: {len(tok_data)} bytes")
    
    # ── Section 5: BYTECODE ──
    # Import the bytecode builder
    from pack_asi import build_bytecode_section
    bytecode_data = build_bytecode_section()
    print(f"  BYTECODE: {len(bytecode_data)} bytes")
    
    # ── Section 6: HARNESS ──
    from pack_asi import build_harness_section
    harness_data = build_harness_section()
    print(f"  HARNESS: {len(harness_data)} bytes")
    
    # ── Section 7: PERSONALITY ──
    from pack_asi import build_personality_section
    personality_data = build_personality_section(interactions=42)
    print(f"  PERSONALITY: {len(personality_data)} bytes")
    
    # ── Section 8: METADATA ──
    from pack_asi import build_metadata_section
    metadata_data = build_metadata_section("test-model", "Test Packer v1")
    print(f"  METADATA: {len(metadata_data)} bytes")
    
    # ── Assemble the .asi file ──
    sections = {
        ASI_SECTION_MODEL_CONFIG: config_data,
        ASI_SECTION_WEIGHTS: weights_data,
        ASI_SECTION_MEMORY_FABRIC: fabric_data,
        ASI_SECTION_TOKENIZER: tok_data,
        ASI_SECTION_BYTECODE: bytecode_data,
        ASI_SECTION_HARNESS: harness_data,
        ASI_SECTION_PERSONALITY: personality_data,
        ASI_SECTION_METADATA: metadata_data,
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
    
    identity_hash = hashlib.sha256(personality_data).digest()
    
    with open(output_path, 'wb') as f:
        # Header (64 bytes)
        f.write(struct.pack("IIIIQq32s",
            ASI_MAGIC, ASI_VERSION, flags, n_sections,
            total_size, header_size, identity_hash
        ))
        
        # Section table
        for i, (stype, sdata) in enumerate(section_list):
            f.write(struct.pack("IIQQQ",
                stype, 0, offsets[i], len(sdata), crc64(sdata)
            ))
        
        # Section data (page-aligned)
        for i, (stype, sdata) in enumerate(section_list):
            pos = f.tell()
            aligned = page_align(pos)
            if aligned > pos:
                f.write(b'\x00' * (aligned - pos))
            f.write(sdata)
    
    actual_size = os.path.getsize(output_path)
    print(f"\n✅ Test .asi created: {output_path}")
    print(f"   Size: {actual_size} bytes ({actual_size/1024:.1f} KB)")
    
    # Verify magic
    with open(output_path, 'rb') as f:
        magic = struct.unpack("I", f.read(4))[0]
        assert magic == ASI_MAGIC, f"Bad magic: 0x{magic:08X}"
        print(f"   Magic: 0x{magic:08X} ✓")
    
    return output_path


if __name__ == "__main__":
    path = create_test_asi("/tmp/test_lila.asi")
