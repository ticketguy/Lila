#!/usr/bin/env python3
"""
ASI Packer v2 — Embeds GGUF for llama.cpp inference

Our custom dequant is broken. llama.cpp's kernels work.
Solution: embed the GGUF inside the .asi, loader extracts it,
llama.cpp does the actual inference.

Usage:
    python pack_asi_v2.py --gguf gemma-4b-Q4_K_M.gguf --output lila.asi
"""

import argparse
import struct
import hashlib
import time
import os
import json
import sys

ASI_MAGIC = 0x41534921
ASI_VERSION = 2
ASI_PAGE_SIZE = 4096

ASI_SECTION_CONFIG        = 0x01
ASI_SECTION_GGUF_BLOB     = 0x02
ASI_SECTION_MEMORY_FABRIC = 0x03
ASI_SECTION_HARNESS       = 0x06
ASI_SECTION_PERSONALITY   = 0x07
ASI_SECTION_METADATA      = 0x08

ASI_FLAG_GGUF_BACKED     = (1 << 8)
ASI_FLAG_HAS_FABRIC      = (1 << 0)
ASI_FLAG_HAS_HARNESS     = (1 << 2)
ASI_FLAG_HAS_PERSONALITY = (1 << 3)


def page_align(x):
    return (x + ASI_PAGE_SIZE - 1) & ~(ASI_PAGE_SIZE - 1)


def pack(gguf_path: str, output_path: str, adapter_path: str = None):
    print(f"\n🌸 ASI Packer v2")
    print(f"   GGUF: {gguf_path}")
    print(f"   Output: {output_path}\n")

    if not os.path.exists(gguf_path):
        sys.exit(f"ERROR: {gguf_path} not found")

    gguf_size = os.path.getsize(gguf_path)
    print(f"   GGUF size: {gguf_size / 1e9:.2f} GB")

    # Verify GGUF magic
    with open(gguf_path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x46554747:
            sys.exit(f"ERROR: Not GGUF (magic 0x{magic:08X})")

    # Build non-GGUF sections
    sections = {}

    sections[ASI_SECTION_CONFIG] = json.dumps({
        "version": 2, "backend": "llama.cpp",
        "gguf_size": gguf_size,
        "mode": "jarvis",
    }).encode()

    if adapter_path and os.path.exists(adapter_path):
        with open(adapter_path, 'rb') as f:
            sections[ASI_SECTION_MEMORY_FABRIC] = f.read()
    else:
        sections[ASI_SECTION_MEMORY_FABRIC] = struct.pack('IIII', 5, 0, 8, 0)

    sections[ASI_SECTION_HARNESS] = json.dumps({
        "tools": ["bash", "file_read", "file_write", "http_request",
                  "tcp_connect", "ssh_exec", "wifi_scan", "wifi_connect",
                  "gpio_write", "gpio_read", "i2c_transfer", "serial_write",
                  "set_volume", "power_action", "notify_user",
                  "memory_store", "memory_recall"],
        "mode": "jarvis",
    }).encode()

    sections[ASI_SECTION_PERSONALITY] = json.dumps({
        "name": "Lila", "family": "Sammie",
        "style": "warm, direct, capable, proactive",
        "mode": "always_on",
    }).encode()

    sections[ASI_SECTION_METADATA] = json.dumps({
        "created": int(time.time()),
        "creator": "Little Fig v2",
        "base": os.path.basename(gguf_path),
    }).encode()

    # Layout: header + table + sections (GGUF last, largest)
    # Order sections so GGUF blob is last (streaming write)
    ordered = [(k, v) for k, v in sorted(sections.items()) if k != ASI_SECTION_GGUF_BLOB]
    ordered.append((ASI_SECTION_GGUF_BLOB, None))

    n_sections = len(ordered)
    header_size = 64
    table_size = n_sections * 32
    first_offset = page_align(header_size + table_size)

    offsets = []
    current = first_offset
    for stype, sdata in ordered:
        offsets.append(current)
        sz = gguf_size if stype == ASI_SECTION_GGUF_BLOB else len(sdata)
        current = page_align(current + sz)
    total_size = current

    flags = ASI_FLAG_GGUF_BACKED | ASI_FLAG_HAS_FABRIC | ASI_FLAG_HAS_HARNESS | ASI_FLAG_HAS_PERSONALITY
    identity_hash = hashlib.sha256(sections[ASI_SECTION_PERSONALITY]).digest()

    print(f"   Writing {output_path} ({total_size/1e9:.2f} GB)...")

    with open(output_path, 'wb') as f:
        # Header (64 bytes)
        f.write(struct.pack('<IIIIQq32s',
            ASI_MAGIC, ASI_VERSION, flags, n_sections,
            total_size, header_size, identity_hash))

        # Section table
        for i, (stype, sdata) in enumerate(ordered):
            sz = gguf_size if stype == ASI_SECTION_GGUF_BLOB else len(sdata)
            ck = 0 if stype == ASI_SECTION_GGUF_BLOB else struct.unpack('<Q', hashlib.sha256(sdata).digest()[:8])[0]
            f.write(struct.pack('<IIQQQ', stype, 0, offsets[i], sz, ck))

        # Write section data
        for i, (stype, sdata) in enumerate(ordered):
            pos = f.tell()
            pad = page_align(pos) - pos
            if pad: f.write(b'\x00' * pad)

            if stype == ASI_SECTION_GGUF_BLOB:
                print(f"   Streaming GGUF...")
                with open(gguf_path, 'rb') as gf:
                    while True:
                        chunk = gf.read(64 * 1024 * 1024)
                        if not chunk: break
                        f.write(chunk)
            else:
                f.write(sdata)

    actual = os.path.getsize(output_path)
    print(f"\n✅ Done: {output_path} ({actual/1e9:.2f} GB)")
    print(f"   Backend: llama.cpp")
    print(f"   Mode: JARVIS (always-on)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--output", default="lila.asi")
    p.add_argument("--adapters", default=None)
    args = p.parse_args()
    pack(args.gguf, args.output, args.adapters)
