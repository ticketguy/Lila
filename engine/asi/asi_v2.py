#!/usr/bin/env python3
"""
ASI v2 Python Cartridge Writer & Validator.
Matches source/engine/asi/asi_v2.h byte-for-byte.
"""

import ctypes
import hashlib
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ASI_V2_MAGIC = 0x41534921
ASI_V2_VERSION = 2
ASI_V2_PAGE_SIZE = 4096

ARCH_IDS = {
    "unknown": 0,
    "dense_llama": 1,
    "hybrid_qwen35": 2,
    "dense_gemma": 3,
    "dense_qwen2": 4,
    "dense_qwen3": 5,
}

SEC_MODEL_CONFIG = 0x01
SEC_WEIGHTS = 0x02
SEC_MEMORY_FABRIC = 0x03
SEC_TOKENIZER = 0x04
SEC_PERSONALITY = 0x07
SEC_METADATA = 0x08
SEC_CHAT_TEMPLATE = 0x09
SEC_VISION = 0x0A
SEC_AUDIO = 0x0B
SEC_ADAPTERS = 0x0C
SEC_RUNTIME_HINTS = 0x0D
SEC_SIGNATURE = 0x0E

TENSOR_LAYOUT_ROW_MAJOR = 1 << 0
TENSOR_LAYOUT_AVX2_REPACK = 1 << 1


def page_align(n: int) -> int:
    return (n + ASI_V2_PAGE_SIZE - 1) & ~(ASI_V2_PAGE_SIZE - 1)


def _build_crc64_table():
    poly = 0x42F0E1EBA9EA3693
    table = []
    for i in range(256):
        crc = i << 56
        for _ in range(8):
            if crc & 0x8000000000000000:
                crc = ((crc << 1) ^ poly) & 0xFFFFFFFFFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFFFFFFFFFF
        table.append(crc)
    return tuple(table)


_CRC64_TABLE = _build_crc64_table()


def _load_c_crc64():
    so_path = Path(__file__).resolve().parent / "libasi_v2.so"
    if not so_path.exists():
        return None
    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.asi_v2_crc64
        fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        fn.restype = ctypes.c_uint64
        return fn
    except OSError:
        return None


_C_CRC64 = _load_c_crc64()


def crc64(data: bytes) -> int:
    if _C_CRC64 is not None:
        return int(_C_CRC64(data, len(data)))
    crc = 0
    table = _CRC64_TABLE
    for byte in data:
        crc = ((crc << 8) ^ table[((crc >> 56) ^ byte) & 0xFF]) & 0xFFFFFFFFFFFFFFFF
    return crc


def pack_model_config(
    *,
    arch_id: int,
    n_layers: int,
    hidden_size: int,
    intermediate_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    max_seq_len: int,
    rope_dim: int = 0,
    rope_theta: float = 10000.0,
    rms_norm_eps: float = 1e-6,
    embed_scale: float = 1.0,
    full_attention_interval: int = 1,
    ssm_conv_kernel: int = 0,
    ssm_state_size: int = 0,
    ssm_group_count: int = 0,
    ssm_inner_size: int = 0,
    ssm_time_step_rank: int = 0,
    default_quant_type: int = 2,
) -> bytes:
    return struct.pack(
        "<IIIIIIIIIIfffIIIIIII4I",
        arch_id,
        n_layers,
        hidden_size,
        intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len,
        rope_dim,
        rope_theta,
        rms_norm_eps,
        embed_scale,
        full_attention_interval,
        ssm_conv_kernel,
        ssm_state_size,
        ssm_group_count,
        ssm_inner_size,
        ssm_time_step_rank,
        default_quant_type,
        0,
        0,
        0,
        0,
    )


def pack_weights_section(tensors: List[Tuple[str, int, int, Tuple[int, int], bytes]]) -> bytes:
    """
    tensors: list of (name, quant_type, layout_flags, (dim0, dim1), raw_bytes)
    """
    n_tensors = len(tensors)
    dir_bytes = 16 + n_tensors * 80
    data_offset = (dir_bytes + 63) & ~63  # 64-byte aligned start for SIMD loads

    entries = []
    blobs = []
    cur = 0
    for name, qtype, lflags, (d0, d1), raw in tensors:
        name_b = name.encode("utf-8")[:39]
        name_padded = name_b + b"\x00" * (40 - len(name_b))
        entries.append(
            struct.pack(
                "<40sIIQQQQ",
                name_padded,
                qtype,
                lflags,
                d0,
                d1,
                cur,
                len(raw),
            )
        )
        aligned_len = (len(raw) + 63) & ~63
        blobs.append(raw + b"\x00" * (aligned_len - len(raw)))
        cur += aligned_len

    header = struct.pack("<IIQ", n_tensors, 0, data_offset)
    pad = b"\x00" * (data_offset - dir_bytes)
    return header + b"".join(entries) + pad + b"".join(blobs)


def write_asi_v2(
    out_path: Path,
    *,
    arch_id: int,
    flags: int,
    sections: List[Tuple[int, int, bytes]],
) -> Path:
    header_size = 64
    table_offset = header_size
    table_size = len(sections) * 32
    cur_off = page_align(table_offset + table_size)

    table_entries = []
    padded_bodies = []
    digest = hashlib.sha256()
    for sec_type, sec_flags, body in sections:
        csum = crc64(body)
        digest.update(body)
        table_entries.append(struct.pack("<IIQQQ", sec_type, sec_flags, cur_off, len(body), csum))
        aligned = page_align(len(body))
        padded_bodies.append(body + b"\x00" * (aligned - len(body)))
        cur_off += aligned

    header = struct.pack(
        "<IIIIIIQQ24s",
        ASI_V2_MAGIC,
        ASI_V2_VERSION,
        arch_id,
        flags,
        len(sections),
        0,
        cur_off,
        table_offset,
        digest.digest()[:24],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        for entry in table_entries:
            f.write(entry)
        f.write(b"\x00" * (page_align(table_offset + table_size) - (table_offset + table_size)))
        for body in padded_bodies:
            f.write(body)
    return out_path
