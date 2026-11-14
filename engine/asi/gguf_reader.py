#!/usr/bin/env python3
"""
Minimal zero-dependency GGUF v2/v3 reader for ASI v2 conversion.
"""

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Tuple

GGUF_MAGIC = b"GGUF"

# ggml_type enum values (standard GGML / llama.cpp)
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    # type_id: (name, block_size_elems, block_size_bytes)
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}

# GGUF metadata value types
_VAL_FMT = {
    0: "<B",   # UINT8
    1: "<b",   # INT8
    2: "<H",   # UINT16
    3: "<h",   # INT16
    4: "<I",   # UINT32
    5: "<i",   # INT32
    6: "<f",   # FLOAT32
    7: "<?",   # BOOL
    10: "<Q",  # UINT64
    11: "<q",  # INT64
    12: "<d",  # FLOAT64
}


@dataclass
class GgufTensor:
    name: str
    dims: Tuple[int, ...]
    ggml_type: int
    offset: int  # relative to tensor_data_start

    @property
    def type_name(self) -> str:
        info = GGML_TYPES.get(self.ggml_type)
        return info[0] if info else f"TYPE_{self.ggml_type}"

    @property
    def n_elems(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        info = GGML_TYPES.get(self.ggml_type)
        if not info:
            raise ValueError(f"Unsupported ggml_type {self.ggml_type} for {self.name}")
        _, blk_elems, blk_bytes = info
        if self.n_elems % blk_elems != 0:
            raise ValueError(
                f"Tensor {self.name} n_elems={self.n_elems} not divisible by block={blk_elems}"
            )
        return (self.n_elems // blk_elems) * blk_bytes


def _read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_str(f: BinaryIO) -> str:
    n = _read_u64(f)
    return f.read(n).decode("utf-8", errors="replace")


def _read_val(f: BinaryIO, vtype: int, *, skip_large_arrays: bool = False) -> Any:
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        if skip_large_arrays and count > 1024:
            # Skip large vocab arrays during fast inspection
            if elem_type in _VAL_FMT:
                sz = struct.calcsize(_VAL_FMT[elem_type])
                f.seek(count * sz, 1)
                return f"<array[{count}] of type {elem_type}>"
            for _ in range(count):
                _read_val(f, elem_type, skip_large_arrays=False)
            return f"<array[{count}] of type {elem_type}>"
        return [_read_val(f, elem_type, skip_large_arrays=False) for _ in range(count)]
    fmt = _VAL_FMT.get(vtype)
    if fmt is None:
        raise ValueError(f"Unknown GGUF value type: {vtype}")
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))[0]


class GgufFile:
    def __init__(self, path: Path, *, skip_large_arrays: bool = False):
        self.path = Path(path)
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[GgufTensor] = []
        self.alignment = 32
        self.tensor_data_offset = 0

        with self.path.open("rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file: {magic!r}")
            self.version = _read_u32(f)
            if self.version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version: {self.version}")
            n_tensors = _read_u64(f)
            n_kv = _read_u64(f)

            for _ in range(n_kv):
                key = _read_str(f)
                vtype = _read_u32(f)
                val = _read_val(f, vtype, skip_large_arrays=skip_large_arrays)
                self.metadata[key] = val

            self.alignment = int(self.metadata.get("general.alignment", 32))

            for _ in range(n_tensors):
                name = _read_str(f)
                n_dims = _read_u32(f)
                dims = tuple(_read_u64(f) for _ in range(n_dims))
                ggml_type = _read_u32(f)
                offset = _read_u64(f)
                self.tensors.append(GgufTensor(name=name, dims=dims, ggml_type=ggml_type, offset=offset))

            pos = f.tell()
            self.tensor_data_offset = (pos + self.alignment - 1) & ~(self.alignment - 1)

    def read_tensor_bytes(self, mm: mmap.mmap, t: GgufTensor) -> bytes:
        start = self.tensor_data_offset + t.offset
        end = start + t.n_bytes
        return bytes(mm[start:end])


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1])
    g = GgufFile(p, skip_large_arrays=True)
    print(f"GGUF v{g.version}: {len(g.tensors)} tensors, alignment={g.alignment}, data_off={g.tensor_data_offset}")
    for k in sorted(g.metadata):
        if not k.startswith("tokenizer.ggml.tokens") and not k.startswith("tokenizer.ggml.merges") and not k.startswith("tokenizer.ggml.scores"):
            v = g.metadata[k]
            if isinstance(v, str) and len(v) > 120:
                v = v[:120] + "..."
            print(f"  KV {k} = {v!r}")
    print("\nFirst 40 tensors:")
    for t in g.tensors[:40]:
        print(f"  {t.name:42s} {t.type_name:6s} {str(t.dims):18s} {t.n_bytes} B")
    # Also show layer 3 (first full attention layer in Qwen3.5!)
    print("\nLayer 3 tensors:")
    for t in g.tensors:
        if t.name.startswith("blk.3."):
            print(f"  {t.name:42s} {t.type_name:6s} {str(t.dims):18s} {t.n_bytes} B")
