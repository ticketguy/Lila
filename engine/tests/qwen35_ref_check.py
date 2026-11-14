#!/usr/bin/env python3
import mmap
import struct
import sys
from pathlib import Path

ASI_DIR = Path(__file__).resolve().parent.parent / "asi"
sys.path.insert(0, str(ASI_DIR))
from gguf_reader import GgufFile

MODEL = Path("/home/ticket/workspaces/lila/models/Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf")

g = GgufFile(MODEL, skip_large_arrays=True)
types = sorted(set((t.type_name, t.dims[0] % 256 == 0) for t in g.tensors))
print("Tensor types & mod256:", types)

with MODEL.open("rb") as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for name in ["blk.0.attn_norm.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight", "blk.3.attn_q_norm.weight", "output_norm.weight"]:
            t = next(x for x in g.tensors if x.name == name)
            raw = g.read_tensor_bytes(mm, t)
            vals = struct.unpack(f"<{len(raw)//4}f", raw)
            mean = sum(vals) / len(vals)
            print(f"{name:28s}: len={len(vals):4d} min={min(vals):.4f} max={max(vals):.4f} mean={mean:.4f} first4={[round(v, 4) for v in vals[:4]]}")
