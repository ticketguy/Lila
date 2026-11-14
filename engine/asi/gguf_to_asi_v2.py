#!/usr/bin/env python3
"""
Convert a GGUF model (and optional mmproj vision GGUF) into an ASI v2 cartridge.
"""

import argparse
import ctypes
import json
import mmap
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ASI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ASI_DIR))

from asi_v2 import (
    ARCH_IDS,
    SEC_CHAT_TEMPLATE,
    SEC_METADATA,
    SEC_MODEL_CONFIG,
    SEC_TOKENIZER,
    SEC_VISION,
    SEC_WEIGHTS,
    TENSOR_LAYOUT_AVX2_REPACK,
    TENSOR_LAYOUT_ROW_MAJOR,
    pack_model_config,
    pack_weights_section,
    write_asi_v2,
)
from gguf_reader import GgufFile

_ARCH_MAP = {
    "llama": "dense_llama",
    "qwen35": "hybrid_qwen35",
    "gemma": "dense_gemma",
    "gemma2": "dense_gemma",
    "qwen2": "dense_qwen2",
    "qwen3": "dense_qwen3",
}


def _meta(md: Dict[str, Any], arch: str, key: str, default: Any = 0) -> Any:
    return md.get(f"{arch}.{key}", default)


def build_model_config(gguf: GgufFile) -> Tuple[int, bytes]:
    md = gguf.metadata
    arch_str = str(md.get("general.architecture", "unknown"))
    asi_arch_name = _ARCH_MAP.get(arch_str, "unknown")
    arch_id = ARCH_IDS[asi_arch_name]

    tokens = md.get("tokenizer.ggml.tokens")
    vocab_size = len(tokens) if isinstance(tokens, list) else int(_meta(md, arch_str, "vocab_size", 0))

    cfg = pack_model_config(
        arch_id=arch_id,
        n_layers=int(_meta(md, arch_str, "block_count", 0)),
        hidden_size=int(_meta(md, arch_str, "embedding_length", 0)),
        intermediate_size=int(_meta(md, arch_str, "feed_forward_length", 0)),
        n_heads=int(_meta(md, arch_str, "attention.head_count", 0)),
        n_kv_heads=int(_meta(md, arch_str, "attention.head_count_kv", 0)),
        head_dim=int(_meta(md, arch_str, "attention.key_length", 0))
        or (
            int(_meta(md, arch_str, "embedding_length", 0))
            // max(1, int(_meta(md, arch_str, "attention.head_count", 1)))
        ),
        vocab_size=vocab_size,
        max_seq_len=int(_meta(md, arch_str, "context_length", 2048)),
        rope_dim=int(_meta(md, arch_str, "rope.dimension_count", 0)),
        rope_theta=float(_meta(md, arch_str, "rope.freq_base", 10000.0)),
        rms_norm_eps=float(_meta(md, arch_str, "attention.layer_norm_rms_epsilon", 1e-6)),
        embed_scale=float(_meta(md, arch_str, "embedding_scale", 1.0)),
        full_attention_interval=int(_meta(md, arch_str, "full_attention_interval", 1)),
        ssm_conv_kernel=int(_meta(md, arch_str, "ssm.conv_kernel", 0)),
        ssm_state_size=int(_meta(md, arch_str, "ssm.state_size", 0)),
        ssm_group_count=int(_meta(md, arch_str, "ssm.group_count", 0)),
        ssm_inner_size=int(_meta(md, arch_str, "ssm.inner_size", 0)),
        ssm_time_step_rank=int(_meta(md, arch_str, "ssm.time_step_rank", 0)),
        default_quant_type=int(md.get("general.file_type", 0)),
    )
    return arch_id, cfg


def build_tokenizer_section(gguf: GgufFile) -> bytes:
    md = gguf.metadata
    payload = {
        "model": md.get("tokenizer.ggml.model", ""),
        "pre": md.get("tokenizer.ggml.pre", ""),
        "bos_token_id": md.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": md.get("tokenizer.ggml.eos_token_id"),
        "padding_token_id": md.get("tokenizer.ggml.padding_token_id"),
        "add_bos_token": md.get("tokenizer.ggml.add_bos_token", False),
        "tokens": md.get("tokenizer.ggml.tokens", []),
        "token_type": md.get("tokenizer.ggml.token_type", []),
        "merges": md.get("tokenizer.ggml.merges", []),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _load_q6k_repacker():
    so = ASI_DIR / "libasi_v2.so"
    if not so.exists():
        return None
    try:
        lib = ctypes.CDLL(str(so))
        fn = lib.asi_v2_repack_q6k
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        fn.restype = None
        return fn
    except (OSError, AttributeError):
        return None


_REPACK_Q6K = _load_q6k_repacker()


def pack_gguf_tensors(gguf: GgufFile, *, repack_avx2: bool = False) -> bytes:
    tensors: List[Tuple[str, int, int, Tuple[int, int], bytes]] = []
    with gguf.path.open("rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for t in gguf.tensors:
                d0 = t.dims[0] if len(t.dims) >= 1 else 1
                d1 = t.dims[1] if len(t.dims) >= 2 else 1
                if len(t.dims) > 2:
                    for extra in t.dims[2:]:
                        d1 *= extra
                raw = gguf.read_tensor_bytes(mm, t)
                lflags = TENSOR_LAYOUT_ROW_MAJOR
                if repack_avx2 and t.ggml_type == 14 and _REPACK_Q6K is not None:
                    n_blocks = len(raw) // 210
                    if n_blocks * 210 == len(raw):
                        dst = ctypes.create_string_buffer(n_blocks * 288)
                        _REPACK_Q6K(raw, dst, n_blocks)
                        raw = dst.raw
                        lflags = TENSOR_LAYOUT_AVX2_REPACK
                tensors.append((t.name, t.ggml_type, lflags, (d0, d1), raw))
    return pack_weights_section(tensors)


def convert_gguf_to_asi_v2(
    gguf_path: Path,
    out_path: Path,
    *,
    mmproj_path: Optional[Path] = None,
) -> Path:
    gguf = GgufFile(gguf_path, skip_large_arrays=False)
    arch_id, cfg_bytes = build_model_config(gguf)
    weights_bytes = pack_gguf_tensors(gguf)
    tok_bytes = build_tokenizer_section(gguf)

    sections: List[Tuple[int, int, bytes]] = [
        (SEC_MODEL_CONFIG, 0, cfg_bytes),
        (SEC_WEIGHTS, 0, weights_bytes),
        (SEC_TOKENIZER, 0, tok_bytes),
    ]

    flags = 0
    chat_tmpl = gguf.metadata.get("tokenizer.chat_template")
    if isinstance(chat_tmpl, str) and chat_tmpl:
        sections.append((SEC_CHAT_TEMPLATE, 0, chat_tmpl.encode("utf-8")))
        flags |= 1 << 4  # ASI_V2_FLAG_HAS_CHAT_TEMPLATE

    if mmproj_path is not None:
        mm_gguf = GgufFile(mmproj_path, skip_large_arrays=False)
        mm_weights = pack_gguf_tensors(mm_gguf)
        sections.append((SEC_VISION, 0, mm_weights))
        flags |= 1 << 0  # ASI_V2_FLAG_HAS_VISION

    meta = {
        "format": "asi-v2",
        "source_gguf": gguf_path.name,
        "architecture": gguf.metadata.get("general.architecture", "unknown"),
        "base_model": gguf.metadata.get("general.base_model.0.name", ""),
        "license": gguf.metadata.get("general.license", ""),
        "n_tensors": len(gguf.tensors),
        "has_vision": mmproj_path is not None,
    }
    sections.append((SEC_METADATA, 0, json.dumps(meta, separators=(",", ":")).encode("utf-8")))

    return write_asi_v2(out_path, arch_id=arch_id, flags=flags, sections=sections)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert GGUF model to ASI v2 cartridge")
    ap.add_argument("gguf", type=Path, help="Input model .gguf")
    ap.add_argument("output", type=Path, help="Output .asi v2 cartridge")
    ap.add_argument("--mmproj", type=Path, default=None, help="Optional vision projector .gguf")
    args = ap.parse_args()

    out = convert_gguf_to_asi_v2(args.gguf, args.output, mmproj_path=args.mmproj)
    print(f"Converted {args.gguf.name} -> {out} ({out.stat().st_size / (1024 * 1024):.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
