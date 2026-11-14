#!/usr/bin/env python3
"""
Benchmark comparing:
  1. llama-cpp-python (llama.cpp backend) on Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf
  2. Custom Lila engine (lila-asi) on a Q4_K .asi model of comparable layer geometry
"""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_DIR / "asi"))
from pack_asi import (
    ASI_MAGIC,
    ASI_VERSION,
    ASI_PAGE_SIZE,
    ASI_SECTION_MODEL_CONFIG,
    ASI_SECTION_WEIGHTS,
    ASI_SECTION_TOKENIZER,
    ASI_SECTION_METADATA,
    ASI_FLAG_TOKENIZER_BPE,
    page_align,
    crc64,
)

QUANT_Q4_K = 2
QUANT_Q6_K_EMBED = 5
Q4_K_BLOCK = 144
Q6_K_BLOCK = 210


def build_synthetic_q4k_asi(
    out_path: Path,
    n_layers: int = 16,
    hidden: int = 1024,
    intermediate: int = 2816,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    vocab_size: int = 32000,
    max_seq: int = 512,
) -> Path:
    """Create a valid Q4_K .asi file with deterministic blocks (no Memory Fabric)."""
    if out_path.exists():
        return out_path

    head_dim = hidden // n_heads
    kv_dim = n_kv_heads * head_dim

    # 1. MODEL_CONFIG (48 bytes)
    config_sec = struct.pack(
        "<IIIIIIIIffII",
        n_layers,
        hidden,
        intermediate,
        n_heads,
        n_kv_heads,
        vocab_size,
        max_seq,
        head_dim,
        1000000.0,  # rope_theta
        1e-6,       # rms_norm_eps
        QUANT_Q4_K,
        128,
    )

    # 2. WEIGHTS (Q6_K tied embedding + Q4_K layers)
    chunks = []
    chunks.append(struct.pack("<III", vocab_size, hidden, QUANT_Q6_K_EMBED))
    embed_row_bytes = (hidden // 256) * Q6_K_BLOCK
    # Non-zero byte pattern so dequant isn't trivial zeros
    embed_row = bytes([0x11] * embed_row_bytes)
    chunks.append(embed_row * vocab_size)

    layer_shapes = [
        (hidden, hidden),        # q_proj
        (kv_dim, hidden),        # k_proj
        (kv_dim, hidden),        # v_proj
        (hidden, hidden),        # o_proj
        (intermediate, hidden),  # gate_proj
        (intermediate, hidden),  # up_proj
        (hidden, intermediate),  # down_proj
    ]
    norm_bytes = struct.pack("<" + "f" * hidden, *([1.0] * hidden))

    # Build a single reusable Q4_K block pattern (non-zero scale/nibbles)
    one_q4k_block = bytes([0x00, 0x3C, 0x00, 0x38] + [0x12] * (Q4_K_BLOCK - 4))
    proj_blobs = []
    for rows, cols in layer_shapes:
        n_blocks = (rows * cols + 255) // 256
        hdr = struct.pack("<III", rows, cols, QUANT_Q4_K)
        proj_blobs.append(hdr + one_q4k_block * n_blocks)

    one_layer = b"".join(proj_blobs) + norm_bytes + norm_bytes
    chunks.append(one_layer * n_layers)
    chunks.append(norm_bytes)  # final_norm
    chunks.append(struct.pack("<I", 0xFFFFFFFF))  # tied LM head
    weights_sec = b"".join(chunks)

    # 3. TOKENIZER (minimal BPE binary header + vocab)
    tok_chunks = [struct.pack("<III", vocab_size, 0, 0)]
    for i in range(vocab_size):
        tok = f"t{i}".encode("utf-8")
        tok_chunks.append(struct.pack("<H", len(tok)) + tok + struct.pack("<f", -float(i)))
    tok_sec = b"".join(tok_chunks)

    # 4. METADATA
    meta_sec = json.dumps(
        {
            "name": "lila-bench-q4k",
            "bos_token": 1,
            "eos_token": 2,
            "vocab_size": vocab_size,
        }
    ).encode("utf-8")

    sections = [
        (ASI_SECTION_MODEL_CONFIG, 0, config_sec),
        (ASI_SECTION_WEIGHTS, 0, weights_sec),
        (ASI_SECTION_TOKENIZER, 0, tok_sec),
        (ASI_SECTION_METADATA, 0, meta_sec),
    ]

    header_size = 64
    table_size = len(sections) * 32
    cur_off = page_align(header_size + table_size)

    table_entries = []
    padded_bodies = []
    for sec_type, flags, data in sections:
        csum = crc64(data)
        table_entries.append((sec_type, flags, cur_off, len(data), csum))
        aligned_len = page_align(len(data))
        padded_bodies.append(data + b"\x00" * (aligned_len - len(data)))
        cur_off += aligned_len

    total_size = cur_off
    id_hash = hashlib.sha256(b"lila-bench-q4k").digest()
    header = struct.pack(
        "<IIIIQQ32s",
        ASI_MAGIC,
        ASI_VERSION,
        ASI_FLAG_TOKENIZER_BPE,
        len(sections),
        total_size,
        header_size,
        id_hash,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        for entry in table_entries:
            f.write(struct.pack("<IIQQQ", *entry))
        f.write(b"\x00" * (page_align(header_size + table_size) - (header_size + table_size)))
        for body in padded_bodies:
            f.write(body)

    return out_path


def bench_llama_cpp(gguf_path: Path, threads_list=(2, 4), max_tokens=32):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[llama-cpp] llama_cpp not installed in this Python")
        return []

    results = []
    prompt = (
        "You are Lila, a concise local AI assistant. "
        "Explain in two short sentences how CPU cache affects token generation speed."
    )
    for threads in threads_list:
        t0 = time.perf_counter()
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=512,
            n_batch=128,
            n_threads=threads,
            n_gpu_layers=0,
            verbose=False,
        )
        load_s = time.perf_counter() - t0

        # Warmup 1 token
        llm(prompt, max_tokens=1, temperature=0.0)

        t1 = time.perf_counter()
        out = llm(prompt, max_tokens=max_tokens, temperature=0.0)
        elapsed = time.perf_counter() - t1
        usage = out["usage"]
        ptok = usage["prompt_tokens"]
        ctok = usage["completion_tokens"]
        tok_s = ctok / elapsed if elapsed > 0 else 0.0
        results.append(
            {
                "threads": threads,
                "load_s": load_s,
                "prompt_tokens": ptok,
                "gen_tokens": ctok,
                "elapsed_s": elapsed,
                "tok_s": tok_s,
            }
        )
    return results


def bench_lila_asi(asi_path: Path, threads: int | None = None):
    bin_path = ENGINE_DIR / "lila-asi"
    env = os.environ.copy()
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
        env["LILA_THREADS"] = str(threads)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(bin_path), str(asi_path), "--bench"],
        capture_output=True,
        text=True,
        env=env,
    )
    wall_s = time.perf_counter() - t0
    out = proc.stdout + "\n" + proc.stderr
    m_gen = re.search(r"Generated:\s+(\d+)\s+tokens", out)
    m_time = re.search(r"Time:\s+([0-9.]+)\s+seconds", out)
    m_speed = re.search(r"Speed:\s+([0-9.]+)\s+tokens/sec", out)
    return {
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "gen_tokens": int(m_gen.group(1)) if m_gen else 0,
        "elapsed_s": float(m_time.group(1)) if m_time else wall_s,
        "tok_s": float(m_speed.group(1)) if m_speed else 0.0,
        "raw": out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gguf",
        default="/home/ticket/workspaces/lila/models/Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf",
    )
    parser.add_argument("--asi", default="/tmp/lila_bench_q4k.asi")
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    asi_path = build_synthetic_q4k_asi(Path(args.asi))
    print(f"ASI bench file: {asi_path} ({asi_path.stat().st_size / (1024*1024):.1f} MB)")

    if not args.skip_llama and Path(args.gguf).exists():
        print("\n--- llama-cpp-python (Qwen3.5-0.8B Q4_K_M) ---")
        for r in bench_llama_cpp(Path(args.gguf), threads_list=(2, 4)):
            print(
                f"  threads={r['threads']} | load={r['load_s']:.2f}s | "
                f"prompt={r['prompt_tokens']}t | gen={r['gen_tokens']}t in {r['elapsed_s']:.2f}s "
                f"({r['tok_s']:.2f} tok/s end-to-end)"
            )

    print(f"\n--- lila-asi (synthetic 16L h=1024 Q4_K, threads={args.threads}) ---")
    r = bench_lila_asi(asi_path, threads=args.threads)
    print(
        f"  rc={r['returncode']} | gen={r['gen_tokens']}t in {r['elapsed_s']:.3f}s "
        f"({r['tok_s']:.2f} tok/s) | total wall={r['wall_s']:.3f}s"
    )
    if r["returncode"] != 0 or r["gen_tokens"] == 0:
        print(r["raw"])


if __name__ == "__main__":
    main()
