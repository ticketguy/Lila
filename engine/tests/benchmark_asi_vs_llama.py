#!/usr/bin/env python3
"""
Apples-to-apples benchmark: Lila ASI v2 (liblila_asi.so) vs llama-cpp-python
on Qwen3.5-0.8B (n_threads=2, n_ctx=512, temperature=0.0, max_tokens=20).
"""

import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from core.asi_backend import AsiLlamaAdapter
from llama_cpp import Llama

GGUF_PATH = "/home/ticket/workspaces/lila/models/Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf"
ASI_PATH = "/tmp/qwen35_0.8b.asi"

MESSAGES = [
    {"role": "system", "content": "You are Lila, a concise local AI assistant."},
    {"role": "user", "content": "Give me a 1-sentence greeting and explain what 2+2 equals."},
]


def bench_backend(name: str, loader):
    t0 = time.monotonic()
    llm = loader()
    load_s = time.monotonic() - t0

    for turn in (1, 2):
        t1 = time.monotonic()
        first_tok_s = None
        parts = []
        stream = llm.create_chat_completion(
            messages=MESSAGES,
            max_tokens=20,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            stream=True,
        )
        for ev in stream:
            delta = (ev.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if first_tok_s is None:
                    first_tok_s = time.monotonic() - t1
                parts.append(delta)
        total_s = time.monotonic() - t1
        text = "".join(parts)
        n_chunks = len(parts)
        decode_s = max(total_s - (first_tok_s or total_s), 1e-6)
        decode_tps = max(n_chunks - 1, 1) / decode_s if n_chunks > 1 else n_chunks / total_s
        print(f"[{name} turn={turn}] load={load_s:.2f}s | ttft={first_tok_s:.2f}s | total={total_s:.2f}s | chunks={n_chunks} | decode={decode_tps:.2f} tok/s")
        print(f"  output: {text!r}\n")
    if hasattr(llm, "close"):
        llm.close()


def main() -> int:
    t0 = time.monotonic()
    asi = AsiLlamaAdapter(ASI_PATH, n_ctx=512, n_threads=2)
    t_load = time.monotonic() - t0
    tp0 = time.monotonic()
    prompt = asi._format_chat(MESSAGES)
    ids = asi.tokenizer.encode(prompt)
    t_tok = time.monotonic() - tp0
    print(f"[Profile ASI] load={t_load:.2f}s | tokenize({len(ids)} toks)={t_tok:.2f}s")
    asi.close()

    bench_backend(
        "Lila ASI v2",
        lambda: AsiLlamaAdapter(ASI_PATH, n_ctx=512, n_threads=2),
    )
    bench_backend(
        "llama.cpp",
        lambda: Llama(GGUF_PATH, n_ctx=512, n_threads=2, verbose=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
