#!/usr/bin/env python3
"""
Correctness & numerical parity gate:
  1. ASI batched prefill (lila_asi_eval_batch) vs token-by-token (lila_asi_eval_token)
  2. ASI vs llama-cpp-python on identical token IDs (N=32 greedy tokens, temperature=0.0)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from llama_cpp import Llama

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.asi_backend import AsiLlamaAdapter

GGUF_PATH = "/home/ticket/workspaces/lila/models/Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf"
ASI_PATH = "/tmp/qwen35_0.8b.asi"
N_CTX = 512
N_THREADS = 2
N_GEN = 32

CORPUS: List[Tuple[str, str]] = [
    (
        "factual",
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "math",
        "<|im_start|>user\nWhat is 17 * 23? Show the final number.<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "code",
        "<|im_start|>user\nWrite a Python function to reverse a string.<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "multiturn",
        "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n"
        "<|im_start|>user\nMy favorite color is blue.<|im_end|>\n"
        "<|im_start|>assistant\nGot it, your favorite color is blue.<|im_end|>\n"
        "<|im_start|>user\nWhat is my favorite color?<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "tool_use",
        "<|im_start|>system\nYou have a tool: get_weather(city). "
        'Respond only with <tool_call>{"name": "get_weather", "arguments": {"city": "..."}}</tool_call>.<|im_end|>\n'
        "<|im_start|>user\nWhat is the weather in Tokyo?<|im_end|>\n<|im_start|>assistant\n",
    ),
]


def _argmax_ctypes(logits_buf: ctypes.Array) -> int:
    arr = np.ctypeslib.as_array(logits_buf)
    return int(np.argmax(arr))


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sum((a - b) ** 2, dtype=np.float64))
    den = float(np.sum(a ** 2, dtype=np.float64)) + 1e-12
    return (num / den) ** 0.5


def _first_divergence(a: List[int], b: List[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def _asi_generate_token_by_token(
    llm: AsiLlamaAdapter, prompt_ids: List[int], n_gen: int
) -> Tuple[np.ndarray, List[int]]:
    vocab_size = llm._cfg.vocab_size
    logits_buf = (ctypes.c_float * vocab_size)()
    llm._lib.lila_asi_session_reset(llm._session)
    for pos, tid in enumerate(prompt_ids):
        rc = llm._lib.lila_asi_eval_token(llm._session, int(tid), pos, logits_buf)
        if rc != 0:
            raise RuntimeError(f"lila_asi_eval_token failed at pos={pos}, rc={rc}")
    prefill_logits = np.ctypeslib.as_array(logits_buf).copy()

    out_ids: List[int] = []
    next_tok = _argmax_ctypes(logits_buf)
    out_ids.append(next_tok)
    cur_pos = len(prompt_ids)
    for _ in range(1, n_gen):
        rc = llm._lib.lila_asi_eval_token(llm._session, next_tok, cur_pos, logits_buf)
        if rc != 0:
            raise RuntimeError(f"lila_asi_eval_token failed at pos={cur_pos}, rc={rc}")
        next_tok = _argmax_ctypes(logits_buf)
        out_ids.append(next_tok)
        cur_pos += 1
    return prefill_logits, out_ids


def _asi_generate_batch(
    llm: AsiLlamaAdapter, prompt_ids: List[int], n_gen: int
) -> Tuple[np.ndarray, List[int]]:
    vocab_size = llm._cfg.vocab_size
    logits_buf = (ctypes.c_float * vocab_size)()
    llm._lib.lila_asi_session_reset(llm._session)
    arr = (ctypes.c_int32 * len(prompt_ids))(*prompt_ids)
    rc = llm._lib.lila_asi_eval_batch(llm._session, arr, len(prompt_ids), 0, logits_buf)
    if rc != 0:
        raise RuntimeError(f"lila_asi_eval_batch failed, rc={rc}")
    prefill_logits = np.ctypeslib.as_array(logits_buf).copy()

    out_ids: List[int] = []
    next_tok = _argmax_ctypes(logits_buf)
    out_ids.append(next_tok)
    cur_pos = len(prompt_ids)
    for _ in range(1, n_gen):
        rc = llm._lib.lila_asi_eval_token(llm._session, next_tok, cur_pos, logits_buf)
        if rc != 0:
            raise RuntimeError(f"lila_asi_eval_token failed at pos={cur_pos}, rc={rc}")
        next_tok = _argmax_ctypes(logits_buf)
        out_ids.append(next_tok)
        cur_pos += 1
    return prefill_logits, out_ids


def _llama_generate(llama: Llama, prompt_ids: List[int], n_gen: int) -> List[int]:
    llama.reset()
    llama.eval(prompt_ids)
    out_ids: List[int] = []
    logits = np.array(llama.scores[llama.n_tokens - 1], dtype=np.float32)
    next_tok = int(np.argmax(logits))
    out_ids.append(next_tok)
    for _ in range(1, n_gen):
        llama.eval([next_tok])
        logits = np.array(llama.scores[llama.n_tokens - 1], dtype=np.float32)
        next_tok = int(np.argmax(logits))
        out_ids.append(next_tok)
    return out_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="ASI vs llama.cpp correctness & numerical parity gate")
    parser.add_argument("--asi", default=ASI_PATH, help=f"Path to ASI model (default: {ASI_PATH})")
    parser.add_argument("--gguf", default=GGUF_PATH, help=f"Path to GGUF model (default: {GGUF_PATH})")
    parser.add_argument("--n-ctx", type=int, default=N_CTX, help=f"Context length (default: {N_CTX})")
    parser.add_argument("--n-threads", type=int, default=N_THREADS, help=f"Thread count (default: {N_THREADS})")
    parser.add_argument("--n-gen", type=int, default=N_GEN, help=f"Greedy tokens to generate per prompt (default: {N_GEN})")
    parser.add_argument("--json-out", default=None, help="Export correctness gate results and corpus summary to JSON at PATH")
    args = parser.parse_args()

    n_gen = args.n_gen
    if n_gen <= 0:
        raise ValueError(f"--n-gen must be positive, got {n_gen}")

    print(f"Loading ASI model: {args.asi} (n_ctx={args.n_ctx}, n_threads={args.n_threads})")
    asi = AsiLlamaAdapter(args.asi, n_ctx=args.n_ctx, n_threads=args.n_threads)
    print(f"Loading llama.cpp model: {args.gguf} (n_ctx={args.n_ctx}, n_threads={args.n_threads})")
    llama = Llama(
        model_path=args.gguf,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        logits_all=True,
        verbose=False,
    )

    results: List[Dict[str, object]] = []
    try:
        for name, prompt in CORPUS:
            prompt_ids = asi.tokenizer.encode(prompt)
            logits_tok, gen_tok = _asi_generate_token_by_token(asi, prompt_ids, n_gen)
            logits_bat, gen_bat = _asi_generate_batch(asi, prompt_ids, n_gen)
            gen_llama = _llama_generate(llama, prompt_ids, n_gen)

            rel_l2 = _rel_l2(logits_tok, logits_bat)
            bt_first_div = _first_divergence(gen_tok, gen_bat)
            bt_agree = sum(1 for a, b in zip(gen_tok, gen_bat) if a == b)

            bat_vs_llama_first_div = _first_divergence(gen_bat, gen_llama)
            bat_vs_llama_agree = sum(1 for a, b in zip(gen_bat, gen_llama) if a == b)
            bat_vs_llama_pct = 100.0 * bat_vs_llama_agree / n_gen

            tok_vs_llama_first_div = _first_divergence(gen_tok, gen_llama)
            tok_vs_llama_agree = sum(1 for a, b in zip(gen_tok, gen_llama) if a == b)
            tok_vs_llama_pct = 100.0 * tok_vs_llama_agree / n_gen

            results.append(
                {
                    "name": name,
                    "prompt_len": len(prompt_ids),
                    "n_gen": n_gen,
                    "rel_l2": rel_l2,
                    "bt_first_div": bt_first_div,
                    "bt_agree": bt_agree,
                    "bat_vs_llama_agree": bat_vs_llama_agree,
                    "bat_vs_llama_pct": bat_vs_llama_pct,
                    "bat_vs_llama_first_div": bat_vs_llama_first_div,
                    "tok_vs_llama_agree": tok_vs_llama_agree,
                    "tok_vs_llama_pct": tok_vs_llama_pct,
                    "tok_vs_llama_first_div": tok_vs_llama_first_div,
                    "text_asi_bat": asi.tokenizer.decode(gen_bat),
                    "text_asi_tok": asi.tokenizer.decode(gen_tok),
                    "text_llama": asi.tokenizer.decode(gen_llama),
                }
            )
    finally:
        asi.close()
        del llama

    total_tokens = len(results) * n_gen
    total_bt_agree = sum(int(r["bt_agree"]) for r in results)
    total_bat_vs_llama_agree = sum(int(r["bat_vs_llama_agree"]) for r in results)
    total_tok_vs_llama_agree = sum(int(r["tok_vs_llama_agree"]) for r in results)

    corpus_bt_pct = 100.0 * total_bt_agree / total_tokens if total_tokens else 0.0
    corpus_bat_vs_llama_pct = 100.0 * total_bat_vs_llama_agree / total_tokens if total_tokens else 0.0
    corpus_tok_vs_llama_pct = 100.0 * total_tok_vs_llama_agree / total_tokens if total_tokens else 0.0

    print(f"\n=== Correctness Gate Results (N={n_gen} greedy tokens, temperature=0.0) ===")
    header = (
        f"{'Prompt':<10} {'Len':>3} | "
        f"{'ASI B/T L2':>10} {'B/T Div':>7} {'B/T Agree':>9} | "
        f"{'ASI(B) vs Llama':>15} {'Div':>4} | "
        f"{'ASI(T) vs Llama':>15} {'Div':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        bt_div = str(r["bt_first_div"]) if r["bt_first_div"] is not None else "none"
        b_div = str(r["bat_vs_llama_first_div"]) if r["bat_vs_llama_first_div"] is not None else "none"
        t_div = str(r["tok_vs_llama_first_div"]) if r["tok_vs_llama_first_div"] is not None else "none"
        print(
            f"{r['name']:<10} {r['prompt_len']:>3} | "
            f"{r['rel_l2']:>10.6f} {bt_div:>7} {r['bt_agree']:>6}/{n_gen} | "
            f"{r['bat_vs_llama_pct']:>13.1f}% {b_div:>4} | "
            f"{r['tok_vs_llama_pct']:>13.1f}% {t_div:>4}"
        )

    print(f"\n=== Corpus Greedy Top-1 Agreement ({len(results)} prompts x {n_gen} = {total_tokens} tokens) ===")
    print(f"  ASI(batch) vs llama.cpp : {total_bat_vs_llama_agree}/{total_tokens} ({corpus_bat_vs_llama_pct:.2f}%)")
    print(f"  ASI(token) vs llama.cpp : {total_tok_vs_llama_agree}/{total_tokens} ({corpus_tok_vs_llama_pct:.2f}%)")
    print(f"  ASI(batch) vs ASI(token): {total_bt_agree}/{total_tokens} ({corpus_bt_pct:.2f}%)")

    print("\n=== Decoded Outputs ===")
    for r in results:
        print(f"[{r['name']}]")
        print(f"  ASI(batch): {r['text_asi_bat']!r}")
        print(f"  ASI(token): {r['text_asi_tok']!r}")
        print(f"  llama.cpp : {r['text_llama']!r}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "asi_path": args.asi,
                "gguf_path": args.gguf,
                "n_ctx": args.n_ctx,
                "n_threads": args.n_threads,
                "n_gen": n_gen,
                "n_prompts": len(results),
            },
            "prompts": results,
            "summary": {
                "total_tokens": total_tokens,
                "bt_agree_tokens": total_bt_agree,
                "bt_agree_pct": corpus_bt_pct,
                "bat_vs_llama_agree_tokens": total_bat_vs_llama_agree,
                "bat_vs_llama_agree_pct": corpus_bat_vs_llama_pct,
                "tok_vs_llama_agree_tokens": total_tok_vs_llama_agree,
                "tok_vs_llama_agree_pct": corpus_tok_vs_llama_pct,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nExported JSON results to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
