#!/usr/bin/env python3
"""
Correctness & numerical parity gate:
  1. Free-running greedy comparison:
     - ASI batched prefill (lila_asi_eval_batch) vs token-by-token (lila_asi_eval_token)
     - ASI vs llama-cpp-python on identical token IDs (greedy tokens, temperature=0.0)
  2. Teacher-forced + FP32-reference audit (LILA_ASI_FP32_REF=1):
     - Teacher-forced top-1 agreement across ASI(Q8_K), llama.cpp, and ASI(FP32-Ref)
     - Per-position logit relative L2 error vs ASI(FP32-Ref)
     - Margin & logit-diff diagnostics at every argmax(ASI) != argmax(llama.cpp) flip
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from llama_cpp import Llama

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.asi_backend import AsiBpeTokenizer, AsiLlamaAdapter

GGUF_PATH = "/home/ticket/workspaces/lila/models/Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf"
ASI_PATH = "/tmp/qwen35_0.8b.asi"
N_CTX = 512
N_THREADS = 2
N_GEN = 32


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

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
    """Relative L2 distance normalized by ||a||_2."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    diff = a64 - b64
    num = float(np.sum(diff * diff, dtype=np.float64))
    den = float(np.sum(a64 * a64, dtype=np.float64)) + 1e-12
    return (num / den) ** 0.5


def _rel_l2_vs_ref(cand: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error of cand against reference vector ref (||cand - ref|| / ||ref||)."""
    return _rel_l2(ref, cand)


def _top1_top2_margin(vec: np.ndarray) -> float:
    """Return sorted[-1] - sorted[-2] margin for a 1-D logit vector."""
    if vec.shape[0] < 2:
        return 0.0
    top2 = np.partition(vec, -2)[-2:]
    hi = float(max(top2[0], top2[1]))
    lo = float(min(top2[0], top2[1]))
    return hi - lo


def _agg_stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compute mean, median, min, max over a sequence, or None fields if empty."""
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _fmt_opt(val: Optional[float], fmt: str = ".4f", default: str = "n/a") -> str:
    return format(val, fmt) if val is not None else default


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


def _asi_teacher_forced_logits(
    llm: AsiLlamaAdapter,
    prompt_ids: Sequence[int],
    gen_ids: Sequence[int],
    vocab: int,
) -> np.ndarray:
    """Collect (len(gen_ids), vocab) teacher-forced logits via eval_batch + eval_token."""
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if not gen_ids:
        raise ValueError("gen_ids must be non-empty")
    vocab_size = int(llm._cfg.vocab_size)
    if vocab > vocab_size:
        raise ValueError(f"requested vocab={vocab} exceeds ASI vocab_size={vocab_size}")

    n_gen = len(gen_ids)
    out = np.empty((n_gen, vocab), dtype=np.float32)
    logits_buf = (ctypes.c_float * vocab_size)()
    llm._lib.lila_asi_session_reset(llm._session)

    arr = (ctypes.c_int32 * len(prompt_ids))(*[int(t) for t in prompt_ids])
    rc = llm._lib.lila_asi_eval_batch(llm._session, arr, len(prompt_ids), 0, logits_buf)
    if rc != 0:
        raise RuntimeError(f"lila_asi_eval_batch failed during teacher-forced eval, rc={rc}")
    out[0, :] = np.ctypeslib.as_array(logits_buf)[:vocab]

    pos = len(prompt_ids)
    for k, tid in enumerate(gen_ids[:-1], start=1):
        rc = llm._lib.lila_asi_eval_token(llm._session, int(tid), pos, logits_buf)
        if rc != 0:
            raise RuntimeError(f"lila_asi_eval_token failed during teacher-forced eval at pos={pos}, rc={rc}")
        out[k, :] = np.ctypeslib.as_array(logits_buf)[:vocab]
        pos += 1
    return out


def _llama_teacher_forced_logits(
    llama: Llama,
    prompt_ids: Sequence[int],
    gen_ids: Sequence[int],
    vocab: int,
) -> np.ndarray:
    """Collect (len(gen_ids), vocab) teacher-forced logits via llama.eval(full_seq[:-1])."""
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    if not gen_ids:
        raise ValueError("gen_ids must be non-empty")
    llama_vocab = int(llama.n_vocab())
    if vocab > llama_vocab:
        raise ValueError(f"requested vocab={vocab} exceeds llama.n_vocab()={llama_vocab}")

    n_gen = len(gen_ids)
    full_seq = [int(t) for t in prompt_ids] + [int(t) for t in gen_ids]
    eval_seq = full_seq[:-1]

    llama.reset()
    llama.eval(eval_seq)

    start = len(prompt_ids) - 1
    end = start + n_gen
    if llama.scores is None or end > llama.scores.shape[0] or vocab > llama.scores.shape[1]:
        raise RuntimeError(
            f"llama.scores shape {None if llama.scores is None else llama.scores.shape} "
            f"cannot satisfy [{start}:{end}, :{vocab}]"
        )
    return np.array(llama.scores[start:end, :vocab], dtype=np.float32, copy=True)


def _run_fp32_worker(manifest_path: str) -> int:
    """Subprocess worker that evaluates ASI with LILA_ASI_FP32_REF=1 from process start."""
    os.environ["LILA_ASI_FP32_REF"] = "1"
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    vocab = int(manifest["vocab"])
    jobs = manifest["jobs"]

    print(
        f"[fp32-worker] Loading ASI(FP32-Ref): {manifest['asi_path']} "
        f"(n_ctx={manifest['n_ctx']}, n_threads={manifest['n_threads']})",
        flush=True,
    )
    asi = AsiLlamaAdapter(
        manifest["asi_path"],
        n_ctx=int(manifest["n_ctx"]),
        n_threads=int(manifest["n_threads"]),
    )
    try:
        for job in jobs:
            print(f"[fp32-worker] Teacher-forcing '{job['name']}' ({len(job['gen_ids'])} steps)...", flush=True)
            arr = _asi_teacher_forced_logits(asi, job["prompt_ids"], job["gen_ids"], vocab)
            np.save(job["out_npy"], arr)
            del arr
    finally:
        asi.close()
    return 0


def _evaluate_teacher_forced_corpus(
    tf_records: Sequence[Dict[str, object]],
    tokenizer: AsiBpeTokenizer,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Compute per-prompt and corpus-wide teacher-forced & FP32-Ref audit metrics."""
    all_asi_rel_l2: List[float] = []
    all_llama_rel_l2: List[float] = []
    all_flips: List[Dict[str, object]] = []
    all_asi_vs_fp32_flip_margins: List[float] = []
    all_llama_vs_fp32_flip_margins: List[float] = []

    total_positions = 0
    total_asi_vs_llama = 0
    total_asi_vs_fp32 = 0
    total_llama_vs_fp32 = 0
    max_abs_q8_vs_fp32 = 0.0

    per_prompt_tf: List[Dict[str, object]] = []
    for rec in tf_records:
        name = str(rec["name"])
        prompt_len = int(rec["prompt_len"])
        n_gen = int(rec["n_gen"])

        llama_logits = np.load(str(rec["llama_npy"]), mmap_mode="r")
        asi_logits = np.load(str(rec["asi_q8_npy"]), mmap_mode="r")
        ref_logits = np.load(str(rec["asi_fp32_npy"]), mmap_mode="r")
        try:
            p_asi_vs_llama = 0
            p_asi_vs_fp32 = 0
            p_llama_vs_fp32 = 0
            p_asi_rel_l2: List[float] = []
            p_llama_rel_l2: List[float] = []
            p_flips: List[Dict[str, object]] = []

            for k in range(n_gen):
                l_vec = llama_logits[k]
                a_vec = asi_logits[k]
                r_vec = ref_logits[k]

                step_max_diff = float(np.max(np.abs(a_vec - r_vec)))
                if step_max_diff > max_abs_q8_vs_fp32:
                    max_abs_q8_vs_fp32 = step_max_diff

                top1_asi = int(np.argmax(a_vec))
                top1_llama = int(np.argmax(l_vec))
                top1_ref = int(np.argmax(r_vec))

                if top1_asi == top1_llama:
                    p_asi_vs_llama += 1
                if top1_asi == top1_ref:
                    p_asi_vs_fp32 += 1
                if top1_llama == top1_ref:
                    p_llama_vs_fp32 += 1

                rel_l2_asi = _rel_l2_vs_ref(a_vec, r_vec)
                rel_l2_llama = _rel_l2_vs_ref(l_vec, r_vec)
                p_asi_rel_l2.append(rel_l2_asi)
                p_llama_rel_l2.append(rel_l2_llama)
                all_asi_rel_l2.append(rel_l2_asi)
                all_llama_rel_l2.append(rel_l2_llama)

                if top1_asi != top1_ref:
                    all_asi_vs_fp32_flip_margins.append(_top1_top2_margin(r_vec))
                if top1_llama != top1_ref:
                    all_llama_vs_fp32_flip_margins.append(_top1_top2_margin(r_vec))

                if top1_asi != top1_llama:
                    margin = _top1_top2_margin(r_vec)
                    asi_gap_at_flip = float(r_vec[top1_ref] - r_vec[top1_asi])
                    llama_gap_at_flip = float(r_vec[top1_ref] - r_vec[top1_llama])
                    abs_diff_at_top1_asi = float(abs(float(a_vec[top1_asi]) - float(l_vec[top1_asi])))
                    abs_diff_at_top1_llama = float(abs(float(a_vec[top1_llama]) - float(l_vec[top1_llama])))
                    abs_diff_at_top1_ref = float(abs(float(a_vec[top1_ref]) - float(l_vec[top1_ref])))
                    abs_top1_logit_diff = float(abs(float(a_vec[top1_asi]) - float(l_vec[top1_llama])))

                    flip_entry: Dict[str, object] = {
                        "prompt": name,
                        "pos": k,
                        "abs_pos": prompt_len + k,
                        "top1_asi": top1_asi,
                        "top1_llama": top1_llama,
                        "top1_ref": top1_ref,
                        "tok_asi": tokenizer.decode([top1_asi]),
                        "tok_llama": tokenizer.decode([top1_llama]),
                        "tok_ref": tokenizer.decode([top1_ref]),
                        "asi_matched_ref": top1_asi == top1_ref,
                        "llama_matched_ref": top1_llama == top1_ref,
                        "fp32_ref_margin": margin,
                        "asi_gap_at_flip": asi_gap_at_flip,
                        "llama_gap_at_flip": llama_gap_at_flip,
                        "asi_logit_at_top1_asi": float(a_vec[top1_asi]),
                        "llama_logit_at_top1_asi": float(l_vec[top1_asi]),
                        "abs_diff_at_top1_asi": abs_diff_at_top1_asi,
                        "asi_logit_at_top1_llama": float(a_vec[top1_llama]),
                        "llama_logit_at_top1_llama": float(l_vec[top1_llama]),
                        "abs_diff_at_top1_llama": abs_diff_at_top1_llama,
                        "abs_diff_at_top1_ref": abs_diff_at_top1_ref,
                        "abs_top1_logit_diff": abs_top1_logit_diff,
                        "max_abs_diff_pair": max(abs_diff_at_top1_asi, abs_diff_at_top1_llama),
                        "rel_l2_asi_vs_fp32": rel_l2_asi,
                        "rel_l2_llama_vs_fp32": rel_l2_llama,
                    }
                    p_flips.append(flip_entry)
                    all_flips.append(flip_entry)
        finally:
            del llama_logits, asi_logits, ref_logits

        total_positions += n_gen
        total_asi_vs_llama += p_asi_vs_llama
        total_asi_vs_fp32 += p_asi_vs_fp32
        total_llama_vs_fp32 += p_llama_vs_fp32

        p_asi_l2_stats = _agg_stats(p_asi_rel_l2)
        p_llama_l2_stats = _agg_stats(p_llama_rel_l2)
        p_flip_margins = [float(f["fp32_ref_margin"]) for f in p_flips]
        p_flip_margin_stats = _agg_stats(p_flip_margins)
        p_asi_gap_stats = _agg_stats([float(f["asi_gap_at_flip"]) for f in p_flips])

        per_prompt_tf.append(
            {
                "n_positions": n_gen,
                "asi_vs_llama_agree": p_asi_vs_llama,
                "asi_vs_llama_pct": 100.0 * p_asi_vs_llama / n_gen,
                "asi_vs_fp32_agree": p_asi_vs_fp32,
                "asi_vs_fp32_pct": 100.0 * p_asi_vs_fp32 / n_gen,
                "llama_vs_fp32_agree": p_llama_vs_fp32,
                "llama_vs_fp32_pct": 100.0 * p_llama_vs_fp32 / n_gen,
                "asi_rel_l2_vs_fp32_mean": p_asi_l2_stats["mean"],
                "asi_rel_l2_vs_fp32_median": p_asi_l2_stats["median"],
                "asi_rel_l2_vs_fp32_max": p_asi_l2_stats["max"],
                "llama_rel_l2_vs_fp32_mean": p_llama_l2_stats["mean"],
                "llama_rel_l2_vs_fp32_median": p_llama_l2_stats["median"],
                "llama_rel_l2_vs_fp32_max": p_llama_l2_stats["max"],
                "n_flips_asi_vs_llama": len(p_flips),
                "flip_fp32_margin_max": p_flip_margin_stats["max"],
                "flip_fp32_margin_median": p_flip_margin_stats["median"],
                "flip_asi_gap_max": p_asi_gap_stats["max"],
                "flip_asi_gap_median": p_asi_gap_stats["median"],
                "flips": p_flips,
            }
        )

    asi_l2_stats = _agg_stats(all_asi_rel_l2)
    llama_l2_stats = _agg_stats(all_llama_rel_l2)
    flip_margins = [float(f["fp32_ref_margin"]) for f in all_flips]
    flip_margin_stats = _agg_stats(flip_margins)
    asi_gap_stats = _agg_stats([float(f["asi_gap_at_flip"]) for f in all_flips])
    diff_at_asi_stats = _agg_stats([float(f["abs_diff_at_top1_asi"]) for f in all_flips])
    diff_at_llama_stats = _agg_stats([float(f["abs_diff_at_top1_llama"]) for f in all_flips])
    max_pair_diff_stats = _agg_stats([float(f["max_abs_diff_pair"]) for f in all_flips])
    asi_vs_fp32_flip_stats = _agg_stats(all_asi_vs_fp32_flip_margins)
    llama_vs_fp32_flip_stats = _agg_stats(all_llama_vs_fp32_flip_margins)

    asi_matched_ref_flips = sum(1 for f in all_flips if bool(f["asi_matched_ref"]))
    llama_matched_ref_flips = sum(1 for f in all_flips if bool(f["llama_matched_ref"]))
    neither_matched_ref_flips = len(all_flips) - asi_matched_ref_flips - llama_matched_ref_flips

    tf_summary: Dict[str, object] = {
        "total_positions": total_positions,
        "fp32_ref_diff_detected": max_abs_q8_vs_fp32 > 0.0,
        "max_abs_q8_vs_fp32_logit_diff": max_abs_q8_vs_fp32,
        "asi_vs_llama_agree_positions": total_asi_vs_llama,
        "asi_vs_llama_agree_pct": 100.0 * total_asi_vs_llama / total_positions if total_positions else 0.0,
        "asi_vs_fp32_agree_positions": total_asi_vs_fp32,
        "asi_vs_fp32_agree_pct": 100.0 * total_asi_vs_fp32 / total_positions if total_positions else 0.0,
        "llama_vs_fp32_agree_positions": total_llama_vs_fp32,
        "llama_vs_fp32_agree_pct": 100.0 * total_llama_vs_fp32 / total_positions if total_positions else 0.0,
        "asi_rel_l2_vs_fp32": asi_l2_stats,
        "llama_rel_l2_vs_fp32": llama_l2_stats,
        "flips_asi_vs_llama": {
            "count": len(all_flips),
            "asi_matched_ref_count": asi_matched_ref_flips,
            "llama_matched_ref_count": llama_matched_ref_flips,
            "neither_matched_ref_count": neither_matched_ref_flips,
            "fp32_ref_margin": flip_margin_stats,
            "asi_gap_at_flip": asi_gap_stats,
            "abs_diff_at_top1_asi": diff_at_asi_stats,
            "abs_diff_at_top1_llama": diff_at_llama_stats,
            "max_abs_diff_pair": max_pair_diff_stats,
        },
        "flips_asi_vs_fp32": {
            "count": len(all_asi_vs_fp32_flip_margins),
            "fp32_ref_margin": asi_vs_fp32_flip_stats,
        },
        "flips_llama_vs_fp32": {
            "count": len(all_llama_vs_fp32_flip_margins),
            "fp32_ref_margin": llama_vs_fp32_flip_stats,
        },
    }
    return per_prompt_tf, tf_summary


def _print_greedy_report(
    results: Sequence[Dict[str, object]],
    n_gen: int,
    total_tokens: int,
    total_bt_agree: int,
    total_bat_vs_llama_agree: int,
    total_tok_vs_llama_agree: int,
    corpus_bt_pct: float,
    corpus_bat_vs_llama_pct: float,
    corpus_tok_vs_llama_pct: float,
) -> None:
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


def _print_teacher_forced_report(results: Sequence[Dict[str, object]], tf_summary: Dict[str, object]) -> None:
    total_pos = int(tf_summary["total_positions"])
    print(
        f"\n=== Teacher-Forced + FP32-Reference Audit "
        f"({len(results)} prompts x {total_pos // len(results) if results else 0} = {total_pos} positions) ==="
    )
    if not tf_summary["fp32_ref_diff_detected"]:
        print(
            "  [NOTE] ASI(Q8_K) and ASI(FP32-Ref) logits were identical (max_abs_diff=0.0); "
            "LILA_ASI_FP32_REF=1 may not be compiled into liblila_asi.so."
        )

    header = (
        f"{'Prompt':<10} {'Len':>3} | "
        f"{'ASI/Llama':>9} {'ASI/FP32':>9} {'Llm/FP32':>9} | "
        f"{'ASI L2 (mean/med)':>19} {'Llm L2 (mean/med)':>19} | "
        f"{'Flips':>5}"
    )
    print(header)
    print("-" * len(header))
    all_flips: List[Dict[str, object]] = []
    for r in results:
        tf = r["teacher_forced"]
        assert isinstance(tf, dict)
        all_flips.extend(tf["flips"])
        asi_l2_str = f"{_fmt_opt(tf['asi_rel_l2_vs_fp32_mean'], '.5f')}/{_fmt_opt(tf['asi_rel_l2_vs_fp32_median'], '.5f')}"
        llm_l2_str = f"{_fmt_opt(tf['llama_rel_l2_vs_fp32_mean'], '.5f')}/{_fmt_opt(tf['llama_rel_l2_vs_fp32_median'], '.5f')}"
        print(
            f"{r['name']:<10} {r['prompt_len']:>3} | "
            f"{tf['asi_vs_llama_pct']:>8.2f}% {tf['asi_vs_fp32_pct']:>8.2f}% {tf['llama_vs_fp32_pct']:>8.2f}% | "
            f"{asi_l2_str:>19} {llm_l2_str:>19} | "
            f"{tf['n_flips_asi_vs_llama']:>5}"
        )

    print("\n=== Teacher-Forced Corpus Summary ===")
    print(
        f"  ASI(Q8_K) vs llama.cpp : "
        f"{tf_summary['asi_vs_llama_agree_positions']}/{total_pos} ({tf_summary['asi_vs_llama_agree_pct']:.2f}%)"
    )
    print(
        f"  ASI(Q8_K) vs FP32-Ref  : "
        f"{tf_summary['asi_vs_fp32_agree_positions']}/{total_pos} ({tf_summary['asi_vs_fp32_agree_pct']:.2f}%)"
    )
    print(
        f"  llama.cpp vs FP32-Ref  : "
        f"{tf_summary['llama_vs_fp32_agree_positions']}/{total_pos} ({tf_summary['llama_vs_fp32_agree_pct']:.2f}%)"
    )

    asi_l2 = tf_summary["asi_rel_l2_vs_fp32"]
    llm_l2 = tf_summary["llama_rel_l2_vs_fp32"]
    print(
        f"  Logit Rel-L2 vs FP32-Ref | ASI(Q8_K): "
        f"mean={_fmt_opt(asi_l2['mean'], '.6f')}, median={_fmt_opt(asi_l2['median'], '.6f')}, "
        f"max={_fmt_opt(asi_l2['max'], '.6f')}"
    )
    print(
        f"  Logit Rel-L2 vs FP32-Ref | llama.cpp: "
        f"mean={_fmt_opt(llm_l2['mean'], '.6f')}, median={_fmt_opt(llm_l2['median'], '.6f')}, "
        f"max={_fmt_opt(llm_l2['max'], '.6f')}"
    )

    flips_info = tf_summary["flips_asi_vs_llama"]
    fm = flips_info["fp32_ref_margin"]
    ag = flips_info["asi_gap_at_flip"]
    da = flips_info["abs_diff_at_top1_asi"]
    dl = flips_info["abs_diff_at_top1_llama"]
    print("\n=== Flip Margin Diagnostics (argmax(ASI) != argmax(llama.cpp)) ===")
    print(
        f"  Flips: {flips_info['count']}/{total_pos} "
        f"(ASI==FP32: {flips_info['asi_matched_ref_count']}, "
        f"Llama==FP32: {flips_info['llama_matched_ref_count']}, "
        f"Neither==FP32: {flips_info['neither_matched_ref_count']})"
    )
    print(
        f"  FP32-Ref top1-top2 margin at flips : "
        f"max={_fmt_opt(fm['max'])}, median={_fmt_opt(fm['median'])}, min={_fmt_opt(fm['min'])}"
    )
    print(
        f"  ASI gap at flip (ref[top1]-ref[asi]): "
        f"max={_fmt_opt(ag['max'])}, median={_fmt_opt(ag['median'])}"
    )
    print(
        f"  |asi_logit - llama_logit| at flips : "
        f"top1_asi(max={_fmt_opt(da['max'])}, med={_fmt_opt(da['median'])}), "
        f"top1_llama(max={_fmt_opt(dl['max'])}, med={_fmt_opt(dl['median'])})"
    )

    if all_flips:
        print("\n=== Per-Flip Details ===")
        limit = 40
        for f in all_flips[:limit]:
            print(
                f"  [{f['prompt']} k={f['pos']}] "
                f"asi={f['top1_asi']}({f['tok_asi']!r}) vs "
                f"llama={f['top1_llama']}({f['tok_llama']!r}) | "
                f"ref={f['top1_ref']} | "
                f"fp32_margin={f['fp32_ref_margin']:.4f}, "
                f"asi_gap={f['asi_gap_at_flip']:.4f}, "
                f"|a-l|@asi={f['abs_diff_at_top1_asi']:.4f}, "
                f"|a-l|@llama={f['abs_diff_at_top1_llama']:.4f}"
            )
        if len(all_flips) > limit:
            print(f"  ... ({len(all_flips) - limit} more flips omitted; see --json-out)")


def main() -> int:
    parser = argparse.ArgumentParser(description="ASI vs llama.cpp correctness & numerical parity gate")
    parser.add_argument("--asi", default=ASI_PATH, help=f"Path to ASI model (default: {ASI_PATH})")
    parser.add_argument("--gguf", default=GGUF_PATH, help=f"Path to GGUF model (default: {GGUF_PATH})")
    parser.add_argument("--n-ctx", type=int, default=N_CTX, help=f"Context length (default: {N_CTX})")
    parser.add_argument("--n-threads", type=int, default=N_THREADS, help=f"Thread count (default: {N_THREADS})")
    parser.add_argument("--n-gen", type=int, default=N_GEN, help=f"Greedy tokens to generate per prompt (default: {N_GEN})")
    parser.add_argument(
        "--teacher-forced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run teacher-forced + FP32-Reference audit (default: True; pass --no-teacher-forced to skip)",
    )
    parser.add_argument("--json-out", default=None, help="Export correctness gate results and corpus summary to JSON at PATH")
    parser.add_argument("--fp32-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-manifest", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.fp32_worker:
        if not args.worker_manifest:
            raise ValueError("--worker-manifest is required with --fp32-worker")
        return _run_fp32_worker(args.worker_manifest)

    n_gen = args.n_gen
    if n_gen <= 0:
        raise ValueError(f"--n-gen must be positive, got {n_gen}")

    # Ensure primary ASI session runs quantized (Q8_K) even if LILA_ASI_FP32_REF was exported in parent shell.
    os.environ.pop("LILA_ASI_FP32_REF", None)

    asi: Optional[AsiLlamaAdapter] = None
    llama: Optional[Llama] = None
    results: List[Dict[str, object]] = []
    tf_summary: Optional[Dict[str, object]] = None
    vocab_compared: Optional[int] = None
    asi_sha256 = _sha256_file(args.asi)
    gguf_sha256 = _sha256_file(args.gguf)

    try:
        print(f"ASI sha256  : {asi_sha256}")
        print(f"GGUF sha256 : {gguf_sha256}")
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
        tokenizer = asi.tokenizer
        asi_vocab = int(asi._cfg.vocab_size)
        llama_vocab = int(llama.n_vocab())
        vocab_compared = min(asi_vocab, llama_vocab)
        if asi_vocab != llama_vocab:
            print(
                f"[NOTE] Vocab width differs: asi={asi_vocab}, llama.n_vocab()={llama_vocab}; "
                f"comparing [: {vocab_compared}]"
            )

        encoded_prompts: List[Tuple[str, List[int]]] = []
        for name, prompt in CORPUS:
            prompt_ids = tokenizer.encode(prompt)
            if not prompt_ids:
                raise ValueError(f"Prompt '{name}' encoded to 0 tokens")
            if len(prompt_ids) + n_gen - 1 > args.n_ctx:
                raise ValueError(
                    f"Prompt '{name}' length {len(prompt_ids)} + n_gen - 1 ({n_gen - 1}) "
                    f"exceeds n_ctx={args.n_ctx}"
                )
            encoded_prompts.append((name, prompt_ids))

        tf_sequences: List[Dict[str, object]] = []
        for name, prompt_ids in encoded_prompts:
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
                    "text_asi_bat": tokenizer.decode(gen_bat),
                    "text_asi_tok": tokenizer.decode(gen_tok),
                    "text_llama": tokenizer.decode(gen_llama),
                }
            )
            tf_sequences.append(
                {
                    "name": name,
                    "prompt_len": len(prompt_ids),
                    "n_gen": n_gen,
                    "prompt_ids": prompt_ids,
                    "gen_ids": gen_llama,
                }
            )

        if args.teacher_forced:
            with tempfile.TemporaryDirectory(prefix="lila_correctness_tf_") as tmp_str:
                tmpdir = Path(tmp_str)
                tf_records: List[Dict[str, object]] = []
                worker_jobs: List[Dict[str, object]] = []

                print("\nCollecting teacher-forced logits from llama.cpp...")
                for idx, seq in enumerate(tf_sequences):
                    llama_npy = tmpdir / f"llama_{idx}.npy"
                    asi_q8_npy = tmpdir / f"asi_q8_{idx}.npy"
                    asi_fp32_npy = tmpdir / f"asi_fp32_{idx}.npy"

                    arr_llama = _llama_teacher_forced_logits(
                        llama, seq["prompt_ids"], seq["gen_ids"], vocab_compared
                    )
                    np.save(llama_npy, arr_llama)
                    del arr_llama

                    tf_records.append(
                        {
                            "name": seq["name"],
                            "prompt_len": seq["prompt_len"],
                            "n_gen": seq["n_gen"],
                            "llama_npy": str(llama_npy),
                            "asi_q8_npy": str(asi_q8_npy),
                            "asi_fp32_npy": str(asi_fp32_npy),
                        }
                    )
                    worker_jobs.append(
                        {
                            "name": seq["name"],
                            "prompt_ids": seq["prompt_ids"],
                            "gen_ids": seq["gen_ids"],
                            "out_npy": str(asi_fp32_npy),
                        }
                    )

                # Free llama.cpp before running ASI teacher-forced & FP32-Ref worker to keep RSS bounded.
                del llama
                llama = None
                gc.collect()

                print("Collecting teacher-forced logits from ASI(Q8_K)...")
                for idx, seq in enumerate(tf_sequences):
                    arr_asi = _asi_teacher_forced_logits(
                        asi, seq["prompt_ids"], seq["gen_ids"], vocab_compared
                    )
                    np.save(str(tf_records[idx]["asi_q8_npy"]), arr_asi)
                    del arr_asi

                # Free parent ASI session before launching the LILA_ASI_FP32_REF=1 worker subprocess.
                asi.close()
                asi = None
                gc.collect()

                manifest_path = tmpdir / "fp32_manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "asi_path": args.asi,
                            "n_ctx": args.n_ctx,
                            "n_threads": args.n_threads,
                            "vocab": vocab_compared,
                            "jobs": worker_jobs,
                        }
                    ),
                    encoding="utf-8",
                )
                worker_env = os.environ.copy()
                worker_env["LILA_ASI_FP32_REF"] = "1"
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--fp32-worker",
                        "--worker-manifest",
                        str(manifest_path),
                    ],
                    env=worker_env,
                    check=True,
                )

                per_prompt_tf, tf_summary = _evaluate_teacher_forced_corpus(tf_records, tokenizer)
                for res, p_tf in zip(results, per_prompt_tf):
                    res["teacher_forced"] = p_tf
    finally:
        if asi is not None:
            asi.close()
        if llama is not None:
            del llama

    total_tokens = len(results) * n_gen
    total_bt_agree = sum(int(r["bt_agree"]) for r in results)
    total_bat_vs_llama_agree = sum(int(r["bat_vs_llama_agree"]) for r in results)
    total_tok_vs_llama_agree = sum(int(r["tok_vs_llama_agree"]) for r in results)

    corpus_bt_pct = 100.0 * total_bt_agree / total_tokens if total_tokens else 0.0
    corpus_bat_vs_llama_pct = 100.0 * total_bat_vs_llama_agree / total_tokens if total_tokens else 0.0
    corpus_tok_vs_llama_pct = 100.0 * total_tok_vs_llama_agree / total_tokens if total_tokens else 0.0

    _print_greedy_report(
        results,
        n_gen,
        total_tokens,
        total_bt_agree,
        total_bat_vs_llama_agree,
        total_tok_vs_llama_agree,
        corpus_bt_pct,
        corpus_bat_vs_llama_pct,
        corpus_tok_vs_llama_pct,
    )

    if tf_summary is not None:
        _print_teacher_forced_report(results, tf_summary)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_payload: Dict[str, object] = {
            "total_tokens": total_tokens,
            "bt_agree_tokens": total_bt_agree,
            "bt_agree_pct": corpus_bt_pct,
            "bat_vs_llama_agree_tokens": total_bat_vs_llama_agree,
            "bat_vs_llama_agree_pct": corpus_bat_vs_llama_pct,
            "tok_vs_llama_agree_tokens": total_tok_vs_llama_agree,
            "tok_vs_llama_agree_pct": corpus_tok_vs_llama_pct,
        }
        if tf_summary is not None:
            summary_payload["teacher_forced"] = tf_summary

        config_payload: Dict[str, object] = {
            "asi_path": args.asi,
            "asi_sha256": asi_sha256,
            "gguf_path": args.gguf,
            "gguf_sha256": gguf_sha256,
            "n_ctx": args.n_ctx,
            "n_threads": args.n_threads,
            "n_gen": n_gen,
            "n_prompts": len(results),
            "teacher_forced": bool(args.teacher_forced),
        }
        if vocab_compared is not None:
            config_payload["vocab_compared"] = vocab_compared

        payload = {
            "config": config_payload,
            "prompts": results,
            "summary": summary_payload,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nExported JSON results to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
