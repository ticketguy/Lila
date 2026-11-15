#!/usr/bin/env python3
"""
Strict, apples-to-apples benchmark harness: Lila ASI (liblila_asi.so C ABI)
vs llama-cpp-python (Llama.eval + Llama.sample) on identical token IDs.

Requirements enforced:
1. Identical integer token IDs tokenized once and fed to both engines.
2. Verified KV/prefix reuse on Turn 2 (only new Turn-2 tokens evaluated).
3. Matched settings: n_ctx=2048, n_threads=2, n_threads_batch=2, n_batch=512,
   use_mmap=True, flash_attn=False, greedy (temperature=0.0 / argmax).
4. Page-cache warmup (1 discarded trial per engine) + N interleaved pairs.
5. Fair load measurement by default: load_open_s (open/map) + populate_s
   (MADV_WILLNEED + touch 1 byte per 4 KiB page right after open) -> load_s
   (pass --no-prepopulate for lazy load only).
6. Primary metrics: t1_ttft, t1_dec, t2_ttft, t2_dec, long_prefill_tps, rss_mb.
   Secondary metrics: vm_hwm_mb, load_s, load_open_s.
7. Pure-stdlib stats: median, [min..max], p95, max, 95% bootstrap CI,
   Mann-Whitney U two-sided p-value, and honest verdicts (PASS,
   ADVANTAGE (overlap), PARITY (p=...), FAIL).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import mmap
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_ASI_PATH = "/tmp/qwen35_0.8b.asi"
DEFAULT_GGUF_PATH = "/home/ticket/workspaces/lila/models/Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf"
DEFAULT_LIB_PATH = str(Path(__file__).resolve().parent.parent / "asi" / "liblila_asi.so")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

N_CTX = 2048
N_THREADS = 2
N_BATCH = 512
N_GEN = 32

T1_PROMPT = (
    "<|im_start|>system\n"
    "You are Lila, a concise local AI assistant. Answer clearly and thoroughly."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Explain in detail how CPU L1, L2, and L3 caches affect local LLM token generation speed."
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

T2_DELTA = (
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Now explain in detail why memory bandwidth is usually the main bottleneck during autoregressive decoding."
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# ~860-token prompt matching LilaCore system + tools length
_TOOL_SPEC = (
    '{"type":"function","function":{"name":"read_file","description":"Read file contents from workspace.",'
    '"parameters":{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"},'
    '"end_line":{"type":"integer"}},"required":["path"]}}}\n'
    '{"type":"function","function":{"name":"write_file","description":"Write or overwrite a file in workspace.",'
    '"parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},'
    '"required":["path","content"]}}}\n'
    '{"type":"function","function":{"name":"run_command","description":"Execute a shell command in workspace.",'
    '"parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout_s":{"type":"integer"}},'
    '"required":["command"]}}}\n'
    '{"type":"function","function":{"name":"search_code","description":"Search codebase with ripgrep.",'
    '"parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},'
    '"required":["pattern"]}}}\n'
)

LONG_PROMPT = (
    "<|im_start|>system\n"
    "You are LilaCore, a local-first AI agent running on the user's workstation. "
    "Follow instructions carefully, verify file paths before editing, and prefer concise, accurate answers.\n\n"
    "## Available Tools\n"
    + (_TOOL_SPEC * 4)
    + "\n## Memory & Context Rules\n"
    "1. Always cite exact file paths and symbol names observed in tool output.\n"
    "2. Never fabricate command results or test outcomes.\n"
    "3. Keep responses structured and avoid unnecessary verbosity.\n"
    "4. Respect workspace boundaries and never modify files outside the active repository.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Summarize the available tools and confirm you are ready to assist.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class _AsiV2ModelConfig(ctypes.Structure):
    _fields_ = [
        ("arch_id", ctypes.c_uint32),
        ("n_layers", ctypes.c_uint32),
        ("hidden_size", ctypes.c_uint32),
        ("intermediate_size", ctypes.c_uint32),
        ("n_heads", ctypes.c_uint32),
        ("n_kv_heads", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("rope_dim", ctypes.c_uint32),
        ("rope_theta", ctypes.c_float),
        ("rms_norm_eps", ctypes.c_float),
        ("embed_scale", ctypes.c_float),
        ("full_attention_interval", ctypes.c_uint32),
        ("ssm_conv_kernel", ctypes.c_uint32),
        ("ssm_state_size", ctypes.c_uint32),
        ("ssm_group_count", ctypes.c_uint32),
        ("ssm_inner_size", ctypes.c_uint32),
        ("ssm_time_step_rank", ctypes.c_uint32),
        ("default_quant_type", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


def _get_mem_mb() -> Tuple[float, float]:
    """Return (VmRSS_MiB, VmHWM_MiB) from /proc/self/status."""
    rss_mb = 0.0
    hwm_mb = 0.0
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss_mb = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                hwm_mb = int(line.split()[1]) / 1024.0
    return rss_mb, hwm_mb


def _prepopulate_file(path: str) -> float:
    """Advise MADV_WILLNEED and touch one byte per 4 KiB page; return elapsed seconds."""
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if hasattr(mm, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
                mm.madvise(mmap.MADV_WILLNEED)
            size = len(mm)
            acc = 0
            for off in range(0, size, 4096):
                acc ^= mm[off]
            if acc == -1:  # never true; keeps loop from being elided
                raise RuntimeError("unreachable")
    return time.perf_counter() - t0


def _bind_asi_lib(lib_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(lib_path)
    lib.lila_asi_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.lila_asi_open.restype = ctypes.c_void_p
    lib.lila_asi_close.argtypes = [ctypes.c_void_p]
    lib.lila_asi_model_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AsiV2ModelConfig)]
    lib.lila_asi_model_info.restype = ctypes.c_int
    lib.lila_asi_session_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.lila_asi_session_create.restype = ctypes.c_void_p
    lib.lila_asi_session_free.argtypes = [ctypes.c_void_p]
    lib.lila_asi_session_reset.argtypes = [ctypes.c_void_p]
    lib.lila_asi_eval_token.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.lila_asi_eval_token.restype = ctypes.c_int
    lib.lila_asi_eval_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.lila_asi_eval_batch.restype = ctypes.c_int
    lib.lila_asi_sample.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.lila_asi_sample.restype = ctypes.c_int32
    return lib


def run_asi_trial(
    asi_path: str,
    lib_path: str,
    t1_ids: List[int],
    t2_ids: List[int],
    long_ids: List[int],
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    n_gen: int,
    prepopulate: bool = True,
) -> Dict:
    lib = _bind_asi_lib(lib_path)
    err = ctypes.create_string_buffer(256)

    t_load0 = time.perf_counter()
    model = lib.lila_asi_open(asi_path.encode("utf-8"), err, 256)
    if not model:
        raise RuntimeError(f"lila_asi_open failed: {err.value.decode('utf-8', 'replace')}")
    cfg = _AsiV2ModelConfig()
    if lib.lila_asi_model_info(model, ctypes.byref(cfg)) != 0:
        raise RuntimeError("lila_asi_model_info failed")
    sess = lib.lila_asi_session_create(model, n_ctx, n_threads, err, 256)
    if not sess:
        raise RuntimeError(f"lila_asi_session_create failed: {err.value.decode('utf-8', 'replace')}")
    load_open_s = time.perf_counter() - t_load0

    populate_s = _prepopulate_file(asi_path) if prepopulate else 0.0
    load_s = load_open_s + populate_s

    logits = (ctypes.c_float * cfg.vocab_size)()
    rng = ctypes.c_uint64(42)

    # --- Turn 1 ---
    lib.lila_asi_session_reset(sess)
    t1_arr = (ctypes.c_int32 * len(t1_ids))(*t1_ids)
    t0 = time.perf_counter()
    for off in range(0, len(t1_ids), n_batch):
        chunk = t1_ids[off : min(len(t1_ids), off + n_batch)]
        c_arr = (ctypes.c_int32 * len(chunk))(*chunk)
        if lib.lila_asi_eval_batch(sess, c_arr, len(chunk), off, logits) != 0:
            raise RuntimeError("ASI Turn 1 eval_batch failed")
    tok = int(lib.lila_asi_sample(logits, cfg.vocab_size, 0.0, 1, 1.0, ctypes.byref(rng)))
    t1_ttft = time.perf_counter() - t0

    pos = len(t1_ids)
    t1_gen = [tok]
    t_dec0 = time.perf_counter()
    for _ in range(n_gen):
        if lib.lila_asi_eval_token(sess, tok, pos, logits) != 0:
            raise RuntimeError("ASI Turn 1 eval_token failed")
        pos += 1
        tok = int(lib.lila_asi_sample(logits, cfg.vocab_size, 0.0, 1, 1.0, ctypes.byref(rng)))
        t1_gen.append(tok)
    t1_dec_s = time.perf_counter() - t_dec0
    t1_dec = n_gen / t1_dec_s

    # --- Turn 2 (reuse Turn 1 KV/DeltaNet state, NO reset) ---
    pos_before_t2 = pos
    assert pos_before_t2 == len(t1_ids) + n_gen, f"ASI pos mismatch: {pos_before_t2}"
    t0 = time.perf_counter()
    t2_eval_count = 0
    for off in range(0, len(t2_ids), n_batch):
        chunk = t2_ids[off : min(len(t2_ids), off + n_batch)]
        c_arr = (ctypes.c_int32 * len(chunk))(*chunk)
        if lib.lila_asi_eval_batch(sess, c_arr, len(chunk), pos_before_t2 + off, logits) != 0:
            raise RuntimeError("ASI Turn 2 eval_batch failed")
        t2_eval_count += len(chunk)
    tok = int(lib.lila_asi_sample(logits, cfg.vocab_size, 0.0, 1, 1.0, ctypes.byref(rng)))
    t2_ttft = time.perf_counter() - t0
    assert t2_eval_count == len(t2_ids), f"ASI Turn 2 eval count {t2_eval_count} != {len(t2_ids)}"

    pos = pos_before_t2 + t2_eval_count
    t2_gen = [tok]
    t_dec0 = time.perf_counter()
    for _ in range(n_gen):
        if lib.lila_asi_eval_token(sess, tok, pos, logits) != 0:
            raise RuntimeError("ASI Turn 2 eval_token failed")
        pos += 1
        tok = int(lib.lila_asi_sample(logits, cfg.vocab_size, 0.0, 1, 1.0, ctypes.byref(rng)))
        t2_gen.append(tok)
    t2_dec_s = time.perf_counter() - t_dec0
    t2_dec = n_gen / t2_dec_s

    # --- Long Prefill (~860 tokens) ---
    lib.lila_asi_session_reset(sess)
    t0 = time.perf_counter()
    for off in range(0, len(long_ids), n_batch):
        chunk = long_ids[off : min(len(long_ids), off + n_batch)]
        c_arr = (ctypes.c_int32 * len(chunk))(*chunk)
        if lib.lila_asi_eval_batch(sess, c_arr, len(chunk), off, logits) != 0:
            raise RuntimeError("ASI long prefill eval_batch failed")
    long_s = time.perf_counter() - t0
    long_prefill_tps = len(long_ids) / long_s

    rss_mb, vm_hwm_mb = _get_mem_mb()

    lib.lila_asi_session_free(sess)
    lib.lila_asi_close(model)

    return {
        "engine": "asi",
        "load_s": load_s,
        "load_open_s": load_open_s,
        "populate_s": populate_s,
        "t1_ttft": t1_ttft,
        "t1_dec": t1_dec,
        "t2_ttft": t2_ttft,
        "t2_dec": t2_dec,
        "long_prefill_tps": long_prefill_tps,
        "rss_mb": rss_mb,
        "vm_hwm_mb": vm_hwm_mb,
        "t2_eval_tokens": t2_eval_count,
        "t2_pos_before": pos_before_t2,
        "t1_first_tok": t1_gen[0],
        "t2_first_tok": t2_gen[0],
    }


def run_llama_trial(
    gguf_path: str,
    t1_ids: List[int],
    t2_ids: List[int],
    long_ids: List[int],
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    n_gen: int,
    prepopulate: bool = True,
) -> Dict:
    from llama_cpp import Llama

    t_load0 = time.perf_counter()
    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_threads_batch=n_threads,
        n_batch=n_batch,
        use_mmap=True,
        flash_attn=False,
        n_gpu_layers=0,
        verbose=False,
    )
    load_open_s = time.perf_counter() - t_load0

    populate_s = _prepopulate_file(gguf_path) if prepopulate else 0.0
    load_s = load_open_s + populate_s

    def sample_greedy() -> int:
        return int(llm.sample(temp=0.0, top_k=1, top_p=1.0, min_p=0.0, repeat_penalty=1.0))

    # --- Turn 1 ---
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(t1_ids)
    tok = sample_greedy()
    t1_ttft = time.perf_counter() - t0

    t1_gen = [tok]
    t_dec0 = time.perf_counter()
    for _ in range(n_gen):
        llm.eval([tok])
        tok = sample_greedy()
        t1_gen.append(tok)
    t1_dec_s = time.perf_counter() - t_dec0
    t1_dec = n_gen / t1_dec_s

    # --- Turn 2 (reuse Turn 1 KV/recurrent state, NO reset) ---
    n_before_t2 = int(llm.n_tokens)
    assert n_before_t2 == len(t1_ids) + n_gen, f"llama n_tokens mismatch: {n_before_t2}"
    t0 = time.perf_counter()
    llm.eval(t2_ids)
    tok = sample_greedy()
    t2_ttft = time.perf_counter() - t0
    n_after_t2 = int(llm.n_tokens)
    t2_eval_count = n_after_t2 - n_before_t2
    assert t2_eval_count == len(t2_ids), f"llama Turn 2 eval count {t2_eval_count} != {len(t2_ids)}"

    t2_gen = [tok]
    t_dec0 = time.perf_counter()
    for _ in range(n_gen):
        llm.eval([tok])
        tok = sample_greedy()
        t2_gen.append(tok)
    t2_dec_s = time.perf_counter() - t_dec0
    t2_dec = n_gen / t2_dec_s

    # --- Long Prefill (~860 tokens) ---
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(long_ids)
    long_s = time.perf_counter() - t0
    long_prefill_tps = len(long_ids) / long_s

    rss_mb, vm_hwm_mb = _get_mem_mb()
    llm.close()

    return {
        "engine": "llama",
        "load_s": load_s,
        "load_open_s": load_open_s,
        "populate_s": populate_s,
        "t1_ttft": t1_ttft,
        "t1_dec": t1_dec,
        "t2_ttft": t2_ttft,
        "t2_dec": t2_dec,
        "long_prefill_tps": long_prefill_tps,
        "rss_mb": rss_mb,
        "vm_hwm_mb": vm_hwm_mb,
        "t2_eval_tokens": t2_eval_count,
        "t2_pos_before": n_before_t2,
        "t1_first_tok": t1_gen[0],
        "t2_first_tok": t2_gen[0],
    }


def _bootstrap_median_ci(xs: Sequence[float], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(xs)
    if n == 1:
        return (xs[0], xs[0])
    medians = sorted(statistics.median(rng.choices(xs, k=n)) for _ in range(n_boot))
    lo = medians[int(0.025 * n_boot)]
    hi = medians[min(n_boot - 1, int(0.975 * n_boot))]
    return (lo, hi)


def _mann_whitney_u_p(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sided Mann-Whitney U p-value in pure Python stdlib."""
    n1, n2 = len(x), len(y)
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])

    n_tot = n1 + n2
    ranks2 = [0] * n_tot  # doubled ranks (integers) to handle ties exactly
    tie_groups: List[int] = []
    i = 0
    while i < n_tot:
        j = i + 1
        while j < n_tot and combined[j][0] == combined[i][0]:
            j += 1
        # 1-based average rank is (i+1 + j)/2, so doubled rank is (i + 1 + j)
        avg_r2 = i + 1 + j
        for k in range(i, j):
            ranks2[k] = avg_r2
        if j - i > 1:
            tie_groups.append(j - i)
        i = j

    r1_2 = sum(ranks2[k] for k in range(n_tot) if combined[k][1] == 0)
    u1 = r1_2 / 2.0 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u_obs = min(u1, u2)

    # Exact DP over doubled rank sums when n_tot <= 20
    if n_tot <= 20:
        dp: Dict[Tuple[int, int], int] = {(0, 0): 1}
        for r2 in ranks2:
            new_dp = dict(dp)
            for (cnt, s), ways in dp.items():
                if cnt < n1:
                    key = (cnt + 1, s + r2)
                    new_dp[key] = new_dp.get(key, 0) + ways
            dp = new_dp

        total_ways = math.comb(n_tot, n1)
        target_max = min(u1, u2)
        extreme = 0
        for (cnt, s), ways in dp.items():
            if cnt == n1:
                cur_u1 = s / 2.0 - n1 * (n1 + 1) / 2.0
                cur_u = min(cur_u1, n1 * n2 - cur_u1)
                if cur_u <= target_max + 1e-9:
                    extreme += ways
        return min(1.0, extreme / total_ways)

    # Normal approximation with tie correction + continuity correction
    mean_u = n1 * n2 / 2.0
    tie_term = sum(t**3 - t for t in tie_groups) / (n_tot * (n_tot - 1)) if n_tot > 1 else 0.0
    var_u = (n1 * n2 / 12.0) * ((n_tot + 1) - tie_term)
    if var_u <= 0.0:
        return 1.0
    z = (abs(u_obs - mean_u) - 0.5) / math.sqrt(var_u)
    z = max(0.0, z)
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def _percentile(xs: Sequence[float], q: float) -> float:
    """Return the q-th percentile (0..100) using linear interpolation (type 7)."""
    if not xs:
        raise ValueError("Cannot compute percentile of an empty sequence")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] + frac * (s[hi] - s[lo]))


def _engine_stats(xs: Sequence[float]) -> Dict[str, float]:
    """Compute pure-stdlib summary statistics for one engine's metric samples."""
    ci_lo, ci_hi = _bootstrap_median_ci(xs)
    return {
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
        "p95": _percentile(xs, 95.0),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
    }


def _evaluate_verdict(
    ax: Sequence[float],
    lx: Sequence[float],
    a_stats: Dict[str, float],
    l_stats: Dict[str, float],
    direction: str,
) -> Tuple[float, str, bool]:
    """Return (mwu_p, verdict_label, strict_passed) without overclaiming ties."""
    pval = _mann_whitney_u_p(ax, lx)
    if pval >= 0.05:
        return pval, f"PARITY (p={pval:.4g})", False

    sign = 1.0 if direction == "higher" else -1.0
    non_overlap_asi = (
        a_stats["min"] > l_stats["max"]
        if direction == "higher"
        else a_stats["max"] < l_stats["min"]
    )
    a_med_s, l_med_s = sign * a_stats["median"], sign * l_stats["median"]
    if a_med_s != l_med_s:
        asi_in_favor = a_med_s > l_med_s
    else:
        wins = sum(1 for a in ax for b in lx if sign * a > sign * b)
        losses = sum(1 for a in ax for b in lx if sign * a < sign * b)
        asi_in_favor = wins > losses

    if non_overlap_asi:
        return pval, "PASS", True
    if asi_in_favor:
        return pval, "ADVANTAGE (overlap)", False
    return pval, "FAIL", False


def _format_engine_summary(stats: Dict[str, float]) -> Tuple[str, str]:
    """Format median [min..max] (95% CI) and tail (p95 / max) strings."""
    med_ci = (
        f"{stats['median']:.3f} [{stats['min']:.3f}..{stats['max']:.3f}] "
        f"({stats['ci95_lo']:.3f}..{stats['ci95_hi']:.3f})"
    )
    tail = f"{stats['p95']:.3f} / {stats['max']:.3f}"
    return med_ci, tail


def _summarize_and_print(
    asi_runs: Sequence[Dict],
    llama_runs: Sequence[Dict],
    metrics: Sequence[Tuple[str, str, str, bool]],
) -> Tuple[Dict[str, Dict], str]:
    """Compute summary stats/verdicts, print the summary table, and return (summary_export, overall_verdict)."""
    rows = []
    all_primary_pass = True
    summary_export: Dict[str, Dict] = {}

    for key, unit, direction, is_primary in metrics:
        ax = [r[key] for r in asi_runs]
        lx = [r[key] for r in llama_runs]
        a_stats = _engine_stats(ax)
        l_stats = _engine_stats(lx)
        pval, verdict, passed = _evaluate_verdict(ax, lx, a_stats, l_stats, direction)

        if is_primary and not passed:
            all_primary_pass = False

        bar_str = verdict + ("" if is_primary else " (sec)")
        a_str, a_tail = _format_engine_summary(a_stats)
        l_str, l_tail = _format_engine_summary(l_stats)
        rows.append((key, unit, direction, a_str, a_tail, l_str, l_tail, f"{pval:.4g}", bar_str))

        summary_export[key] = {
            "unit": unit,
            "direction": direction,
            "primary": is_primary,
            "asi": a_stats,
            "llama": l_stats,
            "mwu_p": pval,
            "bar_verdict": verdict,
        }

    headers = (
        "Metric",
        "Unit",
        "Dir",
        "ASI med [min..max] (95% CI)",
        "ASI p95/max",
        "llama med [min..max] (95% CI)",
        "llama p95/max",
        "MWU p",
        "Bar",
    )
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("\n=== Statistical Summary & Strict Bar Verdict ===")
    header_line = " ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" ".join(f"{val:<{w}}" for val, w in zip(row, widths)))

    overall_verdict = "PASS" if all_primary_pass else "FAIL"
    print(f"\nOVERALL PRIMARY BAR: {overall_verdict}")
    return summary_export, overall_verdict


def _tokenize_once(gguf_path: str) -> Dict[str, List[int]]:
    from llama_cpp import Llama

    llm = Llama(model_path=gguf_path, vocab_only=True, verbose=False)
    t1_ids = llm.tokenize(T1_PROMPT.encode("utf-8"), add_bos=False, special=True)
    t2_ids = llm.tokenize(T2_DELTA.encode("utf-8"), add_bos=False, special=True)
    long_ids = llm.tokenize(LONG_PROMPT.encode("utf-8"), add_bos=False, special=True)
    llm.close()
    return {"t1": list(t1_ids), "t2": list(t2_ids), "long": list(long_ids)}


def _run_worker_subprocess(
    engine: str,
    asi_path: str,
    gguf_path: str,
    lib_path: str,
    tokens_json: str,
    n_ctx: int,
    n_threads: int,
    n_batch: int,
    n_gen: int,
    prepopulate: bool = True,
) -> Dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        engine,
        "--asi",
        asi_path,
        "--gguf",
        gguf_path,
        "--lib",
        lib_path,
        "--tokens-json",
        tokens_json,
        "--n-ctx",
        str(n_ctx),
        "--n-threads",
        str(n_threads),
        "--n-batch",
        str(n_batch),
        "--n-gen",
        str(n_gen),
        "--prepopulate" if prepopulate else "--no-prepopulate",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"Worker {engine} produced no JSON output:\n{proc.stdout}\n{proc.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict apples-to-apples ASI vs llama.cpp benchmark")
    parser.add_argument("--n", type=int, default=15, help="Number of interleaved trial pairs (default: 15)")
    parser.add_argument("--asi", default=DEFAULT_ASI_PATH)
    parser.add_argument("--gguf", default=DEFAULT_GGUF_PATH)
    parser.add_argument("--lib", default=DEFAULT_LIB_PATH)
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument("--n-threads", type=int, default=N_THREADS)
    parser.add_argument("--n-batch", type=int, default=N_BATCH)
    parser.add_argument("--n-gen", type=int, default=N_GEN)
    parser.add_argument(
        "--prepopulate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Advise MADV_WILLNEED and touch model file pages right after open so load_s includes weight residency (default: True; pass --no-prepopulate for lazy load only)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Export raw per-pair numbers and summary statistics to JSON at PATH",
    )
    parser.add_argument("--worker", choices=["asi", "llama"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tokens-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        toks = json.loads(args.tokens_json)
        if args.worker == "asi":
            res = run_asi_trial(
                args.asi, args.lib, toks["t1"], toks["t2"], toks["long"],
                args.n_ctx, args.n_threads, args.n_batch, args.n_gen,
                prepopulate=args.prepopulate,
            )
        else:
            res = run_llama_trial(
                args.gguf, toks["t1"], toks["t2"], toks["long"],
                args.n_ctx, args.n_threads, args.n_batch, args.n_gen,
                prepopulate=args.prepopulate,
            )
        print(json.dumps(res))
        return 0

    asi_sha256 = _sha256_file(args.asi)
    gguf_sha256 = _sha256_file(args.gguf)
    toks = _tokenize_once(args.gguf)
    tokens_json = json.dumps(toks)
    print("=== Strict Apples-to-Apples Benchmark: Lila ASI vs llama.cpp ===")
    print(f"ASI model   : {args.asi}")
    print(f"ASI sha256  : {asi_sha256}")
    print(f"GGUF model  : {args.gguf}")
    print(f"GGUF sha256 : {gguf_sha256}")
    print(
        f"Settings    : n_ctx={args.n_ctx}, n_threads={args.n_threads}, "
        f"n_threads_batch={args.n_threads}, n_batch={args.n_batch}, "
        f"use_mmap=True, flash_attn=False, temp=0.0 (greedy), n_gen={args.n_gen}, "
        f"N={args.n}, prepopulate={args.prepopulate}"
    )
    print(
        f"Token counts: T1={len(toks['t1'])} toks, T2_delta={len(toks['t2'])} toks, "
        f"Long={len(toks['long'])} toks (identical int IDs fed to both)"
    )

    # 1 warmup trial per engine (discarded) to page-cache weights
    print("\n[Warmup] Running 1 discarded warmup trial per engine...")
    w_asi = _run_worker_subprocess(
        "asi", args.asi, args.gguf, args.lib, tokens_json,
        args.n_ctx, args.n_threads, args.n_batch, args.n_gen,
        prepopulate=args.prepopulate,
    )
    w_llama = _run_worker_subprocess(
        "llama", args.asi, args.gguf, args.lib, tokens_json,
        args.n_ctx, args.n_threads, args.n_batch, args.n_gen,
        prepopulate=args.prepopulate,
    )
    print(
        f"[KV/Prefix Reuse Verified] "
        f"ASI T2 evaluated={w_asi['t2_eval_tokens']} toks (pos_before={w_asi['t2_pos_before']}), "
        f"llama.cpp T2 evaluated={w_llama['t2_eval_tokens']} toks (n_tokens_before={w_llama['t2_pos_before']})"
    )

    asi_runs: List[Dict] = []
    llama_runs: List[Dict] = []
    print(f"\n[Trials] Running {args.n} interleaved (ASI, llama.cpp) pairs...")
    for idx in range(1, args.n + 1):
        r_asi = _run_worker_subprocess(
            "asi", args.asi, args.gguf, args.lib, tokens_json,
            args.n_ctx, args.n_threads, args.n_batch, args.n_gen,
            prepopulate=args.prepopulate,
        )
        r_llama = _run_worker_subprocess(
            "llama", args.asi, args.gguf, args.lib, tokens_json,
            args.n_ctx, args.n_threads, args.n_batch, args.n_gen,
            prepopulate=args.prepopulate,
        )
        assert r_asi["t2_eval_tokens"] == len(toks["t2"])
        assert r_llama["t2_eval_tokens"] == len(toks["t2"])
        asi_runs.append(r_asi)
        llama_runs.append(r_llama)
        print(
            f"  pair {idx:2d}/{args.n}: "
            f"ASI(load={r_asi['load_s']:.3f}s[open={r_asi['load_open_s']:.3f}s], "
            f"t1_ttft={r_asi['t1_ttft']:.3f}s, t1_dec={r_asi['t1_dec']:.2f}, "
            f"t2_ttft={r_asi['t2_ttft']:.3f}s, t2_dec={r_asi['t2_dec']:.2f}, "
            f"long={r_asi['long_prefill_tps']:.2f}, rss={r_asi['rss_mb']:.1f}MB, "
            f"hwm={r_asi['vm_hwm_mb']:.1f}MB) | "
            f"llama(load={r_llama['load_s']:.3f}s[open={r_llama['load_open_s']:.3f}s], "
            f"t1_ttft={r_llama['t1_ttft']:.3f}s, t1_dec={r_llama['t1_dec']:.2f}, "
            f"t2_ttft={r_llama['t2_ttft']:.3f}s, t2_dec={r_llama['t2_dec']:.2f}, "
            f"long={r_llama['long_prefill_tps']:.2f}, rss={r_llama['rss_mb']:.1f}MB, "
            f"hwm={r_llama['vm_hwm_mb']:.1f}MB)"
        )

    metrics = [
        ("t1_ttft", "s", "lower", True),
        ("t1_dec", "tok/s", "higher", True),
        ("t2_ttft", "s", "lower", True),
        ("t2_dec", "tok/s", "higher", True),
        ("long_prefill_tps", "tok/s", "higher", True),
        ("rss_mb", "MiB", "lower", True),
        ("vm_hwm_mb", "MiB", "lower", False),
        ("load_s", "s", "lower", False),
        ("load_open_s", "s", "lower", False),
    ]

    summary_export, overall_verdict = _summarize_and_print(asi_runs, llama_runs, metrics)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "asi_path": args.asi,
                "asi_sha256": asi_sha256,
                "gguf_path": args.gguf,
                "gguf_sha256": gguf_sha256,
                "lib_path": args.lib,
                "n_ctx": args.n_ctx,
                "n_threads": args.n_threads,
                "n_threads_batch": args.n_threads,
                "n_batch": args.n_batch,
                "n_gen": args.n_gen,
                "n_pairs": args.n,
                "prepopulate": args.prepopulate,
                "token_counts": {
                    "t1": len(toks["t1"]),
                    "t2": len(toks["t2"]),
                    "long": len(toks["long"]),
                },
            },
            "pairs": [
                {"pair": i + 1, "asi": a, "llama": l}
                for i, (a, l) in enumerate(zip(asi_runs, llama_runs))
            ],
            "summary": summary_export,
            "overall_primary_bar": overall_verdict,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Exported JSON results to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
