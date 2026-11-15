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
6. Primary metrics: t1_ttft, t1_dec, t2_ttft, t2_dec, long_prefill_tps.
   Memory metric: rss_mb. Secondary metrics: vm_hwm_mb, load_s, load_open_s.
7. Pure-stdlib stats: median, [min..max], p5, p95, max, 95% bootstrap CI,
   Mann-Whitney U two-sided p-value, G1 Verdict & Claim Engine
   (STRONG WIN, WIN, PARITY, LOSS -> STRICTLY_BEATS / BEATS /
   PARTIAL_ADVANTAGE / DOES_NOT_BEAT), and backward-compatible strict bar.
8. Per-trial OS/hypervisor telemetry: /proc/stat CPU jiffies deltas
   (steal, iowait, user, system, total) and resource.getrusage(RUSAGE_SELF)
   (ru_minflt, ru_majflt, ru_nvcsw, ru_nivcsw).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import mmap
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import resource
except ImportError:  # pragma: no cover - non-Unix fallback
    resource = None

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


def _read_proc_stat_cpu() -> Optional[Dict[str, int]]:
    """Read aggregate CPU jiffies from /proc/stat (user, system, iowait, steal, total)."""
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = [int(x) for x in line.split()[1:]]
                    # Linux /proc/stat: user nice system idle iowait irq softirq steal [guest guest_nice]
                    # guest/guest_nice are already included in user/nice.
                    core = parts[:8] if len(parts) >= 8 else parts
                    return {
                        "user": parts[0] if len(parts) > 0 else 0,
                        "system": parts[2] if len(parts) > 2 else 0,
                        "iowait": parts[4] if len(parts) > 4 else 0,
                        "steal": parts[7] if len(parts) > 7 else 0,
                        "total": sum(core),
                    }
    except OSError:
        return None
    return None


def _get_rusage_stats() -> Dict[str, int]:
    """Return process page-fault and context-switch counters via resource.getrusage(RUSAGE_SELF)."""
    if resource is None:
        return {"ru_minflt": 0, "ru_majflt": 0, "ru_nvcsw": 0, "ru_nivcsw": 0}
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "ru_minflt": int(ru.ru_minflt),
        "ru_majflt": int(ru.ru_majflt),
        "ru_nvcsw": int(ru.ru_nvcsw),
        "ru_nivcsw": int(ru.ru_nivcsw),
    }


def _collect_lila_asi_env() -> Dict[str, str]:
    """Capture all active LILA_ASI_* environment variables."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("LILA_ASI_")}


def _collect_hardware_passport(n_threads: int) -> Dict[str, Any]:
    """Collect CPU model, core topology, ISA tier, SIMD flags, and RAM info."""
    arch = platform.machine() or "unknown"
    cpu_model = platform.processor() or "unknown"
    vendor_id = "unknown"
    flags: set[str] = set()
    phys_cores: set[Tuple[str, str]] = set()
    cpu_cores_fallback: Optional[int] = None

    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            cpuinfo = f.read()
        m_model = re.search(r"^(?:model name|Model|Processor|Hardware)\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if m_model:
            cpu_model = m_model.group(1).strip()
        m_vendor = re.search(r"^(?:vendor_id|CPU implementer)\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if m_vendor:
            vendor_id = m_vendor.group(1).strip()
        m_flags = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        if m_flags:
            flags = set(m_flags.group(1).split())
        m_cc = re.search(r"^cpu cores\s*:\s*(\d+)$", cpuinfo, re.MULTILINE)
        if m_cc:
            cpu_cores_fallback = int(m_cc.group(1))
        cur_pkg = "0"
        for line in cpuinfo.splitlines():
            if line.startswith("physical id"):
                cur_pkg = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                cur_core = line.split(":", 1)[1].strip()
                phys_cores.add((cur_pkg, cur_core))

    if cpu_model in ("", "unknown", arch):
        try:
            out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=2, check=False)
            if out.returncode == 0:
                m_lscpu = re.search(r"^Model name:\s*(.+)$", out.stdout, re.MULTILINE)
                if m_lscpu:
                    cpu_model = m_lscpu.group(1).strip()
        except OSError:
            pass

    logical_cpus = os.cpu_count() or 1
    core_count = len(phys_cores) if phys_cores else (cpu_cores_fallback or logical_cpus)
    ram_total_gib, ram_avail_gib = 0.0, 0.0
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
            meminfo = f.read()
        mt = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
        ma = re.search(r"^MemAvailable:\s+(\d+)\s+kB", meminfo, re.MULTILINE)
        if mt:
            ram_total_gib = int(mt.group(1)) / (1024.0 * 1024.0)
        if ma:
            ram_avail_gib = int(ma.group(1)) / (1024.0 * 1024.0)

    simd_keys = [
        "avx2", "fma", "f16c", "avx_vnni",
        "avx512f", "avx512bw", "avx512vl", "avx512_vnni",
        "asimd", "neon", "dotprod", "asimddp", "fp16", "i8mm",
    ]
    simd_present = [k for k in simd_keys if k in flags]
    if "avx512f" in flags and "avx512_vnni" in flags:
        isa_tier = "AVX-512 + VNNI"
    elif "avx512f" in flags:
        isa_tier = "AVX-512"
    elif "avx2" in flags:
        isa_tier = "AVX2 + FMA" if "fma" in flags else "AVX2"
    elif "asimd" in flags or "neon" in flags or arch.lower() in ("aarch64", "arm64"):
        isa_tier = "ARM64 NEON + DotProd" if ("dotprod" in flags or "asimddp" in flags) else "ARM64 NEON"
    else:
        isa_tier = arch

    return {
        "arch": arch,
        "os": platform.system() or "unknown",
        "release": platform.release() or "unknown",
        "hostname": platform.node() or "unknown",
        "vendor_id": vendor_id,
        "cpu_model": cpu_model,
        "physical_cores": core_count,
        "logical_cpus": logical_cpus,
        "bench_threads": n_threads,
        "isa_tier": isa_tier,
        "simd_flags": simd_present,
        "hypervisor": "hypervisor" in flags,
        "ram_total_gib": round(ram_total_gib, 3),
        "ram_avail_gib": round(ram_avail_gib, 3),
    }


def _print_hardware_passport(passport: Dict[str, Any], lila_asi_env: Dict[str, str]) -> None:
    """Print the Hardware Passport and active LILA_ASI_* env vars."""
    simd_str = ", ".join(passport.get("simd_flags", [])) or "none"
    env_str = ", ".join(f"{k}={v}" for k, v in lila_asi_env.items()) if lila_asi_env else "none"
    print("=== Hardware Passport ===")
    print(f"Arch / Vendor : {passport['arch']} ({passport['vendor_id']}) | OS={passport['os']} {passport['release']}")
    print(f"CPU Model     : {passport['cpu_model']}")
    print(
        f"Cores/Threads : {passport['physical_cores']} physical / {passport['logical_cpus']} logical "
        f"(bench_threads={passport['bench_threads']}, hypervisor={passport['hypervisor']})"
    )
    print(f"ISA Tier      : {passport['isa_tier']} | SIMD: {simd_str}")
    print(f"System RAM    : {passport['ram_total_gib']:.2f} GiB total ({passport['ram_avail_gib']:.2f} GiB available)")
    print(f"LILA_ASI_* env: {env_str}")


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
    ru_stats = _get_rusage_stats()

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
        "ru_minflt": ru_stats["ru_minflt"],
        "ru_majflt": ru_stats["ru_majflt"],
        "ru_nvcsw": ru_stats["ru_nvcsw"],
        "ru_nivcsw": ru_stats["ru_nivcsw"],
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
    ru_stats = _get_rusage_stats()
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
        "ru_minflt": ru_stats["ru_minflt"],
        "ru_majflt": ru_stats["ru_majflt"],
        "ru_nvcsw": ru_stats["ru_nvcsw"],
        "ru_nivcsw": ru_stats["ru_nivcsw"],
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


PRIMARY_METRICS: Tuple[str, ...] = (
    "t1_ttft",
    "t1_dec",
    "t2_ttft",
    "t2_dec",
    "long_prefill_tps",
)
MEMORY_METRIC = "rss_mb"
SECONDARY_METRICS: Tuple[str, ...] = ("vm_hwm_mb", "load_s", "load_open_s")


def _engine_stats(xs: Sequence[float]) -> Dict[str, float]:
    """Compute pure-stdlib summary statistics for one engine's metric samples."""
    ci_lo, ci_hi = _bootstrap_median_ci(xs)
    return {
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
        "p5": _percentile(xs, 5.0),
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
    """Return (mwu_p, legacy_bar_verdict_label, strict_passed) without overclaiming ties."""
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


def _evaluate_g1_verdict(
    ax: Sequence[float],
    lx: Sequence[float],
    a_stats: Dict[str, float],
    l_stats: Dict[str, float],
    direction: str,
    pval: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate G1 statistical + magnitude + worst-5% tail verdict for one metric."""
    if pval is None:
        pval = _mann_whitney_u_p(ax, lx)

    a_med = a_stats["median"]
    l_med = l_stats["median"]
    rel_diff_pct = ((a_med - l_med) / l_med * 100.0) if l_med != 0.0 else 0.0
    asi_vs_llama_ratio = (a_med / l_med) if l_med != 0.0 else 1.0
    p95_ratio = (a_stats["p95"] / l_stats["p95"]) if l_stats["p95"] != 0.0 else 1.0

    if direction == "higher":
        gain = ((a_med - l_med) / l_med) if l_med != 0.0 else 0.0
        tail_ratio = (a_stats["p5"] / l_stats["p5"]) if l_stats["p5"] != 0.0 else 1.0
        tail_ok = a_stats["p5"] >= 0.90 * l_stats["p5"]
        tail_degradation_pct = (
            ((l_stats["p5"] - a_stats["p5"]) / l_stats["p5"] * 100.0)
            if l_stats["p5"] != 0.0
            else 0.0
        )
        non_overlap_asi = a_stats["min"] > l_stats["max"]
    else:
        gain = ((l_med - a_med) / l_med) if l_med != 0.0 else 0.0
        tail_ratio = p95_ratio
        tail_ok = a_stats["p95"] <= 1.10 * l_stats["p95"]
        tail_degradation_pct = (
            ((a_stats["p95"] - l_stats["p95"]) / l_stats["p95"] * 100.0)
            if l_stats["p95"] != 0.0
            else 0.0
        )
        non_overlap_asi = a_stats["max"] < l_stats["min"]

    improvement_pct = 100.0 * gain
    sign = 1.0 if direction == "higher" else -1.0
    if gain == 0.0:
        wins = sum(1 for a in ax for b in lx if sign * a > sign * b)
        losses = sum(1 for a in ax for b in lx if sign * a < sign * b)
        llama_in_favor = losses > wins
    else:
        llama_in_favor = gain < 0.0

    if gain >= 0.03 and pval < 0.01:
        if tail_ok:
            g1_verdict = "STRONG WIN" if non_overlap_asi else "WIN"
        else:
            g1_verdict = "PARITY (tail>10%)"
    elif gain < -0.03 or (pval < 0.01 and llama_in_favor):
        g1_verdict = "LOSS"
    else:
        g1_verdict = "PARITY"

    return {
        "mwu_p": pval,
        "rel_diff_pct": rel_diff_pct,
        "improvement_pct": improvement_pct,
        "asi_vs_llama_ratio": asi_vs_llama_ratio,
        "tail_ratio": tail_ratio,
        "p95_ratio": p95_ratio,
        "tail_degradation_pct": tail_degradation_pct,
        "tail_ok": tail_ok,
        "non_overlap_asi": non_overlap_asi,
        "g1_verdict": g1_verdict,
    }


def _evaluate_claim_verdict(
    summary_export: Dict[str, Dict[str, Any]],
    n_pairs: int,
    cpu_label: str = "unknown CPU",
    n_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the G1 Claim Engine verdict from per-metric summary verdicts."""
    primary_verdicts = {k: str(summary_export[k]["g1_verdict"]) for k in PRIMARY_METRICS}
    rss_verdict = str(summary_export[MEMORY_METRIC]["g1_verdict"])

    n_strong_win = sum(1 for v in primary_verdicts.values() if v == "STRONG WIN")
    n_win_or_strong = sum(1 for v in primary_verdicts.values() if v in ("WIN", "STRONG WIN"))
    n_parity = sum(1 for v in primary_verdicts.values() if v.startswith("PARITY"))
    n_loss = sum(1 for v in primary_verdicts.values() if v == "LOSS")
    rss_win = rss_verdict in ("WIN", "STRONG WIN")
    rss_loss = rss_verdict == "LOSS"
    dec_ok = primary_verdicts["t1_dec"] != "LOSS" and primary_verdicts["t2_dec"] != "LOSS"

    if n_strong_win == len(PRIMARY_METRICS) and rss_win:
        status = "STRICTLY_BEATS"
    elif n_loss == 0 and dec_ok and n_win_or_strong >= 3 and rss_win:
        status = "BEATS"
    elif not rss_loss and (n_win_or_strong + int(rss_win)) >= 1 and n_loss < max(1, n_win_or_strong):
        status = "PARTIAL_ADVANTAGE"
    else:
        status = "DOES_NOT_BEAT"

    reasons: List[str] = []
    if n_pairs < 15:
        reasons.append(f"N={n_pairs} < 15 required for G1")
    for k in (*PRIMARY_METRICS, MEMORY_METRIC):
        v = str(summary_export[k]["g1_verdict"])
        if v == "LOSS":
            reasons.append(f"{k}=LOSS ({summary_export[k]['improvement_pct']:+.1f}%)")
        elif "tail>10%" in v:
            reasons.append(f"{k}={v} (tail_ratio={summary_export[k]['tail_ratio']:.3f})")
    if not rss_win and rss_verdict != "LOSS":
        reasons.append(f"rss_mb={rss_verdict}")
    if status in ("PARTIAL_ADVANTAGE", "DOES_NOT_BEAT"):
        if n_win_or_strong < 3:
            reasons.append(f"only {n_win_or_strong}/5 primary WIN/STRONG WIN")

    t_suffix = f" ({n_threads} threads)" if n_threads is not None else ""
    if n_pairs < 15:
        allowed_claim = f"Does not beat llama.cpp on {cpu_label}{t_suffix} ({'; '.join(reasons)})"
    elif status == "STRICTLY_BEATS":
        allowed_claim = f"Strictly beats llama.cpp on {cpu_label}{t_suffix}"
    elif status == "BEATS":
        allowed_claim = f"Beats llama.cpp on {cpu_label}{t_suffix}"
    else:
        detail = "; ".join(reasons) if reasons else status
        allowed_claim = f"Does not beat llama.cpp on {cpu_label}{t_suffix} ({detail})"

    all_verdicts = {k: str(v["g1_verdict"]) for k, v in summary_export.items()}
    return {
        "status": status,
        "allowed_claim": allowed_claim,
        "n_pairs": n_pairs,
        "g1_sample_size_met": n_pairs >= 15,
        "primary_metrics": list(PRIMARY_METRICS),
        "memory_metric": MEMORY_METRIC,
        "primary_strong_wins": n_strong_win,
        "primary_wins_or_strong": n_win_or_strong,
        "primary_parities": n_parity,
        "primary_losses": n_loss,
        "rss_verdict": rss_verdict,
        "decode_not_loss": dec_ok,
        "verdicts": all_verdicts,
        "reasons": reasons,
    }


def _summarize_telemetry(asi_runs: Sequence[Dict], llama_runs: Sequence[Dict]) -> Dict[str, Dict[str, Any]]:
    """Summarize OS/hypervisor telemetry across all trials without dropping outliers."""
    out: Dict[str, Dict[str, Any]] = {}
    for label, runs in (("asi", asi_runs), ("llama", llama_runs)):
        steals = [int(r.get("steal_jiffies", 0)) for r in runs]
        iowaits = [int(r.get("iowait_jiffies", 0)) for r in runs]
        majflts = [int(r.get("ru_majflt", 0)) for r in runs]
        nivcsws = [int(r.get("ru_nivcsw", 0)) for r in runs]
        out[label] = {
            "steal_jiffies_total": sum(steals),
            "steal_jiffies_max": max(steals, default=0),
            "steal_trials": sum(1 for s in steals if s > 0),
            "iowait_jiffies_total": sum(iowaits),
            "iowait_jiffies_max": max(iowaits, default=0),
            "ru_majflt_total": sum(majflts),
            "ru_majflt_max": max(majflts, default=0),
            "ru_nivcsw_median": statistics.median(nivcsws) if nivcsws else 0,
            "ru_nivcsw_max": max(nivcsws, default=0),
        }
    return out


def _format_engine_summary(stats: Dict[str, float]) -> Tuple[str, str]:
    """Format median [min..max] (95% CI) and tail (p5 / p95 / max) strings."""
    med_ci = (
        f"{stats['median']:.3f} [{stats['min']:.3f}..{stats['max']:.3f}] "
        f"({stats['ci95_lo']:.3f}..{stats['ci95_hi']:.3f})"
    )
    tail = f"{stats['p5']:.3f}/{stats['p95']:.3f}/{stats['max']:.3f}"
    return med_ci, tail


def _summarize_and_print(
    asi_runs: Sequence[Dict],
    llama_runs: Sequence[Dict],
    metrics: Sequence[Tuple[str, str, str, bool]],
    cpu_label: str = "unknown CPU",
    n_threads: Optional[int] = None,
) -> Tuple[Dict[str, Dict], str, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Compute stats, G1 verdicts, claim verdict, and print summary tables."""
    rows = []
    all_primary_pass = True
    summary_export: Dict[str, Dict] = {}

    for key, unit, direction, is_primary in metrics:
        ax = [r[key] for r in asi_runs]
        lx = [r[key] for r in llama_runs]
        a_stats = _engine_stats(ax)
        l_stats = _engine_stats(lx)
        pval, bar_verdict, passed = _evaluate_verdict(ax, lx, a_stats, l_stats, direction)
        g1_info = _evaluate_g1_verdict(ax, lx, a_stats, l_stats, direction, pval=pval)

        if is_primary and not passed:
            all_primary_pass = False

        role = (
            "primary"
            if key in PRIMARY_METRICS
            else ("memory" if key == MEMORY_METRIC else "secondary")
        )
        bar_str = bar_verdict + ("" if is_primary else " (sec)")
        g1_str = str(g1_info["g1_verdict"]) + ("" if role != "secondary" else " (sec)")
        a_str, a_tail = _format_engine_summary(a_stats)
        l_str, l_tail = _format_engine_summary(l_stats)
        rows.append((
            key,
            unit,
            direction,
            a_str,
            a_tail,
            l_str,
            l_tail,
            f"{g1_info['improvement_pct']:+.1f}%",
            f"{g1_info['tail_ratio']:.3f}",
            f"{pval:.4g}",
            g1_str,
            bar_str,
        ))

        summary_export[key] = {
            "unit": unit,
            "direction": direction,
            "primary": is_primary,
            "role": role,
            "asi": a_stats,
            "llama": l_stats,
            "mwu_p": pval,
            "rel_diff_pct": g1_info["rel_diff_pct"],
            "improvement_pct": g1_info["improvement_pct"],
            "asi_vs_llama_ratio": g1_info["asi_vs_llama_ratio"],
            "tail_ratio": g1_info["tail_ratio"],
            "p95_ratio": g1_info["p95_ratio"],
            "tail_degradation_pct": g1_info["tail_degradation_pct"],
            "tail_ok": g1_info["tail_ok"],
            "non_overlap_asi": g1_info["non_overlap_asi"],
            "g1_verdict": g1_info["g1_verdict"],
            "bar_verdict": bar_verdict,
        }

    headers = (
        "Metric",
        "Unit",
        "Dir",
        "ASI med [min..max] (95% CI)",
        "ASI p5/p95/max",
        "llama med [min..max] (95% CI)",
        "llama p5/p95/max",
        "Gain%",
        "TailR",
        "MWU p",
        "G1",
        "Bar",
    )
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("\n=== Statistical Summary, G1 Claim Engine & Strict Bar Verdict ===")
    header_line = " ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" ".join(f"{val:<{w}}" for val, w in zip(row, widths)))

    telemetry = _summarize_telemetry(asi_runs, llama_runs)
    print("\n=== OS & Hypervisor Trial Telemetry (no outliers dropped) ===")
    for eng in ("asi", "llama"):
        t = telemetry[eng]
        print(
            f"  {eng:<5s}: steal_jiffies total={t['steal_jiffies_total']} max={t['steal_jiffies_max']} "
            f"(trials={t['steal_trials']}/{len(asi_runs)}), "
            f"iowait_total={t['iowait_jiffies_total']} max={t['iowait_jiffies_max']}, "
            f"majflt_total={t['ru_majflt_total']} max={t['ru_majflt_max']}, "
            f"nivcsw med={t['ru_nivcsw_median']:.0f} max={t['ru_nivcsw_max']}"
        )

    overall_verdict = "PASS" if all_primary_pass else "FAIL"
    claim_verdict = _evaluate_claim_verdict(
        summary_export, len(asi_runs), cpu_label=cpu_label, n_threads=n_threads
    )
    print(f"\nOVERALL PRIMARY BAR: {overall_verdict}")
    print(
        f"G1 CLAIM VERDICT   : {claim_verdict['status']} "
        f"(primary_win={claim_verdict['primary_wins_or_strong']}/5, "
        f"strong={claim_verdict['primary_strong_wins']}/5, "
        f"loss={claim_verdict['primary_losses']}, rss={claim_verdict['rss_verdict']})"
    )
    print(f"ALLOWED CLAIM      : {claim_verdict['allowed_claim']}")
    return summary_export, overall_verdict, claim_verdict, telemetry


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
    cpu_before = _read_proc_stat_cpu()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    cpu_after = _read_proc_stat_cpu()
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            res = json.loads(line)
            if cpu_before is not None and cpu_after is not None:
                res["steal_jiffies"] = max(0, cpu_after["steal"] - cpu_before["steal"])
                res["iowait_jiffies"] = max(0, cpu_after["iowait"] - cpu_before["iowait"])
                res["user_jiffies"] = max(0, cpu_after["user"] - cpu_before["user"])
                res["system_jiffies"] = max(0, cpu_after["system"] - cpu_before["system"])
                res["total_jiffies"] = max(0, cpu_after["total"] - cpu_before["total"])
            else:
                res["steal_jiffies"] = 0
                res["iowait_jiffies"] = 0
                res["user_jiffies"] = 0
                res["system_jiffies"] = 0
                res["total_jiffies"] = 0
            return res
    raise RuntimeError(f"Worker {engine} produced no JSON output:\n{proc.stdout}\n{proc.stderr}")


def _parse_threads_list(threads_list_arg: Optional[str], default_threads: int) -> List[int]:
    """Parse comma-separated --threads-list or fall back to [default_threads]."""
    if not threads_list_arg:
        if default_threads < 1:
            raise ValueError(f"--n-threads must be >= 1, got {default_threads}")
        return [default_threads]
    out: List[int] = []
    for raw in threads_list_arg.split(","):
        s = raw.strip()
        if not s:
            continue
        t = int(s)
        if t < 1:
            raise ValueError(f"Invalid thread count in --threads-list: {t}")
        if t not in out:
            out.append(t)
    if not out:
        raise ValueError("--threads-list cannot be empty")
    return out


def _format_steal_tag(run: Dict[str, Any]) -> str:
    steal = int(run.get("steal_jiffies", 0))
    if steal > 0:
        return f" [steal={steal}, nivcsw={int(run.get('ru_nivcsw', 0))}]"
    return ""


def _run_benchmark_for_threads(
    args: argparse.Namespace,
    n_threads: int,
    toks: Dict[str, List[int]],
    tokens_json: str,
    asi_sha256: str,
    gguf_sha256: str,
) -> Dict[str, Any]:
    """Run warmup + N interleaved trials for a single thread count and return export payload."""
    passport = _collect_hardware_passport(n_threads)
    lila_asi_env = _collect_lila_asi_env()
    print("\n=== Strict Apples-to-Apples Benchmark: Lila ASI vs llama.cpp ===")
    _print_hardware_passport(passport, lila_asi_env)
    print(f"ASI model   : {args.asi}")
    print(f"ASI sha256  : {asi_sha256}")
    print(f"GGUF model  : {args.gguf}")
    print(f"GGUF sha256 : {gguf_sha256}")
    print(
        f"Settings    : n_ctx={args.n_ctx}, n_threads={n_threads}, "
        f"n_threads_batch={n_threads}, n_batch={args.n_batch}, "
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
        args.n_ctx, n_threads, args.n_batch, args.n_gen,
        prepopulate=args.prepopulate,
    )
    w_llama = _run_worker_subprocess(
        "llama", args.asi, args.gguf, args.lib, tokens_json,
        args.n_ctx, n_threads, args.n_batch, args.n_gen,
        prepopulate=args.prepopulate,
    )
    print(
        f"[KV/Prefix Reuse Verified] "
        f"ASI T2 evaluated={w_asi['t2_eval_tokens']} toks (pos_before={w_asi['t2_pos_before']}), "
        f"llama.cpp T2 evaluated={w_llama['t2_eval_tokens']} toks (n_tokens_before={w_llama['t2_pos_before']})"
    )

    asi_runs: List[Dict] = []
    llama_runs: List[Dict] = []
    print(f"\n[Trials] Running {args.n} interleaved (ASI, llama.cpp) pairs (n_threads={n_threads})...")
    for idx in range(1, args.n + 1):
        r_asi = _run_worker_subprocess(
            "asi", args.asi, args.gguf, args.lib, tokens_json,
            args.n_ctx, n_threads, args.n_batch, args.n_gen,
            prepopulate=args.prepopulate,
        )
        r_llama = _run_worker_subprocess(
            "llama", args.asi, args.gguf, args.lib, tokens_json,
            args.n_ctx, n_threads, args.n_batch, args.n_gen,
            prepopulate=args.prepopulate,
        )
        assert r_asi["t2_eval_tokens"] == len(toks["t2"])
        assert r_llama["t2_eval_tokens"] == len(toks["t2"])
        asi_runs.append(r_asi)
        llama_runs.append(r_llama)
        asi_steal = _format_steal_tag(r_asi)
        llama_steal = _format_steal_tag(r_llama)
        print(
            f"  pair {idx:2d}/{args.n}: "
            f"ASI(load={r_asi['load_s']:.3f}s[open={r_asi['load_open_s']:.3f}s], "
            f"t1_ttft={r_asi['t1_ttft']:.3f}s, t1_dec={r_asi['t1_dec']:.2f}, "
            f"t2_ttft={r_asi['t2_ttft']:.3f}s, t2_dec={r_asi['t2_dec']:.2f}, "
            f"long={r_asi['long_prefill_tps']:.2f}, rss={r_asi['rss_mb']:.1f}MB, "
            f"hwm={r_asi['vm_hwm_mb']:.1f}MB){asi_steal} | "
            f"llama(load={r_llama['load_s']:.3f}s[open={r_llama['load_open_s']:.3f}s], "
            f"t1_ttft={r_llama['t1_ttft']:.3f}s, t1_dec={r_llama['t1_dec']:.2f}, "
            f"t2_ttft={r_llama['t2_ttft']:.3f}s, t2_dec={r_llama['t2_dec']:.2f}, "
            f"long={r_llama['long_prefill_tps']:.2f}, rss={r_llama['rss_mb']:.1f}MB, "
            f"hwm={r_llama['vm_hwm_mb']:.1f}MB){llama_steal}"
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

    summary_export, overall_verdict, claim_verdict, telemetry = _summarize_and_print(
        asi_runs,
        llama_runs,
        metrics,
        cpu_label=str(passport.get("cpu_model", "unknown CPU")),
        n_threads=n_threads,
    )

    return {
        "config": {
            "asi_path": args.asi,
            "asi_sha256": asi_sha256,
            "gguf_path": args.gguf,
            "gguf_sha256": gguf_sha256,
            "lib_path": args.lib,
            "n_ctx": args.n_ctx,
            "n_threads": n_threads,
            "n_threads_batch": n_threads,
            "n_batch": args.n_batch,
            "n_gen": args.n_gen,
            "n_pairs": args.n,
            "prepopulate": args.prepopulate,
            "token_counts": {
                "t1": len(toks["t1"]),
                "t2": len(toks["t2"]),
                "long": len(toks["long"]),
            },
            "hardware_passport": passport,
            "lila_asi_env": lila_asi_env,
        },
        "pairs": [
            {"pair": i + 1, "asi": a, "llama": l}
            for i, (a, l) in enumerate(zip(asi_runs, llama_runs))
        ],
        "telemetry": telemetry,
        "summary": summary_export,
        "claim_verdict": claim_verdict,
        "overall_primary_bar": overall_verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict apples-to-apples ASI vs llama.cpp benchmark")
    parser.add_argument("--n", type=int, default=15, help="Number of interleaved trial pairs (default: 15)")
    parser.add_argument("--asi", default=DEFAULT_ASI_PATH)
    parser.add_argument("--gguf", default=DEFAULT_GGUF_PATH)
    parser.add_argument("--lib", default=DEFAULT_LIB_PATH)
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument("--n-threads", type=int, default=N_THREADS)
    parser.add_argument(
        "--threads-list",
        default=None,
        help="Optional comma-separated thread counts to sweep (e.g. '1,2'); overrides --n-threads",
    )
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

    if args.n < 1:
        raise ValueError(f"--n must be >= 1, got {args.n}")
    threads_list = _parse_threads_list(args.threads_list, args.n_threads)

    asi_sha256 = _sha256_file(args.asi)
    gguf_sha256 = _sha256_file(args.gguf)
    toks = _tokenize_once(args.gguf)
    tokens_json = json.dumps(toks)

    runs_by_threads: Dict[str, Dict[str, Any]] = {}
    for t in threads_list:
        payload = _run_benchmark_for_threads(args, t, toks, tokens_json, asi_sha256, gguf_sha256)
        runs_by_threads[str(t)] = payload

        if args.json_out and len(threads_list) > 1:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            suffix = out_path.suffix or ".json"
            per_t_path = out_path.with_name(f"{out_path.stem}_t{t}{suffix}")
            per_t_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"Exported thread={t} JSON results to: {per_t_path}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(threads_list) == 1:
            single_payload = runs_by_threads[str(threads_list[0])]
            out_path.write_text(json.dumps(single_payload, indent=2) + "\n", encoding="utf-8")
            print(f"Exported JSON results to: {out_path}")
        else:
            agg_payload = {
                "config": {
                    "asi_path": args.asi,
                    "asi_sha256": asi_sha256,
                    "gguf_path": args.gguf,
                    "gguf_sha256": gguf_sha256,
                    "lib_path": args.lib,
                    "n_ctx": args.n_ctx,
                    "threads_list": threads_list,
                    "n_batch": args.n_batch,
                    "n_gen": args.n_gen,
                    "n_pairs": args.n,
                    "prepopulate": args.prepopulate,
                    "hardware_passport": runs_by_threads[str(threads_list[0])]["config"]["hardware_passport"],
                    "lila_asi_env": runs_by_threads[str(threads_list[0])]["config"]["lila_asi_env"],
                },
                "runs": runs_by_threads,
            }
            out_path.write_text(json.dumps(agg_payload, indent=2) + "\n", encoding="utf-8")
            print(f"Exported combined multi-thread JSON results to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
