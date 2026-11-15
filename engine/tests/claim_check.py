#!/usr/bin/env python3
"""
G0 + G1 Claim Verification Tool for Lila ASI vs llama.cpp.

Pure-stdlib checker that:
1. Verifies G0 correctness/determinism gates from --correctness-json:
   - batch_vs_token exact position agreement == 100.0% over >= 320 positions
   - teacher_forced top-1 agreement ASI vs llama.cpp >= 98.0% over >= 320 positions
   - fp32_ref mean relative-L2 ratio asi_rel_l2 <= 1.05 * llama_rel_l2 (fails if missing)
   - determinism >= 5 runs (passed == True at 1 and N threads if tested; fails if missing)
2. Verifies G1 benchmark claims directly from raw "pairs" in each --bench-json (N >= 15):
   - Recomputes median, min, max, p5, p95, 95% bootstrap CI, two-sided MWU p-value,
     rel_diff_pct, and worst-5% tail_ratio from raw pairs
   - Classifies each metric as STRONG WIN, WIN, PARITY (or PARITY (tail>10%)), or LOSS
   - Emits exact allowed claim text per machine/thread-count:
     "Strictly beats llama.cpp on <CPU> ...", "Beats llama.cpp on <CPU> ...",
     or "Does not beat llama.cpp on <CPU> ... (<reasons>)"
3. Optionally summarizes --roofline-json if provided.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PRIMARY_METRICS: Tuple[str, ...] = (
    "t1_ttft",
    "t1_dec",
    "t2_ttft",
    "t2_dec",
    "long_prefill_tps",
)
MEMORY_METRIC = "rss_mb"
SECONDARY_METRICS: Tuple[str, ...] = ("vm_hwm_mb", "load_s", "load_open_s")

METRIC_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    ("t1_ttft", "s", "lower", "primary"),
    ("t1_dec", "tok/s", "higher", "primary"),
    ("t2_ttft", "s", "lower", "primary"),
    ("t2_dec", "tok/s", "higher", "primary"),
    ("long_prefill_tps", "tok/s", "higher", "primary"),
    ("rss_mb", "MiB", "lower", "memory"),
    ("vm_hwm_mb", "MiB", "lower", "secondary"),
    ("load_s", "s", "lower", "secondary"),
    ("load_open_s", "s", "lower", "secondary"),
)


def _bootstrap_median_ci(xs: Sequence[float], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(xs)
    if n == 1:
        return (float(xs[0]), float(xs[0]))
    medians = sorted(statistics.median(rng.choices(xs, k=n)) for _ in range(n_boot))
    lo = medians[int(0.025 * n_boot)]
    hi = medians[min(n_boot - 1, int(0.975 * n_boot))]
    return (float(lo), float(hi))


def _mann_whitney_u_p(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sided Mann-Whitney U p-value in pure Python stdlib."""
    n1, n2 = len(x), len(y)
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])

    n_tot = n1 + n2
    ranks2 = [0] * n_tot
    tie_groups: List[int] = []
    i = 0
    while i < n_tot:
        j = i + 1
        while j < n_tot and combined[j][0] == combined[i][0]:
            j += 1
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
    ci_lo, ci_hi = _bootstrap_median_ci(xs)
    return {
        "median": float(statistics.median(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "p5": _percentile(xs, 5.0),
        "p95": _percentile(xs, 95.0),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
    }


def _evaluate_g1_metric(
    ax: Sequence[float],
    lx: Sequence[float],
    direction: str,
) -> Dict[str, Any]:
    """Recompute all G1 statistics and verdict from raw sample lists."""
    a_stats = _engine_stats(ax)
    l_stats = _engine_stats(lx)
    pval = _mann_whitney_u_p(ax, lx)

    a_med = a_stats["median"]
    l_med = l_stats["median"]
    rel_diff_pct = ((a_med - l_med) / l_med * 100.0) if l_med != 0.0 else 0.0
    p95_ratio = (a_stats["p95"] / l_stats["p95"]) if l_stats["p95"] != 0.0 else 1.0

    if direction == "higher":
        gain = ((a_med - l_med) / l_med) if l_med != 0.0 else 0.0
        tail_ratio = (a_stats["p5"] / l_stats["p5"]) if l_stats["p5"] != 0.0 else 1.0
        tail_ok = a_stats["p5"] >= 0.90 * l_stats["p5"]
        non_overlap_asi = a_stats["min"] > l_stats["max"]
    else:
        gain = ((l_med - a_med) / l_med) if l_med != 0.0 else 0.0
        tail_ratio = p95_ratio
        tail_ok = a_stats["p95"] <= 1.10 * l_stats["p95"]
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
        "asi": a_stats,
        "llama": l_stats,
        "mwu_p": pval,
        "rel_diff_pct": rel_diff_pct,
        "improvement_pct": improvement_pct,
        "tail_ratio": tail_ratio,
        "p95_ratio": p95_ratio,
        "tail_ok": tail_ok,
        "non_overlap_asi": non_overlap_asi,
        "g1_verdict": g1_verdict,
    }


def _extract_determinism_entries(det_obj: Any) -> List[Tuple[str, Optional[int], bool]]:
    """Normalize determinism schema variants into [(thread_label, n_runs, passed)]."""
    entries: List[Tuple[str, Optional[int], bool]] = []
    if isinstance(det_obj, list):
        for idx, item in enumerate(det_obj):
            if not isinstance(item, dict):
                continue
            t_lbl = str(item.get("n_threads", item.get("threads", f"entry_{idx}")))
            runs = item.get("n_runs", item.get("runs", item.get("num_runs", item.get("n"))))
            passed = item.get("passed") is True
            entries.append((t_lbl, int(runs) if runs is not None else None, passed))
        return entries

    if not isinstance(det_obj, dict):
        return entries

    for container_key in ("per_thread", "by_threads", "results"):
        sub = det_obj.get(container_key)
        if isinstance(sub, dict):
            for k, v in sub.items():
                if isinstance(v, dict):
                    runs = v.get("n_runs", v.get("runs", v.get("num_runs", v.get("n"))))
                    passed = v.get("passed") is True
                    entries.append((str(k), int(runs) if runs is not None else None, passed))
            return entries
        if isinstance(sub, list):
            return _extract_determinism_entries(sub)

    # Dict keyed directly by thread counts (e.g. {"1": {...}, "2": {...}})
    if det_obj and all(isinstance(v, dict) for v in det_obj.values()):
        for k, v in det_obj.items():
            runs = v.get("n_runs", v.get("runs", v.get("num_runs", v.get("n"))))
            passed = v.get("passed") is True
            entries.append((str(k), int(runs) if runs is not None else None, passed))
        return entries

    runs = det_obj.get("n_runs", det_obj.get("runs", det_obj.get("num_runs", det_obj.get("n"))))
    passed = det_obj.get("passed") is True
    threads_val = det_obj.get("n_threads", det_obj.get("threads", "default"))
    entries.append((str(threads_val), int(runs) if runs is not None else None, passed))
    return entries


def verify_g0(correctness_path: Path) -> Dict[str, Any]:
    """Verify all G0 correctness and determinism gates from a correctness JSON file."""
    data = json.loads(correctness_path.read_text(encoding="utf-8"))
    summary = data.get("summary") or {}
    prompts = data.get("prompts") or []
    config = data.get("config") or {}

    checks: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []

    # 1. batch_vs_token exact position agreement == 100.0% over >= 320 positions
    bvt = summary.get("batch_vs_token") if isinstance(summary.get("batch_vs_token"), dict) else {}
    total_tokens = bvt.get("total_positions", summary.get("total_tokens"))
    bt_agree = bvt.get("agree_positions", summary.get("bt_agree_tokens"))
    bt_pct = bvt.get("agree_pct", summary.get("bt_agree_pct"))

    if total_tokens is None and prompts:
        total_tokens = sum(int(p.get("n_gen", 0)) for p in prompts)
    if bt_agree is None and prompts:
        bt_agree = sum(int(p.get("bt_agree", 0)) for p in prompts)
    if bt_pct is None and total_tokens:
        bt_pct = 100.0 * float(bt_agree or 0) / float(total_tokens)

    max_abs_diff = bvt.get("max_abs_diff", summary.get("bt_max_abs_diff", summary.get("max_abs_diff")))
    if max_abs_diff is None and prompts:
        prompt_diffs = [
            p["bt_max_abs_diff"]
            if "bt_max_abs_diff" in p
            else p.get("max_abs_diff")
            for p in prompts
        ]
        prompt_diffs = [float(x) for x in prompt_diffs if x is not None]
        if prompt_diffs:
            max_abs_diff = max(prompt_diffs)
    max_rel_l2 = max((float(p["rel_l2"]) for p in prompts if p.get("rel_l2") is not None), default=None)

    bvt_missing = total_tokens is None or bt_agree is None or bt_pct is None
    bvt_passed = (
        not bvt_missing
        and int(total_tokens) >= 320
        and int(bt_agree) == int(total_tokens)
        and float(bt_pct) == 100.0
    )
    if bvt_missing:
        bvt_reason = "MISSING/FAILED: batch_vs_token summary missing"
    elif not bvt_passed:
        bvt_reason = f"FAILED: {bt_agree}/{total_tokens} ({bt_pct:.2f}%), require 100.0% over >=320"
    else:
        bvt_reason = "PASS"
    if not bvt_passed:
        failures.append(f"batch_vs_token ({bvt_reason})")
    checks["batch_vs_token"] = {
        "passed": bvt_passed,
        "total_positions": int(total_tokens) if total_tokens is not None else None,
        "agree_positions": int(bt_agree) if bt_agree is not None else None,
        "agree_pct": float(bt_pct) if bt_pct is not None else None,
        "max_abs_diff": float(max_abs_diff) if max_abs_diff is not None else None,
        "max_prompt_rel_l2": max_rel_l2,
        "detail": bvt_reason,
    }

    # 2. teacher_forced top-1 agreement ASI vs llama.cpp >= 98.0% over >= 320 positions
    tf = summary.get("teacher_forced") if isinstance(summary.get("teacher_forced"), dict) else None
    if tf is None:
        tf_passed = False
        tf_detail = "MISSING/FAILED: teacher_forced section missing"
        checks["teacher_forced"] = {
            "passed": False,
            "total_positions": None,
            "agree_positions": None,
            "agree_pct": None,
            "detail": tf_detail,
        }
        failures.append(f"teacher_forced ({tf_detail})")
    else:
        tf_pos = tf.get("total_positions")
        tf_agree = tf.get("asi_vs_llama_agree_positions")
        tf_pct = tf.get("asi_vs_llama_agree_pct")
        if tf_pct is None and tf_pos and tf_agree is not None:
            tf_pct = 100.0 * float(tf_agree) / float(tf_pos)
        tf_missing = tf_pos is None or tf_pct is None
        tf_passed = (not tf_missing) and int(tf_pos) >= 320 and float(tf_pct) >= 98.0
        if tf_missing:
            tf_detail = "MISSING/FAILED: teacher_forced agreement fields missing"
        elif not tf_passed:
            tf_detail = f"FAILED: {tf_agree}/{tf_pos} ({float(tf_pct):.2f}%), require >=98.0% over >=320"
        else:
            tf_detail = "PASS"
        if not tf_passed:
            failures.append(f"teacher_forced ({tf_detail})")
        checks["teacher_forced"] = {
            "passed": tf_passed,
            "total_positions": int(tf_pos) if tf_pos is not None else None,
            "agree_positions": int(tf_agree) if tf_agree is not None else None,
            "agree_pct": float(tf_pct) if tf_pct is not None else None,
            "detail": tf_detail,
        }

    # 3. fp32_ref mean relative-L2 ratio asi_rel_l2 <= 1.05 * llama_rel_l2
    fp32_obj = summary.get("fp32_ref") or data.get("fp32_ref") or {}
    asi_l2_mean = ((tf or {}).get("asi_rel_l2_vs_fp32") or {}).get("mean")
    llama_l2_mean = ((tf or {}).get("llama_rel_l2_vs_fp32") or {}).get("mean")
    if asi_l2_mean is None and isinstance(fp32_obj, dict):
        asi_l2_mean = fp32_obj.get("asi_rel_l2", fp32_obj.get("asi_rel_l2_mean"))
    if llama_l2_mean is None and isinstance(fp32_obj, dict):
        llama_l2_mean = fp32_obj.get("llama_rel_l2", fp32_obj.get("llama_rel_l2_mean"))
    fp32_diff_detected = (tf or {}).get("fp32_ref_diff_detected")

    if asi_l2_mean is None or llama_l2_mean is None:
        fp32_passed = False
        fp32_ratio = None
        fp32_detail = "MISSING/FAILED: fp32_ref mean relative-L2 metrics missing from JSON"
    elif fp32_diff_detected is False:
        fp32_passed = False
        fp32_ratio = float(asi_l2_mean) / float(llama_l2_mean) if float(llama_l2_mean) > 0.0 else None
        fp32_detail = "FAILED: fp32_ref_diff_detected is False (FP32-Ref mode inactive)"
    else:
        a_l2 = float(asi_l2_mean)
        l_l2 = float(llama_l2_mean)
        fp32_ratio = (a_l2 / l_l2) if l_l2 > 0.0 else float("inf")
        fp32_passed = a_l2 <= 1.05 * l_l2
        if fp32_passed:
            fp32_detail = "PASS"
        else:
            fp32_detail = f"FAILED: asi_rel_l2={a_l2:.6f} > 1.05 * llama_rel_l2={l_l2:.6f} (ratio={fp32_ratio:.4f})"
    if not fp32_passed:
        failures.append(f"fp32_ref ({fp32_detail})")
    checks["fp32_ref"] = {
        "passed": fp32_passed,
        "asi_rel_l2_mean": float(asi_l2_mean) if asi_l2_mean is not None else None,
        "llama_rel_l2_mean": float(llama_l2_mean) if llama_l2_mean is not None else None,
        "ratio": fp32_ratio,
        "fp32_ref_diff_detected": fp32_diff_detected,
        "detail": fp32_detail,
    }

    # 4. determinism >= 5 runs (passed == True at 1 and N threads if tested)
    det_raw = data.get("determinism") if "determinism" in data else summary.get("determinism")
    if det_raw is None:
        det_passed = False
        det_entries: List[Tuple[str, Optional[int], bool]] = []
        det_detail = "MISSING/FAILED: determinism results missing from correctness JSON (need >=5 runs, passed=True)"
    else:
        det_entries = _extract_determinism_entries(det_raw)
        top_passed_ok = True
        if isinstance(det_raw, dict) and "passed" in det_raw:
            top_passed_ok = det_raw["passed"] is True
        if not det_entries:
            det_passed = False
            det_detail = "MISSING/FAILED: determinism section present but contains no run entries"
        else:
            bad = [
                f"t={t}: n_runs={r}, passed={p}"
                for t, r, p in det_entries
                if (r is None or r < 5 or not p)
            ]
            det_passed = top_passed_ok and not bad
            if det_passed:
                det_detail = "PASS"
            else:
                reason_bits = bad if bad else ["top-level determinism.passed != True"]
                det_detail = "FAILED: " + "; ".join(reason_bits)
    if not det_passed:
        failures.append(f"determinism ({det_detail})")
    checks["determinism"] = {
        "passed": det_passed,
        "entries": [
            {"threads": t, "n_runs": r, "passed": p}
            for t, r, p in det_entries
        ],
        "detail": det_detail,
    }

    g0_passed = all(c["passed"] for c in checks.values())
    return {
        "path": str(correctness_path),
        "n_threads": config.get("n_threads"),
        "passed": g0_passed,
        "checks": checks,
        "failures": failures,
    }


def _load_bench_runs(bench_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Load one or more thread-level benchmark payloads from a --bench-json path."""
    data = json.loads(bench_path.read_text(encoding="utf-8"))
    if isinstance(data.get("runs"), dict):
        runs: List[Tuple[str, Dict[str, Any]]] = []
        for t_key in sorted(data["runs"], key=lambda s: (int(s) if str(s).isdigit() else 0, str(s))):
            runs.append((f"{bench_path} [threads={t_key}]", data["runs"][t_key]))
        return runs
    return [(str(bench_path), data)]


def verify_g1_run(
    label: str,
    data: Dict[str, Any],
    g0_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recompute and verify G1 verdicts directly from raw 'pairs' in one bench JSON."""
    config = data.get("config") or {}
    passport = config.get("hardware_passport") or {}
    cpu_model = (
        passport.get("cpu_model")
        or config.get("cpu_model")
        or "unknown CPU (no hardware_passport)"
    )
    n_threads = config.get("n_threads") or passport.get("bench_threads")
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Benchmark JSON {label} has no raw 'pairs' list")

    n_pairs = len(pairs)
    asi_runs = [p["asi"] for p in pairs]
    llama_runs = [p["llama"] for p in pairs]

    stored_summary = data.get("summary") or {}
    recomputed: Dict[str, Dict[str, Any]] = {}
    stored_mismatches: List[str] = []

    for key, unit, direction, role in METRIC_SPECS:
        ax = [float(r[key]) for r in asi_runs]
        lx = [float(r[key]) for r in llama_runs]
        info = _evaluate_g1_metric(ax, lx, direction)
        info["unit"] = unit
        info["direction"] = direction
        info["role"] = role
        recomputed[key] = info

        if key in stored_summary and "g1_verdict" in stored_summary[key]:
            stored_v = stored_summary[key]["g1_verdict"]
            if stored_v != info["g1_verdict"]:
                stored_mismatches.append(
                    f"{key}: stored={stored_v} != recomputed={info['g1_verdict']}"
                )

    primary_verdicts = {k: recomputed[k]["g1_verdict"] for k in PRIMARY_METRICS}
    rss_verdict = recomputed[MEMORY_METRIC]["g1_verdict"]
    n_strong = sum(1 for v in primary_verdicts.values() if v == "STRONG WIN")
    n_win = sum(1 for v in primary_verdicts.values() if v in ("WIN", "STRONG WIN"))
    n_loss = sum(1 for v in primary_verdicts.values() if v == "LOSS")
    rss_win = rss_verdict in ("WIN", "STRONG WIN")
    rss_loss = rss_verdict == "LOSS"
    dec_ok = primary_verdicts["t1_dec"] != "LOSS" and primary_verdicts["t2_dec"] != "LOSS"

    if n_strong == len(PRIMARY_METRICS) and rss_win:
        g1_status = "STRICTLY_BEATS"
    elif n_loss == 0 and dec_ok and n_win >= 3 and rss_win:
        g1_status = "BEATS"
    elif not rss_loss and (n_win + int(rss_win)) >= 1 and n_loss < max(1, n_win):
        g1_status = "PARTIAL_ADVANTAGE"
    else:
        g1_status = "DOES_NOT_BEAT"

    g1_reasons: List[str] = []
    if n_pairs < 15:
        g1_reasons.append(f"N={n_pairs} < 15 required for G1")
    for k in (*PRIMARY_METRICS, MEMORY_METRIC):
        v = recomputed[k]["g1_verdict"]
        if v == "LOSS":
            g1_reasons.append(f"{k}=LOSS ({recomputed[k]['improvement_pct']:+.1f}%)")
        elif "tail>10%" in v:
            g1_reasons.append(f"{k}={v} (tail_ratio={recomputed[k]['tail_ratio']:.3f})")
    if not rss_win and rss_verdict != "LOSS":
        g1_reasons.append(f"rss_mb={rss_verdict}")
    if g1_status in ("PARTIAL_ADVANTAGE", "DOES_NOT_BEAT") and n_win < 3:
        g1_reasons.append(f"only {n_win}/5 primary WIN/STRONG WIN")

    g1_passed = (n_pairs >= 15) and (g1_status in ("STRICTLY_BEATS", "BEATS"))

    # Telemetry inspection (if present in pairs)
    steal_asi = [int(r.get("steal_jiffies", 0)) for r in asi_runs]
    steal_llama = [int(r.get("steal_jiffies", 0)) for r in llama_runs]
    has_telemetry = any("steal_jiffies" in r for r in asi_runs)

    # Combined G0 + G1 allowed claim text
    t_suffix = f" ({n_threads} threads)" if n_threads is not None else ""
    combined_reasons = list(g1_reasons)
    if g0_result is None:
        g0_ok = False
        combined_reasons.append("G0 correctness JSON not provided")
    else:
        g0_ok = bool(g0_result["passed"])
        if not g0_ok:
            combined_reasons.append("G0 failed: " + "; ".join(g0_result["failures"]))

    if g0_ok and g1_passed:
        if g1_status == "STRICTLY_BEATS":
            allowed_claim = f"Strictly beats llama.cpp on {cpu_model}{t_suffix}"
        else:
            allowed_claim = f"Beats llama.cpp on {cpu_model}{t_suffix}"
    else:
        reason_str = "; ".join(combined_reasons) if combined_reasons else g1_status
        allowed_claim = f"Does not beat llama.cpp on {cpu_model}{t_suffix} ({reason_str})"

    return {
        "source": label,
        "cpu_model": cpu_model,
        "isa_tier": passport.get("isa_tier"),
        "n_threads": n_threads,
        "n_pairs": n_pairs,
        "n_ok": n_pairs >= 15,
        "lila_asi_env": config.get("lila_asi_env", {}),
        "metrics": recomputed,
        "stored_mismatches": stored_mismatches,
        "g1_status": g1_status,
        "g1_passed": g1_passed,
        "g1_reasons": g1_reasons,
        "primary_wins_or_strong": n_win,
        "primary_strong_wins": n_strong,
        "primary_losses": n_loss,
        "rss_verdict": rss_verdict,
        "has_telemetry": has_telemetry,
        "steal_jiffies_total_asi": sum(steal_asi) if has_telemetry else None,
        "steal_jiffies_total_llama": sum(steal_llama) if has_telemetry else None,
        "steal_jiffies_max": max(steal_asi + steal_llama, default=0) if has_telemetry else None,
        "allowed_claim": allowed_claim,
        "overall_passed": g0_ok and g1_passed,
    }


def inspect_roofline(roofline_path: Path) -> Dict[str, Any]:
    """Read and summarize optional roofline JSON without assuming rigid field names."""
    data = json.loads(roofline_path.read_text(encoding="utf-8"))
    return {"path": str(roofline_path), "data": data}


def _print_g0_report(g0: Optional[Dict[str, Any]]) -> None:
    print("=== G0 Correctness & Determinism Gate ===")
    if g0 is None:
        print("  STATUS: MISSING / NOT PROVIDED (--correctness-json not supplied)")
        return
    print(f"  File   : {g0['path']} (n_threads={g0.get('n_threads')})")
    c = g0["checks"]

    bvt = c["batch_vs_token"]
    max_diff_str = (
        f"max_abs_diff={bvt['max_abs_diff']:.6g}"
        if bvt["max_abs_diff"] is not None
        else f"max_abs_diff=N/A (max_rel_l2={bvt['max_prompt_rel_l2']})"
    )
    print(
        f"  [{'PASS' if bvt['passed'] else 'FAIL'}] batch_vs_token : "
        f"{bvt['agree_positions']}/{bvt['total_positions']} ({bvt['agree_pct']:.2f}%), "
        f"{max_diff_str} -> {bvt['detail']}"
    )

    tf = c["teacher_forced"]
    tf_pct_s = f"{tf['agree_pct']:.2f}%" if tf["agree_pct"] is not None else "n/a"
    print(
        f"  [{'PASS' if tf['passed'] else 'FAIL'}] teacher_forced : "
        f"{tf['agree_positions']}/{tf['total_positions']} ({tf_pct_s}) -> {tf['detail']}"
    )

    fp = c["fp32_ref"]
    if fp["asi_rel_l2_mean"] is not None and fp["llama_rel_l2_mean"] is not None:
        fp_vals = (
            f"asi_rel_l2={fp['asi_rel_l2_mean']:.6f}, "
            f"llama_rel_l2={fp['llama_rel_l2_mean']:.6f}, "
            f"ratio={fp['ratio']:.4f} (limit<=1.05)"
        )
    else:
        fp_vals = "missing"
    print(f"  [{'PASS' if fp['passed'] else 'FAIL'}] fp32_ref       : {fp_vals} -> {fp['detail']}")

    det = c["determinism"]
    if det["entries"]:
        det_vals = ", ".join(
            f"t={e['threads']}:runs={e['n_runs']},passed={e['passed']}" for e in det["entries"]
        )
    else:
        det_vals = "missing"
    print(f"  [{'PASS' if det['passed'] else 'FAIL'}] determinism    : {det_vals} -> {det['detail']}")
    print(f"  G0 VERDICT: {'PASS' if g0['passed'] else 'FAIL'}")


def _print_g1_run_report(run_res: Dict[str, Any]) -> None:
    print(f"\n=== G1 Benchmark Audit: {run_res['source']} ===")
    print(
        f"  Hardware : {run_res['cpu_model']}"
        + (f" ({run_res['isa_tier']})" if run_res.get("isa_tier") else "")
        + f" | threads={run_res['n_threads']} | N={run_res['n_pairs']} (req>=15: {'PASS' if run_res['n_ok'] else 'FAIL'})"
    )
    if run_res.get("lila_asi_env"):
        env_s = ", ".join(f"{k}={v}" for k, v in run_res["lila_asi_env"].items())
        print(f"  Env      : {env_s}")
    if run_res["has_telemetry"]:
        print(
            f"  Telemetry: steal_jiffies asi={run_res['steal_jiffies_total_asi']}, "
            f"llama={run_res['steal_jiffies_total_llama']} (max={run_res['steal_jiffies_max']})"
        )

    rows = []
    for key, unit, direction, role in METRIC_SPECS:
        m = run_res["metrics"][key]
        a = m["asi"]
        l = m["llama"]
        a_s = f"{a['median']:.3f} [{a['min']:.3f}..{a['max']:.3f}]"
        l_s = f"{l['median']:.3f} [{l['min']:.3f}..{l['max']:.3f}]"
        tag = "" if role != "secondary" else " (sec)"
        rows.append((
            key,
            direction,
            a_s,
            f"{a['p5']:.3f}/{a['p95']:.3f}",
            l_s,
            f"{l['p5']:.3f}/{l['p95']:.3f}",
            f"{m['improvement_pct']:+.1f}%",
            f"{m['tail_ratio']:.3f}",
            f"{m['mwu_p']:.4g}",
            f"{m['g1_verdict']}{tag}",
        ))

    headers = (
        "Metric",
        "Dir",
        "ASI med [min..max]",
        "ASI p5/p95",
        "llama med [min..max]",
        "llama p5/p95",
        "Gain%",
        "TailR",
        "MWU p",
        "G1 Verdict",
    )
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    hline = " ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print("  " + hline)
    print("  " + "-" * len(hline))
    for row in rows:
        print("  " + " ".join(f"{val:<{w}}" for val, w in zip(row, widths)))

    if run_res["stored_mismatches"]:
        print("  [WARNING] Stored vs recomputed g1_verdict mismatch:")
        for mm in run_res["stored_mismatches"]:
            print(f"    - {mm}")

    print(
        f"  G1 Tier       : {run_res['g1_status']} "
        f"(primary_win={run_res['primary_wins_or_strong']}/5, "
        f"strong={run_res['primary_strong_wins']}/5, "
        f"loss={run_res['primary_losses']}, rss={run_res['rss_verdict']})"
    )
    print(f"  ALLOWED CLAIM : {run_res['allowed_claim']}")


def _print_roofline_summary(roofline_res: Dict[str, Any]) -> None:
    """Print top-level scalars and structured bench_roofline metrics."""
    print(f"\n=== Roofline Summary ({roofline_res['path']}) ===")
    data = roofline_res["data"]
    for k, v in data.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")

    for entry in data.get("synthetic_benchmarks") or []:
        t = entry.get("threads", "?")
        dram_ro = (entry.get("avx2_dram_read_only") or {}).get("median_gb_s")
        s_copy = (entry.get("stream_copy") or {}).get("median_gb_s")
        s_triad = (entry.get("stream_triad") or {}).get("median_gb_s")
        l3_q4k = (entry.get("l3_q4k_dot2") or {}).get("median_gb_s")
        dram_q4k = (entry.get("dram_q4k_dot2") or {}).get("median_gb_s")
        dram_q6k = (entry.get("dram_q6k_dot2") or {}).get("median_gb_s")
        parts = []
        if dram_ro is not None:
            parts.append(f"DRAM-RO={dram_ro:.2f} GB/s")
        if s_copy is not None and s_triad is not None:
            parts.append(f"STREAM Copy/Triad={s_copy:.2f}/{s_triad:.2f} GB/s")
        if l3_q4k is not None:
            parts.append(f"L3 Q4_K dot2={l3_q4k:.2f} GB/s")
        if dram_q4k is not None and dram_q6k is not None:
            parts.append(f"DRAM Q4_K/Q6_K dot2={dram_q4k:.2f}/{dram_q6k:.2f} GB/s")
        if parts:
            print(f"  Synthetic ({t}T): " + " | ".join(parts))

    model = data.get("asi_model") or {}
    if model.get("present"):
        active_b = model.get("active_decode_weight_bytes", 0)
        bb = model.get("byte_breakdown") or {}
        lm_mb = (bb.get("q6k_token_embd_lm_head_bytes", 0) or 0) / 1e6
        lm_pct = (
            (bb.get("q6k_token_embd_lm_head_bytes", 0) / active_b * 100.0)
            if active_b
            else 0.0
        )
        print(
            f"  Model ({model.get('arch', '?')}): active={active_b / 1e6:.2f} MB "
            f"(Q6_K={bb.get('q6k_pct', 0.0):.2f}%, lm_head={lm_mb:.2f} MB/{lm_pct:.2f}%, "
            f"Q4_K={bb.get('q4k_pct', 0.0):.2f}%, F32={bb.get('f32_pct', 0.0):.2f}%)"
        )
        for run in model.get("decode_runs") or []:
            print(
                f"  Decode ({run.get('threads', '?')}T): {run.get('tok_s', 0.0):.2f} tok/s "
                f"({run.get('ms_per_tok', 0.0):.2f} ms/tok, {run.get('effective_gb_s', 0.0):.2f} GB/s eff, "
                f"{run.get('dram_ro_utilization_pct', 0.0):.1f}% DRAM-RO)"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify G0 + G1 claim eligibility from benchmark/correctness JSONs")
    parser.add_argument("--correctness-json", default=None, help="Path to test_correctness_gate.py JSON output (G0)")
    parser.add_argument(
        "--bench-json",
        action="append",
        default=[],
        help="Path to bench_strict_bar.py JSON output (G1); repeatable for multiple thread counts/machines",
    )
    parser.add_argument("--roofline-json", default=None, help="Optional roofline measurement JSON")
    parser.add_argument("--json-out", default=None, help="Optional output path for structured claim-check report JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if G0 fails/missing or any G1 run does not beat llama.cpp",
    )
    args = parser.parse_args()

    if not args.correctness_json and not args.bench_json and not args.roofline_json:
        parser.error("At least one of --correctness-json, --bench-json, or --roofline-json is required")

    g0_res = verify_g0(Path(args.correctness_json)) if args.correctness_json else None
    _print_g0_report(g0_res)

    g1_results: List[Dict[str, Any]] = []
    for bj in args.bench_json:
        for label, run_data in _load_bench_runs(Path(bj)):
            res = verify_g1_run(label, run_data, g0_res)
            g1_results.append(res)
            _print_g1_run_report(res)

    roofline_res = inspect_roofline(Path(args.roofline_json)) if args.roofline_json else None
    if roofline_res is not None:
        _print_roofline_summary(roofline_res)

    print("\n=== Allowed Claims Summary ===")
    if not g1_results:
        print("  No --bench-json inputs provided.")
    else:
        for r in g1_results:
            print(f"  - {r['allowed_claim']}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "g0": g0_res,
            "g1_runs": g1_results,
            "roofline": roofline_res,
            "all_passed": (
                (g0_res is not None and g0_res["passed"])
                and bool(g1_results)
                and all(r["overall_passed"] for r in g1_results)
            ),
        }
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nExported claim check report to: {out_path}")

    if args.strict:
        g0_ok = g0_res is not None and g0_res["passed"]
        g1_ok = bool(g1_results) and all(r["overall_passed"] for r in g1_results)
        if not (g0_ok and g1_ok):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
