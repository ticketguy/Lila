# ASI Runtime — Trials, Experiments & Failure Log

**Target Hardware:** Lenovo ThinkPad T450s (Intel Core i5-5300U, Broadwell, 2C/4T, AVX2/FMA/F16C, DDR3-1600) & Google Colab CPU (`colab_asi_benchmark.ipynb`)  
**Scope:** Chronological engineering log of kernel optimizations, numerical parity discoveries, integration failures, and benchmark harness iterations.

---

## Trial 01 — `.asi` v2 Cartridge Format & Lazy `mmap` Validation

- **Objective:** Replace ad-hoc model loading with a self-describing, 4 KiB page-aligned binary cartridge (`ASI_V2_MAGIC = 0x41534921u`) supporting zero-copy `mmap` and 64-byte aligned tensor payloads.
- **Implementation:** [`asi_v2.h`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.h), [`asi_v2.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.c), [`asi_v2.py`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.py), [`gguf_to_asi_v2.py`](file:///home/ticket/workspaces/lila/source/engine/asi/gguf_to_asi_v2.py).
- **Initial Pitfall (Cold Page Faults vs. Eager CRC-64):** Running CRC-64 ECMA (`0x42F0E1EBA9EA3693ULL`) over the entire `ASI_SEC_WEIGHTS` section on `asi_v2_open()` forced the OS to fault in every weight page eagerly, inflating load latency and RSS.
- **Resolution:** `asi_v2_validate_memory()` always verifies CRC-64 on small metadata/config/tokenizer sections, while skipping bulk blobs (`ASI_SEC_WEIGHTS`, `ASI_SEC_VISION`, `ASI_SEC_AUDIO`) unless `LILA_ASI_VERIFY_CRC=1` is set.
- **Verification:** `make test-v2` ([`test_asi_v2.c`](file:///home/ticket/workspaces/lila/source/engine/tests/test_asi_v2.c)) confirms valid cartridge loading and negative-test rejection when 1 byte of `ASI_SEC_MODEL_CONFIG` is flipped.

---

## Trial 02 — `q6k_repacked_matmul_batch` Stride Bug (`30d89a8` / Colab)

- **Context:** Pre-unpacking 210-byte `Q6_K` blocks into 288-byte `LilaQ6KRepackedBlock` (`u6[256]`, `scales[16]`, FP32 `d`, `pad[12]`) via `asi_v2_repack_q6k()` to eliminate runtime `ql`/`qh` nibble unpacking on the 248k-row `lm_head` and projections.
- **Failure Observed:** On batched prefill (`n_tokens > 1`), `q6k_repacked_matmul_batch` produced corrupted outputs when indexing `qx_all` in the 4-token unrolled loop (`t..t+3`).
- **Root Cause:** `quantize_vec_q8k_strided(qx_all + t, n_tokens, ...)` stores `LilaQ8KBlock` elements in block-major, token-strided order (`qx_all + (size_t)b * n_tokens + t`). Indexing with a non-strided offset or mismatched block stride caused tokens to read neighboring blocks' `qs`/`bsums`/`d`.
- **Resolution:** Standardized block-major stride indexing `qx_all + (size_t)b * n_tokens + (t + k)` across both the 4-token unrolled loop and the tail loop in [`q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h#L2160-L2237).

---

## Trial 03 — 32-YMM Register Spilling on Broadwell AVX2 (16 Architectural YMMs)

- **Context:** Broadwell (i5-5300U) has **16** architectural 256-bit `ymm` registers (`ymm0`–`ymm15`), unlike AVX-512 targets with 32 `zmm` registers.
- **Failure Observed:** Unrolling too many output rows × token columns simultaneously in AVX2 `Q4_K`/`Q6_K` batch microkernels exceeded 16 YMM registers, causing GCC to spill accumulators and mask/unpack temporaries onto the stack inside the inner `b` loop.
- **Resolution:** Constrained AVX2 batched microkernels in [`q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h) to 4-token (`t..t+3`) accumulators per single row (`acc0..acc3` / `sum_vec0..sum_vec3`) and used 2-row (`dot2`) pairing only for single-token `matvec` (`q4k_dot2_q8k_avx2`, `q6k_dot2_q8k_avx2`).

---

## Trial 04 — `dot2` vs. `dot1` on `lm_head` & Multi-Threaded Matvec

- **Context:** Evaluating 2-row `dot2` kernels (`q4k_dot2_q8k_avx2`, `q6k_dot2_q8k_avx2`) vs. 1-row `dot1` (`q4k_dot_q8k_simd`, `q6k_dot_q8k_simd`) on the 248k-row `lm_head` and layer projections.
- **Observation:** In [`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c#L313-L375), `asi_matvec_worker` uses `dot2` when `n_threads <= 1` (reusing `qx` loads across two rows per block), whereas when `n_threads > 1` it switches to `dot1` with atomic dynamic chunking (`chunk_rows = 512` for `rows >= 16384`, `64` otherwise) and explicit `LILA_PREFETCH` (`+ 2 * rb` and `+ 2 * rb + 64`) to avoid contention and register pressure across threads.

---

## Trial 05 — FP32 Reduction Order Causing Layer 4 `Q8_K` Boundary Drift

- **Context:** Comparing `lila_asi_eval_batch` (prefill) against `lila_asi_eval_token` (token-by-token) on `Qwen3.5-0.8B`.
- **Failure Observed:** Even tiny ULPs of FP32 difference from different summation orders (e.g., batched vs. single-token RMSNorm / DeltaNet / projection accumulation) can cross a `Q8_K` quantization rounding boundary (`amax / 127.0f` in `quantize_vec_q8k_strided`) at intermediate layers (observed around Layer 4), amplifying into measurable logit L2 drift or occasional greedy argmax flips on near-tie tokens.
- **Resolution & Gate:**
  - Verified kernel-level parity against FP32 scalar reference in [`test_quant_kernels.c`](file:///home/ticket/workspaces/lila/source/engine/tests/test_quant_kernels.c) (`rel_l2 < 0.01`).
  - Built [`test_correctness_gate.py`](file:///home/ticket/workspaces/lila/source/engine/tests/test_correctness_gate.py) to measure end-to-end prefill logit `rel_l2`, `ASI(batch)` vs. `ASI(token)` first divergence, and corpus-wide greedy top-1 agreement (5 prompts × 32 tokens = 160 tokens) against `llama-cpp-python` on identical token IDs.
  - Unifying the FP32 block-level reduction order in `q4k.h` (accumulating 8-lane `__m256 sum_vec` + scalar `float bias` and reducing via `lila_hsum_f32_avx2(sum_vec) - bias`) reduced `ASI(batch)` vs. `ASI(token)` logit relative L2 error from `0.050–0.154` down to `0.000000` across all 5 prompts, yielding `160/160` (`100.00%`) token agreement at `N=32` and `320/320` (`100.00%`) at `N=64` (`135/160` = `84.38%` at `N=32` and `229/320` = `71.56%` at `N=64` vs. `llama.cpp`).

---

## Trial 06 — `test_5` Daemon Timeout & 860-Token Prompt Sizing

- **Context:** [`qualify_asi_backend.py`](file:///home/ticket/workspaces/lila/source/engine/tests/qualify_asi_backend.py) `test_5_daemon_with_lila_backend_asi()` boots a live `LilaDaemon` over mTLS HTTPS (`/chat`) with `LILA_BACKEND=asi`.
- **Failure Observed:** Early runs with full LilaCore system + tool prompts (~860 tokens) hit client HTTP timeouts on the 2-core Broadwell laptop before prefill + decode completed.
- **Resolution:**
  1. In `qualify_asi_backend.py`, raised the mTLS `/chat` request timeout to `420s` and isolated the ~860-token `LONG_PROMPT` benchmark into [`bench_strict_bar.py`](file:///home/ticket/workspaces/lila/source/engine/tests/bench_strict_bar.py) as `long_prefill_tps`.
  2. Added chunked prefill (`chunk_max = 128` in `qwen35_forward_batch`), persistent scratch (`ensure_batch_scratch`), and `asi_pool_begin`/`asi_pool_end` spin-wait pooling (`AsiThreadPool`) to eliminate per-layer thread wakeups and allocations during long prefills.

---

## Trial 07 — Strict Apples-to-Apples Benchmark Harness (`bench_strict_bar.py`)

- **Objective:** Eliminate confounding variables from earlier `bench_compare.py` / `test_6_controlled_medians` runs (e.g., differing tokenizers, unverified KV reuse, cold-vs-warm cache asymmetry, small `N=5` sample size).
- **Methodology Implemented in [`bench_strict_bar.py`](file:///home/ticket/workspaces/lila/source/engine/tests/bench_strict_bar.py):**
  - Identical `int32` token IDs tokenized once and fed to both `liblila_asi.so` and `llama-cpp-python`.
  - Assertion-verified Turn-2 KV/prefix reuse (`t2_eval_tokens == len(toks["t2"])`).
  - Matched `n_ctx=2048`, `n_threads=2`, `n_threads_batch=2`, `n_batch=512`, `use_mmap=True`, `flash_attn=False`, greedy `temperature=0.0`.
  - Discarded warmup + `N >= 15` interleaved subprocess pairs + optional `--prepopulate` (`MADV_WILLNEED` + 4 KiB page touch).
  - 95% bootstrap CI + two-sided Mann-Whitney U + non-overlapping `[min..max]` strict bar.
- **Status (i5-5300U `N=15`, [`results_broadwell_n15.json`](file:///home/ticket/workspaces/lila/source/paper/results_broadwell_n15.json)):** `t1_dec` (`14.064` vs `13.943 tok/s`, `p=0.9339`), `t2_ttft` (`0.554` vs `0.561 s`, `p=0.901`), and `t2_dec` (`13.812` vs `13.952 tok/s`, `p=0.9339`) reached statistical parity; `rss_mb` (`615.8` vs `978.0 MiB`), `vm_hwm_mb` (`615.8` vs `1101.2 MiB`), and `load_s` (`0.026` vs `2.087 s`) passed with non-overlapping ranges; `t1_ttft` (`1.055` vs `1.002 s`) and `long_prefill_tps` (`45.055` vs `53.747 tok/s`) remain behind `llama.cpp`. Google Colab multi-core verification remains pending execution of [`colab_asi_benchmark.ipynb`](file:///home/ticket/workspaces/lila/source/engine/tests/colab_asi_benchmark.ipynb).

---

## Trial 08 — Prefill GEMM 1×4 Register-Accumulated Microkernel, dTLB-Aware `T_TILE`, `chunk_max=512`, and `T_TILE=32` `conv1d`

- **Hypotheses & Iterations (`q4k.h`, `asi_arch.c`):**
  1. **Exp 1 (`2x4` in-place + `T_TILE=32`):** Improved Q4_K, but Q6_K `2x4` regressed (`20.2 -> 27.6 ms`) from Broadwell 16-YMM register spilling; reverted Q6_K `2x4`. `long_prefill_tps` moved `23.50 -> 25.67 tok/s` (`N=3`).
  2. **Exp 2 (Unpack-2-rows-once into L1d + `2x4`):** Unpacking 2 rows into stack L1d buffers (`u0, u1`, `sc`, `mn`, `d`) once per `T_TILE` lifted `long_prefill_tps` to `27.70 tok/s` (`N=3`), `t1_ttft` `2.250s -> 1.590s`, `t2_ttft` `1.153s -> 0.812s`.
  3. **Exp 3 (`chunk_max = 512`):** Raising `chunk_max` from `128` to `512` in `qwen35_forward_batch` reduced chunk/barrier overhead at `B=512` (`16.96s` / `30.19 tok/s` -> `15.59s` / `32.84 tok/s`).
  4. **Exp 4 (`1x4` register-accumulated + pre-broadcast scales + SIMD `bsums` + dTLB `T_TILE`):** Microbenchmarking showed `2x4` spilled `sv[2][4][8]` per block `b`, whereas `1x4` keeps `sv0..3` and `sumi0..3` resident in YMM registers (`12.52 ms` / `37.54 GMAC/s` vs `23.10 ms` at `T=128`). Floating-point `bsums` accumulation broke `0.000000` batch/token parity (`0.058` L2) and was **rejected** in favor of exact `int32` `_mm256_madd_epi16` `bias0..3`. Sizing `T_TILE = (n_tokens <= 64 && blocks_per_row <= 4) ? 64 : 32` avoided 64-entry L1 dTLB thrashing at `T=512` (`61.00 ms` vs `78.95 ms` for `T_TILE=64`). `B=512` improved to `13.51s` (`37.90 tok/s`).
  5. **Exp 5 (`asi_conv1d_worker` `T_TILE=32` register-cached state):** Tiling `conv_hist == 3 && k_conv == 4 && B >= 2` with `ch` outer / `t` inner (`T_TILE=32`) and register-cached `s0, s1, s2` removed per-token `conv_state` shifts, lifting `B=512` to `13.04s` (`39.27 tok/s`), `B=53` to `1.19s` (`44.58 tok/s`), and `N=15` interleaved `long_prefill_tps` median to **`45.055 [13.927..48.266] tok/s`** while keeping `0.000000` batch/token L2 parity.
