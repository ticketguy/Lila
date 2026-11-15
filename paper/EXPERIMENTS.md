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
  - Unifying the FP32 block-level reduction order in `q4k.h` (accumulating 8-lane `__m256 sum_vec` + scalar `float bias` and reducing via `lila_hsum_f32_avx2(sum_vec) - bias`) reduced `ASI(batch)` vs. `ASI(token)` logit relative L2 error from `0.050–0.154` down to `0.000000` across all 5 prompts (`320/320` = `100.00%` at `N=64`). Adding the Teacher-Forced + `LILA_ASI_FP32_REF=1` audit (`5 x 64 = 320` positions, [`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json)) confirmed **`316/320` (`98.75%`)** `ASI(Q8_K)` vs. `llama.cpp` top-1 agreement (`315/320` vs `FP32-Ref`, `llama.cpp` `316/320` vs `FP32-Ref`), lower mean/median logit rel-L2 vs `FP32-Ref` (`0.047385/0.038031` vs `0.047615/0.038385`), and `4/320` flips all occurring at tiny near-tie `FP32-Ref` margins (`<= 0.0948`).

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
  - Canonical model artifact across all machines (`FadedRedStar/Qwen3.5-0.8B-heretic-imatrix-GGUF` / `Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf`, GGUF SHA-256 `d9662c91079317aee661a1475d61412787696e0dcc5636c39648fb27d49e1edf`, `.asi` SHA-256 `757731964e8a24950f3ae743be9df81ca5671f15b014aa45c31868131679e9f0`), eliminating earlier local imatrix (`d9662c91`) vs. Colab `mradermacher` (`cc90d6ad`) GGUF skew.
  - Identical `int32` token IDs tokenized once and fed to both `liblila_asi.so` and `llama-cpp-python`.
  - Assertion-verified Turn-2 KV/prefix reuse (`t2_eval_tokens == len(toks["t2"])`).
  - Matched `n_ctx=2048`, `n_threads=2`, `n_threads_batch=2`, `n_batch=512`, `use_mmap=True`, `flash_attn=False`, greedy `temperature=0.0`.
  - Discarded warmup + `N >= 15` interleaved subprocess pairs + default `--prepopulate` (`load_s = load_open_s + populate_s` via `madvise(MADV_WILLNEED)` + 4 KiB page touch, plus `--no-prepopulate`).
  - 95% bootstrap CI, two-sided Mann-Whitney U, `p95`/`max` tail reporting, non-overlapping `[min..max]` strict bar, and honest `_evaluate_verdict` labels (`PARITY (p=...)` when `p >= 0.05`, `ADVANTAGE (overlap)`, `PASS`, `FAIL`).
- **Status (i5-5300U `N=15`, [`results_broadwell_n15.json`](file:///home/ticket/workspaces/lila/source/paper/results_broadwell_n15.json)):** Initially (Trial 08, without `--prepopulate`) `t1_dec`, `t2_ttft`, and `t2_dec` showed statistical parity (`p=0.901..0.9339`), while `t1_ttft` and `long_prefill_tps` trailed `llama.cpp`. After Trial 09 optimizations and `--prepopulate` resident-weight enforcement, `long_prefill_tps` (`52.799` vs `46.213 tok/s`, `p=0.0003609`), `t1_ttft` (`0.908` vs `1.094 s`, `p=0.00105`), and `t2_ttft` (`0.487` vs `0.607 s`, `p=0.003691`) show statistically significant `ADVANTAGE (overlap)` on ASI; `t1_dec` (`p=0.3401`) and `t2_dec` (`p=0.3615`) remain `PARITY`; `rss_mb` (`616.266` vs `978.223 MiB`), `vm_hwm_mb` (`638.824` vs `1406.730 MiB`), and `load_s` (`0.053` vs `2.182 s`) `PASS` with non-overlapping ranges (`p < 3.4e-06`), while `OVERALL PRIMARY BAR` remains `FAIL` due to narrow min-max overlaps and decode parity. Second-machine AVX2 verification via [`colab_asi_benchmark.ipynb`](file:///home/ticket/workspaces/lila/source/engine/tests/colab_asi_benchmark.ipynb) (`Intel Xeon @ 2.20GHz`, AVX2-only), wide-SIMD verification (`AVX-512/VNNI`, e.g., Sapphire Rapids VM), and `ARM64 NEON` verification (e.g., Oracle Ampere A1, Raspberry Pi, or Apple Silicon Mac) remain pending hardware runs.

---

## Trial 08 — Prefill GEMM 1×4 Register-Accumulated Microkernel, dTLB-Aware `T_TILE`, `chunk_max=512`, and `T_TILE=32` `conv1d`

- **Hypotheses & Iterations (`q4k.h`, `asi_arch.c`):**
  1. **Exp 1 (`2x4` in-place + `T_TILE=32`):** Improved Q4_K, but Q6_K `2x4` regressed (`20.2 -> 27.6 ms`) from Broadwell 16-YMM register spilling; reverted Q6_K `2x4`. `long_prefill_tps` moved `23.50 -> 25.67 tok/s` (`N=3`).
  2. **Exp 2 (Unpack-2-rows-once into L1d + `2x4`):** Unpacking 2 rows into stack L1d buffers (`u0, u1`, `sc`, `mn`, `d`) once per `T_TILE` lifted `long_prefill_tps` to `27.70 tok/s` (`N=3`), `t1_ttft` `2.250s -> 1.590s`, `t2_ttft` `1.153s -> 0.812s`.
  3. **Exp 3 (`chunk_max = 512`):** Raising `chunk_max` from `128` to `512` in `qwen35_forward_batch` reduced chunk/barrier overhead at `B=512` (`16.96s` / `30.19 tok/s` -> `15.59s` / `32.84 tok/s`).
  4. **Exp 4 (`1x4` register-accumulated + pre-broadcast scales + SIMD `bsums` + dTLB `T_TILE`):** Microbenchmarking showed `2x4` spilled `sv[2][4][8]` per block `b`, whereas `1x4` keeps `sv0..3` and `sumi0..3` resident in YMM registers (`12.52 ms` / `37.54 GMAC/s` vs `23.10 ms` at `T=128`). Floating-point `bsums` accumulation broke `0.000000` batch/token parity (`0.058` L2) and was **rejected** in favor of exact `int32` `_mm256_madd_epi16` `bias0..3`. Sizing `T_TILE = (n_tokens <= 64 && blocks_per_row <= 4) ? 64 : 32` avoided 64-entry L1 dTLB thrashing at `T=512` (`61.00 ms` vs `78.95 ms` for `T_TILE=64`). `B=512` improved to `13.51s` (`37.90 tok/s`).
  5. **Exp 5 (`asi_conv1d_worker` `T_TILE=32` register-cached state):** Tiling `conv_hist == 3 && k_conv == 4 && B >= 2` with `ch` outer / `t` inner (`T_TILE=32`) and register-cached `s0, s1, s2` removed per-token `conv_state` shifts, lifting `B=512` to `13.04s` (`39.27 tok/s`), `B=53` to `1.19s` (`44.58 tok/s`), and `N=15` interleaved `long_prefill_tps` median to **`45.055 [13.927..48.266] tok/s`** while keeping `0.000000` batch/token L2 parity.

---

## Trial 09 — Bounded-Spin Thread Pool, Single-Unpack `tile_out` GEMM, Dynamic Work-Stealing, Teacher-Forced/`FP32-Ref` Audit, and `--prepopulate` Fair Load

- **Root Cause Hypotheses:**
  1. *Extreme tail latency spikes:* Workers and barrier polling in `asi_pool.h` spun indefinitely on shared cache lines (`false sharing` + OS preemption on 2 vCPUs).
  2. *Prefill gap vs `llama.cpp`:* Unpacking Q4_K/Q6_K weight rows inside every token tile repeated 4-bit/6-bit bit-unpacking work, while static row/head splits left one thread idle when preempted, and batch non-GEMM stages (`asi_norm_quant`, `asi_kv_prep`, scalar `conv1d`) ran single-threaded or scalar.
  3. *Correctness opacity:* Free-running `ASI` vs `llama.cpp` agreement (`71.6%` / `74.38%`) collapsed after single near-tie argmax flips without showing logit margins or which engine deviated from unquantized FP32 math.
  4. *Cold-page load asymmetry:* `use_mmap=True` let ASI defer page faults into Turn-1 TTFT while `llama.cpp` absorbed them earlier, obscuring resident startup cost.

- **Changes Implemented (`asi_pool.h`, `q4k.h`, `asi_arch.c`, `test_correctness_gate.py`, `bench_strict_bar.py`):**
  1. *Bounded-spin cache-padded thread pool (`asi_pool.h`):* Aligned hot atomics (`fn`, `gen`, `done_count`, `active`, `sleeping`, `shutdown`) on `_Alignas(64)`; bounded worker wait (`asi_cpu_pause()` < 2048, `sched_yield()` < 4096, then `p->sleeping` + `pthread_cond_wait`) and added `sched_yield()` after 2048 barrier spins in `asi_pool_run`.
  2. *1× Row Unpack + `tile_out` GEMM (`q4k.h`):* Added `lila_q4k_unpack_row_avx2` and `lila_q6k_unpack_row_avx2`, restructuring AVX2 `q4k_matmul_batch_q8k_range` and `q6k_matmul_batch_q8k_range` (`n_tokens >= 4 && blocks_per_row <= 16`) to unpack each weight row once per token chunk across paired `2x4` blocks into contiguous `tile_out`.
  3. *Dynamic work-stealing & parallel non-GEMM prefill (`asi_arch.c`):* Converted batch matmul, DeltaNet, and GQA workers to atomic `next_row`/`next_h` work-stealing; parallelized `asi_norm_quant_pool` and `asi_kv_prep_batch_worker`; vectorized `asi_conv1d_worker` with AVX2 `_MM_TRANSPOSE4_PS` + `lila_expf_avx2` (fixing a 1-ULP `4.3e-8` FMA vs `-std=c11` mismatch).
  4. *Unquantized `FP32-Ref` & Teacher-Forced Audit (`q4k.h`, `asi_arch.c`, `test_correctness_gate.py`):* Added `dequant_q4k_block`, `dequant_q6k_block`, `dequant_q6k_repacked_block`, `lila_quant_dot_fp32`, and `lila_quant_matmul_batch_fp32_range` toggled via `LILA_ASI_FP32_REF=1`, plus isolated `--fp32-worker` evaluation in `test_correctness_gate.py`.
  5. *Fair `--prepopulate` load & tail metrics (`bench_strict_bar.py`):* Enabled `--prepopulate` by default (`load_s = load_open_s + populate_s` via `madvise(MADV_WILLNEED)` + 4 KiB page touch, alongside `--no-prepopulate`) and added `p95`/`max` reporting and explicit `PARITY`/`ADVANTAGE (overlap)` verdicts.

- **Verified Results (`i5-5300U`, `n_threads=2`, `N=64` correctness & `N=15` interleaved `--prepopulate` benchmark):**
  - *Correctness Gate ([`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json)):*
    - Batch vs token-by-token (`N=64`): `0.000000` L2, `320/320 (100.00%)` agreement.
    - Teacher-forced top-1 (`5 x 64 = 320` positions): `ASI(Q8_K) vs llama.cpp = 316/320 (98.75%)`, `ASI(Q8_K) vs FP32-Ref = 315/320 (98.44%)`, `llama.cpp vs FP32-Ref = 316/320 (98.75%)`.
    - Logit rel-L2 vs `FP32-Ref`: `ASI(Q8_K)` mean `0.047385` (median `0.038031`, max `0.404697`) vs `llama.cpp` mean `0.047615` (median `0.038385`, max `0.342743`).
    - All `4/320` `ASI vs llama.cpp` flips occurred on near-tie `FP32-Ref` top-1/top-2 margins (`max=0.0948, median=0.0715, min=0.0500`): `ASI==FP32` (`1`), `Llama==FP32` (`2`), `Neither==FP32` (`1`).
  - *Strict Benchmark ([`results_broadwell_n15.json`](file:///home/ticket/workspaces/lila/source/paper/results_broadwell_n15.json), `--prepopulate`, `T1=53, T2_delta=27, Long=939`):*
    - `long_prefill_tps`: ASI `52.799 [44.701..55.541]` (`p95=54.864, max=55.541`) vs llama `46.213 [40.022..50.272]` (`p95=49.912, max=50.272`) (`p=0.0003609`, `ADVANTAGE (overlap)`).
    - `t1_ttft`: ASI `0.908 [0.862..1.208]` (`p95=1.128, max=1.208`) vs llama `1.094 [1.027..1.810]` (`p95=1.604, max=1.810`) (`p=0.00105`, `ADVANTAGE (overlap)`).
    - `t2_ttft`: ASI `0.487 [0.459..0.953]` (`p95=0.844, max=0.953`) vs llama `0.607 [0.559..0.839]` (`p95=0.794, max=0.839`) (`p=0.003691`, `ADVANTAGE (overlap)`).
    - `t1_dec`: ASI `13.808 [9.232..15.030]` vs llama `13.477 [9.162..14.280]` (`p=0.3401`, `PARITY`).
    - `t2_dec`: ASI `13.062 [10.792..14.722]` vs llama `12.889 [10.098..13.918]` (`p=0.3615`, `PARITY`).
    - `rss_mb` (`616.266` vs `978.223 MiB`, `p=3.341e-06`), `vm_hwm_mb` (`638.824` vs `1406.730 MiB`, `p=3.375e-06`), `load_s` (`0.053` vs `2.182 s`, `p=3.392e-06`), and `load_open_s` (`0.027` vs `2.156 s`, `p=3.392e-06`): `PASS` with non-overlapping ranges.
    - `OVERALL PRIMARY BAR`: `FAIL` (due to decode parity and narrow min-max overlaps on TTFT/prefill).

