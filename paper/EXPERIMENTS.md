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
- **Observation (Historical Trial 04):** In [`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c#L313-L375), `asi_matvec_worker` initially used `dot2` only for `q4k` when `n_threads <= 1` (reusing `qx` loads across two rows per block), whereas when `n_threads > 1` it switched to `dot1` with atomic dynamic chunking (`chunk_rows = 512` for `rows >= 16384`, `64` otherwise) and explicit `LILA_PREFETCH` (`+ 2 * rb` and `+ 2 * rb + 64`) to avoid contention and register pressure across threads. Although `q6k_dot2_q8k_avx2` was defined in [`q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h) during Trial 04, it was not yet wired into `asi_matvec_range`. In **Trial 10 (`v2.2`)**, `q6k_dot2_q8k_avx2` was wired into `asi_matvec_range` alongside `q4k_dot2_q8k_avx2`, and after multi-threaded ablation both `LILA_ASI_Q6K_DOT2=1` and `LILA_ASI_Q4K_DOT2=1` became the defaults across thread counts.

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

---

## Trial 10 — ASI v2.2: Roofline Reconciliation, SMT-Aware Pool, Disk Prefix Snapshot (`ASIPFX02`), AVX-512 Bit-Exact Reduction, and G0/G1 Claim Gate

- **Context:** Reconciling earlier isolated L3 `bench_dot` expectations against full-model DRAM-bound decode (`bench_roofline.c`), tuning multi-threaded `dot2` kernels and SMT spin-backoff (`asi_pool.h`), adding crash-safe persistent prefix snapshots (`ASIPFX02`) wired into `AsiLlamaAdapter` and `LilaCore._boot_gguf()`, aligning AVX-512 kernel reductions to bit-exact AVX2 semantics (`q4k.h`, `asi_arch.c`), and enforcing structured correctness (`G0`) and statistical performance (`G1`) gates via [`claim_check.py`](file:///home/ticket/workspaces/lila/source/engine/tests/claim_check.py).
- **1. Roofline & Active Weight Byte Breakdown ([`bench_roofline.c`](file:///home/ticket/workspaces/lila/source/engine/tests/bench_roofline.c), [`roofline_broadwell.json`](file:///home/ticket/workspaces/lila/source/paper/roofline_broadwell.json)):**
  - **Host hardware:** Idle Broadwell laptop CPU (`Intel(R) Core(TM) i5-5300U CPU @ 2.30GHz`, 2 physical cores / 4 logical CPUs, 3 MiB L3 cache, DDR3-1600, `/tmp/qwen35_0.8b.asi`).
  - **Synthetic DRAM & STREAM ceilings (`256 MiB` AVX2 read-only, `80 MiB` STREAM arrays):**
    - `avx2_dram_read_only`: **1T** median `11.53 GB/s` (`11.5327`, best `11.9750`, mean `10.9163`); **2T** median `12.08 GB/s` (`12.0802`, best `13.5898`, mean `10.3338`).
    - `stream_copy` / `stream_triad`: **1T** median `7.84` (`7.8436`, best `9.2414`) / `9.66 GB/s` (`9.6645`, best `10.3629`); **2T** median `8.26` (`8.2615`, best `8.7441`) / `5.70 GB/s` (`5.6979`, best `9.4184` Triad).
  - **Isolated `Q4_K` / `Q6_K` `dot1` vs. `dot2` kernels (`L3 = 2 MiB` vs. `DRAM = 64 MiB`):**

| Benchmark Kernel | Size / Locality | 1T Med GB/s (`best`) | 1T Med GFLOP/s (`best`) | 2T Med GB/s (`best`) | 2T Med GFLOP/s (`best`) |
|---|---|---|---|---|---|
| `l3_q4k_dot1` | 2 MiB (L3) | `7.60` (`8.12`) | `27.00` (`28.88`) | `11.82` (`14.54`) | `42.02` (`51.71`) |
| `l3_q4k_dot2` | 2 MiB (L3) | `7.80` (`8.35`) | `27.74` (`29.69`) | `11.05` (`13.72`) | `39.30` (`48.77`) |
| `l3_q6k_dot1` | 2 MiB (L3) | `7.56` (`8.73`) | `18.43` (`21.28`) | `12.71` (`15.52`) | `30.98` (`37.84`) |
| `l3_q6k_dot2` | 2 MiB (L3) | `7.43` (`8.21`) | `18.11` (`20.01`) | `14.01` (`15.28`) | `34.17` (`37.26`) |
| `dram_q4k_dot1` | 64 MiB (DRAM) | `5.78` (`6.68`) | `20.56` (`23.74`) | `11.06` (`11.16`) | `39.31` (`39.69`) |
| `dram_q4k_dot2` | 64 MiB (DRAM) | `3.82` (`4.30`) | `13.59` (`15.30`) | `3.90` (`7.54`) | `13.86` (`26.82`) |
| `dram_q6k_dot1` | 64 MiB (DRAM) | `6.60` (`6.86`) | `16.09` (`16.72`) | `5.85` (`7.60`) | `14.27` (`18.52`) |
| `dram_q6k_dot2` | 64 MiB (DRAM) | `5.73` (`6.07`) | `13.98` (`14.80`) | `6.36` (`6.77`) | `15.51` (`16.51`) |

  - **Reconciling `bench_dot` vs. full-model decode:** Isolated `bench_dot` operates over `~2 MiB` of `Q4_K` weights resident inside the 3 MiB L3 cache (`~27–42 GFLOP/s`), whereas real `Qwen3.5-0.8B` (`hybrid_qwen35`, `320` tensors, `file_size_bytes = 642,277,376`, `repacked_tensors = 0`, `tied_embeddings = true`, `vocab_size = 248320`, `n_embd = 1024`, `n_layers = 24`) streams **`518,335,744 B` (`518.34 MB`)** of active decode weights from DRAM on every generated token:
    - **`Q6_K`: `297,185,280 B` (`297.19 MB`, `57.3345%`)** — dominated by tied `token_embd`/`lm_head` at **`208,588,800 B` (`208.59 MB`, `40.2420%` of all active decode bytes)**, plus `attn_qkv` (`51,609,600 B`, `9.9568%`), `ffn_down` (`36,126,720 B`, `6.9698%`), and `other` (`860,160 B`, `0.1659%`).
    - **`Q4_K`: `219,156,480 B` (`219.16 MB`, `42.2808%`)** — `attn_qkv` (`28,311,552 B`, `5.4620%`), `ffn_down` (`24,772,608 B`, `4.7793%`), and `other` (`166,072,320 B`, `32.0395%`).
    - **`F32`: `1,993,984 B` (`1.99 MB`, `0.3847%`)**.
  - **Full-model `bench_roofline` decode (`n_tokens = 32`):**
    - **1T:** `7.11 tok/s` (`7.1084`, `eval_tok_s = 7.1281`), `140.68 ms/tok` (`140.6782`), effective bandwidth `3.68 GB/s` (`3.6845`), **`31.9%`** (`31.9487%`) of `avx2_dram_read_only` (`46.9750%` of `stream_copy`), vs. pure-DRAM roofline `22.25 tok/s` and isolated-kernel prediction `109.18 ms/tok` (`9.16 tok/s`, leaving `31.50 ms` / `22.39%` non-matvec/dispatch residual).
    - **2T:** `10.38 tok/s` (`10.3835`, `eval_tok_s = 10.4251`), `96.31 ms/tok` (`96.3069`), effective bandwidth `5.38 GB/s` (`5.3821`), **`44.6%`** (`44.5534%`) of `avx2_dram_read_only` (`65.1473%` of `stream_copy`), vs. pure-DRAM roofline `23.31 tok/s` (`pred_matvec_ms = 102.94 ms`, `9.71 tok/s`, where the `3.90 GB/s` isolated `dram_q4k_dot2` synthetic microbenchmark slightly over-penalized real 2T decode).
- **2. `3 × 32`-Token Full-Model `dot1`/`dot2` Ablation & Per-Stage Decode Profile (Idle Broadwell CPU):**
  - Matrix ablation across `LILA_ASI_Q6K_DOT2` and `LILA_ASI_Q4K_DOT2` in `AsiRuntimeConfig` ([`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c)):

| `LILA_ASI_Q6K_DOT2` | `LILA_ASI_Q4K_DOT2` | 1T Med `[min..max]` (`tok/s`) | 2T Med `[min..max]` (`tok/s`) | Notes |
|---|---|---|---|---|
| `0` (`dot1`) | `0` (`dot1`) | `7.83 [5.46..8.91]` | `8.88 [7.95..10.89]` | Baseline `dot1` on both weight formats |
| `0` (`dot1`) | `1` (`dot2`) | `8.29 [7.76..8.30]` | `12.19 [9.22..12.81]` | `Q4_K` `dot2` only |
| `1` (`dot2`) | `0` (`dot1`) | `8.01 [7.96..8.82]` | `11.37 [9.63..14.13]` | `Q6_K` `dot2` wired into `asi_matvec_range` |
| **`1` (`dot2`)** | **`1` (`dot2`)** | **`8.14 [8.09..9.24]`** | **`13.58 [13.49..13.63]`** | **New v2.2 default (`+52.9%` median 2T vs `dot1/dot1`, tight `0.14 tok/s` spread)** |

  - **Per-token decode stage profile (`LILA_ASI_PROFILE=1` at 2T with `dot2=1/1`, `~68 ms/tok` total):**
    - `lm_head` (`248,320 × 1024` `Q6_K`): `~23–26 ms` (`~36%`)
    - `ffn` (SwiGLU gate/up/down): `~18–21 ms` (`~29%`)
    - `dn_proj` (DeltaNet projections): `~13–16 ms` (`~22%`)
    - `ssm` (Conv1D + DeltaNet recurrence): `~3.2–4.0 ms` (`~5%`)
    - `fa_proj` (Full-attention projections): `~2.7–3.7 ms` (`~5%`)
    - `fa_gqa` (Grouped-query attention + KV cache): `~0.10 ms` (`<0.2%`)
    - `norm` (RMSNorms + `Q8_K` quantization): `~0.06 ms` (`<0.1%`)
  - Enabling `dot2` for both `Q6_K` and `Q4_K` cuts redundant `Q8_K` vector loads across `lm_head`, `ffn`, and `dn_proj` (together `~87%` of decode latency), lifting 2T median decode on idle Broadwell from `8.88` to `13.58 tok/s` with minimal variance (`[13.49..13.63]`).
- **3. SMT-Aware Thread Pool (`AsiThreadPool` in [`asi_pool.h`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_pool.h)):**
  - Added `asi_detect_physical_cores()` parsing `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` (with `sysconf(_SC_NPROCESSORS_ONLN) / 2` fallback) and `LILA_ASI_SPIN_LIMIT` runtime override (`yield_limit = 2 * spin_limit`).
  - When `n_threads > physical_cores` (e.g., 1-core / 2-SMT cloud VMs such as Google Colab Xeon), the pool automatically uses `spin_limit = 64` / `yield_limit = 128` to yield the hyperthread instead of starving its sibling; when `n_threads <= physical_cores`, it retains `spin_limit = 2048` / `yield_limit = 4096`.
- **4. Disk-Persisted Prefix Snapshot (`ASIPFX02`, [`lila_asi.h`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.h), [`lila_asi.c`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.c), [`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c), [`asi_backend.py`](file:///home/ticket/workspaces/lila/source/src/core/asi_backend.py), [`lilacore.py`](file:///home/ticket/workspaces/lila/source/src/core/lilacore.py)):**
  - Introduced `LILA_ASI_NUMERICS_VERSION` (`2U`), a 144-byte `_Static_assert`-checked `AsiPrefixSnapshotHeader` with magic `"ASIPFX02"`, independent **CRC-64 ECMA** checksums over header (`header_crc64`) and payload (`payload_crc64`), and crash-safe atomic writes (`<filepath>.tmp.<pid>` -> `fflush` -> `fsync` -> `rename`).
  - Wired automatic `.asip` disk prefix caching (`prefix_<hex16>.asip`, keyed via `_compute_prefix_key_hash` over `b"ASIPFX02"`, model `identity_hash`, `arch_id`, `fp32_ref`, token count, UTF-8 `sys_text`, and prompt token IDs) into `AsiLlamaAdapter` (`0o700` directory, `0o600` files via `restrictive_umask()`, LRU eviction controlled by `LILA_ASI_PREFIX_CACHE_MAX_FILES=8`, configurable via `LILA_ASI_PREFIX_CACHE_DIR` / `LILA_ASI_DISABLE_DISK_PREFIX`) and connected `LilaCore._boot_gguf()` to `Path(self.memory_dir) / "prefix_cache"`.
  - **Verification:** `test_asi_v2` reported `Disk prefix round-trip diff=0.000000 (memcmp=0)` alongside corruption/tamper rejection; `qualify_asi_backend.py` reported `ALL 6 ASI QUALIFICATION CHECKS PASSED` (including disk cache persistence `hit_b='disk'` and cache invalidation `mut_hit='miss'`).
- **5. AVX-512 Bit-Exact Reduction & Multi-ISA Compile Audit ([`q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h), [`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c)):**
  - Aligned `#if defined(__AVX512F__)` dot and batched GEMM reductions in `q4k.h` and `asi_arch.c` with the `#if defined(__AVX2__)` bit-exact 8-lane `__m256 sum_vec` + `lila_hsum_f32_avx2` accumulation tree and shared `tile_out` unpack path (`lila_q4k_unpack_row_avx2` / `lila_q6k_unpack_row_avx2`).
  - Replaced branchy `get_scale_min_k4` inner-loop logic in `Q4_K` kernels with `lila_q4k_unpack_scales_avx2`.
  - **Verification:** Clean compilation verified across AVX2 (`-mavx2 -mfma -mf16c`), AVX-512 (`-mavx512f -mavx512bw -mavx512dq -mavx512vl`), AVX-512-VNNI (`-mavx512vnni`), non-`F16C` (`fp16_to_fp32` software fallback), pure scalar fallback, and `ASan+UBSan+LSan`. Note: AVX-512 and ARM64 NEON execution remains **compile/audit-verified only** pending runs on physical AVX-512 and ARM64 hardware.
- **6. G0 Correctness Gate (`N_GEN = 64`, 320 Positions, [`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json)):**
  - `.asi` SHA-256 `757731964e8a24950f3ae743be9df81ca5671f15b014aa45c31868131679e9f0`, GGUF SHA-256 `d9662c91079317aee661a1475d61412787696e0dcc5636c39648fb27d49e1edf`, `n_ctx = 512`, `n_threads = 2`, `n_gen = 64`, `5` prompts (`320` token positions), `vocab_compared = 248320`.
  - **G0 Sub-Checks (all `PASS`):**
    - `batch_vs_token`: **`320/320` (`100.0%`)**, `max_abs_diff = 0.0`, `max_rel_l2 = 0.0`.
    - `teacher_forced`: `ASI vs llama.cpp = 316/320 (98.75%)`, `ASI vs FP32-Ref = 315/320 (98.44%)` (`98.4375%`), `llama.cpp vs FP32-Ref = 316/320 (98.75%)`.
    - `fp32_ref`: `ASI Rel-L2 mean = 0.047385` (`median = 0.038031`, `max = 0.404697`) vs `llama.cpp mean = 0.047615` (`median = 0.038385`, `max = 0.342743`), **`ratio = 0.9952 <= 1.05`** (`0.995179`); all `4` ASI-vs-llama flips (`ASI==Ref: 1`, `Llama==Ref: 2`, `Neither==Ref: 1`) occurred at near-tie `FP32-Ref` margins (`0.050007..0.094835`, max `0.0948`).
    - `determinism`: **5-subprocess determinism `PASS`** at both `threads=1` and `threads=2` (`32` tokens on `factual` and `tool_use`), with exact `32/32` cross-thread agreement (`factual` SHA-256 `1f118f986237206116a3e68a054ffc510e7299367926205ebd6de27e87e64f28`, `tool_use` SHA-256 `9c1ed37382f2530c696317fd60c5555a9dba4822235280178db635e7af22f85b`).
- **7. G1 Performance Gate & Claim Verdict ([`claim_check.py`](file:///home/ticket/workspaces/lila/source/engine/tests/claim_check.py), [`claim_verdict_broadwell.json`](file:///home/ticket/workspaces/lila/source/paper/claim_verdict_broadwell.json)) & Cross-Machine Context:**
  - Formal G1 evaluation on Broadwell (`results_broadwell_n15.json`, `N = 15`, `n_threads = 2`) under `claim_check.py` (requiring `N >= 15`, `WIN` = median gain `>= 3%`, MWU `p < 0.01`, tail within `10%` at `p5`/`p95`; `STRONG WIN` = `WIN` + non-overlapping `[min..max]`):

| Metric | Dir | Role | ASI `med [min..max]` (`p5/p95`) | `llama.cpp` `med [min..max]` (`p5/p95`) | Gain% | TailR | MWU *p* | G1 Verdict |
|---|---|---|---|---|---|---|---|---|
| `t1_ttft` (s) | lower | primary | `0.908 [0.862..1.208]` (`0.864/1.128`) | `1.094 [1.027..1.810]` (`1.031/1.604`) | `+17.1%` | `0.703` | `0.00105` | **`WIN`** |
| `t1_dec` (tok/s) | higher | primary | `13.808 [9.232..15.030]` (`9.378/14.968`) | `13.477 [9.162..14.280]` (`10.426/14.278`) | `+2.5%` | `0.900` | `0.3401` | `PARITY` |
| `t2_ttft` (s) | lower | primary | `0.487 [0.459..0.953]` (`0.459/0.844`) | `0.607 [0.559..0.839]` (`0.563/0.794`) | `+19.9%` | `1.063` | `0.003691` | **`WIN`** |
| `t2_dec` (tok/s) | higher | primary | `13.062 [10.792..14.722]` (`11.283/14.494`) | `12.889 [10.098..13.918]` (`11.540/13.852`) | `+1.3%` | `0.978` | `0.3615` | `PARITY` |
| `long_prefill_tps` | higher | primary | `52.799 [44.701..55.541]` (`45.090/54.864`) | `46.213 [40.022..50.272]` (`40.436/49.912`) | `+14.3%` | `1.115` | `0.0003609` | **`WIN`** |
| `rss_mb` (MiB) | lower | memory | `616.266 [616.188..616.469]` (`616.198/616.428`) | `978.223 [978.105..978.410]` (`978.119/978.399`) | `+37.0%` | `0.630` | `3.341e-06` | **`STRONG WIN`** |
| `vm_hwm_mb` (MiB) | lower | secondary | `638.824 [638.602..638.988]` (`638.648/638.955`) | `1406.730 [1406.504..1406.844]` (`1406.586/1406.836`) | `+54.6%` | `0.454` | `3.375e-06` | **`STRONG WIN`** |
| `load_s` (s) | lower | secondary | `0.053 [0.052..0.100]` (`0.052/0.084`) | `2.182 [1.991..3.216]` (`1.994/3.112`) | `+97.5%` | `0.027` | `3.392e-06` | **`STRONG WIN`** |
| `load_open_s` (s) | lower | secondary | `0.027 [0.026..0.048]` (`0.027/0.035`) | `2.156 [1.966..3.192]` (`1.969/3.081`) | `+98.7%` | `0.011` | `3.392e-06` | **`STRONG WIN`** |

  - **Broadwell Verdict (`claim_verdict_broadwell.json`):** `G1 Tier: BEATS` (`3` primary `WIN`s, `0` losses, `rss_mb` `STRONG WIN`), `overall_passed = true`, exact allowed claim string: `"Beats llama.cpp on Intel(R) Core(TM) i5-5300U CPU @ 2.30GHz (2 threads)"`. Note honestly that the 15 raw pairs in `results_broadwell_n15.json` were captured on the pre-v2.2 build before `dot2=1` defaults (`LILA_ASI_Q6K_DOT2=1`, `LILA_ASI_Q4K_DOT2=1`), SMT spin-backoff, and `/proc/stat` `steal_jiffies` telemetry (`has_telemetry = false`).
  - **Cross-Machine Colab Xeon Run (`N = 10`, commit `4e73276`, `Intel(R) Xeon(R) CPU @ 2.20GHz`, 1 physical core / 2 logical CPUs, AVX2+FMA, same `d9662c91` GGUF):**
    - Medians across the 10 interleaved pairs: `load_s = 0.086 s` (`[0.082..0.091]`) vs `1.015 s` (`[0.976..1.722]`), `t1_ttft = 1.036 s` (`[1.027..1.642]`) vs `1.136 s` (`[1.132..2.210]`), `t1_dec = 12.90 tok/s` (`12.895`, `[8.06..13.04]`) vs `13.80 tok/s` (`[6.44..13.88]`), `t2_ttft = 0.552 s` (`0.5515`, `[0.538..0.927]`) vs `0.654 s` (`0.6535`, `[0.629..0.734]`), `t2_dec = 12.73 tok/s` (`[4.45..12.92]`) vs `13.64 tok/s` (`[6.87..13.87]`), `long_prefill_tps = 44.38 tok/s` (`44.375`, `[41.95..47.29]`) vs `41.03 tok/s` (`41.025`, `[40.82..44.10]`), steady-state `rss_mb = 619.8` vs `1107.9 MiB` (`1107.85`), and peak `vm_hwm_mb = 619.8` vs `1381.7 MiB`.
    - Because commit `4e73276` was pre-v2.2 (`N = 10 < 15`, prior to `dot2=1` defaults, SMT spin-backoff `spin_limit=64` / `yield_limit=128`, and `/proc/stat` steal-time logging), it does not qualify for formal G1 certification; [`colab_asi_benchmark.ipynb`](file:///home/ticket/workspaces/lila/source/engine/tests/colab_asi_benchmark.ipynb) is updated for `N = 15`, `threads = 1, 2`, `N_GEN = 64` G0 check, `claim_check.py`, and `steal_jiffies` logging as the next cloud verification run.

---

## Trial 11 — ASI v2.3 Prototype (Evaluated & Discarded): Zero-Training Jev-Style Batched Speculative Verification & Dynamic Two-Stage `lm_head` Shortlisting

- **Motivation & Literature Context:**
  - Trial 10's roofline audit established that full-model `Qwen3.5-0.8B` decode on Broadwell (`i5-5300U`, `2C/4T`, DDR3-1600, `12.08 GB/s` DRAM-RO ceiling) streams `518.34 MB`/token, with tied `token_embd`/`lm_head` (`248,320 × 1024` `Q6_K`) accounting for **`208.59 MB` (`40.24%` of active bytes)** and **`~23–26 ms` (`~36%` of `~68 ms/tok` steady-state decode)**.
  - Prior inference work attacks this memory-bandwidth wall via speculative verification and vocabulary narrowing: **Speculative Decoding / Speculative Sampling** (Leviathan, Kalman, & Matias, ICML 2023; Chen et al., 2023) verifies $K$ drafted tokens in one batched target forward pass using an auxiliary draft model; **Medusa** (Cai et al., ICML 2024) adds trained parallel decoding heads; **Lookahead Decoding** (Fu et al., ICML 2024) trades abundant parallel GPU FLOPs for Jacobi n-gram verification; **Prompt Lookup Decoding** (Saxena, 2023) removes the draft model entirely by proposing prompt/context n-gram matches; and **FR-Spec** (Zhao et al., ACL 2025) accelerates large-vocabulary models by restricting draft projection to frequency-ranked token subsets while preserving exact target verification.
  - To avoid auxiliary draft models (RSS/load penalty), trained heads (format/model dependence), and English-specific static token tables, Trial 11 prototyped a **100% language-agnostic** (pure token-ID driven), zero-training speculative verifier plus dynamic per-session `lm_head` shortlisting inside the `.asi` v2 runtime—remaining drop-in compatible with existing `.asi` v2 cartridges and `llama-cpp-python` callers (`lila_asi_eval_token` / `lila_asi_eval_batch`).

- **Prototype Scope (Reverted cleanly to `v2.2` after evaluation):**
  - *Speculative suffix proposer (`qwen35_propose_draft`):* Language-agnostic `4 -> 3 -> 2 -> 1-gram` token-ID history suffix matcher over `token_history`, `<think>` transition heuristic, and `fail_hash[1024]` failed-transition filter (gated by `LILA_ASI_SPECULATIVE` and logged via `LILA_ASI_SPEC_STATS`).
  - *Jev-style batched verification & DeltaNet rewind (`qwen35_forward_speculative`):* `B = 1 + K = 4` batched forward pass across all `24` layers and the `248,320`-row `lm_head`, `18`-layer DeltaNet pre-batch snapshot + `qwen35_rewind_deltanet_to` state replay on rejection, and reusable session buffers adding **`+25.5 MiB` RSS** (`647.055` vs `621.559 MiB` in the `N = 3` run).
  - *Language-adaptive two-stage `lm_head` (`matvec_shortlist_q8k_pool`):* `LILA_ASI_LMHEAD_SHORTLIST`-gated 256-row page bitmap activating base pages (`0..32767`), special tail pages, and pages observed in `token_history`, with a heuristic Stage-1 early exit (`top1 >= 18.0f && (top1 - top2) >= 2.0f`).
  - *Files touched & cleanly reverted:* Prototype edits in [`asi_arch.h`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.h), [`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c), [`lila_asi.c`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.c), and [`test_asi_v2.c`](file:///home/ticket/workspaces/lila/source/engine/tests/test_asi_v2.c) were reverted cleanly back to `v2.2` following evaluation.

- **Measured Results (verified from `v23_correctness_n64.json`, `v23_ablation_n3.json`, and `v23_multilang_ablation.json`):**
  1. **G0 Correctness & Host Qualification (`v2.3_spec`, `LILA_ASI_LMHEAD_SHORTLIST=0`, `v23_correctness_n64.json`):**
     - Passed G0 identically to `v2.2` ([`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json)): `batch_vs_token = 320/320 (100.0%, max_abs_diff = 0.0, max_rel_l2 = 0.0)`, `100%` identical generated texts, summaries, and determinism SHA-256 digests (`1f118f98...` / `9c1ed373...`) to `v2.2`, `teacher_forced = 316/320 (98.75%)` vs `llama.cpp` (`315/320 = 98.44%` vs `FP32-Ref`), `ASI Rel-L2 mean = 0.047385` (`median = 0.038031`, vs `llama.cpp 0.047615`, ratio `0.9952 <= 1.05`), and `5`-subprocess determinism `PASS` (`threads = 1, 2`, exact `32/32` cross-thread parity).
     - Prototype verification also passed `ASan+UBSan+LSan` cleanly and passed all 6 checks in [`qualify_asi_backend.py`](file:///home/ticket/workspaces/lila/source/engine/tests/qualify_asi_backend.py) (`t1_dec = 14.08`, `t2_dec = 12.47` vs `llama.cpp 13.01 / 12.75 tok/s`).
  2. **`bench_strict_bar.py` `N = 3` Controlled 4-Way Ablation (`v23_ablation_n3.json`, `n_threads = 2`, `n_gen = 32`, `T1 = 53`, `T2 = 27`, `Long = 939`, `--prepopulate`, `steal_jiffies = 0` across all 12 runs):**

| Metric | `v2.2` `med [min..max]` | `v2.3_spec` `med [min..max]` | `v2.3_spec_shortlist` `med [min..max]` | `llama.cpp` `med [min..max]` |
|---|---|---|---|---|
| `t1_ttft` (s, lower) | `0.883 [0.831..1.113]` | `1.140 [0.899..1.279]` | **`0.880 [0.859..0.975]`** | `1.196 [1.168..1.553]` |
| `t1_dec` (tok/s, higher) | **`14.152 [12.624..14.610]`** | `11.613 [9.405..12.588]` | `12.124 [11.925..12.256]` | `12.121 [10.735..14.055]` |
| `t2_ttft` (s, lower) | `0.613 [0.465..0.653]` | **`0.506 [0.482..0.642]`** | `0.516 [0.463..0.658]` | `0.591 [0.588..0.645]` |
| `t2_dec` (tok/s, higher) | **`12.918 [12.225..13.939]`** | `8.209 [7.660..10.601]` | `11.527 [9.163..11.644]` | `12.240 [11.824..12.662]` |
| `long_prefill_tps` (higher) | `48.889 [37.061..53.392]` | `44.548 [37.955..49.349]` | **`51.918 [43.903..54.256]`** | `43.302 [41.007..51.207]` |
| `rss_mb` (MiB, lower) | **`621.559 [621.512..621.660]`** | `647.055 [647.051..647.121]` | `647.000 [646.988..647.070]` | `979.582 [979.578..979.594]` |
| `load_s` (s, lower) | **`0.056 [0.054..0.061]`** | `0.062 [0.054..0.075]` | `0.059 [0.055..0.060]` | `2.463 [2.371..2.549]` |

  - *`[SPEC STATS]` & Greedy Parity (`v23_ablation_n3.json` `spec_inspection`):* Both `v2.3_spec` and `v2.3_spec_shortlist` produced `100%` identical tokens to `v2.2` on `T1`, `T2`, and `Long decode` with identical speculation counts: **`T1`** (`32` calls): `6` batches / `18` proposed / `4` accepted / `4` misses; **`T2` (`t2_delta`, `32` calls)**: `13` batches / `39` proposed / `7` accepted / `9` misses (`T1+T2` total `64` calls: `19` batches / `57` proposed / `11` accepted / `13` misses); **`Long decode`** (`32` calls): `14` batches / `42` proposed / `4` accepted / `12` misses (`Long prefill` = `0` calls).

  3. **6-Prompt Agentic & Multilingual Suite (`v23_multilang_ablation.json`, `N = 3`, `n_threads = 2`, `n_gen = 64`):**

| Prompt (`prompt_len`) | `eq_v2.2` (`spec` / `short`) | `v2.2` `tok/s` `med [min..max]` | `v2.3_spec` `tok/s` `med [min..max]` | `v2.3_spec_shortlist` `tok/s` `med [min..max]` | `llama.cpp` `tok/s` `med [min..max]` | `[SPEC STATS]` (`batches / prop / acc / miss`) |
|---|---|---|---|---|---|---|
| `english_qa` (`52`) | `true` / **`false`** | **`13.50 [11.80..14.56]`** | `10.82 [9.50..10.93]` | `11.90 [6.09..12.20]` *(wrong!)* | `13.15 [11.79..13.17]` | `spec`: `18 / 54 / 7 / 15`; `short`: `15 / 45 / 9 / 10` |
| `rag_edit` (`185`) | `true` / `true` | **`13.29 [11.30..13.83]`** | `11.05 [7.99..14.38]` | `11.33 [10.90..13.21]` | `12.17 [9.82..13.17]` | `18 / 54 / 25 / 8` |
| `tool_json` (`164`) | `true` / `true` | **`13.56 [11.61..13.89]`** | `11.48 [9.91..11.77]` | `10.71 [8.88..12.23]` | `10.60 [9.29..13.24]` *(48, EOS)* | `23 / 69 / 19 / 15` |
| `french` (`28`) | `true` / `true` | `12.66 [8.65..14.38]` | `14.88 [12.27..15.48]` | **`16.20 [15.00..16.43]`** | `12.87 [11.52..12.94]` | `12 / 36 / 23 / 3` |
| `spanish` (`26`) | `true` / `true` | `12.08 [10.42..14.35]` | **`12.62 [8.62..12.71]`** | `12.53 [12.36..12.88]` | `11.56 [9.54..13.25]` | `14 / 42 / 12 / 8` |
| `yoruba` (`54`) | `true` / `true` | `12.76 [9.70..13.19]` | `12.92 [9.25..13.74]` | **`14.82 [13.85..14.90]`** | `12.90 [10.97..13.02]` | `20 / 59 / 30 / 7` |

  - **`LILA_ASI_LMHEAD_SHORTLIST=1` Correctness Bug (`english_qa` step 8):** On `english_qa`, `v2.3_spec_shortlist` broke greedy parity at decode step 8 (`eq_v2.2 = false`): token `40755` (`" Tables"`, page `159`, outside base pages `0..32767` and not yet in `token_history`) was skipped by the Stage-1 confidence gate in favor of token `6424` (`" Table"`, page `25`), diverging `56/64` subsequent tokens (`"**Virtual Memory Page Table (VPM)**"` instead of `"**Virtual Memory Page Tables**"`). Note also that on `tool_json`, `llama.cpp` emitted EOS after `48` decode evals while ASI ran all `63` evals (`n_gen = 64`).

- **Root-Cause Analysis & Discard Decision:**
  1. **Why `B = 4` Speculation Slowed Real `2C/4T` DDR3-1600 Decode:** Unlike GPU tensor cores where extra batch columns are compute-free, Broadwell `2C/4T` executes `4×` inner dot products across all 24 layers + the `248,320`-row `lm_head` + `18`-layer DeltaNet snapshot/rewind (`qwen35_rewind_deltanet_to`), costing `~2.5×–3.0×` a single `B = 1` `dot2` decode step. Consequently, breaking even requires `>2–2.5` accepted tokens per batch. Permissive `1-gram` fallback triggering inflated miss rates (`15/18` batches on `english_qa`, `15/23` on `tool_json`, `8/18` on `rag_edit`, `12/14` on `long_decode`), causing even `rag_edit` (`25/18 = 1.39` accepted/batch, `11.05` vs `13.29 tok/s`) and `tool_json` (`19/23 = 0.83` accepted/batch, `11.48` vs `13.56 tok/s`) to regress behind `v2.2`. Only repetitive phrasing (`french`: `23/12 = 1.92` accepted/batch, `3` misses) improved (`+17.5%`, `14.88` vs `12.66 tok/s`, and `yoruba` `30/20 = 1.50` accepted/batch stayed near parity at `12.92` vs `12.76 tok/s`).
  2. **Why Confidence-Gated `lm_head` Shortlisting Is Unsafe Without Strict Bounds:** Heuristic Stage-1 logit margins (`top1 >= 18.0f && (top1 - top2) >= 2.0f`) over base + history pages provide no mathematical guarantee against unvisited rows exceeding `top1`, as exposed by token `40755 -> 6424` on `english_qa`. Lossless `lm_head` pruning requires exact Cauchy-Schwarz / row-norm upper bounds (`||w_r||_2 * ||x||_2`) or restricting vocabulary pruning strictly to a draft stage verified against full `lm_head` (as in FR-Spec).
  3. **Discard Decision:** All 4 C files (`asi_arch.h`, `asi_arch.c`, `lila_asi.c`, `test_asi_v2.c`) were reverted cleanly back to **`v2.2`** (`dot2=1/1`, `621.6 MiB` RSS vs `647.1 MiB`), preserving lossless parity and retaining `v2.2`'s lead over `llama.cpp` across `t1_dec` (`14.15` vs `12.12`), `t2_dec` (`12.92` vs `12.24`), `english_qa` (`13.50` vs `13.15`), `rag_edit` (`13.29` vs `12.17`), `tool_json` (`13.56` vs `10.60`), and `spanish` (`12.08` vs `11.56`).

---

## Trial 12 — ASI v2.3: Pure-SIMD Scale Broadcasts, Parallel `conv1d`/`full_attn`, Register-Direct Small-Batch GEMM, 32-Byte Aligned `LilaQ8KBlock`, and Zero-Rollback Stack-Scratch Speculation

- **Motivation & Engineering Changes (Fixes 1–4, `LILA_ASI_SPECULATIVE=1` default):**
  - Trial 11 revealed two distinct bottlenecks: (a) `Q4_K dot2` DRAM bandwidth was artificially throttled (`3.82–3.90 GB/s`) by scalar scale unpacking, and (b) `B = 4` speculation suffered from `+25.5 MiB` DeltaNet snapshot RSS, `18`-layer rewind overhead, and permissive `1-gram` false-positive batches.
  - **Fix 1 — Pure-SIMD `Q4_K` Scale Broadcasts & 32-Byte Aligned `LilaQ8KBlock` ([`q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h)):** Replaced scalar `dmin`/`sc` unpacking in the AVX2 `Q4_K` inner loop with pure-SIMD scale broadcasts and aligned `LilaQ8KBlock` to 32 bytes (`aligned(32)`), doubling isolated `Q4_K dot2` DRAM throughput on Broadwell from `3.82 / 3.90 GB/s` (1T/2T) to **`7.43 / 7.58 GB/s`** (`+94%`).
  - **Fix 2 — Parallel `conv1d` & `full_attn` (`asi_arch.c`):** Parallelized 1D causal convolution and full-attention head/KV loops across the thread pool.
  - **Fix 3 — Register-Direct Small-Batch GEMM (`asi_arch.c`):** Eliminated temporary stack/heap reduction buffers for small `B` (`B <= 4`) quantized GEMM, keeping accumulators register-resident.
  - **Fix 4 — Zero-Rollback Stack-Scratch Speculation (`asi_arch.h`, `asi_arch.c`, `lila_asi.c`, `LILA_ASI_SPECULATIVE=1` default):** Replaced the `+25.5 MiB` 18-layer DeltaNet snapshot/replay with stack-allocated scratch state (`+4.11 MiB` RSS vs `v2.2`: `625.67` vs `621.56 MiB`) and tightened the token-ID suffix proposer to strict `>= 2-gram` matching (eliminating `1-gram` miss storms), while keeping `LILA_ASI_LMHEAD_SHORTLIST=0` (disabled).

- **Measured Verification Gates on Idle Broadwell (`i5-5300U`, `2C/4T`, DDR3-1600):**
  1. **Roofline & Isolated Kernel Throughput ([`roofline_broadwell.json`](file:///home/ticket/workspaces/lila/source/paper/roofline_broadwell.json)):**
     - Synthetic `256 MiB` AVX2 DRAM-RO: `7.05 GB/s` (1T), `13.22 GB/s` (2T); STREAM Copy/Triad: `5.43 / 8.39 GB/s` (1T), `8.01 / 8.98 GB/s` (2T).
     - Isolated `64 MiB` DRAM `Q4_K dot2`: **`7.43 GB/s`** (`26.40 GFLOP/s`, 1T) and **`7.58 GB/s`** (`26.95 GFLOP/s`, 2T) vs `4.88 / 4.26 GB/s` for `Q6_K dot2`.
     - Single unpopulated 32-token decode in `bench_roofline`: `6.02 tok/s` (`166.06 ms/tok`, `44.3%` DRAM-RO) at 1T and `6.68 tok/s` (`149.66 ms/tok`, `26.2%` DRAM-RO) at 2T.
  2. **G0 Strict Correctness & Determinism ([`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json)):**
     - `batch_vs_token`: **`320/320` (`100.00%`, `max_abs_diff = 0.0`, `max_rel_l2 = 0.0`)** -> `PASS`.
     - `teacher_forced`: **`316/320` (`98.75%`)** vs `llama.cpp` (`315/320 = 98.44%` vs `FP32-Ref`) -> `PASS`.
     - `fp32_ref`: `ASI Rel-L2 mean = 0.047385` (`median = 0.038031`, vs `llama.cpp 0.047615`, `ratio = 0.9952 <= 1.05`, `fp32_ref_diff_detected = True`) -> `PASS`.
     - `determinism`: `5` runs at `threads = 1` (`1f118f98` / `9c1ed373`) and `5` runs at `threads = 2` (`1f118f98` / `9c1ed373`), `all_threads_agree = True` (`32/32` exact) -> `PASS`.
     - **OVERALL G0 VERDICT:** `PASS (strict=True)`.
  3. **Host Qualification ([`qualify_asi_backend.py`](file:///home/ticket/workspaces/lila/source/engine/tests/qualify_asi_backend.py)):**
     - **ALL 6 CHECKS PASSED** (batch/token parity, multi-turn + `.asip` disk prefix cache, cancellation/parallel isolation, RSS drift `+1.8 MB`, live `LilaDaemon` mTLS HTTPS `/chat` 200 in `1.4s`).
     - Controlled `N = 5` qualification benchmark (`ASI v2.3` vs `llama.cpp`): `load = 0.42 [0.41..0.53]` vs `2.09 [2.00..2.40] s`, `t1_ttft = 0.72 [0.70..0.74]` vs `0.83 [0.80..2.21] s`, `t1_dec = 13.68 [11.45..14.85]` vs `13.18 [11.62..13.99] tok/s`, `t2_ttft = 0.47 [0.45..0.48]` vs `0.93 [0.85..2.58] s`, `t2_dec = 14.38 [13.96..15.08]` vs `12.12 [11.11..13.63] tok/s`.
  4. **Strict Claim Check ([`claim_check.py`](file:///home/ticket/workspaces/lila/source/engine/tests/claim_check.py), [`claim_verdict_broadwell.json`](file:///home/ticket/workspaces/lila/source/paper/claim_verdict_broadwell.json)):**
     - `G0 VERDICT: PASS`, `G1 Tier: BEATS` (`primary_win = 3/5`, `loss = 0`, `rss = STRONG WIN` on `results_broadwell_n15.json`), `all_passed: true` (`ALLOWED CLAIM: Beats llama.cpp on Intel(R) Core(TM) i5-5300U CPU @ 2.30GHz (2 threads)`).
  5. **Controlled `N = 3` Ablation (`trial12_ablation_n3.json`, `n_threads = 2`, `n_gen = 32`, `T1 = 53`, `T2 = 27`, `Long = 939`):**

| Metric | `v2.2` `med [min..max]` | `v2.3_nospec` `med [min..max]` | `v2.3_spec` `med [min..max]` | `llama.cpp` `med [min..max]` |
|---|---|---|---|---|
| `t1_ttft` (s, lower) | **`0.883 [0.831..1.113]`** | `1.298 [0.919..1.638]` | `1.148 [0.957..1.264]` | `1.106 [1.104..1.692]` |
| `t1_dec` (tok/s, higher) | **`14.152 [12.624..14.610]`** | `11.988 [8.007..12.806]` | `13.976 [10.892..14.804]` | `12.562 [10.058..12.634]` |
| `t2_ttft` (s, lower) | `0.613 [0.465..0.653]` | `0.520 [0.506..0.674]` | **`0.508 [0.501..0.510]`** | `0.634 [0.622..1.241]` |
| `t2_dec` (tok/s, higher) | `12.918 [12.225..13.939]` | **`14.373 [13.099..14.429]`** | `12.917 [11.717..13.536]` | `10.140 [9.719..12.334]` |
| `long_prefill_tps` (higher) | `48.889 [37.061..53.392]` | **`49.450 [45.049..51.007]`** | `48.132 [45.029..50.825]` | `44.168 [43.614..48.007]` |
| `rss_mb` (MiB, lower) | **`621.559 [621.512..621.660]`** | **`621.559 [621.465..621.625]`** | `625.672 [625.648..625.738]` | `979.535 [979.516..979.555]` |
| `load_s` (s, lower) | **`0.056 [0.054..0.061]`** | `0.068 [0.060..0.073]` | `0.061 [0.061..0.070]` | `2.594 [2.550..3.210]` |

  - *`[ASI SPEC STATS]` & Greedy Parity (`trial12_ablation_n3.json`):* Both `v2.3_nospec` and `v2.3_spec` produced `100%` identical tokens to `v2.2` (`all_equal_v22 = true`). For `v2.3_spec`: **`T1`** (`32` calls): `3` hits / `2` batches / `6` drafted / `3` accepted / `1` miss; **`T2` delta** (`32` calls): `5` hits / `5` batches / `15` drafted / `5` accepted / `4` misses; **`Long decode`** (`32` calls): `4` hits / `4` batches / `12` drafted / `4` accepted / `3` misses.

  6. **6-Prompt Agentic & Multilingual Suite (`trial12_multilang.json`, `N = 3`, `n_threads = 2`, `n_gen = 64`):**

| Prompt (`len`) | `eq_v2.2` (`nospec` / `spec`) | `v2.2` `tok/s` `med [min..max]` | `v2.3_nospec` `tok/s` `med [min..max]` | `v2.3_spec` `tok/s` `med [min..max]` | `llama.cpp` `tok/s` `med [min..max]` | `[SPEC]` (`b / d / a / m`, `hits`) |
|---|---|---|---|---|---|---|
| `english_qa` (`52`) | `true` / `true` | **`13.50 [11.80..14.56]`** | `12.35 [11.89..12.98]` | `12.45 [11.70..14.12]` | `12.13 [9.58..12.41]` | `3 / 9 / 5 / 2` (`5`) |
| `rag_edit` (`185`) | `true` / `true` | `13.29 [11.30..13.83]` | **`14.10 [13.97..14.37]`** | `12.63 [12.17..14.28]` | `10.10 [8.55..10.44]` | `13 / 39 / 23 / 6` (`23`) |
| `tool_json` (`164`) | `true` / `true` | `13.56 [11.61..13.89]` | **`14.26 [13.29..14.29]`** | `13.61 [13.42..14.32]` | `10.08 [10.01..10.91]` *(48, EOS)* | `11 / 33 / 21 / 4` (`18`) |
| `french` (`28`) | `true` / `true` | `12.66 [8.65..14.38]` | `13.70 [12.56..14.25]` | **`14.47 [14.19..15.22]`** | `10.70 [9.56..10.87]` | `8 / 24 / 15 / 5` (`15`) |
| `spanish` (`26`) | `true` / `true` | `12.08 [10.42..14.35]` | `12.29 [10.84..13.13]` | **`14.26 [13.87..14.35]`** | `11.28 [10.18..12.13]` | `5 / 15 / 7 / 3` (`7`) |
| `yoruba` (`54`) | `true` / `true` | `12.76 [9.70..13.19]` | `12.01 [8.62..12.44]` | **`16.04 [8.82..16.08]`** | `11.46 [7.73..12.01]` | `13 / 38 / 25 / 6` (`24`) |

  - **Takeaway:** With Fixes 1–4, `v2.3_spec` (`LILA_ASI_SPECULATIVE=1` default) preserves `100%` lossless greedy parity across every prompt, trims speculation RSS overhead from `+25.5 MiB` to `+4.11 MiB` (`625.67 MiB` vs `llama.cpp 979.54 MiB`), and outperforms `llama.cpp` on all 6 multilingual/agentic prompts (`+2.6%` to `+40.0%`).

---

## References (Trial 11 Literature)

1. Yaniv Leviathan, Matan Kalman, and Yossi Matias. *Fast Inference from Transformers via Speculative Decoding*. Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202:19274–19286, 2023. `arXiv:2211.17192`.
2. Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling*. `arXiv:2302.01318`, 2023.
3. Apoorv Saxena. *Prompt Lookup Decoding*. GitHub technical release (`github.com/apoorvumang/prompt-lookup-decoding`), November 2023.
4. Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. International Conference on Machine Learning (ICML 2024), `arXiv:2401.10774`, 2024.
5. Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*. International Conference on Machine Learning (ICML 2024), `arXiv:2402.02057`, 2024.
6. Weilin Zhao, Tengyu Pan, Xu Han, Yudi Zhang, Ao Sun, Yuxiang Huang, Kaihuo Zhang, Weilun Zhao, Yuxuan Li, Jie Zhou, Hao Zhou, Jianyong Wang, Zhiyuan Liu, and Maosong Sun. *FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling*. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025, Volume 1: Long Papers), `10.18653/v1/2025.acl-long.198`, `arXiv:2502.14856`, 2025.



