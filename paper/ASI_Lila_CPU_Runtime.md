# ASI: A Portable, CPU-Optimized Inference Runtime for Quantized Language Models

**Lila-ASI Technical Report #1**

**Authors:** Sammie (`ticketguy`) and the Gemini 4 Hydrogen Engineering Agents  
**Target & Evaluation Hardware:** CPU-only Lenovo ThinkPad T450s (Intel Core i5-5300U, Broadwell, 2 cores / 4 threads, AVX2 / FMA / F16C, DDR3-1600)  
**Primary Implementation:** [`source/engine/asi/`](file:///home/ticket/workspaces/lila/source/engine/asi/) and [`source/engine/runtime/q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h)

---

## 1. Abstract

Local-first AI agents running on commodity laptops face strict memory, thermal, and startup-latency budgets without discrete GPUs. We present **ASI (Active System Image)**, a standalone C11/SIMD inference runtime and portable binary cartridge format (`.asi` v2) designed for CPU-only execution and integrated as an inference backend for the Lila agent host. ASI separates model distribution from runtime execution via a page-aligned, memory-mappable cartridge (`AsiV2Header`, `AsiV2SectionEntry`, `AsiV2TensorEntry`) and executes hybrid state-space/attention architectures—specifically Qwen3.5 Gated DeltaNet + Grouped-Query Attention (`ASI_ARCH_HYBRID_QWEN35`)—using persistent scratch arenas, a spin-then-sleep thread pool (`AsiThreadPool`), integer-SIMD quantized dot products (`Q4_K × Q8_K` and `Q6_K × Q8_K`), and in-session system-prefix state snapshots.

On the tested configuration (Intel Core i5-5300U, 2 threads, `Qwen3.5-0.8B` Q4_K_M), we establish a teacher-forced and exact-FP32-reference correctness gate (`test_correctness_gate.py --n-gen 64 --teacher-forced`) alongside an interleaved (`N=15`) apples-to-apples benchmark harness (`bench_strict_bar.py`) with mandatory full 4 KiB page-population on both engines. In teacher-forced evaluation across `320` token positions (`5` prompts × `64` tokens), `ASI(Q8_K)` achieves **`320/320` (`100.00%`)** batch-vs-token parity (`0.000000` relative L2), **`316/320` (`98.75%`)** top-1 agreement with `llama.cpp`, and lower mean/median logit relative L2 error against an unquantized `FP32-Ref` baseline than `llama.cpp` (`0.047385 / 0.038031` vs `0.047615 / 0.038385`), with all `4/320` flips occurring at near-tie `FP32-Ref` margins (`<= 0.0948`). In `N=15` interleaved prepopulated trials (`results_broadwell_n15.json`), bounded spin+yield pooling and 1× row-unpack GEMM tiling give ASI statistically significant median advantages on 939-token prefill (`52.799` vs `46.213 tok/s`, MWU `p=0.0003609`), Turn-1 TTFT (`0.908` vs `1.094 s`, `p=0.00105`), Turn-2 TTFT (`0.487` vs `0.607 s`, `p=0.003691`), steady-state RSS (`616.3` vs `978.2 MiB`, `p=3.341e-6`, non-overlapping), peak RSS (`638.8` vs `1406.7 MiB`), and prepopulated load (`0.053` vs `2.182 s`), while Turn-1 decode (`13.808` vs `13.477 tok/s`, `p=0.3401`) and Turn-2 decode (`13.062` vs `12.889 tok/s`, `p=0.3615`) sit at statistical parity. Because speed ranges overlap on Turn-1/Turn-2 TTFT and prefill and decode sits at parity, the strict zero-overlap primary bar across all 6 metrics is not yet cleared; all claims remain scoped to the verified Broadwell AVX2/FMA/F16C target, while wide-SIMD and ARM64 paths are compile-checked only.

---

## 2. Introduction & Motivation

Deploying an interactive local assistant such as Lila on an older, CPU-only laptop (2 cores / 4 threads, DDR3-1600, no GPU) exposes three bottlenecks that general-purpose serving frameworks rarely optimize together:

1. **Cold-start and memory-footprint overhead:** Parsing unstructured metadata, eagerly touching or copying weight tensors, and allocating per-call scratch buffers inflate resident set size (RSS) and delay time-to-first-token (TTFT).
2. **Turn-to-turn system-prompt re-evaluation:** Agentic workloads prepend long system prompts and tool schemas (often 800+ tokens) on every turn. Recomputing or re-materializing prefix states wastes scarce Broadwell compute.
3. **Host-engine coupling:** Embedding model-specific tensor layouts, tokenizer logic, and runtime dispatch directly inside the host application complicates deployment and makes cross-engine parity testing difficult.

ASI addresses these constraints by introducing:
- A self-describing, 4 KiB page-aligned `.asi` v2 cartridge format ([`asi_v2.h`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.h)) that can be directly `mmap`-ed on POSIX (`MAP_PRIVATE`) and Windows (`CreateFileMappingA` / `MapViewOfFile`).
- A minimal C ABI ([`lila_asi.h`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.h)) exposing model lifecycle, session management, prefix snapshot/restore, single-token decode, batched prefill, and sampling.
- Specialized CPU quantization kernels ([`q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h)) and a hybrid Qwen3.5 runtime ([`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c)) tuned for small-core-count AVX2/FMA/F16C CPUs.

---

## 3. The `.asi` v2 Cartridge Format

The `.asi` v2 specification is defined in [`source/engine/asi/asi_v2.h`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.h) and implemented in [`asi_v2.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.c), [`asi_v2.py`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.py), and [`gguf_to_asi_v2.py`](file:///home/ticket/workspaces/lila/source/engine/asi/gguf_to_asi_v2.py). All on-disk structures are little-endian and enforced at compile time with `_Static_assert`.

### 3.1 File Header & Section Directory

- **Header (`AsiV2Header`, 64 bytes):**
  - `magic = 0x41534921u` (`"ASI!"` little-endian), `version = 2u`.
  - `arch_id`: `AsiV2ArchId` enum (`1 = dense_llama`, `2 = hybrid_qwen35`, `3 = dense_gemma`, `4 = dense_qwen2`, `5 = dense_qwen3`).
  - `flags`: capability bitmask (`ASI_V2_FLAG_HAS_VISION`, `HAS_AUDIO`, `HAS_ADAPTERS`, `HAS_SIGNATURE`, `HAS_CHAT_TEMPLATE`).
  - `n_sections`, `total_size`, `section_table_offset`, and a 24-byte `identity_hash` (first 24 bytes of SHA-256 over section payloads in `write_asi_v2`).
- **Section Table (`AsiV2SectionEntry`, 32 bytes per entry):**
  - Each section (`ASI_SEC_MODEL_CONFIG = 0x01`, `ASI_SEC_WEIGHTS = 0x02`, `ASI_SEC_TOKENIZER = 0x04`, `ASI_SEC_METADATA = 0x08`, `ASI_SEC_CHAT_TEMPLATE = 0x09`, `ASI_SEC_VISION = 0x0A`, etc.) has a 4 KiB page-aligned `offset` (`ASI_V2_PAGE_SIZE = 4096u`), `size`, and 64-bit `checksum`.
  - Legacy sections (`ASI_SEC_BYTECODE_LEGACY = 0x05`, `ASI_SEC_HARNESS_LEGACY = 0x06`) are ignored in the v2 load path; the live Lila host owns tools and permissions.

### 3.2 Integrity Verification & Lazy `mmap`

Section checksums use **CRC-64 ECMA** (`0x42F0E1EBA9EA3693ULL` polynomial in `asi_v2_crc64`). During `asi_v2_validate_memory`:
- Small metadata/config/tokenizer sections are always CRC-64 verified on open.
- Bulk binary blobs (`ASI_SEC_WEIGHTS`, `ASI_SEC_VISION`, `ASI_SEC_AUDIO`) skip eager CRC-64 scanning by default unless `LILA_ASI_VERIFY_CRC=1` is set in the environment. This preserves lazy page-faulting under `mmap` so opening a cartridge does not force hundreds of megabytes of cold disk reads before inference begins.

### 3.3 Model Config & 64-Byte Aligned Weights Directory

- **`AsiV2ModelConfig` (96 bytes):** Stores explicit architecture hyperparameters (`n_layers`, `hidden_size`, `intermediate_size`, `n_heads`, `n_kv_heads`, `head_dim`, `vocab_size`, `max_seq_len`, `rope_dim`, `rope_theta`, `rms_norm_eps`, `embed_scale`), plus hybrid SSM fields (`full_attention_interval`, `ssm_conv_kernel`, `ssm_state_size`, `ssm_group_count`, `ssm_inner_size`, `ssm_time_step_rank`).
- **`ASI_SEC_WEIGHTS`:** Begins with a 16-byte `AsiV2WeightsHeader` (`n_tensors`, `flags`, `data_offset`), followed by `n_tensors` × 80-byte `AsiV2TensorEntry` records (`name[40]`, `quant_type`, `layout_flags`, `shape[2]`, `data_offset`, `data_size`).
- **64-byte Alignment:** In `pack_weights_section`, `data_offset` and every individual tensor payload length are rounded up to 64-byte boundaries (`(n + 63) & ~63`) so SIMD loads from mapped weight pointers never cross unaligned boundaries at tensor starts.
- **Optional AVX2 Q6_K Repacking (`ASI_TENSOR_LAYOUT_AVX2_REPACK`):** `gguf_to_asi_v2.py` supports calling `asi_v2_repack_q6k` (`libasi_v2.so`) to pre-unpack 210-byte GGML `Q6_K` blocks into 288-byte `LilaQ6KRepackedBlock` structs (`u6[256]`, `scales[16]`, FP32 `d`, `pad[12]`).

---

## 4. Runtime Architecture

The runtime is structured into three layers:

1. **Public C ABI ([`lila_asi.c`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.c), [`lila_asi.h`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.h)):**
   - `lila_asi_open` / `lila_asi_close`, `lila_asi_model_info`, `lila_asi_get_section` (zero-copy pointer into `SEC_TOKENIZER` / `SEC_CHAT_TEMPLATE` for the Python adapter [`asi_backend.py`](file:///home/ticket/workspaces/lila/source/src/core/asi_backend.py)).
   - `lila_asi_session_create` / `free` / `reset`, `lila_asi_eval_token`, `lila_asi_eval_batch`, `lila_asi_sample`, and `lila_asi_session_save_prefix` / `restore_prefix`.
2. **Thread Pool ([`asi_pool.h`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_pool.h)):**
   - `AsiThreadPool` avoids OpenMP barrier overhead on tight per-layer dispatches. Hot atomics (`fn`, `gen`, `done_count`, `active`, `sleeping`, `shutdown`) are aligned on 64-byte cache-line boundaries (`_Alignas(64)`) to prevent cross-core false sharing.
   - To eliminate OS-contention tail-latency spikes on 2-core CPUs, workers use a bounded spin (`asi_cpu_pause()` for `2048` iterations, then `sched_yield()` up to `4096`), before sleeping on `pthread_cond_wait(&p->cv_work, &p->mu)`; `asi_pool_run` likewise yields after `2048` barrier spins and wakes any sleeping workers cleanly.
   - Multi-threaded decode matvec (`asi_matvec_worker` / `asi_matvec_range`) uses `dot2` kernels (`q4k_dot2_q8k_avx2`, `q6k_dot2_q8k_avx2`) with atomic dynamic chunking (`asi_prefetch_row`), while batch GEMM (`asi_matmul_batch_worker`, `asi_fused_matmul_batch_worker`) and attention/DeltaNet workers (`asi_deltanet_worker`, `asi_full_attn_batch_worker`) use atomic `next_row`/`next_h` work-stealing.
3. **Hybrid Qwen3.5 Execution ([`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c)):**
   - Dispatches via `AsiArchHandler` (`g_asi_arch_hybrid_qwen35`).
   - **Layer Heterogeneity:** Each layer is classified as `QWEN35_LAYER_DELTANET` (1D causal convolution via AVX2 `_MM_TRANSPOSE4_PS` + `lila_expf_avx2` in `asi_conv1d_worker` + tiled recurrent DeltaNet state update) or `QWEN35_LAYER_FULL_ATTN` (Gated Q/K/V attention with Q/K RMSNorm, partial RoPE, grouped GQA KV reuse in `asi_kv_prep_batch_worker`/`asi_full_attn_batch_worker`, and FP16 `k_cache`/`v_cache`).
   - **Chunked Prefill, Parallel Norms, & 1× Weight-Unpack GEMM:** `qwen35_forward_batch` processes prompts in chunks of up to `chunk_max = 512` tokens (`qwen35_forward_chunk`), using parallel `asi_norm_quant_pool` + persistent 64-byte aligned scratch (`ensure_batch_scratch`), while `q4k_matmul_batch_q8k_range` and `q6k_matmul_batch_q8k_range` unpack each weight row (`lila_q4k_unpack_row_avx2` / `lila_q6k_unpack_row_avx2`) **once** per token chunk into L1d and accumulate into a contiguous `tile_out` transpose buffer.
   - **System-Prefix Snapshot Caching & `LILA_ASI_FP32_REF=1`:** `qwen35_save_prefix` / `qwen35_restore_prefix` snapshot DeltaNet + KV state across turns, and setting `LILA_ASI_FP32_REF=1` bypasses `Q8_K` activation quantization to evaluate exact FP32-dequantized weight dot products (`lila_quant_dot_fp32` / `lila_quant_matmul_batch_fp32_range`) for numerical reference auditing.

---

## 5. Quantization & SIMD Kernel Design

All quantized dot products and batched GEMMs live in [`source/engine/runtime/q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h).

### 5.1 Activation Quantization (`LilaQ8KBlock`)

Input FP32 activation vectors are quantized on the fly into 256-element `LilaQ8KBlock` structures (`float d`, `int8_t qs[256]`, `int16_t bsums[16]`). For batched prefill (`n_tokens > 1`), `quantize_vec_q8k_strided` writes blocks in **block-major, token-strided** layout (`qx_all + (size_t)b * n_tokens + tok`) so that weight block `b` can be loaded once and multiplied against 4 consecutive tokens (`t..t+3`) in the inner loop.

### 5.2 Hardware-Verified vs. Compile-Checked Paths

| Kernel / ISA Path | Target Instructions | Status |
|---|---|---|
| **AVX2 + FMA + F16C** (`__AVX2__`) | `_mm256_maddubs_epi16`, `_mm256_madd_epi16`, `_mm256_fmadd_ps`, `_cvtsh_ss` | **Hardware-verified** on Intel Core i5-5300U (Broadwell) |
| **AVX-512F / BW** (`__AVX512F__ && __AVX512BW__`) | 512-bit `q4k_dot_q8k_simd`, `q6k_matmul_batch_q8k_range`, `quantize_vec_q8k_strided` | **Compile-checked / unverified on target hardware** (guarded by `#if`; not executed on Broadwell) |
| **AVX-512 VNNI / AVX-VNNI** (`__AVX512VNNI__`, `__AVXVNNI__`) | `_mm512_dpbusd_epi32`, `_mm256_dpbusd_epi32` | **Compile-checked / unverified on target hardware** |
| **ARM64 NEON / DotProd** (`__aarch64__`, `__ARM_FEATURE_DOTPROD`) | `vdotq_s32` / `vmull_s8` | **Compile-checked / unverified on target hardware** |

> [!NOTE]
> Only the AVX2 + FMA + F16C code path has been executed and verified against scalar references on physical hardware in this study. Wide-SIMD (AVX-512, VNNI) and ARM64 NEON paths are present in `q4k.h` for portability but require hardware validation on their respective CPUs.

---

## 6. Correctness & Equivalence Gate

Quantized integer-SIMD inference cannot be validated solely by unit-testing isolated matrix multiplies; small floating-point differences propagate across 24+ layers and can be amplified by per-block `Q8_K` quantization boundaries (`amax / 127.0f`).

### 6.1 Kernel-Level Parity (`test_quant_kernels.c`)

[`test_quant_kernels.c`](file:///home/ticket/workspaces/lila/source/engine/tests/test_quant_kernels.c) verifies `q4k_matvec` and `q6k_matvec` against scalar FP32 dequantization (`dequant_q4k_block`, `dequant_q6k_block`) on 128×512 random blocks with clean FP16 scales, requiring relative L2 error `< 0.01`.

### 6.2 End-to-End Parity Gate (`test_correctness_gate.py`)

[`test_correctness_gate.py`](file:///home/ticket/workspaces/lila/source/engine/tests/test_correctness_gate.py) evaluates `Qwen3.5-0.8B` (`n_ctx=512`, `n_threads=2`) across a 5-prompt corpus (`factual`, `math`, `code`, `multiturn`, `tool_use`), generating `N=32` greedy tokens (`temperature=0.0`, 160 total tokens) on **identical integer token IDs** across three paths:
1. `ASI(token)`: `lila_asi_eval_token` for every prompt and decode token.
2. `ASI(batch)`: `lila_asi_eval_batch` for prompt prefill, followed by `lila_asi_eval_token` for decode.
3. `llama.cpp`: `llama-cpp-python` (`Llama.eval` + `np.argmax`).

The gate reports prefill logit relative L2 error (`ASI(batch)` vs `ASI(token)`), first divergence index, and corpus-wide greedy top-1 token agreement.

---

## 7. Experimental Methodology & Reproducibility

To eliminate measurement artifacts common in local LLM benchmarks (tokenization mismatches, unverified KV reuse, cold-cache bias, and thermal drift), [`bench_strict_bar.py`](file:///home/ticket/workspaces/lila/source/engine/tests/bench_strict_bar.py) enforces:

1. **Byte-Identical Artifact & Single Tokenization:** Both local and remote runs share the exact same canonical GGUF (`FadedRedStar/Qwen3.5-0.8B-heretic-imatrix-GGUF` / `Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf`, GGUF SHA-256 `d9662c91079317aee661a1475d61412787696e0dcc5636c39648fb27d49e1edf`, `.asi` SHA-256 `757731964e8a24950f3ae743be9df81ca5671f15b014aa45c31868131679e9f0`), and prompts (`T1_PROMPT`, `T2_DELTA`, and `LONG_PROMPT` at ~860 tokens) are tokenized once and passed as identical `int32` token-ID arrays to both `liblila_asi.so` and `llama-cpp-python`.
2. **Verified KV/Prefix Reuse:** Turn 2 (`T2_DELTA`) asserts `t2_eval_tokens == len(toks["t2"])` on every trial for both engines.
3. **Matched Runtime Settings:** `n_ctx=2048`, `n_threads=2`, `n_threads_batch=2`, `n_batch=512`, `use_mmap=True`, `flash_attn=False`, greedy (`temperature=0.0`).
4. **Page-Cache Warmup, Mandatory Pre-Population & Subprocess Isolation:** Each engine runs 1 discarded warmup trial in a fresh subprocess, followed by `N >= 15` interleaved `(ASI, llama.cpp)` subprocess pairs. By default (`--prepopulate`, disableable via `--no-prepopulate`), every trial calls `madvise(MADV_WILLNEED)` and touches one byte per 4 KiB page of the model file immediately after opening so `load_s` (`load_open_s + populate_s`) measures full resident-weight startup honestly without pushing cold page faults into `t1_ttft`.
5. **Statistical Rigor & Tail Visibility:** Reports median, `[min..max]` spread, `p95`, `max`, 95% bootstrap confidence interval (2,000 resamples), and two-sided Mann-Whitney U p-value. Statistically tied rows (`p >= 0.05`) are explicitly labeled `PARITY`, advantages with overlapping ranges as `ADVANTAGE (overlap)`, and the **Strict Bar** requires `p < 0.05` plus non-overlapping `[min..max]` intervals in ASI's favor across all 6 primary metrics (`t1_ttft`, `t1_dec`, `t2_ttft`, `t2_dec`, `long_prefill_tps`, `rss_mb`).

---

## 8. Results & Ablations

### 8.1 Qualification & Correctness Status (`N=64` + Teacher-Forced `FP32-Ref` Audit)

- **Cartridge & Integrity (`test_asi_v2.c`):** Verified `asi_v2_open`, tensor lookup, `ASI_SEC_CHAT_TEMPLATE` presence, and CRC-64 corruption rejection.
- **Lila Integration (`qualify_asi_backend.py`):** Covers 6 end-to-end checks: batch-vs-token parity, multi-turn & tool-call generation, cancellation & concurrent session isolation, prefix-cache invalidation & 5-cycle RSS drift (`< 8.0 MB`), live `LilaDaemon` mTLS HTTPS `/chat` with `LILA_BACKEND=asi`, and controlled medians.
- **Free-Running Correctness Gate (`test_correctness_gate.py --n-gen 64`, [`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json)):**
  - `ASI(batch)` vs `ASI(token)`: **`0.000000` relative L2 error** and **`320/320` (`100.00%`)** exact token agreement across all 5 prompts (`factual`, `math`, `code`, `multiturn`, `tool_use`).
  - Free-running greedy vs `llama.cpp` over `64` tokens/prompt (`320` total): `tool_use` **`100.0%`**, `multiturn` **`95.3%`** (first divergence token 34), `math` **`87.5%`** (token 23), `code` **`46.9%`** (token 29), `factual` **`42.2%`** (token 24), for **`238/320` (`74.38%`)** overall (`80/80` = `100.00%` at `N=16`).
- **Teacher-Forced + Unquantized `FP32-Ref` Audit (`5` prompts × `64` = `320` positions):**
  - Teacher-forced top-1 agreement: **`ASI(Q8_K) vs llama.cpp = 316/320 (98.75%)`**, **`ASI(Q8_K) vs FP32-Ref = 315/320 (98.44%)`**, `llama.cpp vs FP32-Ref = 316/320 (98.75%)`.
  - Logit relative L2 vs `FP32-Ref`: `ASI(Q8_K)` **`mean=0.047385, median=0.038031`** (`max=0.404697`) vs `llama.cpp` `mean=0.047615, median=0.038385` (`max=0.342743`).
  - **Flip margin diagnostics (`4/320` flips; `ASI==FP32: 1`, `Llama==FP32: 2`, `Neither==FP32: 1`):** Every flip occurred at a tiny near-tie `FP32-Ref` top-1/top-2 margin (`min=0.0500`, `median=0.0715`, `max=0.0948`), confirming that greedy branching reflects near-tie `Q8_K` quantization noise rather than an implementation error.

### 8.2 Strict Apples-to-Apples Benchmark (`N=15`, Prepopulated, i5-5300U, 2 Threads, [`results_broadwell_n15.json`](file:///home/ticket/workspaces/lila/source/paper/results_broadwell_n15.json))

| Metric | Unit | Dir | ASI `med [min..max]` (`95% CI`) | ASI `p95 / max` | `llama.cpp` `med [min..max]` (`95% CI`) | `llama` `p95 / max` | MWU *p* | Verdict |
|---|---|---|---|---|---|---|---|---|
| `t1_ttft` | s | lower | `0.908 [0.862..1.208]` (`0.877..1.061`) | `1.128 / 1.208` | `1.094 [1.027..1.810]` (`1.061..1.292`) | `1.604 / 1.810` | `0.00105` | `ADVANTAGE (overlap)` |
| `t1_dec` | tok/s | higher | `13.808 [9.232..15.030]` (`13.047..14.903`) | `14.968 / 15.030` | `13.477 [9.162..14.280]` (`12.777..13.969`) | `14.278 / 14.280` | `0.3401` | `PARITY (p=0.3401)` |
| `t2_ttft` | s | lower | `0.487 [0.459..0.953]` (`0.466..0.509`) | `0.844 / 0.953` | `0.607 [0.559..0.839]` (`0.596..0.663`) | `0.794 / 0.839` | `0.003691` | `ADVANTAGE (overlap)` |
| `t2_dec` | tok/s | higher | `13.062 [10.792..14.722]` (`12.568..13.710`) | `14.494 / 14.722` | `12.889 [10.098..13.918]` (`12.625..13.614`) | `13.852 / 13.918` | `0.3615` | `PARITY (p=0.3615)` |
| `long_prefill_tps` | tok/s | higher | `52.799 [44.701..55.541]` (`50.187..53.837`) | `54.864 / 55.541` | `46.213 [40.022..50.272]` (`42.966..48.855`) | `49.912 / 50.272` | `0.0003609` | `ADVANTAGE (overlap)` |
| `rss_mb` | MiB | lower | `616.266 [616.188..616.469]` (`616.246..616.309`) | `616.428 / 616.469` | `978.223 [978.105..978.410]` (`978.145..978.344`) | `978.399 / 978.410` | `3.341e-06` | `PASS` |
| `vm_hwm_mb` (sec) | MiB | lower | `638.824 [638.602..638.988]` (`638.707..638.887`) | `638.955 / 638.988` | `1406.730 [1406.504..1406.844]` (`1406.645..1406.789`) | `1406.836 / 1406.844` | `3.375e-06` | `PASS (sec)` |
| `load_s` (sec) | s | lower | `0.053 [0.052..0.100]` (`0.053..0.058`) | `0.084 / 0.100` | `2.182 [1.991..3.216]` (`2.021..2.315`) | `3.112 / 3.216` | `3.392e-06` | `PASS (sec)` |
| `load_open_s` (sec) | s | lower | `0.027 [0.026..0.048]` (`0.027..0.028`) | `0.035 / 0.048` | `2.156 [1.966..3.192]` (`1.996..2.290`) | `3.081 / 3.192` | `3.392e-06` | `PASS (sec)` |

---

## 9. Honest Limitations & Future Work

1. **Compute-bound Broadwell 2C/4T regime:** On a 15 W dual-core Broadwell CPU (i5-5300U, DDR3-1600), prefill and 248k-row `lm_head` projections are sensitive to integer-SIMD throughput and L1/L2 cache residency. Findings on this chip do not automatically generalize to high-core-count servers or memory-bandwidth-saturated GPUs.
2. **Lazy `mmap` vs. Turn-1 TTFT:** Skipping `LILA_ASI_VERIFY_CRC=1` makes `lila_asi_open` nearly instantaneous, but cold page faults shift I/O latency into Turn-1 prefill/decode unless pages are warmed (`--prepopulate`).
3. **ISA Coverage Tiers (`AVX2` vs. Wide-SIMD vs. `ARM64 NEON`):** Standard Google Colab CPU runtimes (`Intel Xeon @ 2.20GHz`) expose **AVX2 / FMA / F16C only** (no AVX-512), so [`colab_asi_benchmark.ipynb`](file:///home/ticket/workspaces/lila/source/engine/tests/colab_asi_benchmark.ipynb) validates independent second-machine AVX2 reproducibility on the canonical imatrix GGUF (`d9662c91...`), not wide-SIMD code paths. The AVX-512F/BW, AVX-512VNNI, AVX-VNNI, and ARM64 NEON/DotProd branches in `q4k.h` remain compile-checked only and require dedicated hardware runs—e.g., an AVX-512/VNNI cloud VM (Intel Sapphire Rapids or AMD Zen 4 EPYC) for wide-SIMD x86, and an Oracle Cloud Free Ampere A1 VM, Raspberry Pi, or Apple Silicon Mac for `ARM64 NEON`.
4. **Architecture Coverage:** In `asi_arch.c`, `g_asi_arch_hybrid_qwen35` is fully implemented, whereas `g_asi_arch_dense_transformer` (for `dense_llama`, `dense_gemma`, `dense_qwen2`, `dense_qwen3`) currently registers `NULL` handler pointers and remains to be implemented.
