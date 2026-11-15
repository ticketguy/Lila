# ASI: A Portable, CPU-Optimized Inference Runtime for Quantized Language Models

**Lila-ASI Technical Report #1**

**Authors:** Sammie (`ticketguy`) and the Gemini 4 Hydrogen Engineering Agents  
**Target & Evaluation Hardware:** CPU-only Lenovo ThinkPad T450s (Intel Core i5-5300U, Broadwell, 2 cores / 4 threads, AVX2 / FMA / F16C, DDR3-1600)  
**Primary Implementation:** [`source/engine/asi/`](file:///home/ticket/workspaces/lila/source/engine/asi/) and [`source/engine/runtime/q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h)

---

## 1. Abstract

Local-first AI agents running on commodity laptops face strict memory, thermal, and startup-latency budgets without discrete GPUs. We present **ASI (Active System Image)**, a standalone C11/SIMD inference runtime and portable binary cartridge format (`.asi` v2) designed for CPU-only execution and integrated as an inference backend for the Lila agent host. ASI separates model distribution from runtime execution via a page-aligned, memory-mappable cartridge (`AsiV2Header`, `AsiV2SectionEntry`, `AsiV2TensorEntry`) and executes hybrid state-space/attention architectures—specifically Qwen3.5 Gated DeltaNet + Grouped-Query Attention (`ASI_ARCH_HYBRID_QWEN35`)—using persistent scratch arenas, a spin-then-sleep thread pool (`AsiThreadPool`), integer-SIMD quantized dot products (`Q4_K × Q8_K` and `Q6_K × Q8_K`), and in-session system-prefix state snapshots.

On the tested configuration (Intel Core i5-5300U, 2 threads, `Qwen3.5-0.8B` Q4_K_M), we establish a strict correctness gate comparing batched prefill (`lila_asi_eval_batch`) against token-by-token evaluation (`lila_asi_eval_token`) and `llama-cpp-python` on identical token IDs, and define a reproducible, interleaved (`N >= 15`) apples-to-apples benchmark harness (`bench_strict_bar.py`). Across `N=15` interleaved apples-to-apples trials on the i5-5300U (`results_broadwell_n15.json`), ASI reaches statistical parity with `llama.cpp` on Turn-1 decode (`14.064` vs `13.943 tok/s`, MWU `p=0.9339`), Turn-2 TTFT (`0.554` vs `0.561 s`, `p=0.901`), and Turn-2 decode (`13.812` vs `13.952 tok/s`, `p=0.9339`), while reducing steady-state RSS by 37% (`615.8` vs `978.0 MiB`, `p=3.35e-6`), peak RSS (`VmHWM`) by 44% (`615.8` vs `1101.2 MiB`), and cold load time (`0.026` vs `2.087 s`). On Turn-1 TTFT (`1.055` vs `1.002 s`) and 939-token prefill (`45.055` vs `53.747 tok/s`, up 8.2x from our `5.48 tok/s` baseline), `llama.cpp` retains an edge. We scope all claims strictly to the verified Broadwell AVX2/FMA/F16C target and explicitly distinguish hardware-verified kernels from compile-checked wide-SIMD and ARM64 paths.

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
   - `AsiThreadPool` avoids OpenMP barrier overhead on tight per-layer dispatches. Workers sleep on `pthread_cond_wait` when idle; `asi_pool_begin` sets `active = 1` across a prefill chunk so workers spin briefly using `_mm_pause()` (`asi_cpu_pause()`) between layer projections, and `asi_pool_end` returns them to sleep.
   - Single-row-pair matvec (`n_threads <= 1`) uses 2-row `dot2` kernels (`q4k_dot2_q8k_avx2`, `q6k_dot2_q8k_avx2`), while multi-threaded matvec uses atomic dynamic chunking (`chunk_rows = 512` for `rows >= 16384`, `64` otherwise).
3. **Hybrid Qwen3.5 Execution ([`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c)):**
   - Dispatches via `AsiArchHandler` (`g_asi_arch_hybrid_qwen35`).
   - **Layer Heterogeneity:** Each layer is classified as `QWEN35_LAYER_DELTANET` (1D causal convolution over `conv_state` + recurrent DeltaNet `ssm_state` update) or `QWEN35_LAYER_FULL_ATTN` (Gated Q/K/V attention with Q/K RMSNorm, partial RoPE, and FP16 `k_cache`/`v_cache`).
   - **Chunked Prefill & Persistent Scratch:** `qwen35_forward_batch` processes prompts in chunks of up to `chunk_max = 512` tokens (`qwen35_forward_chunk`). `ensure_batch_scratch` grows persistent buffers (`xb`, `xnb`, `projb`, `qkvb`, `zb`, `ab`, `bb`, `qgb`, `kb`, `vb`, `aob`, `fgb`, `fub`, `qx_h`) once per session up to the largest chunk size `B <= 512`, eliminating per-token or per-call `malloc`/`free` churn.
   - **System-Prefix Snapshot Caching:** `qwen35_save_prefix` / `qwen35_restore_prefix` snapshot the compact DeltaNet `conv_state` + `ssm_state` and the first `n_tokens` of FP16 `k_cache`/`v_cache` into session buffers, enabling instant turn-to-turn prefix restoration without re-evaluating system/tool tokens.

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

1. **Single Tokenization:** Prompts (`T1_PROMPT`, `T2_DELTA`, and `LONG_PROMPT` at ~860 tokens) are tokenized once and passed as identical `int32` token-ID arrays to both `liblila_asi.so` and `llama-cpp-python`.
2. **Verified KV/Prefix Reuse:** Turn 2 (`T2_DELTA`) asserts `t2_eval_tokens == len(toks["t2"])` on every trial for both engines.
3. **Matched Runtime Settings:** `n_ctx=2048`, `n_threads=2`, `n_threads_batch=2`, `n_batch=512`, `use_mmap=True`, `flash_attn=False`, greedy (`temperature=0.0`).
4. **Page-Cache Warmup & Subprocess Isolation:** Each engine runs 1 discarded warmup trial in a fresh subprocess, followed by `N >= 15` interleaved `(ASI, llama.cpp)` subprocess pairs. Optional `--prepopulate` calls `madvise(MADV_WILLNEED)` and touches one byte per 4 KiB page before timing.
5. **Statistical Rigor:** Reports median, `[min..max]` spread, 95% bootstrap confidence interval (2,000 resamples), and two-sided Mann-Whitney U p-value. The **Strict Bar** requires both a better median and non-overlapping `[min..max]` intervals across all 6 primary metrics (`t1_ttft`, `t1_dec`, `t2_ttft`, `t2_dec`, `long_prefill_tps`, `rss_mb`).

---

## 8. Results & Ablations

### 8.1 Qualification & Correctness Status

- **Cartridge & Integrity (`test_asi_v2.c`):** Verified `asi_v2_open`, tensor lookup, `ASI_SEC_CHAT_TEMPLATE` presence, and CRC-64 corruption rejection.
- **Lila Integration (`qualify_asi_backend.py`):** Covers 6 end-to-end checks: batch-vs-token parity, multi-turn & tool-call generation, cancellation & concurrent session isolation, prefix-cache invalidation & 5-cycle RSS drift (`< 8.0 MB`), live `LilaDaemon` mTLS HTTPS `/chat` with `LILA_BACKEND=asi`, and `N=5` controlled medians.
- **Correctness Gate (`test_correctness_gate.py`, `N=64` greedy tokens, [`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json)):** `ASI(batch)` vs `ASI(token)` achieves **`0.000000` relative L2 error** and **`320/320` (`100.00%`)** exact token agreement across all 5 prompts (`factual`, `math`, `code`, `multiturn`, `tool_use`). Against `llama.cpp` over 64 greedy tokens per prompt, `code` and `tool_use` match at **`100.0%`** (`64/64`), `math` at **`95.3%`** (first divergence at token 23), `factual` at **`42.2%`** (token 27), and `multiturn` at **`20.3%`** (token 12), for **`229/320` (`71.56%`)** overall corpus agreement (`135/160` = `84.38%` at `N=32`).

### 8.2 Strict Apples-to-Apples Benchmark (`N=15`, i5-5300U, 2 Threads, [`results_broadwell_n15.json`](file:///home/ticket/workspaces/lila/source/paper/results_broadwell_n15.json))

| Metric | Unit | Direction | ASI median `[min..max]` (95% CI) | `llama.cpp` median `[min..max]` (95% CI) | MWU *p* | Verdict |
|---|---|---|---|---|---|---|
| `t1_ttft` | s | lower | `1.055 [0.997..2.600]` (`1.014..1.378`) | `1.002 [0.840..1.489]` (`0.943..1.179`) | `0.04649` | `FAIL` |
| `t1_dec` | tok/s | higher | `14.064 [4.814..15.828]` (`12.739..15.340`) | `13.943 [11.469..15.671]` (`13.624..14.821`) | `0.9339` | `FAIL` (ASI median wins; overlapping) |
| `t2_ttft` | s | lower | `0.554 [0.492..1.644]` (`0.521..0.681`) | `0.561 [0.500..0.914]` (`0.521..0.631`) | `0.901` | `FAIL` (ASI median wins; overlapping) |
| `t2_dec` | tok/s | higher | `13.812 [0.466..15.867]` (`12.465..15.377`) | `13.952 [10.622..15.436]` (`13.468..15.100`) | `0.9339` | `FAIL` (statistical parity) |
| `long_prefill_tps` | tok/s | higher | `45.055 [13.927..48.266]` (`42.123..47.268`) | `53.747 [40.275..57.548]` (`48.034..56.035`) | `0.0009058` | `FAIL` |
| `rss_mb` | MiB | lower | `615.773 [615.715..615.906]` (`615.762..615.797`) | `978.008 [977.949..978.148]` (`977.992..978.090`) | `3.35e-06` | `PASS` |
| `vm_hwm_mb` (sec) | MiB | lower | `615.773 [615.715..615.906]` (`615.762..615.797`) | `1101.188 [1100.961..1101.352]` (`1101.055..1101.266`) | `3.358e-06` | `PASS (sec)` |
| `load_s` (sec) | s | lower | `0.026 [0.025..0.063]` (`0.026..0.028`) | `2.087 [1.878..3.485]` (`1.925..2.418`) | `3.392e-06` | `PASS (sec)` |

---

## 9. Honest Limitations & Future Work

1. **Compute-bound Broadwell 2C/4T regime:** On a 15 W dual-core Broadwell CPU (i5-5300U, DDR3-1600), prefill and 248k-row `lm_head` projections are sensitive to integer-SIMD throughput and L1/L2 cache residency. Findings on this chip do not automatically generalize to high-core-count servers or memory-bandwidth-saturated GPUs.
2. **Lazy `mmap` vs. Turn-1 TTFT:** Skipping `LILA_ASI_VERIFY_CRC=1` makes `lila_asi_open` nearly instantaneous, but cold page faults shift I/O latency into Turn-1 prefill/decode unless pages are warmed (`--prepopulate`).
3. **Unverified Wide-SIMD & ARM64 Paths:** The AVX-512F/BW, AVX-512VNNI, AVX-VNNI, and ARM64 NEON/DotProd branches in `q4k.h` are compile-checked only and have not yet been hardware-verified on physical Xeon/EPYC or Apple/ARM64 silicon (e.g., via `colab_asi_benchmark.ipynb`).
4. **Architecture Coverage:** In `asi_arch.c`, `g_asi_arch_hybrid_qwen35` is fully implemented, whereas `g_asi_arch_dense_transformer` (for `dense_llama`, `dense_gemma`, `dense_qwen2`, `dense_qwen3`) currently registers `NULL` handler pointers and remains to be implemented.
