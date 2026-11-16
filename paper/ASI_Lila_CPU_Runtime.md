# ASI: A Portable, CPU-Optimized Inference Runtime for Quantized Language Models

**Lila-ASI Technical Report #1 (v2.2)**

**Authors:** Sammie (`ticketguy`) and the Gemini 4 Hydrogen Engineering Agents  
**Primary Evaluation Hardware:** CPU-only Lenovo ThinkPad T450s (`Intel(R) Core(TM) i5-5300U CPU @ 2.30GHz`, Broadwell, 2 physical cores / 4 logical CPUs, 3 MiB L3, AVX2 / FMA / F16C, DDR3-1600)  
**Cross-Machine Reference:** Google Colab CPU VM (`Intel(R) Xeon(R) CPU @ 2.20GHz`, 1 physical core / 2 logical CPUs, AVX2 / FMA)  
**Primary Implementation:** [`source/engine/asi/`](file:///home/ticket/workspaces/lila/source/engine/asi/) and [`source/engine/runtime/q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h)

---

## 1. Abstract

Local-first AI agents running on commodity laptops face strict memory, thermal, and startup-latency budgets without discrete GPUs. We present **ASI (Active System Image)**, a standalone C11/SIMD inference runtime and portable binary cartridge format (`.asi` v2) engineered for CPU-only execution and integrated into the Lila agent host ([`AsiLlamaAdapter`](file:///home/ticket/workspaces/lila/source/src/core/asi_backend.py) and [`LilaCore._boot_gguf()`](file:///home/ticket/workspaces/lila/source/src/core/lilacore.py)). ASI decouples model distribution from runtime execution via a 4 KiB page-aligned, `mmap`-able binary cartridge (`AsiV2Header`, `AsiV2SectionEntry`, `AsiV2TensorEntry`) and executes hybrid state-space/attention architectures—specifically Qwen3.5 Gated DeltaNet + Grouped-Query Attention (`ASI_ARCH_HYBRID_QWEN35`)—using persistent scratch arenas, an SMT-aware spin-then-sleep thread pool (`AsiThreadPool`), integer-SIMD 2-row `dot2` and 1× row-unpack GEMM kernels (`Q4_K × Q8_K` and `Q6_K × Q8_K`), and crash-safe disk-persisted system-prefix snapshots (`ASIPFX02`).

In ASI v2.2 on `Intel(R) Core(TM) i5-5300U CPU @ 2.30GHz` (`Qwen3.5-0.8B` Q4_K_M, `n_threads = 2`), we combine a full **G0 correctness gate** ([`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json)), a hardware **roofline & weight-byte audit** ([`roofline_broadwell.json`](file:///home/ticket/workspaces/lila/source/paper/roofline_broadwell.json)), and a statistical **G1 performance claim gate** ([`claim_verdict_broadwell.json`](file:///home/ticket/workspaces/lila/source/paper/claim_verdict_broadwell.json)). Under G0 across `320` token positions (`N_GEN = 64` across `5` prompts), `ASI(Q8_K)` achieves **`320/320` (`100.0%`)** batch-vs-token parity (`0.0` relative L2), **`316/320` (`98.75%`)** teacher-forced top-1 agreement with `llama.cpp`, **`315/320` (`98.44%`)** agreement and lower mean logit relative L2 error against unquantized `FP32-Ref` than `llama.cpp` (`0.047385` vs `0.047615`, ratio `0.9952 <= 1.05`, all `4/320` flips at near-tie `FP32-Ref` margins `<= 0.0948`), and `5`-subprocess bitwise determinism (`PASS` at `threads = 1, 2` with `32/32` cross-thread agreement). Roofline measurements reconcile earlier L3 microbenchmarks (`~27–42 GFLOP/s` on `2 MiB`) against full-model decode (`518.34 MB` active weights streamed per token, `57.33% Q6_K` with `40.24%` in `token_embd`/`lm_head`, vs `11.53 / 12.08 GB/s` DRAM read-only ceiling) and motivate enabling `LILA_ASI_Q6K_DOT2=1` and `LILA_ASI_Q4K_DOT2=1` by default, lifting idle 2T `3 × 32`-token decode from `8.88` to `13.58 tok/s`. Under G1 (`claim_check.py` over `N = 15` prepopulated pairs), ASI earns tier **`BEATS`** with the verified claim **"Beats llama.cpp on Intel(R) Core(TM) i5-5300U CPU @ 2.30GHz (2 threads)"**: `WIN` on Turn-1 TTFT (`0.908` vs `1.094 s`, `+17.1%`, `p = 0.00105`), Turn-2 TTFT (`0.487` vs `0.607 s`, `+19.9%`, `p = 0.003691`), and 939-token prefill (`52.799` vs `46.213 tok/s`, `+14.3%`, `p = 0.0003609`), statistical `PARITY` on Turn-1/Turn-2 decode, and non-overlapping `STRONG WIN` on RSS (`616.266` vs `978.223 MiB`, `-37.0%`, `p = 3.341e-06`), peak HWM (`638.824` vs `1406.730 MiB`, `-54.6%`), and load latency (`0.053` vs `2.182 s`, `-97.5%`). Note that the `N = 15` Broadwell timing pairs and a companion `N = 10` Colab Xeon run (`4e73276`) predate the v2.2 `dot2=1` defaults and SMT spin-backoff, and AVX-512/NEON kernels are compile- and audit-verified only.

---

## 2. Introduction, Motivation & Contributions

Deploying an interactive local assistant such as Lila on an older, CPU-only laptop (2 physical cores / 4 threads, DDR3-1600, no GPU) exposes three bottlenecks that general-purpose serving frameworks rarely optimize together:

1. **Cold-start and memory-footprint overhead:** Parsing unstructured metadata, eagerly allocating duplicated weight buffers, and creating transient per-call scratch memory inflate resident set size (RSS) and delay startup.
2. **Turn-to-turn system-prompt re-evaluation:** Agentic workloads prepend long system prompts and tool schemas (800+ tokens) on every turn; without persistent prefix state across process restarts or sessions, scarce laptop CPU cycles are repeatedly wasted recomputing identical hybrid DeltaNet and KV states.
3. **Host-engine coupling and uncalibrated microbenchmarks:** Coupling tokenizer, tensor layout, and chat templates to a monolithic server complicates cross-engine numerical auditing, while isolated L3-resident kernel benchmarks can severely mispredict DRAM-bound full-model decode performance unless checked against a hardware roofline.

ASI v2.2 makes four concrete engineering contributions:

- **Zero-Copy `.asi` v2 Cartridge & Minimal C ABI:** A 4 KiB page-aligned binary format ([`asi_v2.h`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.h)) with 64-byte aligned weights (`MAP_PRIVATE` / `MapViewOfFile`) and a focused C11 ABI ([`lila_asi.h`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.h)) integrated into [`asi_backend.py`](file:///home/ticket/workspaces/lila/source/src/core/asi_backend.py) and [`lilacore.py`](file:///home/ticket/workspaces/lila/source/src/core/lilacore.py).
- **Crash-Safe Disk Prefix Snapshot (`ASIPFX02`) & SMT-Aware Thread Pool:** Deterministic `.asip` prefix persistence (`AsiPrefixSnapshotHeader`, dual CRC-64 ECMA checksums, atomic `<filepath>.tmp.<pid>` rename, `0o700`/`0o600` private permissions, LRU cache eviction) paired with topology-aware `AsiThreadPool` spin-backoff ([`asi_pool.h`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_pool.h)) for SMT-oversubscribed cloud VMs.
- **Roofline-Reconciled Kernels & Bit-Exact AVX-512 Alignment:** Full DRAM-vs-L3 characterization explaining decode bottlenecks (`57.33% Q6_K` active bytes, `40.24%` in tied `token_embd`/`lm_head`), default `dot2` `Q4_K`/`Q6_K` matvec kernels in [`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c), branchless scale unpacking (`lila_q4k_unpack_scales_avx2`), and bit-exact 8-lane reduction matching AVX2 and AVX-512 code paths in [`q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h).
- **Automated G0 Correctness & G1 Statistical Claim Gates:** Enforced validation via [`test_correctness_gate.py`](file:///home/ticket/workspaces/lila/source/engine/tests/test_correctness_gate.py), [`bench_roofline.c`](file:///home/ticket/workspaces/lila/source/engine/tests/bench_roofline.c), and [`claim_check.py`](file:///home/ticket/workspaces/lila/source/engine/tests/claim_check.py) separating verified hardware claims from untested ISA tiers and historical runs.

---

## 3. The `.asi` v2 Cartridge Format

The `.asi` v2 specification is defined in [`source/engine/asi/asi_v2.h`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.h) and implemented in [`asi_v2.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.c), [`asi_v2.py`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_v2.py), and [`gguf_to_asi_v2.py`](file:///home/ticket/workspaces/lila/source/engine/asi/gguf_to_asi_v2.py). All on-disk structures are little-endian and enforced at compile time with `_Static_assert`.

### 3.1 File Header & Section Directory

- **Header (`AsiV2Header`, 64 bytes):**
  - `magic = 0x41534921u` (`"ASI!"` little-endian), `version = 2u`.
  - `arch_id`: `AsiV2ArchId` enum (`1 = dense_llama`, `2 = hybrid_qwen35`, `3 = dense_gemma`, `4 = dense_qwen2`, `5 = dense_qwen3`).
  - `flags`: capability bitmask (`ASI_V2_FLAG_HAS_VISION`, `HAS_AUDIO`, `HAS_ADAPTERS`, `HAS_SIGNATURE`, `HAS_CHAT_TEMPLATE`).
  - `n_sections`, `total_size`, `section_table_offset`, and a 24-byte `identity_hash` (first 24 bytes of SHA-256 over section payloads computed in `write_asi_v2`).
- **Section Table (`AsiV2SectionEntry`, 32 bytes per entry):**
  - Each section (`ASI_SEC_MODEL_CONFIG = 0x01`, `ASI_SEC_WEIGHTS = 0x02`, `ASI_SEC_TOKENIZER = 0x04`, `ASI_SEC_METADATA = 0x08`, `ASI_SEC_CHAT_TEMPLATE = 0x09`, `ASI_SEC_VISION = 0x0A`, etc.) has a 4 KiB page-aligned `offset` (`ASI_V2_PAGE_SIZE = 4096u`), `size`, and 64-bit `checksum`.
  - Legacy sections (`ASI_SEC_BYTECODE_LEGACY = 0x05`, `ASI_SEC_HARNESS_LEGACY = 0x06`) are ignored in the v2 load path; the live Lila host owns tools and permissions.

### 3.2 Integrity Verification & Lazy `mmap`

Section checksums use **CRC-64 ECMA** (`0x42F0E1EBA9EA3693ULL` polynomial in `asi_v2_crc64`). During `asi_v2_validate_memory`:
- Small metadata, config, chat-template, and tokenizer sections are always CRC-64 verified on open.
- Bulk binary blobs (`ASI_SEC_WEIGHTS`, `ASI_SEC_VISION`, `ASI_SEC_AUDIO`) skip eager CRC-64 scanning by default unless `LILA_ASI_VERIFY_CRC=1` is set in the environment. This preserves lazy page-faulting under `mmap` so `lila_asi_open` completes in milliseconds without forcing cold disk reads before inference begins.

### 3.3 Model Config & 64-Byte Aligned Weights Directory

- **`AsiV2ModelConfig` (96 bytes):** Stores explicit architecture hyperparameters (`n_layers`, `hidden_size`, `intermediate_size`, `n_heads`, `n_kv_heads`, `head_dim`, `vocab_size`, `max_seq_len`, `rope_dim`, `rope_theta`, `rms_norm_eps`, `embed_scale`), plus hybrid SSM fields (`full_attention_interval`, `ssm_conv_kernel`, `ssm_state_size`, `ssm_group_count`, `ssm_inner_size`, `ssm_time_step_rank`).
- **`ASI_SEC_WEIGHTS`:** Begins with a 16-byte `AsiV2WeightsHeader` (`n_tensors`, `flags`, `data_offset`), followed by `n_tensors` × 80-byte `AsiV2TensorEntry` records (`name[40]`, `quant_type`, `layout_flags`, `shape[2]`, `data_offset`, `data_size`).
- **64-byte Alignment:** In `pack_weights_section`, `data_offset` and every individual tensor payload length are rounded up to 64-byte boundaries (`(n + 63) & ~63`) so SIMD loads from mapped weight pointers never cross unaligned cache-line starts.
- **Optional AVX2 Q6_K Repacking (`ASI_TENSOR_LAYOUT_AVX2_REPACK`):** `gguf_to_asi_v2.py` supports calling `asi_v2_repack_q6k` (`libasi_v2.so`) to pre-unpack 210-byte GGML `Q6_K` blocks into 288-byte `LilaQ6KRepackedBlock` structs (`u6[256]`, `scales[16]`, FP32 `d`, `pad[12]`); all benchmarked models in this study use standard unpacked GGML layouts (`repacked_tensors = 0`).

---

## 4. Runtime Architecture (v2.2)

The runtime separates C execution from host orchestration across four cooperating subsystems:

### 4.1 Public C ABI & Host Integration

[`lila_asi.h`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.h) and [`lila_asi.c`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.c) export model lifecycle (`lila_asi_open`, `lila_asi_close`, `lila_asi_model_info`, `lila_asi_model_header`, `lila_asi_get_section`), session management (`lila_asi_session_create`, `free`, `reset`), execution (`lila_asi_eval_token`, `lila_asi_eval_batch`, `lila_asi_sample`), and both in-memory (`lila_asi_session_save_prefix` / `restore_prefix`) and disk-backed (`lila_asi_session_save_prefix_file` / `load_prefix_file`) prefix snapshots tied to `LILA_ASI_NUMERICS_VERSION` (`2U`).

On the host side, [`AsiLlamaAdapter`](file:///home/ticket/workspaces/lila/source/src/core/asi_backend.py) loads `liblila_asi.so`, extracts `ASI_SEC_TOKENIZER` / `ASI_SEC_CHAT_TEMPLATE` zero-copy, manages session reuse and prefix caching, and integrates directly with [`LilaCore._boot_gguf()`](file:///home/ticket/workspaces/lila/source/src/core/lilacore.py).

### 4.2 SMT-Aware Thread Pool (`asi_pool.h`)

[`AsiThreadPool`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_pool.h) avoids OpenMP barrier overhead on tight per-layer dispatches. Hot synchronization atomics (`fn`, `gen`, `done_count`, `active`, `sleeping`, `shutdown`) are aligned on 64-byte cache-line boundaries (`_Alignas(64)`) to prevent cross-core false sharing.

In v2.2, `asi_detect_physical_cores()` parses `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` (falling back to `sysconf(_SC_NPROCESSORS_ONLN) / 2`) at pool initialization to adapt spin-backoff policy to hardware SMT oversubscription:
- When `n_threads <= physical_cores` (e.g., 2 threads on 2 physical Broadwell cores), workers spin via `asi_cpu_pause()` for `spin_limit = 2048` iterations and `sched_yield()` up to `yield_limit = 4096` before sleeping on `pthread_cond_wait`.
- When `n_threads > physical_cores` (e.g., 2 threads on a 1-core / 2-SMT cloud VM such as Google Colab Xeon), spinning starves the hyperthread sibling running peer work or OS tasks; ASI automatically drops to `spin_limit = 64` / `yield_limit = 128`.
- `LILA_ASI_SPIN_LIMIT` explicitly overrides `spin_limit` (setting `yield_limit = 2 * spin_limit`) for controlled scheduling experiments.

### 4.3 Hybrid Qwen3.5 Execution & `AsiRuntimeConfig` (`asi_arch.c`)

[`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c) dispatches via `AsiArchHandler` (`g_asi_arch_hybrid_qwen35`):
- **Layer Heterogeneity:** Each layer runs either `QWEN35_LAYER_DELTANET` (1D causal convolution via AVX2 `_MM_TRANSPOSE4_PS` + `lila_expf_avx2` in `asi_conv1d_worker` + tiled recurrent DeltaNet state update) or `QWEN35_LAYER_FULL_ATTN` (Gated Q/K/V attention with Q/K RMSNorm, partial RoPE, grouped GQA KV reuse in `asi_kv_prep_batch_worker` / `asi_full_attn_batch_worker`, and FP16 KV caches).
- **Chunked Prefill & 1× Weight-Unpack GEMM:** `qwen35_forward_batch` processes prompts in chunks of up to `chunk_max = 512` tokens (`qwen35_forward_chunk`), using parallel `asi_norm_quant_pool` + persistent 64-byte aligned scratch (`ensure_batch_scratch`), while `q4k_matmul_batch_q8k_range` and `q6k_matmul_batch_q8k_range` unpack each weight row (`lila_q4k_unpack_row_avx2` / `lila_q6k_unpack_row_avx2`) **once** per token chunk into L1d and accumulate into a contiguous `tile_out` transpose buffer.
- **v2.2 `dot2` Default & Runtime Configuration (`AsiRuntimeConfig`):** Multi-threaded `asi_matvec_worker` / `asi_matvec_range` now wires both `q4k_dot2_q8k_avx2` and `q6k_dot2_q8k_avx2` with defaults `LILA_ASI_Q6K_DOT2 = 1`, `LILA_ASI_Q4K_DOT2 = 1`, `LILA_ASI_MATVEC_CHUNK = 64`, `LILA_ASI_LMHEAD_CHUNK = 512`, `LILA_ASI_FUSED_CHUNK = 32`, and `LILA_ASI_BATCH_CHUNK_MAX = 512`. `LILA_ASI_PROFILE=1` enables per-stage microsecond attribution (`dn_proj`, `ssm`, `fa_proj`, `fa_gqa`, `ffn`, `norm`, `lm_head`), and `LILA_ASI_FP32_REF=1` bypasses `Q8_K` activation quantization to evaluate exact FP32-dequantized dot products (`lila_quant_dot_fp32` / `lila_quant_matmul_batch_fp32_range`).

### 4.4 Crash-Safe Disk-Persisted Prefix Snapshot (`ASIPFX02`)

To persist evaluated system prompts across cold subprocesses and agent restarts without recomputing DeltaNet/KV states:
- **Binary Container ([`lila_asi.c`](file:///home/ticket/workspaces/lila/source/engine/asi/lila_asi.c), [`asi_arch.c`](file:///home/ticket/workspaces/lila/source/engine/asi/asi_arch.c)):** `AsiPrefixSnapshotHeader` (144 bytes, `_Static_assert`-checked) opens with magic `"ASIPFX02"`, embeds `LILA_ASI_NUMERICS_VERSION` (`2U`), model geometry/identity binding, and two independent **CRC-64 ECMA** checksums—`header_crc64` over the header prefix and `payload_crc64` over serialized DeltaNet/KV state buffers—preventing silent restore across numeric versions, modified models, or truncated files.
- **Atomic Persistence:** `lila_asi_session_save_prefix_file` writes `<filepath>.tmp.<pid>`, calls `fflush` and `fsync`, and atomically promotes via `rename`, preventing partial snapshots on crash or cancellation. `lila_asi_session_load_prefix_file` validates magic, version, header CRC-64, and payload CRC-64 before mutating live session state.
- **Automatic Host Cache (`AsiLlamaAdapter` & `LilaCore._boot_gguf()`):** `_compute_prefix_key_hash` hashes `b"ASIPFX02"`, model `identity_hash`, `arch_id`, `fp32_ref`, token count, UTF-8 `sys_text`, and exact integer prompt token IDs into filenames `prefix_<hex16>.asip`. Cache directories are created with `0o700` (`secure_private_directory`) and `.asip` files with `0o600` (`restrictive_umask()`), bounded by LRU pruning (`LILA_ASI_PREFIX_CACHE_MAX_FILES=8`, configurable via `LILA_ASI_PREFIX_CACHE_DIR` or `LILA_ASI_DISABLE_DISK_PREFIX`). `LilaCore._boot_gguf()` configures `Path(self.memory_dir) / "prefix_cache"` automatically on boot.

---

## 5. Quantization & SIMD Kernel Design

All quantized dot products and batched GEMMs live in [`source/engine/runtime/q4k.h`](file:///home/ticket/workspaces/lila/source/engine/runtime/q4k.h).

### 5.1 Activation Quantization (`LilaQ8KBlock`) & 2-Row `dot2` Reuse

Input FP32 activation vectors are quantized on the fly into 256-element `LilaQ8KBlock` structures (`float d`, `int8_t qs[256]`, `int16_t bsums[16]`). For batched prefill (`n_tokens > 1`), `quantize_vec_q8k_strided` writes blocks in **block-major, token-strided** layout (`qx_all + (size_t)b * n_tokens + tok`) so that weight block `b` loads once and feeds 4 consecutive tokens (`t..t+3`). During single-token decode, `q4k_dot2_q8k_avx2` and `q6k_dot2_q8k_avx2` load each quantized activation block `qx[b]` once in registers and compute two output rows simultaneously (`row`, `row+1`), halving activation traffic and amortizing `Q8_K` unpacking across memory-bound `lm_head`, `ffn`, and `dn_proj` layers.

### 5.2 AVX-512 Bit-Exact Reduction, Branchless Scales & ISA Compile Matrix

In v2.2, `#if defined(__AVX512F__)` reduction trees in `q4k.h` and `asi_arch.c` were aligned with `#if defined(__AVX2__)` so wide-SIMD dot products fold their upper/lower 256-bit halves into the identical 8-lane `__m256 sum_vec` and `lila_hsum_f32_avx2` horizontal sum, and batched GEMMs share the bit-exact `tile_out` unpack path (`lila_q4k_unpack_row_avx2`, `lila_q6k_unpack_row_avx2`) without introducing divergence across SIMD widths. Inside `Q4_K` kernels, branchy `get_scale_min_k4` indexing was replaced with `lila_q4k_unpack_scales_avx2`.

| Kernel / ISA Path | Key Instructions / Fallback | Verification Status |
|---|---|---|
| **AVX2 + FMA + F16C** (`__AVX2__`) | `_mm256_maddubs_epi16`, `_mm256_madd_epi16`, `_mm256_fmadd_ps`, `_cvtsh_ss` | **Hardware-verified** on `i5-5300U` (`AVX2/FMA/F16C`) & compiled clean |
| **AVX-512F / BW / DQ / VL** (`__AVX512F__`) | Bit-exact `__m256 sum_vec` + `lila_hsum_f32_avx2`, shared `tile_out` unpack | **Compile- and audit-verified** (`-mavx512f -mavx512bw -mavx512dq -mavx512vl`; requires AVX-512 hardware run) |
| **AVX-512-VNNI / AVX-VNNI** | `_mm512_dpbusd_epi32`, `_mm256_dpbusd_epi32` | **Compile- and audit-verified** (`-mavx512vnni`; requires VNNI hardware run) |
| **Non-`F16C` & Scalar Fallback** | Software `fp16_to_fp32`, scalar dot/GEMM paths | **Compile-verified** + **Sanitizer-clean** (`ASan+UBSan+LSan`) |
| **ARM64 NEON / DotProd** (`__aarch64__`) | `vdotq_s32` / `vmull_s8` | **Audit/code-verified** (requires ARM64 hardware execution) |

---

## 6. Correctness & G0 Claim Gate

Quantized integer-SIMD inference cannot be validated solely by isolated matrix-multiply checks; small floating-point differences propagate across 24+ layers and can flip greedy decisions when top-1/top-2 margins are tight. ASI v2.2 enforces both unit checks and an automated **G0 Gate** in [`claim_check.py`](file:///home/ticket/workspaces/lila/source/engine/tests/claim_check.py):

1. **Kernel & Cartridge Unit Integrity ([`test_quant_kernels.c`](file:///home/ticket/workspaces/lila/source/engine/tests/test_quant_kernels.c), [`test_asi_v2.c`](file:///home/ticket/workspaces/lila/source/engine/tests/test_asi_v2.c)):** Verifies `Q4_K`/`Q6_K` matvec kernels against scalar FP32 dequantization (`rel_l2 < 0.01`), `asi_v2_open`, tensor lookups, CRC-64 corruption rejection, and exact `ASIPFX02` disk prefix snapshot save/load parity (`memcmp == 0`) with header/payload tamper rejection.
2. **Backend Host Qualification ([`qualify_asi_backend.py`](file:///home/ticket/workspaces/lila/source/engine/tests/qualify_asi_backend.py)):** Executes 6 end-to-end checks on `AsiLlamaAdapter` and `LilaDaemon`: batch-vs-token parity, multi-turn & tool-call generation, cancellation & concurrent session isolation, `.asip` disk prefix persistence (`hit_b='disk'`) plus invalidation (`mut_hit='miss'`) and 5-cycle RSS drift, live mTLS `/chat` under `LILA_BACKEND=asi`, and latency checks.
3. **Formal G0 Gate ([`test_correctness_gate.py`](file:///home/ticket/workspaces/lila/source/engine/tests/test_correctness_gate.py) + [`claim_check.py`](file:///home/ticket/workspaces/lila/source/engine/tests/claim_check.py)):** Evaluates `Qwen3.5-0.8B` (`n_ctx = 512`, `n_threads = 2`, `N_GEN = 64` across 5 prompts = `320` token positions) on identical integer token IDs across `ASI(token)`, `ASI(batch)`, `llama.cpp`, and `ASI(FP32-Ref)` (`LILA_ASI_FP32_REF=1`). Passing G0 requires:
   - `batch_vs_token == 100.0%` over `>= 320` positions (`max_rel_l2 = 0`),
   - `teacher_forced` top-1 agreement `>= 98.0%` vs `llama.cpp` over `>= 320` positions,
   - `fp32_ref` mean logit Rel-L2 ratio `(ASI / llama.cpp) <= 1.05`, and
   - `determinism`: `>= 5` independent subprocess runs per thread count (`threads = 1, 2`) with `0` token divergence and exact cross-thread agreement (`all_threads_agree == true`).

---

## 7. Experimental Methodology, Roofline Harness & G1 Gate

### 7.1 Benchmark Hygiene ([`bench_strict_bar.py`](file:///home/ticket/workspaces/lila/source/engine/tests/bench_strict_bar.py))

1. **Byte-Identical Artifact & Single Tokenization:** All Broadwell and Colab comparisons use `Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf` (GGUF SHA-256 `d9662c91079317aee661a1475d61412787696e0dcc5636c39648fb27d49e1edf`; `.asi` SHA-256 `757731964e8a24950f3ae743be9df81ca5671f15b014aa45c31868131679e9f0`, `642,277,376` bytes). Prompts (`T1_PROMPT` `53` tokens, `T2_DELTA` `27` tokens, `LONG_PROMPT` `939` tokens) are tokenized once into identical `int32` token-ID arrays.
2. **Matched Settings & Pre-Population:** `n_ctx = 2048`, `n_threads = n_threads_batch = 2`, `n_batch = 512`, `use_mmap = True`, `flash_attn = False`, greedy sampling (`temperature = 0.0`), verified Turn-2 KV/prefix reuse (`t2_eval_tokens == 27`). After 1 discarded warmup per engine, `N` interleaved `(ASI, llama.cpp)` subprocess pairs execute with mandatory `--prepopulate` (`madvise(MADV_WILLNEED)` + touching 1 byte per 4 KiB page) so `load_s` measures resident model startup honestly without deferring page faults into `t1_ttft`.

### 7.2 Hardware Roofline Harness ([`bench_roofline.c`](file:///home/ticket/workspaces/lila/source/engine/tests/bench_roofline.c))

`bench_roofline` measures: (a) 256 MiB AVX2 sequential read-only bandwidth (`avx2_dram_read_only`), (b) 80 MiB STREAM Copy and STREAM Triad bandwidth, (c) isolated `Q4_K` and `Q6_K` `dot1`/`dot2` throughput at L3-resident (`2 MiB`) and DRAM-resident (`64 MiB`) footprints, (d) an exact tensor-by-tensor active decode weight audit over `.asi` weights, and (e) full-model 32-token decode compared against the measured DRAM-RO and isolated-matvec ceilings.

### 7.3 Formal G1 Statistical Claim Gate ([`claim_check.py`](file:///home/ticket/workspaces/lila/source/engine/tests/claim_check.py))

While our earlier `results_broadwell_n15.json` summary used an un-tiered zero-overlap bar (`overall_primary_bar`), `claim_check.py` formalizes statistical claim tiers using `N >= 15` pairs, Mann-Whitney U (`p`), 95% bootstrap CIs, and `p5`/`p95` tail ratios (`TailR`, capped at `10%` worst-case regression relative to `llama.cpp`):
- **`WIN`:** Median gain `>= 3.0%`, `MWU p < 0.01`, and `p5`/`p95` tail within `10%` of `llama.cpp`.
- **`STRONG WIN`:** Meets `WIN` plus non-overlapping `[min..max]` ranges.
- **`BEATS` Tier:** Requires `>= 1` primary `WIN` (`t1_ttft`, `t1_dec`, `t2_ttft`, `t2_dec`, or `long_prefill_tps`), `0` primary losses, and a `WIN`/`STRONG WIN` on `rss_mb` (where `STRICTLY_BEATS` additionally requires non-overlapping wins across every primary metric).
- `claim_check.py` also records `/proc/stat` `steal_jiffies` telemetry on virtualized runners (prepared in [`colab_asi_benchmark.ipynb`](file:///home/ticket/workspaces/lila/source/engine/tests/colab_asi_benchmark.ipynb)) so hypervisor CPU starvation is observable.

---

## 8. Experimental Results, Roofline Reconciliation & Ablations

### 8.1 Roofline Reconciliation & Active Decode Weight Breakdown ([`roofline_broadwell.json`](file:///home/ticket/workspaces/lila/source/paper/roofline_broadwell.json))

Isolated `bench_dot` benchmarks (`~2 MiB` of `Q4_K` weights fitting inside Broadwell's 3 MiB L3 cache) reach **`27–42 GFLOP/s`**, whereas real `Qwen3.5-0.8B` (`320` tensors, `repacked_tensors = 0`, `tied_embeddings = true`, `vocab_size = 248320`, `n_embd = 1024`, `n_layers = 24`) must stream **`518,335,744 B` (`518.34 MB`)** of active decode weights from DDR3-1600 DRAM per generated token:
- **`Q6_K`: `297,185,280 B` (`297.19 MB`, `57.3345%`)** — dominated by tied `token_embd`/`lm_head` at **`208,588,800 B` (`208.59 MB`, `40.2420%` of all active decode bytes)**, plus `attn_qkv` (`51,609,600 B`, `9.9568%`), `ffn_down` (`36,126,720 B`, `6.9698%`), and `other` (`860,160 B`, `0.1659%`).
- **`Q4_K`: `219,156,480 B` (`219.16 MB`, `42.2808%`)** — `attn_qkv` (`28,311,552 B`, `5.4620%`), `ffn_down` (`24,772,608 B`, `4.7793%`), and `other` (`166,072,320 B`, `32.0395%`).
- **`F32`: `1,993,984 B` (`1.99 MB`, `0.3847%`)**.

| Benchmark / Metric | Scope / Size | 1T Median (`best`) | 2T Median (`best`) |
|---|---|---|---|
| `avx2_dram_read_only` | 256 MiB DRAM | `11.53 GB/s` (`11.98`) | `12.08 GB/s` (`13.59`) |
| `stream_copy` / `stream_triad` | 80 MiB DRAM | `7.84` (`9.24`) / `9.66` (`10.36`) GB/s | `8.26` (`8.74`) / `5.70` (`9.42`) GB/s |
| `l3_q4k_dot1` / `l3_q4k_dot2` | 2 MiB (L3) | `7.60` (`27.00`) / `7.80` (`27.74`) GB/s (GFLOP/s) | `11.82` (`42.02`) / `11.05` (`39.30`) GB/s (GFLOP/s) |
| `l3_q6k_dot1` / `l3_q6k_dot2` | 2 MiB (L3) | `7.56` (`18.43`) / `7.43` (`18.11`) GB/s (GFLOP/s) | `12.71` (`30.98`) / `14.01` (`34.17`) GB/s (GFLOP/s) |
| `dram_q4k_dot1` / `dram_q4k_dot2` | 64 MiB (DRAM) | `5.78` (`6.68`) / `3.82` (`4.30`) GB/s | `11.06` (`11.16`) / `3.90` (`7.54`) GB/s |
| `dram_q6k_dot1` / `dram_q6k_dot2` | 64 MiB (DRAM) | `6.60` (`6.86`) / `5.73` (`6.07`) GB/s | `5.85` (`7.60`) / `6.36` (`6.77`) GB/s |
| Full `bench_roofline` decode (`32 tok`) | 518.34 MB/tok | `7.11 tok/s` (`140.68 ms/tok`, `3.68 GB/s`, `31.9%` DRAM-RO) | `10.38 tok/s` (`96.31 ms/tok`, `5.38 GB/s`, `44.6%` DRAM-RO) |

Even at 100% of Broadwell's `avx2_dram_read_only` ceiling (`12.08 GB/s` at 2T), `518.34 MB/tok` caps pure DRAM-bound decode at `23.31 tok/s` (`22.25 tok/s` at 1T). At 1T, `bench_roofline` attributes `109.18 ms/tok` (`9.16 tok/s`) to isolated matvec speeds (`47.5%` `Q6_K`, `52.5%` `Q4_K`) plus `31.50 ms/tok` (`22.4%`) non-matvec/dispatch residual; at 2T, isolated `dram_q4k_dot2` (`3.90 GB/s`) underpredicts full-model decode (`9.71 tok/s` predicted vs `10.38 tok/s` measured).

### 8.2 `3 × 32`-Token `dot1`/`dot2` Ablation & Per-Stage Decode Profile (Idle Broadwell)

Wiring `q6k_dot2_q8k_avx2` alongside `q4k_dot2_q8k_avx2` inside `asi_matvec_range` (`asi_arch.c`) was ablated across `3 × 32`-token full-model decode runs on idle Broadwell:

| `LILA_ASI_Q6K_DOT2` | `LILA_ASI_Q4K_DOT2` | 1T Med `[min..max]` (`tok/s`) | 2T Med `[min..max]` (`tok/s`) | Configuration Role |
|---|---|---|---|---|
| `0` | `0` | `7.83 [5.46..8.91]` | `8.88 [7.95..10.89]` | Both `dot1` |
| `0` | `1` | `8.29 [7.76..8.30]` | `12.19 [9.22..12.81]` | `Q4_K` `dot2` only |
| `1` | `0` | `8.01 [7.96..8.82]` | `11.37 [9.63..14.13]` | `Q6_K` `dot2` only |
| **`1`** | **`1`** | **`8.14 [8.09..9.24]`** | **`13.58 [13.49..13.63]`** | **v2.2 default (`+52.9%` median 2T vs `0/0`)** |

Enabling `dot2=1/1` delivers `+52.9%` higher 2T median decode throughput (`8.88` to `13.58 tok/s`) and collapses run-to-run jitter (`[13.49..13.63]` vs `[7.95..10.89]`). Under `LILA_ASI_PROFILE=1` at 2T with `dot2=1/1` (`~68 ms/tok` steady-state), per-token time is concentrated in three quantized projection groups (`~87%` total): `lm_head` (`248,320 × 1024` `Q6_K`) = **`~23–26 ms` (`~36%`)**, `ffn` = **`~18–21 ms` (`~29%`)**, and `dn_proj` = **`~13–16 ms` (`~22%`)**, followed by `ssm` (Conv1D + DeltaNet recurrence) = `~3.2–4.0 ms` (`~5%`), `fa_proj` = `~2.7–3.7 ms` (`~5%`), `fa_gqa` = `~0.10 ms` (`<0.2%`), and `norm` = `~0.06 ms` (`<0.1%`).

### 8.3 Qualification & G0 Correctness Gate (`N_GEN = 64`, [`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json))

- **Snapshot & Backend Qualification:** `test_asi_v2` verified CRC-64 cartridge integrity and `ASIPFX02` snapshots (`Disk prefix round-trip diff=0.000000 (memcmp=0)` plus tamper rejection). `qualify_asi_backend.py` reported `ALL 6 ASI QUALIFICATION CHECKS PASSED`, including disk prefix persistence (`hit_b='disk'`) and key mutation invalidation (`mut_hit='miss'`).
- **G0 Gate Sub-Checks (`320` positions = `5` prompts × `64` tokens, `vocab_compared = 248320`):**
  - `batch_vs_token`: **`320/320` (`100.0%`)**, `max_abs_diff = 0.0`, `max_rel_l2 = 0.0` (`PASS`). Free-running greedy vs `llama.cpp` matches `238/320` (`74.38%`), with per-prompt agreement `tool_use = 100.0%`, `multiturn = 95.3%` (div 34), `math = 87.5%` (div 23), `code = 46.9%` (div 29), and `factual = 42.2%` (div 24).
  - `teacher_forced`: **`ASI vs llama.cpp = 316/320 (98.75%)`**, **`ASI vs FP32-Ref = 315/320 (98.44%)`**, `llama.cpp vs FP32-Ref = 316/320 (98.75%)` (`PASS`).
  - `fp32_ref`: `ASI Rel-L2 mean = 0.047385` (`median = 0.038031`, `max = 0.404697`) vs `llama.cpp mean = 0.047615` (`median = 0.038385`, `max = 0.342743`), **`ratio = 0.9952 <= 1.05`** (`0.995179`, `PASS`). All `4/320` ASI-vs-`llama.cpp` flips (`ASI==Ref: 1`, `Llama==Ref: 2`, `Neither==Ref: 1`) occurred at tight `FP32-Ref` margins (`0.050007..0.094835`, max `0.0948`).
  - `determinism`: **`PASS`** across `5` independent subprocesses at both `threads = 1` and `threads = 2` (`32` tokens on `factual` and `tool_use`), with exact `32/32` cross-thread agreement (`factual` SHA-256 `1f118f986237206116a3e68a054ffc510e7299367926205ebd6de27e87e64f28`, `tool_use` SHA-256 `9c1ed37382f2530c696317fd60c5555a9dba4822235280178db635e7af22f85b`).

### 8.4 G1 Claim Verdict (`claim_verdict_broadwell.json`) & Cross-Machine Context

Evaluated via `claim_check.py` against [`results_broadwell_n15.json`](file:///home/ticket/workspaces/lila/source/paper/results_broadwell_n15.json) (`N = 15`, `i5-5300U`, `n_threads = 2`, [`claim_verdict_broadwell.json`](file:///home/ticket/workspaces/lila/source/paper/claim_verdict_broadwell.json)):

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

- **Verified Claim Verdict:** `G1 Tier: BEATS` (`3` primary `WIN`s, `0` primary losses, `rss_mb` `STRONG WIN`), `overall_passed = true`, allowed claim: **`"Beats llama.cpp on Intel(R) Core(TM) i5-5300U CPU @ 2.30GHz (2 threads)"`**. *Scope note:* the 15 raw timing pairs in `results_broadwell_n15.json` were captured on the pre-v2.2 build before `dot2=1` defaults, SMT spin-backoff, and `steal_jiffies` telemetry (`has_telemetry = false`).
- **Cross-Machine Colab Xeon Reference (`N = 10`, commit `4e73276`, `Intel(R) Xeon(R) CPU @ 2.20GHz`, 1 physical core / 2 logical CPUs, AVX2+FMA, same `d9662c91` GGUF):** Across 10 pre-v2.2 interleaved pairs, ASI achieved lower load latency (`0.086` vs `1.015 s`), faster Turn-1 TTFT (`1.036` vs `1.136 s`) and Turn-2 TTFT (`0.552` vs `0.654 s`), higher 939-token prefill (`44.38` vs `41.03 tok/s`), lower steady-state RSS (`619.8` vs `1107.9 MiB`) and peak HWM (`619.8` vs `1381.7 MiB`), while showing SMT tail dips (`min = 4.45 / 8.06 tok/s`) on Turn-1/Turn-2 decode (`12.90` vs `13.80 tok/s` and `12.73` vs `13.64 tok/s`). Because commit `4e73276` had `N = 10 < 15` and predated v2.2's SMT spin-backoff (`spin_limit=64` / `yield_limit=128`), `dot2=1` defaults, and `steal_jiffies` logging, it is reported as cross-machine historical context rather than a G1 pass; [`colab_asi_benchmark.ipynb`](file:///home/ticket/workspaces/lila/source/engine/tests/colab_asi_benchmark.ipynb) is prepared for a full `N = 15`, `threads = 1, 2` v2.2 cloud rerun.

### 8.5 Negative Results — Zero-Training N-Gram Speculative Verification and Dynamic `lm_head` Shortlisting on Hybrid CPUs

To test whether zero-training speculative verification and vocabulary narrowing could break past Broadwell's `518.34 MB/tok` DRAM decode ceiling and `208.59 MB` `lm_head` bottleneck without requiring an auxiliary draft model [1, 2], trained Medusa heads [4], GPU-oriented Jacobi FLOP overprovisioning [5], or English-biased static frequency tables [6], we prototyped and evaluated an `ASI v2.3` candidate (Trial 11, subsequently reverted cleanly to `v2.2`):

- **Design (100% Language-Agnostic & Cartridge-Compatible):**
  1. *Suffix proposer (`qwen35_propose_draft`):* Following Prompt Lookup Decoding [3], a pure token-ID `4 -> 3 -> 2 -> 1-gram` suffix matcher over `token_history`, paired with a `<think>` transition heuristic and a `fail_hash[1024]` failed-transition filter (no natural-language tables).
  2. *Batched verification & DeltaNet rewind (`qwen35_forward_speculative`):* Single `B = 1 + K = 4` batched forward pass across all `24` layers and the `248,320`-row `lm_head`, backed by an `18`-layer DeltaNet state snapshot and `qwen35_rewind_deltanet_to` replay on rejection (`+25.5 MiB` reusable session RSS: `647.055` vs `621.559 MiB`).
  3. *Language-adaptive two-stage `lm_head` (`matvec_shortlist_q8k_pool`):* Inspired by FR-Spec [6], a 256-row page bitmap activating base rows (`0..32767`), special tail pages, and `token_history` pages with heuristic Stage-1 early exit (`top1 >= 18.0f && (top1 - top2) >= 2.0f`).

- **G0 Correctness & Qualification (`v2.3_spec`, `SHORTLIST=0`):** Passed G0 with `320/320` (`100.0%`, `0.0` diff) batch-vs-token parity, `100%` identical outputs to `v2.2` ([`correctness_broadwell_n64.json`](file:///home/ticket/workspaces/lila/source/paper/correctness_broadwell_n64.json)), `316/320` (`98.75%`) teacher-forced agreement vs `llama.cpp`, `ASI Rel-L2 = 0.047385`, `5`-subprocess determinism `PASS`, clean `ASan+UBSan+LSan`, and `6/6` in [`qualify_asi_backend.py`](file:///home/ticket/workspaces/lila/source/engine/tests/qualify_asi_backend.py) (`t1_dec = 14.08`, `t2_dec = 12.47` vs `llama.cpp 13.01 / 12.75 tok/s`).

- **`bench_strict_bar.py` `N = 3` Controlled 4-Way Ablation (`n_threads = 2`, `n_gen = 32`, `steal_jiffies = 0` across all 12 runs):**

| Metric | `v2.2` `med [min..max]` | `v2.3_spec` `med [min..max]` | `v2.3_spec_shortlist` `med [min..max]` | `llama.cpp` `med [min..max]` |
|---|---|---|---|---|
| `t1_ttft` (s, lower) | `0.883 [0.831..1.113]` | `1.140 [0.899..1.279]` | **`0.880 [0.859..0.975]`** | `1.196 [1.168..1.553]` |
| `t1_dec` (tok/s, higher) | **`14.152 [12.624..14.610]`** | `11.613 [9.405..12.588]` | `12.124 [11.925..12.256]` | `12.121 [10.735..14.055]` |
| `t2_ttft` (s, lower) | `0.613 [0.465..0.653]` | **`0.506 [0.482..0.642]`** | `0.516 [0.463..0.658]` | `0.591 [0.588..0.645]` |
| `t2_dec` (tok/s, higher) | **`12.918 [12.225..13.939]`** | `8.209 [7.660..10.601]` | `11.527 [9.163..11.644]` | `12.240 [11.824..12.662]` |
| `long_prefill_tps` (higher) | `48.889 [37.061..53.392]` | `44.548 [37.955..49.349]` | **`51.918 [43.903..54.256]`** | `43.302 [41.007..51.207]` |
| `rss_mb` (MiB, lower) | **`621.559 [621.512..621.660]`** | `647.055 [647.051..647.121]` | `647.000 [646.988..647.070]` | `979.582 [979.578..979.594]` |
| `load_s` (s, lower) | **`0.056 [0.054..0.061]`** | `0.062 [0.054..0.075]` | `0.059 [0.055..0.060]` | `2.463 [2.371..2.549]` |

  `[SPEC STATS]` (`eq_v2.2 = true` on `T1`, `T2`, `Long`): **`T1`**: `6` batches / `18` proposed / `4` accepted / `4` misses; **`T2` delta**: `13` batches / `39` proposed / `7` accepted / `9` misses; **`Long decode`**: `14` batches / `42` proposed / `4` accepted / `12` misses.

- **6-Prompt Agentic & Multilingual Suite (`N = 3`, `n_threads = 2`, `n_gen = 64`):**

| Prompt (`len`) | `eq_v2.2` (`spec`/`short`) | `v2.2` `tok/s` | `v2.3_spec` `tok/s` | `v2.3_spec_shortlist` `tok/s` | `llama.cpp` `tok/s` | `[SPEC STATS]` (`b / p / a / m`) |
|---|---|---|---|---|---|---|
| `english_qa` (`52`) | `true` / **`false`** | **`13.50 [11.80..14.56]`** | `10.82 [9.50..10.93]` | `11.90 [6.09..12.20]` *(bug)* | `13.15 [11.79..13.17]` | `18/54/7/15` (`short`: `15/45/9/10`) |
| `rag_edit` (`185`) | `true` / `true` | **`13.29 [11.30..13.83]`** | `11.05 [7.99..14.38]` | `11.33 [10.90..13.21]` | `12.17 [9.82..13.17]` | `18 / 54 / 25 / 8` |
| `tool_json` (`164`) | `true` / `true` | **`13.56 [11.61..13.89]`** | `11.48 [9.91..11.77]` | `10.71 [8.88..12.23]` | `10.60 [9.29..13.24]` *(48, EOS)* | `23 / 69 / 19 / 15` |
| `french` (`28`) | `true` / `true` | `12.66 [8.65..14.38]` | `14.88 [12.27..15.48]` | **`16.20 [15.00..16.43]`** | `12.87 [11.52..12.94]` | `12 / 36 / 23 / 3` |
| `spanish` (`26`) | `true` / `true` | `12.08 [10.42..14.35]` | **`12.62 [8.62..12.71]`** | `12.53 [12.36..12.88]` | `11.56 [9.54..13.25]` | `14 / 42 / 12 / 8` |
| `yoruba` (`54`) | `true` / `true` | `12.76 [9.70..13.19]` | `12.92 [9.25..13.74]` | **`14.82 [13.85..14.90]`** | `12.90 [10.97..13.02]` | `20 / 59 / 30 / 7` |

- **Root Causes & Why `v2.2` Was Retained:**
  1. *`B = 4` compute & rewind overhead on 2-core DDR3-1600:* Computing `4×` inner dot products across all 24 layers + `248,320`-row `lm_head` + `18`-layer DeltaNet snapshot/rewind costs `~2.5×–3.0×` a single `B = 1` `dot2` step, demanding `>2–2.5` accepted tokens per batch just to break even. Permissive `1-gram` triggers inflated miss counts (`15/18` on `english_qa`, `15/23` on `tool_json`, `8/18` on `rag_edit`, `12/14` on `long_decode`), slowing even high-acceptance agentic workloads (`rag_edit`: `11.05` vs `13.29 tok/s`, `tool_json`: `11.48` vs `13.56 tok/s`); only repetitive `french` (`23/12`, `3` misses) improved (`+17.5%`).
  2. *Unsound heuristic `lm_head` margin gating:* Under `LILA_ASI_LMHEAD_SHORTLIST=1`, `english_qa` step 8 replaced token `40755` (`" Tables"`, page `159`, outside `0..32767` and unvisited in `token_history`) with `6424` (`" Table"`, page `25`), breaking greedy parity (`eq_v2.2 = false`). Exact `lm_head` pruning requires provable Cauchy-Schwarz / row-norm upper bounds rather than empirical Stage-1 logit margins.
  3. *Revert to `v2.2`:* All 4 prototype C files were reverted cleanly to **`v2.2`** (`dot2=1/1`, `621.6 MiB` RSS), preserving `100%` greedy parity and maintaining `v2.2`'s clear decode lead over `llama.cpp` across `t1_dec` (`14.15` vs `12.12`), `t2_dec` (`12.92` vs `12.24`), `english_qa` (`13.50` vs `13.15`), `rag_edit` (`13.29` vs `12.17`), `tool_json` (`13.56` vs `10.60`), and `spanish` (`12.08` vs `11.56`).

---

## 9. Honest Limitations & Future Work

1. **Pre-v2.2 `N = 15` / `N = 10` Raw Captures:** Both `results_broadwell_n15.json` (`G1 Tier: BEATS`) and the Colab `N = 10` (`4e73276`) baseline predate v2.2 (`LILA_ASI_Q6K_DOT2=1`, `LILA_ASI_Q4K_DOT2=1`, SMT `spin_limit=64` backoff, and `/proc/stat` `steal_jiffies` telemetry). Re-running `N = 15` interleaved G1 pairs on Broadwell and Colab Xeon via [`colab_asi_benchmark.ipynb`](file:///home/ticket/workspaces/lila/source/engine/tests/colab_asi_benchmark.ipynb) is the next immediate empirical step.
2. **Untested Wide-SIMD & ARM64 Hardware Execution:** Only AVX2 + FMA + F16C has been executed on physical hardware (`i5-5300U` and Colab `Xeon`). Although `#if defined(__AVX512F__)` paths were refactored for bit-exact `__m256` reduction alignment and clean builds were verified across AVX-512, AVX-512-VNNI, non-`F16C`, scalar fallback, and `ASan+UBSan+LSan`, physical execution on AVX-512 (e.g., Sapphire Rapids / Zen 4) and ARM64 NEON/DotProd hardware remains to be validated.
3. **DRAM Bandwidth Ceiling & `lm_head` Dominance:** On Broadwell (`12.08 GB/s` 2T DRAM-RO ceiling), `518.34 MB` of active decode weights bounds theoretical maximum decode to `23.31 tok/s`, and `lm_head` (`248,320 × 1024` `Q6_K`) alone accounts for `40.24%` of active bytes (`~36%` of decode latency). As shown in Section 8.5, naive `B = 4` n-gram speculation and heuristic `lm_head` shortlisting fail to clear the `2C/4T` compute/rewind bar or maintain exact parity; further speedup requires exact Cauchy-Schwarz row-norm-bounded `lm_head` pruning, strictly draft-scoped vocabulary compression, or weight repacking tradeoffs beyond pure `dot2`.
4. **Lazy `mmap` Cold-Fault Tradeoff:** Disabling eager `LILA_ASI_VERIFY_CRC=1` keeps `lila_asi_open` near `27 ms`, but without `--prepopulate` (`madvise(MADV_WILLNEED)`), cold page faults shift disk latency into Turn-1 prefill/decode.
5. **Architecture Coverage:** `g_asi_arch_hybrid_qwen35` is complete and G0-verified, whereas `g_asi_arch_dense_transformer` (`dense_llama`, `dense_gemma`, `dense_qwen2`, `dense_qwen3`) still registers `NULL` handler pointers in `asi_arch.c`.

---

## 10. References

1. **Leviathan, Y., Kalman, M., & Matias, Y. (2023).** *Fast Inference from Transformers via Speculative Decoding.* Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202:19274–19286. `arXiv:2211.17192`.
2. **Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023).** *Accelerating Large Language Model Decoding with Speculative Sampling.* `arXiv:2302.01318`.
3. **Saxena, A. (2023).** *Prompt Lookup Decoding.* Open-source technical release, `github.com/apoorvumang/prompt-lookup-decoding` (November 2023).
4. **Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024).** *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* International Conference on Machine Learning (ICML 2024). `arXiv:2401.10774`.
5. **Fu, Y., Bailis, P., Stoica, I., & Zhang, H. (2024).** *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.* International Conference on Machine Learning (ICML 2024). `arXiv:2402.02057`.
6. **Zhao, W., Pan, T., Han, X., Zhang, Y., Sun, A., Huang, Y., Zhang, K., Zhao, W., Li, Y., Zhou, J., Zhou, H., Wang, J., Liu, Z., & Sun, M. (2025).** *FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling.* Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025, Volume 1: Long Papers), `10.18653/v1/2025.acl-long.198`. `arXiv:2502.14856`.
