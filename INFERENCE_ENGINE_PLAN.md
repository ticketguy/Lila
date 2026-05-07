# Lila Inference Engine — Build Plan

## What This Is

A custom inference engine for Lila written in assembly + C. No Python at runtime.
No dependency on llama.cpp, vLLM, transformers, or any third-party inference library.
Lila runs as native machine code — the fastest possible execution on any hardware.

## Why Custom

1. **Speed** — Hand-tuned assembly for the hot path (matmul, attention) beats any compiler output
2. **Portability** — Assembly kernels per architecture means it runs on anything: x86 desktop, ARM phone, RISC-V edge device
3. **Control** — Memory Fabric (multi-LoRA) is native to the engine, not bolted on
4. **Identity** — Lila speaks machine language. Her own inference IS machine language. Aligned.
5. **Independence** — No supply chain risk. No one else's bugs. No license constraints.

## Architecture

```
lila-engine/
├── kernels/                    # Assembly kernels (THE hot path)
│   ├── x86_64/
│   │   ├── matmul_avx512.S    # Matrix multiply (AVX-512, INT4 fused)
│   │   ├── matmul_avx2.S     # Fallback for older CPUs
│   │   ├── rmsnorm.S          # RMS normalization
│   │   ├── softmax.S          # Softmax with online computation
│   │   ├── silu.S             # SiLU activation
│   │   ├── dequant_int4.S    # FigQuant INT4 dequantization
│   │   ├── lora_fused.S      # Dequant + base matmul + LoRA in one pass
│   │   └── rope.S            # Rotary position embeddings
│   ├── arm64/
│   │   ├── matmul_neon.S     # Matrix multiply (NEON)
│   │   ├── matmul_sve.S      # Matrix multiply (SVE — newer ARM)
│   │   ├── rmsnorm.S
│   │   ├── softmax.S
│   │   ├── silu.S
│   │   ├── dequant_int4.S
│   │   ├── lora_fused.S
│   │   └── rope.S
│   └── riscv/                 # Future: RISC-V vector extension
│       └── (same pattern)
│
├── runtime/                   # C runtime (orchestrates kernels)
│   ├── model.c               # Model struct, weight loading (mmap)
│   ├── model.h
│   ├── inference.c           # Token generation loop
│   ├── attention.c           # Multi-head attention (dispatches to kernels)
│   ├── transformer.c        # Full transformer block
│   ├── kv_cache.c           # KV cache management
│   ├── memory_fabric.c      # Multi-adapter LoRA routing + gating
│   ├── tokenizer.c          # BPE tokenizer (sentencepiece compatible)
│   ├── quantize.c           # FigQuant format reader
│   ├── detect.c             # Hardware detection (which kernels to use)
│   └── allocator.c          # Custom memory allocator (arena-based)
│
├── format/                    # Model file format
│   ├── lila_format.h         # Custom binary format (or GGUF-compatible)
│   └── convert.py           # Convert safetensors → lila format
│
├── interface/                 # How the outside world talks to the engine
│   ├── cli.c                 # Command-line interface
│   ├── server.c             # HTTP/WebSocket server (for harness)
│   ├── voice_bridge.c       # Audio I/O bridge
│   └── python_bind.c        # Optional: Python bindings for testing
│
├── tests/                    # Correctness tests
│   ├── test_matmul.c
│   ├── test_attention.c
│   ├── test_dequant.c
│   ├── test_lora.c
│   └── bench/               # Performance benchmarks
│       ├── bench_matmul.c
│       ├── bench_attention.c
│       └── bench_e2e.c
│
├── Makefile                  # Build system (detect arch, assemble, link)
└── README.md
```

## Build Phases

### Phase 1: Foundation (get tokens flowing)
1. Hardware detection (x86_64 feature flags: AVX2, AVX-512, AMX)
2. Model loading via mmap (zero-copy from disk)
3. Basic matmul kernel (AVX2 — works on most x86 CPUs)
4. RMSNorm, SiLU, Softmax kernels
5. Single transformer block forward pass
6. Full model forward pass
7. Token generation loop (greedy decode)
8. **TEST: generate coherent text from Gemma 4B weights**

### Phase 2: INT4 Quantization (shrink the model)
1. FigQuant INT4 dequantization kernel
2. Fused dequant + matmul (never fully dequantize to RAM)
3. Custom model format with packed INT4 weights
4. Converter: safetensors → lila INT4 format
5. **TEST: same output quality as FP16, 4x less memory**

### Phase 3: Memory Fabric (multi-LoRA)
1. LoRA forward pass kernel (base + A×B correction)
2. Multi-adapter routing (5 namespaces, gated)
3. Gate computation (sigmoid projection)
4. Fused: dequant_base + lora_correction in one kernel
5. **TEST: Memory Fabric produces correct output, adapters influence generation**

### Phase 4: Optimization (make it fast)
1. AVX-512 matmul (2x over AVX2 on supported hardware)
2. Cache-optimal tiling (L1/L2/L3 aware)
3. Prefetch hints in assembly
4. Thread parallelism (one thread per transformer block layer)
5. KV cache with paged allocation (no reallocation during generation)
6. **BENCHMARK: tokens/second vs llama.cpp on same model**

### Phase 5: ARM (mobile/edge)
1. Port all kernels to ARM64 NEON
2. ARM SVE kernels for newer chips (Apple M-series, Snapdragon)
3. Memory-constrained mode (streaming layers from disk)
4. **TEST: runs on Raspberry Pi 5 / phone-class hardware**

### Phase 6: Voice + Interface
1. Audio I/O bridge (ALSA/CoreAudio/WASAPI)
2. WebSocket server for harness communication
3. Hot-reload: load new weights without restarting
4. **TEST: Lila speaks and listens in real-time**

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Language | Assembly + C | Maximum speed, zero overhead |
| Weight format | Custom (FigQuant INT4) | Optimized for our fused kernels |
| Memory model | mmap + arena allocator | Zero-copy loading, no malloc fragmentation |
| Threading | pthread (1 thread per layer group) | Simple, predictable, no framework |
| Build system | Makefile with arch detection | No cmake/meson complexity |
| Architecture dispatch | Runtime CPU feature detection | One binary runs on any x86/ARM |

## Performance Targets

| Metric | Target | Why |
|---|---|---|
| Tokens/sec (4B, INT4, CPU) | >30 tok/s | Conversational speed |
| Tokens/sec (4B, INT4, GPU) | >100 tok/s | Real-time interaction |
| First token latency | <200ms | Feels instant |
| Memory (4B, INT4) | <3 GB | Runs on 4GB RAM machines |
| Binary size | <1 MB | Minimal footprint |
| Startup time | <500ms | Near-instant boot |

## Experiments Needed

Before writing final kernels, we need to test:

1. **Tiling strategy** — What tile size gives best L1/L2 cache hit rate?
   Test: 64×64, 128×128, 256×256 tiles, measure throughput per architecture
   
2. **INT4 dequant placement** — Dequant to register (per-element) vs dequant to L1 (per-tile)?
   Test: both approaches, measure memory bandwidth vs compute utilization

3. **LoRA fusion overhead** — Is fused (base+LoRA in one kernel) actually faster than separate?
   Test: fused vs split, across different LoRA ranks (4, 8, 16, 32)

4. **Thread scaling** — How many threads before diminishing returns on 4/8/16 core machines?
   Test: 1, 2, 4, 8, 16 threads on matmul of different sizes

5. **ARM NEON vs SVE** — SVE has variable vector length. Is it worth the complexity?
   Test: same kernel in NEON (128-bit fixed) vs SVE (scalable), measure on M-series

6. **Memory layout** — Row-major vs column-major vs tiled storage for weights?
   Test: all three, measure matmul throughput (cache line utilization)

7. **Codebook in register** — FigQuant's 16-value codebook fits in one 512-bit register.
   Test: keep codebook permanently in zmm register vs reload per group

## Dependencies

- NASM or GAS (assembler)
- GCC or Clang (C compiler, for runtime/interface)
- No other dependencies. No libraries. No frameworks. Pure code.

## Relationship to Little Fig

```
Little Fig (Python, runs offline)
    │
    │ TRAINS the model
    │ Produces: model weights (safetensors)
    │
    ▼
format/convert.py
    │
    │ CONVERTS to Lila format
    │ Produces: .lila binary (INT4 packed with FigQuant)
    │
    ▼
Lila Engine (assembly + C, runs always)
    │
    │ LOADS and RUNS the model
    │ Pure machine code, no Python
    │
    ▼
LILA (the intelligence)
```

---

*Private. Not open source. Built by Sammie for Sammie.*
