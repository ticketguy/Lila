# .asi Format Specification — Lila Active System Image

## Overview

A `.asi` file is a **complete, self-contained intelligence**. Not just weights — it's the entire system:
model weights + memory adapters + compute kernels + tokenizer + personality + runtime.

Load one file. Lila runs. No Python. No frameworks. No dependencies.

## Design Principles

1. **One file, everything** — Weights, adapters, tokenizer, kernels, harness, personality state
2. **Platform independent** — Compute kernels stored as portable bytecode (LilaVM ISA)
3. **mmap-friendly** — Sections aligned to page boundaries for zero-copy loading
4. **Trainable** — Little Fig can extract adapters, train them, and repack
5. **Hot-reloadable** — Replace adapters without restarting the engine
6. **Streamable** — Section table at the front allows lazy loading of individual sections

## Binary Layout

```
┌─────────────────────────────────────────────────────────┐
│  ASI Header (64 bytes)                                  │
├─────────────────────────────────────────────────────────┤
│  Section Table (variable, 32 bytes per entry)           │
├───────────────────────────── page-aligned ──────────────┤
│  Section 0: MODEL_CONFIG (architecture params)          │
├───────────────────────────── page-aligned ──────────────┤
│  Section 1: WEIGHTS (FigQuant INT4 packed layers)       │
├───────────────────────────── page-aligned ──────────────┤
│  Section 2: MEMORY_FABRIC (5 namespace LoRA adapters)   │
├───────────────────────────── page-aligned ──────────────┤
│  Section 3: TOKENIZER (BPE vocab + merge rules)         │
├───────────────────────────── page-aligned ──────────────┤
│  Section 4: BYTECODE (portable compute kernels)         │
├───────────────────────────── page-aligned ──────────────┤
│  Section 5: HARNESS (tool definitions + execution DAG)  │
├───────────────────────────── page-aligned ──────────────┤
│  Section 6: PERSONALITY (state vector + identity)       │
├───────────────────────────── page-aligned ──────────────┤
│  Section 7: METADATA (creation time, version, hash)     │
└─────────────────────────────────────────────────────────┘
```

## Header (64 bytes)

| Offset | Size | Field         | Description                         |
|--------|------|---------------|-------------------------------------|
| 0      | 4    | magic         | `0x41534921` ("ASI!")               |
| 4      | 4    | version       | Format version (1)                  |
| 8      | 4    | flags         | Bit flags (see below)               |
| 12     | 4    | n_sections    | Number of sections                  |
| 16     | 8    | total_size    | Total file size in bytes            |
| 24     | 8    | section_table_offset | Offset to section table      |
| 32     | 32   | identity_hash | SHA-256 of personality section      |

### Flags

| Bit | Meaning                               |
|-----|---------------------------------------|
| 0   | Has Memory Fabric (trainable)         |
| 1   | Has bytecode kernels                  |
| 2   | Has harness (tool-capable)            |
| 3   | Has personality state                 |
| 4   | Weights are FigQuant INT4             |
| 5   | Tokenizer uses BPE                    |
| 6   | Hot-reload supported                  |
| 7   | Encrypted (household key required)    |

## Section Table (32 bytes per entry)

| Offset | Size | Field       | Description                       |
|--------|------|-------------|-----------------------------------|
| 0      | 4    | type        | Section type enum                 |
| 4      | 4    | flags       | Section-specific flags            |
| 8      | 8    | offset      | Byte offset from file start       |
| 16     | 8    | size        | Section size in bytes             |
| 24     | 8    | checksum    | CRC-64 of section data            |

### Section Types

```
ASI_SECTION_MODEL_CONFIG   = 0x01
ASI_SECTION_WEIGHTS        = 0x02
ASI_SECTION_MEMORY_FABRIC  = 0x03
ASI_SECTION_TOKENIZER      = 0x04
ASI_SECTION_BYTECODE       = 0x05
ASI_SECTION_HARNESS        = 0x06
ASI_SECTION_PERSONALITY    = 0x07
ASI_SECTION_METADATA       = 0x08
```

## Section Details

### MODEL_CONFIG (0x01)

Model architecture parameters. Same as the existing .lila header but extended:

```c
struct AsiModelConfig {
    uint32_t n_layers;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t head_dim;
    float    rope_theta;
    float    rms_norm_eps;
    uint32_t quant_type;        // 0=FP32, 1=FP16, 2=INT8, 3=INT4_FIGQUANT
    uint32_t group_size;        // Quantization group size (128)
};
```

### WEIGHTS (0x02)

FigQuant INT4 packed model weights. Layout:

```
[Token Embedding — FP32: vocab_size × hidden_size × 4 bytes]
[Layer 0]
  [q_proj: rows, cols, codebook[16], scales[n_groups], packed_indices]
  [k_proj: ...]
  [v_proj: ...]
  [o_proj: ...]
  [gate_proj: ...]
  [up_proj: ...]
  [down_proj: ...]
  [input_layernorm: hidden_size × 4 bytes FP32]
  [post_attention_layernorm: hidden_size × 4 bytes FP32]
[Layer 1]
  ...
[Final Norm — FP32: hidden_size × 4 bytes]
[LM Head — FP32 or 0xFFFFFFFF if tied]
```

### MEMORY_FABRIC (0x03)

The 5 namespace adapters (A Thousand Pearls). Trainable section.

```
[Fabric Header]
  uint32_t n_namespaces;       // 5
  uint32_t n_layers;           // matches model
  
[Per namespace × per layer]
  uint32_t rank;               // LoRA rank
  uint32_t in_features;
  uint32_t out_features;
  float    gate;               // [0,1] activation
  float[]  A;                  // [in_features × rank] FP32
  float[]  B;                  // [rank × out_features] FP32
```

Little Fig writes ONLY this section when training. The rest stays frozen.

### TOKENIZER (0x04)

Complete tokenizer. No external files needed.

```
[Tokenizer Header]
  uint32_t vocab_size;
  uint32_t n_merges;
  uint32_t bos_id;
  uint32_t eos_id;
  uint32_t pad_id;

[Vocab: vocab_size entries]
  uint16_t token_len;
  char[]   token_bytes;        // UTF-8

[Merges: n_merges entries]
  uint32_t left_id;
  uint32_t right_id;
  uint32_t merged_id;
  float    priority;           // BPE merge priority
```

### BYTECODE (0x05)

Portable compute kernels in LilaVM bytecode. This is the key innovation —
instead of shipping x86/ARM assembly, we ship bytecode that gets JIT'd on load.

```
[Bytecode Header]
  uint32_t n_kernels;
  uint32_t vm_version;         // LilaVM ISA version

[Kernel Table: n_kernels entries]
  uint32_t kernel_id;          // e.g. KERNEL_MATMUL, KERNEL_RMSNORM
  uint32_t bytecode_offset;
  uint32_t bytecode_size;
  uint32_t flags;              // SIMD hints, parallelism hints

[Kernel Bytecode]
  // LilaVM instructions (see lilavm.h)
```

When a native kernel is available (detected at runtime), it's used instead.
Bytecode is the portable fallback that guarantees it runs EVERYWHERE.

### HARNESS (0x06)

Tool execution capabilities embedded in the intelligence.

```
[Harness Header]
  uint32_t n_tools;
  uint32_t n_patterns;

[Tool Definitions]
  // Each tool: name, description, argument schema, execution pattern
  
[Execution Patterns]
  // DAG of tool composition patterns (stored as bytecode)
```

### PERSONALITY (0x07)

Emergent identity state. Updated as Lila grows.

```
[Personality Header]
  uint64_t interactions_count;
  uint64_t last_active;        // Unix timestamp
  uint32_t state_dim;          // Personality vector dimension

[State Vector]
  float[]  personality;        // Learned personality embedding

[Identity Fragments]
  // Key-value pairs of identity facts
  // "name" → "Lila"
  // "family" → "Sammie's household"
  // etc.
```

### METADATA (0x08)

```
  uint64_t created_at;
  uint64_t modified_at;
  char[64] creator;            // "Little Fig v1.0"
  char[64] base_model;         // "gemma-3-4b-it"
  uint8_t[32] content_hash;   // SHA-256 of all other sections
```

## Operations

### Load (boot Lila)
```
1. mmap the .asi file
2. Verify magic + checksum
3. Parse section table
4. JIT-compile bytecode kernels (or use native if detected)
5. Map weight section (zero-copy)
6. Load tokenizer into memory
7. Load personality state
8. Initialize KV cache
9. Ready for inference
```

### Train (Little Fig writes adapters)
```
1. Open .asi file
2. Read MODEL_CONFIG + WEIGHTS (frozen)
3. Read MEMORY_FABRIC (current adapters)
4. Train new adapter weights
5. Write ONLY the MEMORY_FABRIC section back
6. Update METADATA timestamps
7. Update header checksum
```

### Hot-reload (update adapters while running)
```
1. Lila is running from mmap'd .asi
2. New .asi arrives (after training)
3. mmap new file
4. Verify magic + version match
5. Swap MEMORY_FABRIC pointers atomically
6. munmap old file
7. Zero downtime
```

## File Extension

`.asi` — **Active System Image**

The name says it all. It's not a model file. It's not a weight dump.
It's an active system. Load it and it thinks.

## Compatibility

The .asi format is designed to be:
- **Forward compatible** — new section types can be added without breaking old loaders
- **Backward compatible** — old .asi files load in new engines (missing sections use defaults)
- **Convertible** — existing .lila files convert to .asi by adding sections

## Size Estimates (Gemma 4B)

| Section          | Size        |
|-----------------|-------------|
| MODEL_CONFIG    | 48 bytes    |
| WEIGHTS (INT4)  | ~2.2 GB     |
| MEMORY_FABRIC   | ~50 MB      |
| TOKENIZER       | ~4 MB       |
| BYTECODE        | ~200 KB     |
| HARNESS         | ~50 KB      |
| PERSONALITY     | ~10 KB      |
| METADATA        | ~256 bytes  |
| **TOTAL**       | **~2.3 GB** |

One file. Everything. On a 4 GB machine.
