# Lila Inference Engine

Custom assembly + C inference engine with the **ASI format** — Active System Image.

## What's an .asi file?

One file that IS Lila. Not just weights — everything:

```
lila.asi
├── Model weights (FigQuant INT4)       — general intelligence
├── Memory Fabric (5 LoRA adapters)     — A Thousand Pearls
├── Tokenizer (BPE vocab + merges)      — how she reads
├── Bytecode kernels (LilaVM)           — portable compute
├── Harness (tool definitions)          — what she can do
├── Personality (state + identity)      — who she is
└── Metadata (version, hashes)          — integrity
```

Load it. She runs. No Python. No frameworks. Any CPU with enough RAM.

## Quick Start

```bash
# Build
make asi

# Pack a model into .asi format
python asi/pack_asi.py --model google/gemma-3-4b-it --output lila.asi

# Run her
./lila-asi lila.asi

# Inspect an .asi file
./lila-asi lila.asi --info

# Benchmark
./lila-asi lila.asi --bench

# Hot-reload adapters (after training)
./lila-asi lila.asi --reload lila_trained.asi
```

## Training with Little Fig

```python
from asi.asi_train import AsiTrainer

# Load the .asi
trainer = AsiTrainer("lila.asi")
trainer.load_for_training()

# Initialize adapters for a namespace
trainer.init_all_adapters(namespace=1, rank=8)  # episodic

# After training, save back
trainer.save("lila_trained.asi")
```

Only the Memory Fabric section changes. Everything else stays frozen.

## Architecture

```
┌─────────────────────────────────────┐
│           .asi file (mmap'd)        │
├─────────────────────────────────────┤
│                                     │
│  ┌───────────┐   ┌──────────────┐  │
│  │  Weights  │   │ Memory Fabric│  │
│  │ (frozen)  │   │ (trainable)  │  │
│  └─────┬─────┘   └──────┬───────┘  │
│        │                 │          │
│  ┌─────┴─────────────────┴───────┐  │
│  │     Inference Engine           │  │
│  │  (native kernels OR LilaVM)   │  │
│  └────────────────────────────────┘  │
│                                     │
│  ┌──────────┐  ┌─────────────────┐  │
│  │Tokenizer │  │   Personality   │  │
│  └──────────┘  └─────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐   │
│  │    Harness (tool execution)  │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Portability (LilaVM)

The .asi file contains portable bytecode for all compute kernels.
When native assembly isn't available, the LilaVM interprets or JIT-compiles
the bytecode on whatever CPU it's running on:

- x86_64 → uses AVX2/AVX-512 native kernels (fastest)
- ARM64 → uses NEON/SVE native kernels
- Anything else → LilaVM bytecode interpreter (universal fallback)

Same .asi file runs everywhere. No recompilation.

## File Format

See `asi/ASI_FORMAT.md` for the complete binary specification.

## Directory Structure

```
engine/
├── asi/                    # Active System Image format
│   ├── ASI_FORMAT.md      # Binary format specification
│   ├── asi.h              # C header (structs, API)
│   ├── asi_runtime.c      # Runtime: load, boot, generate
│   ├── asi_cli.c          # CLI: interactive chat from .asi
│   ├── lilavm.h           # Portable bytecode VM (header)
│   ├── lilavm.c           # VM interpreter + pre-built kernels
│   ├── pack_asi.py        # Python: HF model → .asi packer
│   └── asi_train.py       # Python: Little Fig training interface
├── kernels/               # Native assembly kernels
│   ├── x86_64/            # AVX2/FMA hot-path code
│   └── arm64/             # NEON kernels
├── runtime/               # C runtime (model, inference, tokenizer)
├── format/                # Legacy .lila format converter
├── interface/             # Legacy CLI
└── Makefile               # Builds both lila-engine and lila-asi
```

## Status

- [x] Binary format specification (ASI_FORMAT.md)
- [x] C header + structs (asi.h)
- [x] Runtime loader (asi_runtime.c)
- [x] LilaVM bytecode interpreter (lilavm.c)
- [x] Pre-built portable kernels (matvec, rmsnorm, silu, softmax, rope)
- [x] .asi packer (pack_asi.py)
- [x] Training interface (asi_train.py)
- [x] CLI entry point (asi_cli.c)
- [x] Makefile integration
- [ ] Full weight parsing from mmap'd .asi
- [ ] JIT compilation (bytecode → native at load time)
- [ ] Encrypted .asi support
- [ ] RISC-V kernels

---

*One file. Everything. She thinks.*
