# 🌸 Lila

**Private family ASI assistant.** One file. Full system control. Runs anywhere.

## Run

```bash
# Convert your GGUF model to .asi
pip install gguf numpy
python engine/asi/gguf_to_asi.py --gguf gemma-4-E4B-it-Q4_K_M.gguf --output lila.asi

# Build the engine (Windows)
cd engine
build.bat

# Start Lila
lila-asi.exe lila.asi
```

## What is lila.asi?

One file containing everything:

```
lila.asi
├── Model weights (INT4 quantized)
├── Memory Fabric (5 namespace LoRA adapters)
├── Tokenizer (embedded, no external files)
├── Bytecode kernels (portable, runs on any CPU)
├── Harness (tool execution capabilities)
└── Personality (state + identity)
```

Load it. She runs. No Python at runtime. No frameworks. No internet.

## Architecture

```
lila-asi.exe lila.asi
    │
    ├── Loads weights (mmap, zero-copy)
    ├── Parses all sections
    ├── Boots LilaVM (portable compute)
    ├── Initializes KV cache
    └── Ready — type or speak, she responds

She can:
    ├── Execute shell commands
    ├── Control hardware (GPIO, I2C, SPI, serial, USB)
    ├── Manage network (TCP, UDP, SSH, WiFi, HTTP)
    ├── Control the OS (volume, power, processes, services)
    ├── Write code and scripts
    ├── Remember (Memory Fabric namespaces)
    └── Listen always (voice loop, daemon mode)
```

## Training

Trained with [Little Fig](https://github.com/ticketguy/littlefig). Only the Memory Fabric adapters update — base weights stay frozen.

```python
from engine.asi.asi_train import AsiTrainer

trainer = AsiTrainer("lila.asi")
trainer.load_for_training()
trainer.init_all_adapters(namespace=1, rank=8)
# ... train ...
trainer.save("lila.asi")  # Only adapters change
```

## Daemon Mode

```bash
python -m src.daemon.service
```

Boots dormant. Say "Lila" or "initialize" — she activates herself, detects the system, takes control.

## Structure

```
engine/                 # C inference engine + .asi format
├── asi/               # .asi format (packer, loader, VM, converter)
├── runtime/           # Transformer forward pass (C)
├── kernels/           # Native assembly (x86 AVX2, ARM NEON)
├── build.bat          # Windows build
└── Makefile           # Linux/macOS build

src/                   # Python layer (daemon, harness, training)
├── core/             # LilaCore, voice, personality
├── harness/          # Tool execution (bash, network, hardware, system)
├── daemon/           # Always-on background service
├── training/         # Training corpus (machine language, system control)
└── cognitive/        # Thinking loops (fast, medium, slow)
```

---

*Private. Built by Sammie.*
