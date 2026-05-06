# 🌸 Lila

**Private family ASI assistant.** Not a chatbot. Not a product. A persistent intelligence that serves Sammie and family.

She runs on **Gemma 4B** trained with [Little Fig](https://github.com/ticketguy/littlefig) — memory lives in her weights via the Memory Fabric (A Thousand Pearls). She speaks, listens, remembers, learns, and grows.

## Quick Start

```bash
python lila.py              # Text mode
python lila.py --voice      # Voice mode
```

## Architecture

```
Lila (Gemma 4B, trained with Little Fig)
├── Cognitive Core (frozen INT4) — general intelligence
├── Memory Fabric (5 namespace adapters) — A Thousand Pearls
│   ├── personal/   — Sammie facts, family, preferences
│   ├── episodic/   — conversation history, events  
│   ├── wiki/       — verified permanent knowledge (LKB)
│   ├── schedule/   — time-sensitive info
│   └── contested/  — unresolved conflicts
├── Machine Language — assembly, binary protocols in weights
└── Personality — emergent from interaction, never configured
```

## Structure

```
lila.py                 # Start Lila
src/
├── core/               # LILA HERSELF
│   ├── lilacore.py     # Central intelligence, inference loop
│   ├── voice.py        # Speech I/O
│   └── personality.py  # Emergent identity
├── cognitive/          # HER THINKING
│   ├── fast_loop.py    # Reactive: input → respond
│   ├── consolidation.py # Medium: promote memories
│   └── emergence.py    # Slow: reflection, growth
├── harness/            # HER HANDS (tool execution)
├── perception/         # HER SENSES (listening, monitoring)
└── training/           # HOW SHE LEARNS
    └── machine_lang.py # Assembly/binary training corpus
```

## Principles

1. **Completion over reporting** — Execute fully, don't stop when stuck
2. **Memory is cognition** — Intelligence lives in the quality of A Thousand Pearls
3. **Personality is emergent** — Never predefined, grows from relationship
4. **LilaCore is the self** — The thread of identity through everything
5. **Nothing leaves the household** — All data stays private

---

*Private. Not open source. Not for commercial use.*
*Built by Sammie.*
