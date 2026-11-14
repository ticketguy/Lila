# Lila

Lila is a local Python assistant runtime with configurable local model
backends, a centralized permission-aware tool executor, optional workstation
tools, and push-to-talk speech input. The Python runtime is the active
integration path; the C engine under `engine/` is separate and is not
currently loaded by `LilaCore`.

## Run

From the repository root, using the existing project environment:

```bash
../environment/.venv/bin/python lila.py
../environment/.venv/bin/python lila.py --voice
```

Voice mode is terminal-based push-to-talk: press Enter to record, speak, then
press Enter to finish; type `q` to exit. A local model must be configured for
Lila to generate responses. `--check` exercises startup without entering a
conversation loop.

Remote Hugging Face model identifiers are blocked by default. Deliberately
allow one only by specifying both `--model <identifier>` and
`--allow-remote-model`; local model directories remain local-only.

The daemon is an optional runtime entry point:

```bash
../environment/.venv/bin/python -m src.daemon.service --no-voice
```

This mode uses an interactive terminal. The optional authenticated HTTP API is
disabled by default; do not enable it or expose it beyond loopback without an
explicitly reviewed design.

## Current boundaries

- Every model tool call goes through `HarnessExecutor`, schema validation, and
  the central permission policy.
- Workstation tools are an optional personal extension. Enable them with
  `LILA_WORKSTATION_ENABLED=1`; they use the fixed Command Center registry
  described in the architecture guide and are disabled by default.
- Windows tray and global keyboard voice input are available through the
  separate .NET desktop client. See [voice input setup](docs/voice-input/README.md).
- Memory Fabric, autonomous learning, and self-modification are not active
  features.
- Speech uses the Parakeet command-line backend. Captured audio and original
  transcripts are retained outside this repository.

## Documentation

- [System architecture](docs/LILA_ARCHITECTURE.md)
- [Codebase guide](docs/LILA_CODEBASE_GUIDE.md)
- [Voice architecture](docs/LILA_VOICE_ARCHITECTURE.md)
- [Voice research roadmap](docs/LILA_VOICE_RESEARCH_ROADMAP.md)
- [System design decisions](docs/LILA_SYSTEM_DESIGN.md)
