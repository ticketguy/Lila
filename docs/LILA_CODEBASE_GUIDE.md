# Lila Codebase Guide

This guide describes the active Python runtime and the separate engine
subproject. The checked-out tree contains user work and backups; do not remove
or overwrite files simply because they are not described here.

## Runtime paths

### Typed CLI

`lila.py` → `LilaCore.boot()` → model and harness initialization →
`LilaCore.think(text)` → model output → `HarnessExecutor.process()` when a
tool call is present → tool results return to LilaCore → normal response.

### Speech CLI

`lila.py --voice` → `LilaVoice.start_listening()` → `parecord` capture →
`ParakeetTranscriber.transcribe()` → raw transcript saved and passed unchanged
to `_handle()` → `LilaCore.think(transcript)` → normal response.

### Daemon

`python -m src.daemon.service` → `LilaDaemon.start()` → one `LilaCore`.
Text, transcript, and authenticated HTTP input call `process_input()`, which
calls `LilaCore.think()`. `--headless --port 8765` starts the authenticated
loopback API before model loading so clients can poll readiness; the model is
loaded once per process. `/voice/start`, `/voice/stop`, `/voice/cancel`, and
`/voice/status` control one bounded WSL microphone session. Sensitive tools
have no owner-approval callback in the daemon and therefore fail closed.

### Optional Workstation tool

Set `LILA_WORKSTATION_ENABLED=1` to register `src.workstation.tools`. A tool
call is validated and authorized by HarnessExecutor → the tool loads
`WorkstationRegistry` from its fixed personal production path → schema/path
validation → bounded metadata response. The core runtime does not import or
require this personal extension by default. No new tool accepts a filesystem
path.

## Repository map

### Root

- `lila.py` — Python CLI, typed conversation loop, optional voice entry point,
  and CLI approval callback. Calls `LilaCore`; it is not an executor.
- `README.md` — project entry point and links to this documentation.
- `docs/` — architecture, codebase, voice, research-roadmap, and design
  documentation.
- `tests/` — focused automated tests for execution authority, workstation
  registry/tools, and voice input.
- `engine/` — separate C and assembly inference-engine/ASI-format project.
  `engine/asi/` contains ASI packing/conversion/VM code,
  `engine/runtime/` contains the C inference implementation,
  `engine/kernels/` contains architecture-specific assembly,
  `engine/interface/` contains its CLI, and `engine/tests/` contains engine
  tests. Its Makefiles and build scripts are local to that subproject; current
  Python LilaCore does not call it.

### `src/core/`

- `lilacore.py` — central model loading, conversation construction, tool
  registration, HarnessExecutor orchestration, response cleanup, and
  interaction logging. Its direct dependency includes the harness;
  workstation registration is optional and disabled by default. GGUF uses `llama-cpp-python` without a
  fallback; remote Transformers identifiers require explicit opt-in.
- `private_storage.py` — targeted POSIX permission and restrictive-umask
  helpers. It protects exact log paths and explicitly selected private files;
  securing a directory never chmods its contents.
- `runtime_lock.py` — cross-process lock shared by the CLI and daemon to
  prevent the supported entry points from loading simultaneous model copies.
- `voice.py` — audio capture, replaceable `Transcriber` protocol, Parakeet
  subprocess adapter, raw artifact persistence, error bounds, and shutdown of
  active capture. Called by the CLI and daemon.
- `personality.py` — personality/identity data classes. It is not part of the
  active `think()` execution pipeline.
- `__init__.py` — core package marker; runtime behavior is implemented in the
  modules above.

### `src/harness/`

- `tools.py` — `Tool`, `ToolArg`, `ToolResult`, the shared `ToolRegistry`,
  registration helpers, call parsing, and result formatting.
- `executor.py` — canonical executor, schema validation, authorization,
  result formatting, metadata logging, and execution history.
- `permissions.py` — centralized risk/mode policies and path-sensitive
  handling for filesystem reads.
- `system_tools.py` — filesystem, shell, schedule, web, and system tool
  implementations and their schemas. Registered through LilaCore; not called
  directly by CLI or daemon.
- `extended_tools.py` — network and hardware tool definitions. Registered
  through LilaCore and still subject to the central permission policies.
- `__init__.py` — package exports.

### `src/workstation/`

- `models.py` — frozen typed representations for workstation configuration,
  categories, workspaces, intake, software, taxonomy, and overview data.
- `registry.py` — fixed-path, read-only JSON loader and validator. Public
  queries include `load()`, `get_overview()`, `list_categories()`,
  `find_project()`, `get_project()`, path resolvers, and `project_exists()`.
- `tools.py` — six bounded, read-only Harness tools; workspace lookup accepts
  a registered name, not a path.
- `__init__.py` — public registry/model exports.

### `src/daemon/`

- `service.py` — daemon lifecycle and text/voice/optional authenticated
  loopback HTTP interfaces. It delegates requests to LilaCore, owns a
  duplicate-instance lock, and supports headless voice session control. The
  service unit template is in `deploy/systemd/`; it is installed only by the
  explicit activation script. The Windows tray client and global keyboard
  input implementation are in `windows/`.
- `__init__.py` — lazy daemon exports.

### `windows/`

- `Lila.Desktop/` — .NET 10 Windows Forms tray client, Raw Input keyboard
  diagnostic, Credential Manager access, and WSL/API client. It sends text and
  voice session controls to the existing loopback API; it contains no model,
  ASR, or tool-execution implementation.
- `Lila.Desktop.Core/` — platform-neutral API models/client, WSL process
  orchestration, keyboard-event interpretation, and voice-session state
  machine.
- `Lila.Desktop.Tests/` — unit tests for key interpretation, start/release
  ordering, cancellation, API authentication/errors/timeouts, and WSL status.
- `README.md` — Windows build, credential, diagnostic, and run instructions.

### Other Python packages

- `src/__init__.py` — source package initializer.
- `src/cognitive/__init__.py` and `src/training/__init__.py` — package markers.
- `src/training/machine_lang.py` — machine-language corpus generation.
- `src/training/system_control.py` — system-control training-example
  generation. These files are not invoked by the runtime interaction loop.
- `src/cognitive/fast_loop.py`, `consolidation.py`, `emergence.py` — cognitive
  experiments; not wired into LilaCore's active response path.
- `src/perception/__init__.py` — perception package marker; no active speech
  or image pipeline is implemented there.

## Workspace-level runtime data

In the original development checkout, the Git repository is nested under
`/home/ticket/workspaces/lila/source` and these assets live in its parent
workspace. These paths describe that machine's layout, not requirements for a
migrated checkout; see [`voice-input/portability.md`](./voice-input/portability.md).

- `environment/.venv/` — the project-scoped Python environment used to launch
  and test the runtime.
- `models/` — default local language-model location. `LilaCore.boot()` first
  checks the workspace `models/qwen3-0.6b-q4_k_m.gguf`, then legacy
  `~/.lila/model.gguf` and `~/.lila/model` locations unless an explicit model
  path is supplied.
- `experiments/voice/` — external Parakeet and Whisper.cpp assets, plus
  retained WAV/transcript artifacts under `recordings/`.
- `logs/` — interaction history and Harness metadata logs. The interaction
  log includes user and assistant text; Harness logs redact tool payloads and
  outputs.
- `memory/` — the core creates this workspace directory, but persistent
  memory operations are not active in the runtime.

## Where changes belong

- Conversation/model behavior or tool orchestration: `src/core/lilacore.py`.
- Tool schema parsing or central execution/logging: `src/harness/`.
- Permission policy changes: `src/harness/permissions.py`, with focused
  tests; do not add interface-specific execution paths.
- Workstation registry validation: `src/workstation/registry.py` and
  `models.py`; do not duplicate registry contents in code/config.
- Workstation capabilities: `src/workstation/tools.py`, registered only via
  LilaCore and tested through HarnessExecutor.
- Speech capture/backend selection: `src/core/voice.py`; transcript delivery
  belongs in the existing CLI/daemon callback to LilaCore.
- Daemon lifecycle or HTTP behavior: `src/daemon/service.py`; preserve its
  loopback/authentication boundary and central execution path.
- C/ASI experimentation: `engine/`; do not assume it is the Python runtime.
- Documentation: update the relevant files in `docs/` whenever an implemented
  runtime behavior changes.

## Tests and environment

The focused suite currently consists of:

- `tests/test_execution_authority.py`
- `tests/test_workstation_registry.py`
- `tests/test_workstation_tools.py`
- `tests/test_voice_input.py`

Use the Python environment configured for the checkout. Speech executables,
models, recordings, transcripts, and main model weights are replaceable local
assets and should remain outside Git. See [`voice-input/build.md`](./voice-input/build.md)
for setup and [`voice-input/portability.md`](./voice-input/portability.md) for migration.
