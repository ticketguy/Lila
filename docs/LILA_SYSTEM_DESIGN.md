# Lila System Design

This document records the current architecture and remaining operational
limits. Earlier design proposals are preserved in the voice research roadmap;
the implementation and migration guides are the operational source of truth.

## Implemented decisions

| Area | Current implementation |
|---|---|
| Orchestration | `LilaCore` loads the configured local model and owns conversation orchestration. |
| Tool authority | `HarnessExecutor` validates schemas and applies centralized permission policy to every model tool call. |
| Permissions | Approval-required operations fail closed when no approval callback exists; the daemon has no approval UI. |
| Workstation | Personal registry tools are optional and disabled by default with `LILA_WORKSTATION_ENABLED=1`. They retain fixed paths for this user's Command Center and are not required by LilaCore. |
| Voice pipeline | Linux `parecord` capture feeds the external Parakeet CLI; the unmodified transcript goes through the same LilaCore and Harness path. |
| Windows input | The .NET tray client owns global keyboard input: Right Ctrl hold-to-talk, Pause/Fn+P toggle-to-talk, and Escape cancellation. See the shortcut guide for the supported configuration limits. |
| API | Authenticated headless HTTP binds to loopback and supports text and bounded voice-session control. API credentials are stored outside Git and in Windows Credential Manager on the desktop. |
| Lifecycle | The user-systemd unit is installed by explicit Linux activation. The Windows client supervises WSL while running; systemd alone does not keep WSL alive. |
| Model loading | `LILA_MODEL_PATH` configures the main GGUF or local Transformers model separately from Parakeet's executable/model configuration. Weights remain external. |

## Privacy and security limits

- Interaction logs retain user and assistant text, including voice transcripts.
- Harness logs redact tool payloads and results; this does not redact the
  separate interaction history.
- Private runtime directories target POSIX mode `0700`; interaction and
  Harness logs and Lila-owned audio/transcript files target `0600`. These
  permissions are access controls, not encryption, and depend on filesystem
  support.
- Recordings, transcripts, and interaction history are not automatically
  rotated or deleted. Treat them as private user data and keep them outside Git.
- Windows and Linux activation are explicit user actions. Linux activation
  installs a user-level service; it does not enable lingering, change `/etc`,
  alter WSL configuration, or expose the API to LAN. Windows deployment does
  not register automatic startup unless the user enables it in settings.
- The desktop credentials are stored in Windows Credential Manager. The Linux
  service token belongs in the user's private config directory, not source
  control.

## Migration and operational limits

- The core runtime is portable across checkout paths and usernames. The
  optional Workstation extension is not generic: its registry and trusted
  workspace paths are intentionally specific to this personal setup.
- Main model weights, Parakeet executable/model, Python environment, API token,
  recordings, and logs are separate from the source checkout. Recreate or
  copy them using the procedures in `voice-input/build.md` and
  `voice-input/portability.md`; the repository does not bundle these assets.
- WSL and microphone behavior must be checked on each target computer. The
  current ThinkPad T450s observation is that Fn+P emits a short Pause pulse,
  which is suitable for toggle mode rather than hold mode. Other keyboards
  may expose different events.
- The C/assembly `engine/` is a separate experimental inference project and
  is not loaded by the active Python LilaCore.
- Parakeet is the only wired ASR backend. TTS, streaming ASR, VAD, automatic
  transcript cleanup, and autonomous learning are not active features.
