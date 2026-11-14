# Portability and clean-clone setup

## Source and local data are separate

| Category | Location | Source control |
|---|---|---|
| Windows/daemon source, tests, scripts and docs | repository clone | commit normally after review |
| Windows publish | `%LOCALAPPDATA%\LilaDesktop\VoiceInput\app` | generated, installer-owned |
| Windows API credential | Windows Credential Manager | never committed |
| Windows settings | `%LOCALAPPDATA%\LilaDesktop\settings.json` | private local configuration |
| Linux service credential | `~/.config/lila/service.env` | private, mode 0600 |
| Optional Linux voice overrides | `~/.config/lila/voice.env` | private local configuration |
| Parakeet executable/model | `experiments/voice/whisper.cpp/...` or configured paths | replaceable local assets; ignored |
| Recordings/transcripts | `${XDG_DATA_HOME:-~/.local/share}/lila/recordings` | private personal data; never commit |
| API and inference logs | existing Lila workspace logs | private personal data |
| Build output | `bin/`, `obj/`, per-user publish directory | generated; never commit |

The repository `.gitignore` excludes voice recordings, the Parakeet build tree
and `.bin` model assets. Do not place credentials or personal data in a clone.
Recordings already stored in the previous voice-assets directory are left
there; changing the default does not move or delete existing audio/transcripts.

## Future personal-machine migration

1. Clone the repository in Windows for the desktop build and in the WSL Linux
   filesystem for efficient Linux runtime use. Keep the clones on the same
   reviewed revision.
2. Install/update WSL and create a supported Linux distribution. Confirm
   systemd and WSL audio input are available for the Linux user.
3. Install the existing Lila Python runtime dependencies. Keep language-model
   weights outside source control and choose their location with
   `LILA_MODEL_PATH` in `~/.config/lila/service.env`; the runtime also retains
   its existing default model lookup. This voice work does not download model
   weights or change model selection.
4. Build/provide the existing Parakeet CLI and its model. Place them under
   `experiments/voice/whisper.cpp/build/bin` and
   `experiments/voice/whisper.cpp/models`, or set explicit overrides.
5. Run `bash deploy/linux/activate-voice-input.sh` from the WSL clone. Resolve
   any prerequisite errors before continuing.
6. Run `deploy/windows/Activate-Lila.ps1` from PowerShell in the Windows clone.
   This publishes the self-contained x64 app but does not enable Windows
   startup.
7. Launch Lila, save the API credential in Windows Credential Manager and
   verify authenticated readiness. Opt into Windows sign-in startup only in
   Settings if wanted.
8. Complete the [acceptance checklist](acceptance.md), including physical
   keyboard focus and microphone tests.

## Per-machine voice overrides

The Linux service reads optional assignments from
`~/.config/lila/voice.env`:

```text
LILA_VOICE_HOME=/home/your-linux-user/lila-assets/voice
LILA_VOICE_ASR_EXECUTABLE=/absolute/path/to/parakeet-cli
LILA_VOICE_ASR_MODEL=/absolute/path/to/parakeet-model.bin
LILA_VOICE_CAPTURE_DEVICE=optional-pulse-source-name
LILA_RECORDINGS_DIR=/absolute/path/to/private-recordings
```

The language model location is independent of the voice ASR model. Set
`LILA_MODEL_PATH` in `service.env` to a local GGUF file or supported model
directory. Replace the file/path in place or update the setting when changing
models; the activation scripts neither copy nor remove model weights.

Omit `LILA_VOICE_CAPTURE_DEVICE` to use the PulseAudio default source. Keep the
file private and never commit it. The Linux activation script preserves an
existing `voice.env` and references it as an optional systemd environment file.

The Windows/WSL voice integration does not require Lila's optional Workstation
registry or any particular project/category layout. Those personal tools may
be configured differently or omitted on another machine.
