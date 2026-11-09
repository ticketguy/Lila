# Discovery record

## Existing path

The Windows app already used a persistent Raw Input window with background
input, sent authenticated requests to the loopback API, and controlled voice
sessions through `/voice/start`, `/voice/stop`, and `/voice/cancel`. Its voice
session controller serialized requests and associated a session with the
physical key identity. The previous settings accepted arbitrary single virtual
keys but had no modifier-chord model or safety validation.

The Linux implementation in `src/core/voice.py` already captured mono 16 kHz
WAV with `parecord`, bounded capture and ASR durations, ran the existing CPU
Parakeet CLI, stored the original transcript, and sent that exact text to
LilaCore. `src/daemon/service.py` exposed the authenticated API on
`127.0.0.1:8765`. The user systemd unit supplied the headless service process.

## Findings and changes

- Reused the Raw Input listener, API client, voice state machine, WSL client,
  `parecord`, Parakeet and user systemd service.
- Added a shared validated `ShortcutBinding` policy and modifier-aware key
  interpretation. Unsupported keys are rejected rather than passed through as
  arbitrary global shortcuts.
- Changed the microphone default from the machine-specific `RDPSource` to the
  WSL/PulseAudio default source. `LILA_VOICE_CAPTURE_DEVICE` can select a named
  source for a particular machine.
- Made voice asset paths repository-relative for clean clones, with
  `LILA_VOICE_HOME`, `LILA_VOICE_ASR_EXECUTABLE` and `LILA_VOICE_ASR_MODEL`
  overrides. Recordings now default to the user's XDG data directory or
  `~/.local/share/lila/recordings`.
- Added explicit Windows publish and Linux service activation scripts. Neither
  enables Windows startup implicitly; the tray app's Settings checkbox is an
  explicit opt-in.

## Hardware and compatibility notes

The ThinkPad T450s Fn+P behavior is a hardware observation supplied for this
work: it produces a short Pause-key pulse. Raw Input sees Windows virtual-key,
scan-code, E0/E1 and device information; it does not expose a generic physical
Fn key on every laptop. A key that produces no stable Raw Input event cannot be
configured by this client. Right Ctrl and Fn+P were reported working on the
T450s during prior use, but this code revision has not been independently
retested on that device in this implementation run.

## Compatibility findings

The service still requires a Linux user systemd manager, `parecord`, Python
dependencies, Parakeet executable and model. The activation script checks these
instead of installing a different ASR engine or enabling systemd lingering.
WSL audio source names differ by machine, so the default source is used unless
an explicit per-user override is needed.
