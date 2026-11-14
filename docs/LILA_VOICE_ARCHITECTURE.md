# Lila Voice Architecture

## Implemented pipeline

```text
Microphone (configured PulseAudio source)
  → parecord
  → mono 16 kHz signed 16-bit WAV
  → Parakeet CLI
  → original stdout transcript
  → transcript file and existing Lila input callback
  → LilaCore.think()
  → HarnessExecutor, if the model requests a tool
  → normal Lila response
```

There is one assistant and one text-processing path. The transcript is not
corrected, paraphrased, or sent through a second model before `LilaCore`.

## Capture and interaction

`src/core/voice.py` contains `VoiceConfig`, the `Transcriber` protocol,
`ParakeetTranscriber`, and `LilaVoice`.

- **Observed environment:** capture was verified in this WSLg setup through
  PulseAudio using the input source `RDPSource`; `parecord` is available in
  the WSL environment. These are deployment facts, not universal hardware or
  platform requirements.
- **Current defaults:** `VoiceConfig.capture_command` is `parecord`,
  `capture_device` is configurable (the current WSL setup uses `RDPSource`),
  `sample_rate` is 16,000 Hz, and the maximum
  recording duration is 120 seconds. Sample rate is currently constrained to
  16 kHz by configuration validation.
- `parecord` is invoked for one completed utterance and writes a mono,
  signed 16-bit PCM WAV (`--channels=1`, `--format=s16le`,
  `--file-format=wav`).
- The CLI provides completed-utterance push-to-talk: press Enter to start,
  speak, then press Enter to stop; type `q` to quit.
- Captured files are saved outside the source repository. The default location
  is under the Lila workspace's `experiments/voice/recordings` directory and
  can be configured through `VoiceConfig`.
- A capture process is stopped on normal completion, cancellation, timeout,
  or shutdown.

The CLI remains terminal-driven. The Windows tray client provides global
Right Ctrl hold-to-talk, Pause/Fn+P toggle-to-talk, and Escape cancellation;
the Linux daemon provides the authenticated voice API. Headless daemon mode provides
authenticated `/voice/start`, `/voice/stop`, `/voice/cancel`, and
`/voice/status` routes. Start invokes the same configured capture; stop ends
capture, transcribes with Parakeet, and sends the returned transcript unchanged
through LilaCore. Cancellation prevents transcript delivery. See
[`LILA_SERVICE_OPERATIONS.md`](./LILA_SERVICE_OPERATIONS.md) for request shape
and [`voice-input/README.md`](./voice-input/README.md) for the Windows client.

## Transcription and backend boundary

`Transcriber.transcribe(audio_path: Path) -> str` is the replaceable backend
contract. `ParakeetTranscriber` invokes the existing
`experiments/voice/whisper.cpp/build/bin/parakeet-cli` with the Parakeet
TDT 0.6B Q4_K model, `--no-gpu`, and two CPU threads by default. The process
has a 180-second default timeout. The returned stdout is passed through
unchanged.

Each utterance starts a fresh `parakeet-cli` subprocess. The CLI loads the
configured Parakeet model for that invocation; Lila does not keep a resident
ASR model between recordings. `VoiceConfig.transcription_threads` defaults to
2 and `transcription_timeout_seconds` defaults to 180 seconds. These are
current configuration defaults chosen for the available CPU/RAM environment,
not universal Parakeet requirements. Both `parecord` and the Parakeet
executable must be installed/present in the runtime environment; no Python
speech-recognition package is used for this path.

The CLI saves the exact returned transcript beside its WAV as
`<recording>.transcript.txt`. Empty transcripts and missing assets are
reported as errors. Startup failures, non-zero ASR exits, capture errors,
invalid/empty WAV files, maximum-duration timeouts, and ASR timeouts have
explicit failure paths. A failed transcription does not delete the captured
WAV.

Whisper Small and Small Q5_0 assets are experimental research artifacts, not
active Lila backends. No Whisper transcription adapter is wired into the
runtime. Parakeet is the current backend. The remaining local voice model
inventory at the time of this documentation is:

| Asset | Bytes | Runtime status |
|---|---:|---|
| `ggml-small.bin` | 487,601,967 | Retained research asset; not wired |
| `ggml-small-q5_0.bin` | 175,209,663 | Retained research asset; not wired |
| `ggml-parakeet-tdt-0.6b-v3-q4_k.bin` | 415,611,879 | Active Parakeet model |

Parakeet executable and Whisper.cpp build remain outside the canonical source
repository. The small test-model fixtures in the Whisper.cpp checkout are test
artifacts, not the full Tiny/Base speech model files.

## Transcript delivery

In CLI mode, the callback in `lila.py` calls `lila.think(text)` with the
unmodified transcript and sends the normal response to `LilaVoice.speak()`.
That method currently prints text; it is not TTS.

The daemon's injected transcript handler calls `process_input(transcript)`,
which calls the same `LilaCore.think()` method. Tool calls then follow the
existing HarnessExecutor and permission flow.

## Persistence and privacy

Audio and transcript artifacts are intentionally retained outside Git in the
configured recordings directory (the original development checkout uses
`<workspace>/experiments/voice/recordings/`). Conversation entries are stored
in `<workspace>/logs/interactions.jsonl`; Harness execution metadata is stored
in `<workspace>/logs/harness/executions.jsonl`. Conversation
entries contain user and assistant text, so spoken transcripts also persist
there. Harness logs contain execution metadata and redaction placeholders, not
tool argument or result contents.

The runtime secures its data directories to POSIX mode `0700`; interaction and
Harness logs and Lila-owned audio/transcript files are protected individually
to `0600`. The shared recordings directory is not traversed to chmod its
contents: unrelated benchmark prompts, ASR outputs, and research files keep
their current modes. New audio capture inherits a restrictive `077` umask and
the resulting Lila-owned file is explicitly secured.

In the original WSL setup these paths are in the Linux filesystem. The
workspace location follows the checkout's parent layout; use the voice-input
migration guide to configure external locations on another computer.

Linux filesystem permissions are expected to support POSIX modes. Windows
mounted filesystems such as `/mnt/c` can have different mode behavior depending
on mount configuration. The runtime verifies effective modes and raises an
error when enforcement fails; it does not silently continue with unprotected
private output. Existing research files whose permissions were restricted by
an earlier implementation may remain at `0600`; their prior modes are unknown
and are not restored automatically.

These permissions restrict ordinary other-account access only when honored by
the filesystem. They are access controls, not encryption. Root, host
administrators, backups, and processes running as the same Unix account can
still access the data.

Recordings and transcripts have no automatic rotation or deletion. Retention,
encrypted backups, and whether future evaluation/training may use a recording
remain decisions requiring owner policy. Do not copy private audio or
transcripts into Git.

## Resource and replacement constraints

Only Parakeet is loaded for transcription, through a bounded CPU subprocess;
the Whisper models are not loaded concurrently. Thread count, timeout, model,
and executable are configured through `VoiceConfig`. A future backend can
implement `Transcriber` without changing transcript delivery or LilaCore.

Streaming, VAD, barge-in, echo cancellation, model selection UI, Whisper
integration, and TTS are not implemented.

## Service and Windows client

The tested capture source in the current WSL installation is `RDPSource`,
but source selection remains configurable. The .NET Windows tray client owns
global Raw Input and controls voice through the authenticated loopback API.
The defaults are Right Ctrl hold-to-talk, Pause/Fn+P toggle-to-talk, and Esc
cancellation. Windows startup remains opt-in. The client supervises a tracked
WSL keepalive child while it runs; systemd alone does not keep WSL alive.

The repository includes a user-systemd unit template and explicit activation
scripts. They do not install or start services until invoked. User lingering
is not enabled; if the systemd user manager cannot be reached after a fresh
Windows sign-in, verify the target distro's manager behavior before deciding
whether lingering is needed. Hardware acceptance, including physical keyboard
timings and post-restart service readiness, remains a ThinkPad test item.
See [`voice-input/README.md`](./voice-input/README.md) and
[`WINDOWS_CLIENT_INTEGRATION.md`](./WINDOWS_CLIENT_INTEGRATION.md).

Wake-word detection and TTS are not part of the current implementation.
