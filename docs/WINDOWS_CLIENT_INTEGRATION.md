# Windows Desktop Client Integration

The WinForms tray client lives in `../windows/`. The API remains bound to
`127.0.0.1:8765`; model inference, Parakeet, transcript handling and
HarnessExecutor remain in WSL. See the [Windows README](../windows/README.md)
for build and operation details.
The complete portable setup and troubleshooting guide is in
[`voice-input/README.md`](voice-input/README.md).

## Startup, lifecycle and connection

Windows sign-in startup is off by default. When the user opts in through
Settings, the app writes/removes a value in the current user's
`HKCU\Software\Microsoft\Windows\CurrentVersion\Run` key. It needs no
administrator privileges and creates no machine-wide entry. The `--startup`
launch hides the window to the tray. A per-user global named mutex prevents
duplicate tray instances across interactive sessions.

The tray process owns its WSL lifecycle:

1. Query `wsl.exe --list --running --quiet` for the configured distribution
   (`Ubuntu-24.04` by default). If stopped, start it with `/bin/true`.
2. Start one tracked `wsl.exe` child running `/bin/sh -c 'IFS= read -r _ || :'`.
   It waits on standard input. Windows owns its process handle, checks liveness
   on the five-second recovery cycle and starts it again after an unexpected
   exit. This managed process keeps WSL active while the tray app runs.
3. Start/check `lila.service` through the configured user's systemd manager.
   If `systemctl --user` cannot reach the manager, report a startup failure.
4. Poll authenticated `GET http://127.0.0.1:8765/health`. Report connected and
   enable shortcut handling only after `ready: true`.
5. On loss, cancel active voice safely and recheck WSL, helper, service and
   health. Do not replay text or voice requests with ambiguous outcomes.

Microsoft notes that systemd services do not keep a WSL instance alive
([WSL systemd documentation](https://learn.microsoft.com/windows/wsl/systemd/)).
The app stops `lila.service` and sends a shutdown line to its managed helper on
**Exit**. Closing the conversation window only hides the app in the tray.

The app does not enable user lingering, edit `/etc/wsl.conf` or `.wslconfig`,
change firewall/network settings or install machine-wide startup. The managed
helper is a Linux process, not a Linux login session. A newly started distro
may not expose a usable systemd user manager to a Windows-launched command
without a Linux login. This app does not enable lingering silently; confirm
`systemctl --user` availability after a clean Windows sign-in on the target
ThinkPad.

## Authentication and privacy

Every endpoint uses `Authorization: Bearer <token>`. Settings stores the token
as a generic Windows Credential Manager credential. Do not embed it in source,
arguments, URLs, crash reports, telemetry or logs. Never accept redirects to a
different host while forwarding the bearer token. Limit requests to the fixed
loopback origin and reject oversized JSON bodies.

The token authenticates a local client but is not per-user authorization. Any
process that can read it can submit text, use the microphone session and request
model tools. HarnessExecutor remains the tool execution and permission
authority. Since headless operation has no interactive approval surface,
sensitive tool actions continue to fail closed.

## Text and voice requests

- Health: `GET /health` returns readiness and service state.
- Text: `POST /chat` with `{"text":"..."}` follows the existing LilaCore path.
- Start voice: `POST /voice/start` with `{}` starts WSL microphone capture and
  returns an opaque `session_id`.
- Stop voice: `POST /voice/stop` with that ID stops `parecord`, runs Parakeet,
  retains the raw transcript and sends it unchanged to LilaCore. Windows does
  not implement ASR or rewrite transcripts.
- Cancel voice: `POST /voice/cancel` with the ID prevents transcript delivery.
- Status: `GET /voice/status` reports bounded session state.

Only one recording/transcription session may be active. Start while busy returns
HTTP 409. On an ambiguous client timeout, query `/voice/status` before retrying;
never replay a transcript after an uncertain stop. Microphone/ASR errors return
HTTP 502, unready service requests return HTTP 503.

POST bodies use UTF-8 JSON with known `Content-Length` and
`Content-Type: application/json; charset=utf-8`; chunked request framing is
rejected. Responses use snake_case fields such as `session_id` and
`elapsed_seconds`. The service remains loopback-only.

The service retains WAV and transcript files; LilaCore logs conversation text.
The app shows voice state, transcript and response. Retention/deletion controls
remain outside this integration.

## Global keyboard behavior

Raw Input uses `RIDEV_INPUTSINK` and a persistent hidden message target, with
device notifications enabled. It stays registered while the app is in the tray.
The dispatcher remains disabled until service health is ready.

Both defaults work simultaneously: **Right Ctrl** (`VK_RCONTROL`, `0xA3`) is
hold-to-talk; **Fn + P** (`VK_PAUSE`, `0x13`) is toggle-to-talk; **Esc** cancels
either session without submitting. Bindings can be disabled or changed
independently in Settings using the conservative whitelist in
[`voice-input/shortcuts.md`](voice-input/shortcuts.md). Right Ctrl arms for
180 ms and starts only if still held alone. A non-modifier down during arming
suppresses activation, preserving Ctrl+C/V/S. The non-modifier key is used
transiently and never logged or retained. Matching device/scan key-up stops
hold recording once. Pause starts on first down, ignores key-up, and submits on
the next physical press. Repeats are ignored; short duplicate pulses are
debounced. Pressing the other binding during a session does nothing. Device
removal and a 115-second client watchdog handle missing releases; the service
enforces its 120-second maximum.

The Start voice button remains click-to-toggle. The UI shows the active source,
recording state, cancel control and connection state.

## Verification boundary

Automated tests cover shortcut-state and WSL lifecycle policy. Actual registry
startup/removal, a clean Windows restart/sign-in, WSL user-manager availability,
managed helper recovery, background Raw Input, Right Ctrl holds of 1/3/5/10
seconds, Fn + P toggle timing, Esc, ordinary Ctrl shortcuts, other-app focus,
and exit cleanup require a hands-on run on the ThinkPad. Do not claim hardware
verification until that run succeeds.
