# Lila Background Service Operations

The repository includes a **user systemd unit template** at
`deploy/systemd/lila.service`. The explicit Linux activation script expands its
repository/Python/environment placeholders and installs the unit for the
current user. The unit starts one headless daemon; its process owns one
LilaCore/model for its lifetime and binds the authenticated API to
`127.0.0.1:8765`. See the [portable voice-input guide](voice-input/README.md).

## Prerequisites and configuration

The Linux activation script discovers the repository path and accepts an
explicit Python path. It requires the project's runtime environment,
`parecord`, Parakeet CLI and model to be installed already. It creates
`~/.config/lila/service.env` with a randomly generated `LILA_API_TOKEN` only
when the file does not exist; an existing credential is preserved. The token
must be at least 32 characters and the file is protected with mode 0600. Do not
pass tokens on command lines or put them in source/logs.
Set `LILA_MODEL_PATH` in that file to choose a local, replaceable GGUF or
supported model directory. Activation never copies or removes model weights.

## Install and operate

```bash
bash deploy/linux/activate-voice-input.sh
systemctl --user status lila.service
journalctl --user -u lila.service -f
systemctl --user restart lila.service
systemctl --user stop lila.service
```

Activation validates the systemd user manager, Python imports, `parecord`,
Parakeet executable/model and a usable config before writing the unit. It uses
the user's systemd manager and does not use sudo. Re-running it with an
unchanged install is safe. To remove only the unchanged unit created by the
script:

```bash
bash deploy/linux/activate-voice-input.sh --deactivate
```

The install marker includes the installed unit hash. If the unit was edited
after activation, deactivation preserves it. Credentials, voice overrides,
models, recordings and logs are never removed by deactivation.

User lingering (`loginctl enable-linger "$USER"`) starts the user's systemd
manager outside a Linux login session. It can be required if
`systemctl --user` is unavailable to the Windows-launched service command; it
is not needed to keep WSL alive while the desktop's supervised helper is
running. The inspected configuration previously reported `Linger=no`; it has
not been changed or revalidated here. Do not enable lingering as a substitute
for the Windows-owned WSL lifecycle, and do not claim startup works until the
target distro's user manager is verified after Windows sign-in. Lingering is a
persistent per-user setting with resource implications.

## WSL lifecycle and Windows tray ownership

The Windows tray app registers per-user startup under the current user's Run
key and starts only after Windows sign-in. While the tray process lives, it
starts/checks Ubuntu-24.04 and owns one tracked WSL child process that waits on
stdin. The Windows process polls its `wsl.exe` handle and restarts it if it
exits unexpectedly. Once WSL is available, the app starts/checks the user unit
and enables voice shortcuts only after authenticated health reports ready.

Microsoft documents that systemd services do not keep a WSL instance alive
([WSL systemd documentation](https://learn.microsoft.com/windows/wsl/systemd/)).
The managed process keeps a WSL process alive while the desktop tray is active;
it is not an untracked background shell. On tray **Exit**, the client cancels
voice, requests `systemctl --user stop lila.service`, sends the helper's
shutdown line, and waits for the Windows process handle to exit. Closing only
the conversation window does not stop the service/helper.

This design does not enable systemd lingering, modify `/etc/wsl.conf`,
`.wslconfig`, firewall/network settings, or install a system-wide service.
Lingering is not needed to keep WSL alive because the tray-owned child does
that. It may still be needed if this Ubuntu setup does not create a systemd user
manager for commands launched from Windows without a Linux login session. The
app reports failure in that case; validate the current distro's
`systemctl --user` path after a clean Windows sign-in before claiming
autostart-ready operation.

## API contract

All endpoints require `Authorization: Bearer <token>` and JSON POST bodies are
limited to 1,000,000 bytes. The server binds only to `127.0.0.1`.

| Method and path | Request | Result |
|---|---|---|
| `GET /health` or `/status` | — | Service status, readiness, voice availability and model label |
| `POST /chat` | `{"text":"..."}` | `{"response":"..."}` via the existing LilaCore |
| `GET /voice/status` | — | State, opaque session ID, elapsed time and bounded error |
| `POST /voice/start` | `{}` | Starts a single bounded microphone recording and returns session ID |
| `POST /voice/stop` | `{"session_id":"..."}` | Stops, transcribes, preserves raw transcript and sends it unchanged to LilaCore |
| `POST /voice/cancel` | `{"session_id":"..."}` | Cancels active capture/ASR without sending transcript to LilaCore |

Voice states are `idle`, `recording`, `transcribing`, `cancelling`,
`completed`, `cancelled`, or `failed`. Start while a session is active returns
HTTP 409; microphone/ASR failures return HTTP 502 and unready service requests
return HTTP 503. Clients should use the returned session ID for stop/cancel and poll
status after connection interruptions; a client timeout does not mean the
server operation was cancelled. `/chat` remains compatible with the existing
text API. Requests reach the existing HarnessExecutor authority through
LilaCore; the daemon does not execute tools directly. Sensitive actions remain
fail-closed when no approval UI is attached.

The service does not log request bodies, authentication headers, or transcripts
itself. Existing LilaCore interaction logging persists user and assistant text,
including voice transcripts. Parakeet output is also stored beside the private
recording. Retention remains indefinite; see the voice architecture document.

## Shutdown and failures

SIGTERM/SIGINT stops accepting API work, cancels active capture/ASR when
possible, closes the HTTP listener, releases the workspace model lock shared
with the supported CLI, and unloads LilaCore references. A model load failure is reported to the journal
and exits nonzero; systemd retries failures with a bounded rate. The service
does not claim success until `systemctl --user status` reports it active and an
authenticated `/health` request reports `ready: true`.
