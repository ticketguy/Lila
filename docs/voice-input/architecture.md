# Architecture and data flow

## Responsibilities

- **Windows tray app:** settings UI, foreground conversation window, per-user
  startup option, duplicate-instance guard, supervised WSL helper, Raw Input,
  shortcut interpretation, voice state UI and authenticated HTTP requests.
- **Linux user service:** one LilaCore/model process, loopback HTTP API,
  microphone capture, WAV validation, Parakeet subprocess, private recording
  and transcript storage, and raw transcript delivery to LilaCore.
- **HarnessExecutor:** remains the authority for model tool execution and
  permission checks. Voice input does not bypass it.

## Keyboard flow

The app registers a hidden message target with `RIDEV_INPUTSINK` and device
notifications. It tracks only configured primary keys, Escape and modifier
identities needed to recognize configured chords. Other non-modifier key-downs
are sent only as transient Right Ctrl arming-suppression signals; their
identities and text are not retained or logged. Raw Input never suppresses
ordinary keyboard input.

The interpreter requires the exact configured modifier set. A matching
hold-to-talk key-down arms and starts the voice session; its matching physical
key-up submits it. Toggle-to-talk starts on key-down, ignores its key-up and
submits on the next fresh press. Escape cancels either session. Device removal,
the session watchdog and service disconnect paths prevent a stuck recording.
Only one session can own the voice API at a time.

## Voice-session lifecycle

1. Windows sends authenticated `POST /voice/start` with `{}`.
2. Linux creates a private recording file and launches `parecord` on the
   configured/default PulseAudio source.
3. On hold release or toggle-stop, Windows sends authenticated
   `POST /voice/stop` with the returned session ID.
4. Linux stops capture, validates the WAV, runs the existing Parakeet CLI,
   preserves the original transcript, and passes it unchanged to LilaCore.
5. Escape uses `/voice/cancel`; cancellation does not submit a transcript.

The server bounds recording to 120 seconds and transcription to its configured
timeout. The Windows client has a shorter missed-release safety stop. After an
ambiguous HTTP failure, clients query status rather than replaying a stop or
transcript blindly.

## WSL lifecycle and recovery

The tray process owns a tracked `wsl.exe` child that blocks on stdin while the
app remains open. It checks the process, restarts it after an unexpected exit,
starts/checks `lila.service`, then enables shortcuts only after the authenticated
health endpoint reports ready. Exit stops voice, requests service shutdown,
then closes the helper. Closing the conversation window hides the app in the
tray. Microsoft documents that systemd services alone do not keep WSL alive:
[WSL systemd documentation](https://learn.microsoft.com/windows/wsl/systemd/).

The helper does not create a Linux login session. If the Linux user's
`systemctl --user` manager is not reachable, the app reports disconnected; the
scripts do not enable lingering or alter `/etc/wsl.conf`, `.wslconfig`,
firewall rules or LAN exposure.

## Security and retention

The API binds only to `127.0.0.1` and requires a bearer credential. Windows
stores the token in Credential Manager. Linux stores its token in a mode-0600
user environment file. Credentials are not put in source, command arguments or
logs. The service retains recordings/transcripts and LilaCore logs conversation
text; retention remains user-managed. Voice does not relax Harness permission
policy, and approval-required operations remain fail-closed in headless mode.
