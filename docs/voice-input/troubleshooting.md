# Troubleshooting

## Shortcut does not activate

1. Confirm the tray says Lila is connected; keyboard activation is disabled
   until authenticated health is ready.
2. Confirm the binding is enabled and selected in Settings.
3. Open **Keyboard diagnostic** and verify the expected VK, scan code, E0/E1
   flags, device and down/up events. Unrelated keys are not captured into the
   diagnostic log.
4. Check that the exact configured modifiers are held. Extra modifiers block
   activation by design.
5. Some keyboard firmware does not emit a stable global event for Fn keys.
   ThinkPad T450s Fn+P is known to produce a Pause pulse; use it in toggle mode.

Right Ctrl starts only after the arming delay. Ctrl+C/V/S remains ordinary
input. A quick Right Ctrl tap or another key pressed during arming will not
start recording.

## Lila disconnected or service unavailable

- Use **Retry Lila** and inspect the tray connection state.
- In WSL, check `systemctl --user status lila.service` and
  `journalctl --user -u lila.service`.
- The service requires the systemd user manager. The scripts do not enable
  lingering. If Windows-launched commands cannot reach `systemctl --user`, log
  into the Linux user and validate the manager; do not infer readiness from the
  WSL distro merely being running.
- Check that the tray-owned WSL helper remains alive and that the distro name
  matches Settings. The app retries after helper/service failure.

## Authentication errors

Save the existing API token in Windows Credential Manager. In WSL, check that
`~/.config/lila/service.env` exists, has mode 0600 and contains a token of at
least 32 characters. Do not paste token values into diagnostic output.

## Microphone and ASR failures

- Confirm `parecord` is installed and that `pactl list short sources` shows a
  usable source. The default source is used unless `LILA_VOICE_CAPTURE_DEVICE`
  is set in the optional `~/.config/lila/voice.env`.
- Configure `LILA_VOICE_ASR_EXECUTABLE` and `LILA_VOICE_ASR_MODEL` if the
  Parakeet executable/model are not at their documented default locations.
- Review the bounded service error and `journalctl` output. WAVs and original
  transcripts are retained in the configured recordings directory for
  troubleshooting; delete them only through the user's own retention process.
- Empty/invalid WAVs are rejected and not sent to LilaCore. ASR timeout and
  microphone start errors are returned as safe API errors.

## Exit and recovery

Closing the main window hides it in the tray. Choose **Exit** to cancel active
voice, stop the Linux user service and gracefully close the managed helper. If
the systemd manager is unavailable, the helper still receives its shutdown
request; inspect the service manually if necessary.
