# Portable Windows/WSL Voice Input

This guide covers the Windows tray client and the existing Linux voice service.
Windows owns the UI, authenticated API client and global keyboard input. WSL
owns microphone capture, Parakeet transcription, recordings, transcripts and
LilaCore delivery.

## Documents

- [Discovery record](discovery.md)
- [Architecture and data flow](architecture.md)
- [Build and test guide](build.md)
- [Install, activate and deactivate](installation.md)
- [Shortcut guide](shortcuts.md)
- [Troubleshooting](troubleshooting.md)
- [Portability and clean-clone setup](portability.md)
- [Acceptance checklist](acceptance.md)

The current defaults are Right Ctrl hold-to-talk and Pause/Fn+P toggle-to-talk.
The app does not register Windows startup by default. The ThinkPad T450s Fn+P
key has been observed to emit a short Pause pulse; that makes it appropriate
for toggle mode, not hold mode. Other keyboard models need hardware validation.

The voice service uses the existing Parakeet backend; this work does not add
another ASR engine or alter Harness authority.
