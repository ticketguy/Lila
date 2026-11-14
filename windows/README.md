# Lila Windows Desktop

The .NET 10 WinForms tray app provides the Windows UI and global keyboard
input. The authenticated API, microphone capture, Parakeet ASR, recordings,
transcripts and LilaCore remain in WSL. See the [portable voice-input guide](../docs/voice-input/README.md)
for discovery, architecture, build/test, installation, shortcuts, troubleshooting
and acceptance.

## Startup and WSL

Windows sign-in startup is **off by default**. The user may opt in from Settings;
only the current user's Run key is used, without administrator privileges. The
startup launch hides the conversation window in the tray. A per-user mutex
prevents a duplicate tray process.

While the tray app runs, it owns a tracked WSL process, checks/restarts it,
starts/checks the Linux `lila.service`, and enables keyboard shortcuts only
after authenticated health reports ready. Closing the conversation window hides
the app; **Exit** cancels voice and releases the managed WSL resources. The app
does not enable lingering or change system-wide WSL/network settings.

## Default keyboard bindings

- Right Ctrl: hold-to-talk after the arming delay.
- Pause/Fn+P: toggle-to-talk; first press starts and second press submits.
- Esc: cancel either session without transcript delivery.

Settings allows independent hold/toggle bindings from a conservative whitelist
of single keys and modifier chords. Unsafe or unsupported keys are blocked.
Raw Input runs while focused or in the tray and does not suppress ordinary
keyboard input.

## Development commands

Run in PowerShell from this repository root:

```powershell
dotnet build .\windows\Lila.Desktop\Lila.Desktop.csproj -c Release
dotnet test .\windows\Lila.Desktop.Tests\Lila.Desktop.Tests.csproj -c Release
dotnet publish .\windows\Lila.Desktop\Lila.Desktop.csproj `
  -c Release -r win-x64 --self-contained true `
  -p:PublishSingleFile=false -p:PublishTrimmed=false `
  -o "$env:LOCALAPPDATA\LilaDesktop\publish\win-x64"
```

The per-user deployment script is `deploy/windows/Activate-Lila.ps1`; it does
not enable startup. Linux service activation is
`bash deploy/linux/activate-voice-input.sh` from the WSL clone. Both scripts
have safe, marked deactivation paths. See the installation guide for details.

Actual device behavior must be recorded separately from automated tests. The
T450s Fn+P/Pause pulse is a known observation; other keyboard and audio hardware
needs its own acceptance run.
