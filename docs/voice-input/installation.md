# Install, activate and deactivate

Activation is explicit and per-user. It does not require administrator rights,
register Windows startup, modify firewall/network configuration, or install a
machine-wide service.

## Windows desktop deployment

From PowerShell in the repository clone:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\deploy\windows\Activate-Lila.ps1
```

The script checks .NET 10 and the selected WSL distribution, builds a
self-contained `win-x64` publish, validates the executable, then installs it
under `%LOCALAPPDATA%\LilaDesktop\VoiceInput\app`. It refuses to replace an
unmarked directory. It does not register startup or write credentials. Launch
`Lila.Desktop.exe`, save the existing API token in Settings, and use the
**Start Lila at my Windows sign-in** option only if desired.

## Linux/WSL service activation

In the WSL clone, make sure the Python environment, `parecord`, Parakeet CLI and
model are already installed. The default asset paths are under
`experiments/voice/whisper.cpp`; override them with the environment variables
documented in [portability.md](portability.md). Then run:

```bash
bash deploy/linux/activate-voice-input.sh
```

Use `--python /absolute/path/to/python` if the environment is not at
`.venv/bin/python` or the legacy sibling `../environment/.venv/bin/python`.
The script checks the user systemd manager, audio capture command, Python
imports and ASR files before installing. It creates a strong API token in
`~/.config/lila/service.env` only if the file is absent, protects it as mode
0600, and never prints it. Existing credentials/configuration are preserved.
The generated per-user systemd unit uses the actual repository and Python
paths. It enables and starts only `lila.service` for that Linux user.

Verify with:

```bash
systemctl --user status lila.service
systemctl --user show-environment
journalctl --user -u lila.service -n 80 --no-pager
```

The Windows client then stores the same API token in Windows Credential
Manager. Retrieve the value from the private Linux service environment file
using a local editor and paste it only into Lila Settings; close the editor
afterward. Do not put it in command arguments, source, or logs. The client
reports connected only after authenticated `/health` returns ready.

## Deactivation

Remove the Windows deployment (only its marked application directory and a
matching startup entry, if one exists):

```powershell
.\deploy\windows\Activate-Lila.ps1 -Deactivate
```

Remove the Linux service unit only if it was installed by the Linux script:

```bash
bash deploy/linux/activate-voice-input.sh --deactivate
```

The Linux script verifies its install marker and unit hash before removal. Both
operations preserve app settings, Credential Manager entries, API credentials,
model files, recordings, transcripts and logs. Existing unmarked installations
are not removed.

## Startup option

Windows startup is off by default. Explicitly opt in through the app's Settings;
the app writes only the current user's Run entry. Deactivation removes that
entry only when it points to the deployment being removed.
