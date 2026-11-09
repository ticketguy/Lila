# Build and migration setup

This guide uses the existing Lila Python application, Parakeet voice backend,
and Windows desktop project. The repository does not currently contain a
complete Python dependency lock file, Parakeet source checkout, Parakeet model,
or main GGUF weights. Those assets must be provided from a separately verified
installation; the activation scripts do not download them.

## Linux/WSL prerequisites

Use a supported WSL Linux distribution with systemd enabled, a working WSL
audio input source, and the repository checkout on the Linux filesystem. The
Linux voice activation script itself does not use sudo. On a fresh Ubuntu
distribution, the standard OS packages used to create the Python environment
and provide `parecord` are:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip pulseaudio-utils
```

`parecord` is supplied by the PulseAudio utilities package. Confirm that the
WSL audio server exposes the intended microphone before continuing:

```bash
command -v parecord
command -v pactl
pactl info
pactl list short sources
systemctl --user show-environment
```

The last command must succeed for the current Linux user. This setup does not
enable lingering or change system WSL configuration.

## Recreate the Python environment

For a move from an existing working machine, first capture its installed
Python-package versions outside the repository. From the source machine, using
the environment that actually runs Lila:

```bash
../environment/.venv/bin/python -m pip freeze > /tmp/lila-python-requirements.txt
```

Transfer that inventory privately with the source checkout, review platform
specific packages, then create a new environment in the clone:

```bash
cd /path/to/lila/source
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r /path/to/lila-python-requirements.txt
```

If the source environment is elsewhere, use its actual Python path instead of
`../environment/.venv/bin/python`. The package inventory is a migration aid,
not a source-controlled project lock: compiled packages may need a compatible
wheel or a separately managed build on the destination. This repository does
not currently specify a complete, reproducible Python package set, so do not
guess versions from this guide.

Check that the imports used by the service and voice path work:

```bash
.venv/bin/python -c 'import src.daemon.service, src.core.voice'
```

## Obtain and configure Parakeet assets

The Parakeet CLI and model are separate local assets; neither is bundled here.
The existing default layout, when you already have that asset tree, is:

```text
experiments/voice/whisper.cpp/build/bin/parakeet-cli
experiments/voice/whisper.cpp/models/ggml-parakeet-tdt-0.6b-v3-q4_k.bin
```

The repository does not record a verified upstream download location, source
revision, or checksum for these files. On a migration, use the same verified
source/build and model artifact recorded for your current installation, or copy
the already verified executable and model from that installation; do not infer
a download URL from the filename. Record and compare their hashes during the
copy:

```bash
sha256sum /path/to/parakeet-cli /path/to/parakeet-model.bin
```

The destination paths can be set in `~/.config/lila/voice.env`:

```text
LILA_VOICE_ASR_EXECUTABLE=/absolute/path/to/parakeet-cli
LILA_VOICE_ASR_MODEL=/absolute/path/to/parakeet-model.bin
```

The executable must be executable and the model must be readable. Confirm the
capture tool and paths before activating the service:

```bash
test -x /absolute/path/to/parakeet-cli
test -r /absolute/path/to/parakeet-model.bin
command -v parecord
```

Also verify the selected ASR build against a known local WAV using the same
Parakeet command recorded by the existing installation. No Parakeet version or
model hash is asserted by this repository.

## Configure the separate main language model

Set `LILA_MODEL_PATH` in `~/.config/lila/service.env`. Use an absolute local
path to a GGUF file, or to a local Transformers model directory:

```text
LILA_MODEL_PATH=/absolute/path/to/models/your-model.gguf
```

The GGUF inference path requires `llama-cpp-python` in this Python environment;
a local Transformers directory requires the existing `torch` and `transformers`
runtime dependencies. The main model and Parakeet ASR model are independent:
`LILA_MODEL_PATH` does not set either voice ASR path. Keep all weights outside
source control. The activation script checks that a configured path exists and
that a file has a `.gguf` suffix; it does not load the model or verify model
provenance.

Before activation, check the configured path and the required loader import in
the selected environment. For a GGUF:

```bash
test -r /absolute/path/to/models/your-model.gguf
.venv/bin/python -c 'import llama_cpp'
```

For a Transformers model, check that the configured path is a directory and
that the dependencies used by the existing model configuration import. The
repository does not provide a model download or a standalone model-load
preflight command.

## Activate only after preflight

From the Linux checkout, select the environment explicitly if it is not at the
default location:

```bash
LILA_PYTHON="$PWD/.venv/bin/python" bash deploy/linux/activate-voice-input.sh
```

The script repeats its prerequisite checks before installing the per-user
systemd unit. Then verify:

```bash
systemctl --user status lila.service
journalctl --user -u lila.service -n 80 --no-pager
```

Use Windows Settings to save the same API token from the private Linux
`service.env` in Windows Credential Manager. Lila reports connected only after
the authenticated API health check is ready. Test an actual short recording
and transcription before relying on voice input.

## Windows build and tests

From PowerShell in the repository clone:

```powershell
dotnet --list-sdks
dotnet build .\windows\Lila.Desktop\Lila.Desktop.csproj -c Release
dotnet test .\windows\Lila.Desktop.Tests\Lila.Desktop.Tests.csproj -c Release
dotnet publish .\windows\Lila.Desktop\Lila.Desktop.csproj `
  -c Release -r win-x64 --self-contained true `
  -p:PublishSingleFile=false -p:PublishTrimmed=false `
  -o "$env:LOCALAPPDATA\LilaDesktop\publish\win-x64"
```

The explicit activation script publishes to its managed per-user directory.
It does not register Windows startup.

## Python tests and shell validation

From the repository root using the existing Python environment:

```bash
../environment/.venv/bin/python -m unittest discover -s tests -p 'test_voice_input.py'
../environment/.venv/bin/python -m unittest discover -s tests -p 'test_service_voice_api.py'
../environment/.venv/bin/python -m unittest discover -s tests -p 'test_voice_activation_script.py'
../environment/.venv/bin/python -m unittest discover -s tests
bash -n deploy/linux/activate-voice-input.sh
```

On Windows, the rollback transaction tests can be run without Pester:

```powershell
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File .\tests\test_windows_activation_rollback.ps1
```

The automated Python tests use temporary files and fake capture/transcriber
processes. They do not validate a physical microphone, WSL audio, Parakeet
artifact provenance, or keyboard firmware. Run the hardware checklist in
[acceptance.md](acceptance.md) separately.
