#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deploy/linux/activate-voice-input.sh [--python PATH] [--adopt-existing] [--check-only]
  bash deploy/linux/activate-voice-input.sh --deactivate

Install/remove the current user's Lila systemd voice service. No sudo is used.
Activation requires an existing Linux Python environment, parecord, the
Parakeet CLI/model, and a reachable systemd user manager.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
python_path="${LILA_PYTHON:-}"
deactivate=0
adopt_existing=0
check_only=0
while (($#)); do
  case "$1" in
    --python)
      (($# >= 2)) || { echo "Missing value for --python." >&2; exit 2; }
      python_path="$2"; shift 2 ;;
    --adopt-existing) adopt_existing=1; shift ;;
    --check-only) check_only=1; shift ;;
    --deactivate|--uninstall) deactivate=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

config_home="${XDG_CONFIG_HOME:-$HOME/.config}"
lila_config="$config_home/lila"
unit_dir="$config_home/systemd/user"
unit_path="$unit_dir/lila.service"
marker="$lila_config/voice-input-install.json"
service_env="$lila_config/service.env"

if ((deactivate)); then
  if [[ ! -f "$marker" ]]; then
    echo "No Lila voice-input installation marker exists; nothing was removed."
    exit 0
  fi
  readarray -t install_info < <(python3 - "$marker" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value.get("unit_path", ""))
print(value.get("unit_sha256", ""))
PY
)
  marked_unit="${install_info[0]:-}"
  marked_hash="${install_info[1]:-}"
  if [[ "$marked_unit" != "$unit_path" || -z "$marked_hash" ]]; then
    echo "Install marker does not match this user's Lila unit path; refusing removal." >&2
    exit 1
  fi
  if [[ -f "$unit_path" ]]; then
    current_hash="$(sha256sum "$unit_path" | cut -d ' ' -f 1)"
    if [[ "$current_hash" == "$marked_hash" ]]; then
      systemctl --user disable --now lila.service >/dev/null 2>&1 || true
      rm -- "$unit_path"
      systemctl --user daemon-reload
      echo "Removed the Lila service unit created by this activation."
    else
      echo "The service unit changed after activation; it was preserved."
    fi
  fi
  rm -- "$marker"
  echo "Preserved credentials, settings, recordings, logs, and model assets."
  exit 0
fi

if [[ -z "$python_path" ]]; then
  for candidate in "$repo_root/.venv/bin/python" "$repo_root/../environment/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then python_path="$candidate"; break; fi
  done
fi
if [[ -z "$python_path" || ! -x "$python_path" ]]; then
  echo "Python environment not found. Create a Linux venv or pass --python PATH." >&2
  exit 1
fi
python_dir="$(cd "$(dirname "$python_path")" && pwd -P)"
python_path="$python_dir/$(basename "$python_path")"
cd "$repo_root"

for command_name in python3 systemctl parecord; do
  command -v "$command_name" >/dev/null || {
    echo "Required command '$command_name' is missing in this WSL distribution." >&2
    exit 1
  }
done
systemctl --user show-environment >/dev/null 2>&1 || {
  echo "The systemd user manager is unavailable. Log into this Linux user and verify systemd; this script does not enable lingering." >&2
  exit 1
}

if [[ -n "${LILA_VOICE_HOME:-}" ]]; then
  voice_home="$LILA_VOICE_HOME"
elif [[ -d "$repo_root/experiments/voice" ]]; then
  voice_home="$repo_root/experiments/voice"
elif [[ -f "$repo_root/../experiments/voice/whisper.cpp/models/ggml-parakeet-tdt-0.6b-v3-q4_k.bin" ]]; then
  voice_home="$(realpath "$repo_root/../experiments/voice")"
else
  voice_home="$repo_root/experiments/voice"
fi
asr_executable="${LILA_VOICE_ASR_EXECUTABLE:-$voice_home/whisper.cpp/build/bin/parakeet-cli}"
asr_model="${LILA_VOICE_ASR_MODEL:-$voice_home/whisper.cpp/models/ggml-parakeet-tdt-0.6b-v3-q4_k.bin}"
[[ -x "$asr_executable" ]] || { echo "Parakeet executable is missing or not executable: $asr_executable" >&2; exit 1; }
[[ -f "$asr_model" ]] || { echo "Parakeet model is missing: $asr_model" >&2; exit 1; }
"$python_path" -c 'import src.daemon.service, src.core.voice' >/dev/null || {
  echo "Python dependencies are incomplete in $python_path." >&2
  exit 1
}
if ((check_only)); then
  echo "Lila Linux setup prerequisites are available."
  exit 0
fi

mkdir -p -m 700 "$lila_config" "$unit_dir"
chmod 700 "$lila_config"
if [[ -e "$marker" ]]; then
  readarray -t install_info < <(python3 - "$marker" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    value = json.load(stream)
print(value.get("unit_path", ""))
print(value.get("unit_sha256", ""))
PY
  )
  current_hash=""
  [[ ! -f "$unit_path" ]] || current_hash="$(sha256sum "$unit_path" | cut -d ' ' -f 1)"
  if [[ "${install_info[0]:-}" == "$unit_path" &&
        "${install_info[1]:-}" == "$current_hash" && -n "$current_hash" ]]; then
    systemctl --user daemon-reload
    systemctl --user enable --now lila.service
    echo "Lila voice input is already activated and the service is enabled."
    exit 0
  fi
  echo "The managed unit differs from its activation record; preserving it and refusing replacement." >&2
  exit 1
fi
if [[ -e "$unit_path" ]]; then
  if ((adopt_existing)); then
    actual_fragment="$(systemctl --user show --property=FragmentPath --value lila.service)"
    actual_working="$(systemctl --user show --property=WorkingDirectory --value lila.service)"
    actual_environment="$(systemctl --user show --property=EnvironmentFiles --value lila.service)"
    actual_exec="$(systemctl --user show --property=ExecStart --value lila.service)"
    if [[ "$actual_fragment" != "$unit_path" ||
          "$actual_working" != "$repo_root" ||
          "$actual_environment" != *"$service_env"* ||
          "$actual_exec" != *"$python_path"* ||
          "$actual_exec" != *"src.daemon.service"* ]]; then
      echo "The existing lila.service does not match this repository and service.env; it was left unchanged." >&2
      exit 1
    fi
    systemctl --user daemon-reload
    systemctl --user restart lila.service
    systemctl --user is-active --quiet lila.service || {
      echo "The matching lila.service did not become active; inspect its journal." >&2
      exit 1
    }
    echo "Restarted the matching existing Lila service without replacing its unit."
    exit 0
  fi
  echo "A user lila.service already exists; it will not be overwritten. Inspect it and deactivate only if it was created by this script." >&2
  exit 1
fi

if [[ ! -e "$service_env" ]]; then
  umask 077
  "$python_path" - "$service_env" <<'PY'
import os, secrets, sys
path = sys.argv[1]
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    stream.write("LILA_API_TOKEN=" + secrets.token_urlsafe(48) + "\n")
PY
  echo "Created a private API credential file at $service_env. Its value was not displayed."
fi
"$python_path" - "$service_env" <<'PY'
import sys
values = {}
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        key, sep, value = line.rstrip("\n").partition("=")
        if sep and key == "LILA_API_TOKEN":
            values[key] = value.strip().strip('"').strip("'")
token = values.get("LILA_API_TOKEN", "")
if len(token) < 32:
    raise SystemExit("LILA_API_TOKEN is missing or shorter than 32 characters in service.env.")
PY
python3 - "$service_env" <<'PY'
import sys
from pathlib import Path
configured = None
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        key, separator, value = line.rstrip("\n").partition("=")
        if separator and key == "LILA_MODEL_PATH":
            configured = value.strip().strip('"').strip("'")
if configured:
    model = Path(configured)
    if not model.is_absolute():
        raise SystemExit("LILA_MODEL_PATH must be an absolute local path.")
    if not model.exists():
        raise SystemExit("LILA_MODEL_PATH does not exist; update service.env before activation.")
    if model.is_file() and model.suffix.casefold() != ".gguf":
        raise SystemExit("LILA_MODEL_PATH file must be a GGUF; Transformers models must be directories.")
PY
chmod 600 "$service_env"

unit_tmp="$(mktemp "$unit_dir/.lila.service.XXXXXX")"
trap 'rm -f -- "$unit_tmp"' EXIT
python3 - "$unit_tmp" "$repo_root" "$python_path" "$service_env" \
  "$repo_root/deploy/systemd/lila.service" <<'PY'
import sys
target, root, python, env, template_path = sys.argv[1:]
def quote(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
with open(template_path, encoding="utf-8") as source:
    contents = source.read()
contents = contents.replace("@LILA_REPOSITORY_ROOT@", quote(root))
contents = contents.replace("@LILA_PYTHON@", quote(python))
contents = contents.replace("@LILA_SERVICE_ENV@", quote(env))
contents = contents.replace("@LILA_OPTIONAL_VOICE_ENV@", "-" + quote(env.rsplit("/", 1)[0] + "/voice.env"))
if "@LILA_" in contents:
    raise SystemExit("The systemd unit template contains an unknown placeholder.")
with open(target, "w", encoding="utf-8") as unit:
    unit.write(contents)
PY
chmod 600 "$unit_tmp"
mv -- "$unit_tmp" "$unit_path"
trap - EXIT
unit_hash="$(sha256sum "$unit_path" | cut -d ' ' -f 1)"
python3 - "$marker" "$unit_path" "$unit_hash" <<'PY'
import json, os, sys
path, unit, digest = sys.argv[1:]
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump({"unit_path": unit, "unit_sha256": digest}, stream, indent=2)
    stream.write("\n")
PY
systemctl --user daemon-reload
systemctl --user enable --now lila.service
echo "Lila service activated for this Linux user. Windows startup was not changed."
echo "Check with: systemctl --user status lila.service"
