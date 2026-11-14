#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deploy/linux/setup-lila.sh [--python PATH]

Provision per-user mTLS material and install or safely restart this user's
Lila systemd service. This Linux entry point does not configure Windows.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
python_path="${LILA_PYTHON:-}"
while (($#)); do
  case "$1" in
    --python)
      (($# >= 2)) || { echo "Missing value for --python." >&2; exit 2; }
      python_path="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$python_path" ]]; then
  export LILA_PYTHON="$python_path"
fi

bash "$repo_root/deploy/linux/provision-mtls.sh" >/dev/null
if [[ -n "$python_path" ]]; then
  bash "$repo_root/deploy/linux/activate-voice-input.sh" \
    --python "$python_path" --adopt-existing
else
  bash "$repo_root/deploy/linux/activate-voice-input.sh" --adopt-existing
fi
echo "Lila Linux setup completed. Private TLS material remains under ~/.config/lila/tls."
