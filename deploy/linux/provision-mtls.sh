#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

config_home="${XDG_CONFIG_HOME:-$HOME/.config}"
lila_config="$config_home/lila"
tls_dir="$lila_config/tls"
material_dir="$tls_dir/material"
service_env="$lila_config/service.env"

for command_name in openssl python3 sha256sum flock; do
  command -v "$command_name" >/dev/null || {
    echo "Required command '$command_name' is missing in this WSL distribution." >&2
    exit 1
  }
done

if [[ -L "$lila_config" || -L "$tls_dir" ]]; then
  echo "Lila TLS configuration paths must not be symlinks." >&2
  exit 1
fi
mkdir -p -- "$tls_dir"
chmod 700 -- "$lila_config" "$tls_dir"
[[ "$(stat -c '%a' "$lila_config")" == 700 &&
   "$(stat -c '%a' "$tls_dir")" == 700 ]] || {
  echo "Could not enforce private permissions on the Lila TLS directories." >&2
  exit 1
}

if [[ -L "$tls_dir/.setup.lock" ]]; then
  echo "The Lila TLS setup lock must not be a symlink." >&2
  exit 1
fi
exec 9>"$tls_dir/.setup.lock"
chmod 600 "$tls_dir/.setup.lock"
flock -n 9 || {
  echo "Another Lila TLS setup process is running." >&2
  exit 1
}

python3 - "$service_env" <<'PY'
import os
import secrets
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_symlink():
    raise SystemExit("service.env must not be a symlink.")
try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
except FileExistsError:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise SystemExit("service.env must be a regular file owned by the current Linux user.")
    os.chmod(path, 0o600)
    with path.open(encoding="utf-8") as stream:
        lines = stream.readlines()
    token = next((line.partition("=")[2].strip().strip('"').strip("'")
                  for line in lines if line.partition("=")[0] == "LILA_API_TOKEN"), "")
    if len(token) < 32:
        raise SystemExit("LILA_API_TOKEN is missing or shorter than 32 characters in service.env.")
    raise SystemExit(0)
else:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("LILA_API_TOKEN=" + secrets.token_urlsafe(48) + "\n")
PY
service_env_status=$?
if ((service_env_status != 0)); then
  exit "$service_env_status"
fi
chmod 600 "$service_env"
[[ "$(stat -c '%a' "$service_env")" == 600 ]] || {
  echo "Could not enforce private permissions on service.env." >&2
  exit 1
}

if [[ -L "$material_dir" ]]; then
  echo "The existing Lila TLS material path must not be a symlink." >&2
  exit 1
fi
if [[ ! -e "$material_dir" ]]; then
  stage="$(mktemp -d "$tls_dir/.material.XXXXXX")"
  trap 'rm -rf -- "$stage"' EXIT

  openssl req -x509 -newkey rsa:3072 -nodes -days 3650 \
    -keyout "$stage/ca.key" -out "$stage/ca.pem" \
    -subj '/CN=Lila Local mTLS CA' \
    -addext 'basicConstraints=critical,CA:TRUE,pathlen:0' \
    -addext 'keyUsage=critical,keyCertSign,cRLSign' >/dev/null 2>&1

  openssl req -new -newkey rsa:3072 -nodes \
    -keyout "$stage/server.key" -out "$stage/server.csr" \
    -subj '/CN=localhost' >/dev/null 2>&1
  cat >"$stage/server.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:localhost,IP:127.0.0.1
EOF
  openssl x509 -req -in "$stage/server.csr" -CA "$stage/ca.pem" \
    -CAkey "$stage/ca.key" -CAcreateserial -days 3650 \
    -extfile "$stage/server.ext" -out "$stage/server.pem" >/dev/null 2>&1

  openssl req -new -newkey rsa:3072 -nodes \
    -keyout "$stage/client.key" -out "$stage/client.csr" \
    -subj '/CN=Lila Island Client' >/dev/null 2>&1
  cat >"$stage/client.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=clientAuth
EOF
  openssl x509 -req -in "$stage/client.csr" -CA "$stage/ca.pem" \
    -CAkey "$stage/ca.key" -CAcreateserial -days 3650 \
    -extfile "$stage/client.ext" -out "$stage/client.pem" >/dev/null 2>&1

  rm -f -- "$stage/server.csr" "$stage/client.csr" \
    "$stage/server.ext" "$stage/client.ext" "$stage/ca.srl"
  chmod 600 "$stage"/*
  openssl verify -purpose sslserver -CAfile "$stage/ca.pem" "$stage/server.pem" >/dev/null
  openssl verify -purpose sslclient -CAfile "$stage/ca.pem" "$stage/client.pem" >/dev/null
  for key in ca.key server.key client.key; do
    openssl pkey -in "$stage/$key" -check -noout >/dev/null 2>&1
  done
  mv -- "$stage" "$material_dir"
  trap - EXIT
elif [[ ! -d "$material_dir" ]]; then
  echo "The existing Lila TLS material path is not a regular directory; refusing replacement." >&2
  exit 1
fi
[[ "$(stat -c '%a' "$material_dir")" == 700 &&
   "$(stat -c '%u' "$material_dir")" == "$(id -u)" ]] || {
  echo "The Lila TLS material directory must be owned by this user and have mode 0700." >&2
  exit 1
}

required_files=(ca.key ca.pem server.key server.pem client.key client.pem)
for name in "${required_files[@]}"; do
  file="$material_dir/$name"
  if [[ -L "$file" || ! -f "$file" ]]; then
    echo "Existing Lila TLS material is incomplete at '$material_dir'; refusing replacement." >&2
    exit 1
  fi
done
for name in ca.key server.key client.key ca.pem server.pem client.pem; do
  file="$material_dir/$name"
  if [[ "$(stat -c '%a' "$file")" != 600 ||
        "$(stat -c '%u' "$file")" != "$(id -u)" ]]; then
    echo "Lila TLS file '$file' must have mode 0600." >&2
    exit 1
  fi
done
openssl verify -purpose sslserver -CAfile "$material_dir/ca.pem" "$material_dir/server.pem" >/dev/null
openssl verify -purpose sslclient -CAfile "$material_dir/ca.pem" "$material_dir/client.pem" >/dev/null
for name in ca.key server.key client.key; do
  openssl pkey -in "$material_dir/$name" -check -noout >/dev/null 2>&1 || {
    echo "Lila TLS private key '$name' is invalid." >&2
    exit 1
  }
done

client_sha256="$(openssl x509 -in "$material_dir/client.pem" -outform DER | sha256sum | cut -d ' ' -f 1)"
server_sha256="$(openssl x509 -in "$material_dir/server.pem" -outform DER | sha256sum | cut -d ' ' -f 1)"
pfx="$tls_dir/island-client.pfx"
if [[ -e "$pfx" || -L "$pfx" ]]; then
  if [[ -L "$pfx" || ! -f "$pfx" ]]; then
    echo "The existing Island client PFX path is not a regular file; refusing replacement." >&2
    exit 1
  fi
  pfx_sha256="$(openssl pkcs12 -in "$pfx" -clcerts -nokeys -passin pass: 2>/dev/null |
    openssl x509 -outform DER 2>/dev/null | sha256sum | cut -d ' ' -f 1)"
  [[ "$pfx_sha256" == "$client_sha256" ]] || {
    echo "The existing Island client PFX does not match the provisioned certificate." >&2
    exit 1
  }
else
  pfx_tmp="$(mktemp "$tls_dir/.island-client.XXXXXX")"
  trap 'rm -f -- "$pfx_tmp"' EXIT
  openssl pkcs12 -export -out "$pfx_tmp" \
    -inkey "$material_dir/client.key" -in "$material_dir/client.pem" \
    -certfile "$material_dir/ca.pem" -passout pass: >/dev/null 2>&1
  chmod 600 "$pfx_tmp"
  mv -- "$pfx_tmp" "$pfx"
  trap - EXIT
fi
[[ "$(stat -c '%a' "$pfx")" == 600 &&
   "$(stat -c '%u' "$pfx")" == "$(id -u)" ]] || {
  echo "The Island client PFX must be owned by this user and have mode 0600." >&2
  exit 1
}

python3 - "$service_env" "$material_dir" "$client_sha256" <<'PY'
import os
import stat
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
tls_dir = Path(sys.argv[2])
client_pin = sys.argv[3]
values = {
    "LILA_TLS_SERVER_CERT": str(tls_dir / "server.pem"),
    "LILA_TLS_SERVER_KEY": str(tls_dir / "server.key"),
    "LILA_TLS_CLIENT_CA": str(tls_dir / "ca.pem"),
    "LILA_TLS_ALLOWED_CLIENT_SHA256": client_pin,
}
rendered = {
    key: f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34)).replace("$", chr(92) + "$")}"'
    for key, value in values.items()
}

info = path.stat(follow_symlinks=False)
if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
    raise SystemExit("service.env must be a regular file owned by the current Linux user.")
lines = path.read_text(encoding="utf-8").splitlines()
positions = {}
for index, line in enumerate(lines):
    key, separator, _ = line.partition("=")
    if separator and key in values:
        if key in positions:
            raise SystemExit(f"Duplicate {key} entries in service.env; refusing to edit.")
        positions[key] = index

present = set(positions)
if present and present != set(values):
    raise SystemExit("service.env has a partial mTLS configuration; refusing to overwrite it.")
if present:
    for key in values:
        if lines[positions[key]] != rendered[key]:
            raise SystemExit(f"Existing {key} differs from the provisioned certificate; refusing replacement.")
else:
    lines.extend(rendered.values())

text = "\n".join(lines) + "\n"
fd, temp_name = tempfile.mkstemp(prefix=".service.env.", dir=path.parent)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_name, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
except BaseException:
    try:
        os.unlink(temp_name)
    except FileNotFoundError:
        pass
    raise
os.chmod(path, 0o600)
if stat.S_IMODE(path.stat().st_mode) != 0o600:
    raise SystemExit("Could not enforce private permissions on service.env.")
PY

python3 - "$pfx" "$client_sha256" "$server_sha256" "$service_env" <<'PY'
import json
import sys
from pathlib import Path

pfx, client, server, env = sys.argv[1:]
print(json.dumps({
    "client_pfx": str(Path(pfx).resolve()),
    "client_sha256": client,
    "server_sha256": server,
    "service_env": str(Path(env).resolve()),
}))
PY
