#!/usr/bin/env python3
import os
import secrets
import stat
import tempfile
from pathlib import Path


def main() -> None:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    service_env = config_home / "lila" / "service.env"
    if service_env.is_symlink():
        raise SystemExit("service.env must not be a symlink.")
    info = service_env.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise SystemExit("service.env must be a regular file owned by the current Linux user.")

    lines = service_env.read_text(encoding="utf-8").splitlines()
    token_indexes = []
    for index, line in enumerate(lines):
        key, separator, _ = line.partition("=")
        if separator and key == "LILA_API_TOKEN":
            token_indexes.append(index)
    if len(token_indexes) != 1:
        raise SystemExit("service.env must contain exactly one LILA_API_TOKEN.")
    lines[token_indexes[0]] = "LILA_API_TOKEN=" + secrets.token_urlsafe(48)

    fd, temp_name = tempfile.mkstemp(prefix=".service.env.", dir=service_env.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, service_env)
        directory_fd = os.open(
            service_env.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
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
    os.chmod(service_env, 0o600)
    if stat.S_IMODE(service_env.stat().st_mode) != 0o600:
        raise SystemExit("Could not enforce private permissions on service.env.")


if __name__ == "__main__":
    main()
