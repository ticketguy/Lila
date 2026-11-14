#!/usr/bin/env python3
import os
import stat
from pathlib import Path


def main() -> None:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    service_env = config_home / "lila" / "service.env"
    if service_env.is_symlink():
        raise SystemExit("service.env must not be a symlink.")
    info = service_env.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise SystemExit("service.env must be a regular file owned by the current Linux user.")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit("service.env must have mode 0600.")

    tokens = []
    for line in service_env.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "LILA_API_TOKEN":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            tokens.append(value)
    if len(tokens) != 1 or len(tokens[0]) < 32:
        raise SystemExit("service.env must contain exactly one valid LILA_API_TOKEN.")
    print(tokens[0])


if __name__ == "__main__":
    main()
