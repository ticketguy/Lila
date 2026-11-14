"""Cross-process lock shared by supported Lila CLI and daemon runtimes."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from src.core.private_storage import secure_private_directory


def acquire_runtime_lock() -> int:
    """Acquire the workspace model-runtime lock or reject a second instance."""
    workspace = Path(__file__).resolve().parents[3]
    lock_path = workspace / "logs" / "lila-runtime.lock"
    secure_private_directory(lock_path.parent)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError(
            "Another Lila runtime already owns the workspace model lock."
        ) from exc
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    return descriptor


def release_runtime_lock(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
