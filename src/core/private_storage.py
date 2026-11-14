"""Helpers for keeping runtime conversation and voice data private."""

from __future__ import annotations

import os
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

_UMASK_LOCK = threading.RLock()


@contextmanager
def restrictive_umask() -> Iterator[None]:
    """Create private files with a process umask of 077, then restore it."""
    with _UMASK_LOCK:
        previous = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous)


def secure_private_directory(path: Path) -> Path:
    """Create or secure one runtime data directory without traversing it."""
    directory = Path(path)
    if directory.is_symlink():
        raise PermissionError(
            f"Private data directory must not be a symlink: {directory}"
        )

    with restrictive_umask():
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)

    _require_mode(directory, 0o700, "directory")
    return directory


@contextmanager
def open_private_text(path: Path, mode: str = "a") -> Iterator[TextIO]:
    """Open a private regular text file without following a symlink."""
    if mode not in {"a", "w"}:
        raise ValueError("Private text files support only append or write mode.")

    target = Path(path)
    secure_private_directory(target.parent)
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if mode == "a" else os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0)

    with restrictive_umask():
        descriptor = os.open(target, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PermissionError(f"Private data file is not regular: {target}")
            os.fchmod(descriptor, 0o600)
            _require_mode(target, 0o600, "file")
            stream = os.fdopen(descriptor, mode, encoding="utf-8")
        except Exception:
            os.close(descriptor)
            raise

    with stream:
        yield stream


def secure_private_file(path: Path) -> Path:
    """Restrict an existing regular private file to owner read/write."""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise PermissionError(f"Private data file must be a regular file: {target}")
    os.chmod(target, 0o600)
    _require_mode(target, 0o600, "file")
    return target


def _require_mode(path: Path, expected: int, kind: str) -> None:
    actual = stat.S_IMODE(path.stat().st_mode)
    if actual != expected:
        raise PermissionError(
            f"Unable to secure private {kind} {path}: "
            f"expected mode {expected:04o}, got {actual:04o}"
        )
