"""Linux file-target leases for Harness file reads.

The approval contract is platform-neutral data. This adapter uses Linux
directory-relative opens (available in WSL) to pin directory components and
the final file object before handing a read-only handle to the registered
handler. Unsupported platforms or filesystems fail closed.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import stat
from typing import Iterator, Mapping, Any


class FileTargetError(RuntimeError):
    """A file target could not be opened as the exact approved object."""

    def __init__(self, reason_code: str = "target.file_identity_mismatch"):
        self.reason_code = reason_code
        super().__init__(reason_code)


def describe_file_target(path: str) -> dict[str, Any]:
    """Resolve one existing regular file and snapshot identity from the OS."""
    if not isinstance(path, str) or not path or "\x00" in path:
        raise FileTargetError("target.file_path_invalid")
    try:
        requested = Path(os.path.expanduser(path))
        if ".." in requested.parts:
            raise FileTargetError("target.file_traversal_rejected")
        canonical = requested if requested.is_absolute() else Path.cwd() / requested
        canonical = Path(os.path.abspath(str(canonical)))
        parent = canonical.parent
    except FileTargetError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise FileTargetError("target.file_unavailable") from None
    fd, parent_stat = _open_canonical(canonical, parent)
    try:
        target_stat = os.fstat(fd)
        if not stat.S_ISREG(target_stat.st_mode) or stat.S_ISLNK(target_stat.st_mode):
            raise FileTargetError("target.file_type_unsupported")
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise FileTargetError("target.parent_type_unsupported")
    finally:
        os.close(fd)
    return {
        "target_type": "filesystem.file",
        "canonical_location": str(canonical),
        "approved_root": str(parent),
        "device": int(target_stat.st_dev),
        "inode": int(target_stat.st_ino),
        "mode": int(target_stat.st_mode),
        "file_type": int(stat.S_IFMT(target_stat.st_mode)),
        "size": int(target_stat.st_size),
        "mtime_ns": int(target_stat.st_mtime_ns),
        "ctime_ns": int(target_stat.st_ctime_ns),
        "symlink": False,
        "symlink_policy": "reject-all-components-v1",
        "parent_device": int(parent_stat.st_dev),
        "parent_inode": int(parent_stat.st_ino),
        "resolver_version": "2",
    }


def _verify_stat(value: os.stat_result, expected: Mapping[str, Any]) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or int(value.st_dev) != expected.get("device")
        or int(value.st_ino) != expected.get("inode")
        or int(value.st_mode) != expected.get("mode")
        or int(stat.S_IFMT(value.st_mode)) != expected.get("file_type")
        or int(value.st_size) != expected.get("size")
        or int(value.st_mtime_ns) != expected.get("mtime_ns")
        or int(value.st_ctime_ns) != expected.get("ctime_ns")
    ):
        raise FileTargetError("target.file_identity_mismatch")


class VerifiedFileHandle:
    """Per-invocation read lease. Its file descriptor never leaves this object."""

    __slots__ = ("_fd", "_identity", "_closed", "canonical_location")

    def __init__(self, *, _issuer, fd: int, identity: Mapping[str, Any]):
        if _issuer is not _HANDLE_ISSUER:
            raise TypeError("Verified file handles can only be issued by the file-target adapter.")
        self._fd = fd
        self._identity = dict(identity)
        self._closed = False
        self.canonical_location = identity["canonical_location"]

    def read_text(self, *, max_chars: int = 50_000) -> tuple[str, bool, int]:
        if self._closed:
            raise FileTargetError("target.file_handle_closed")
        try:
            before = os.fstat(self._fd)
            _verify_stat(before, self._identity)
            if not hasattr(os, "pread"):
                raise FileTargetError("target.file_read_unsupported")
            chunks = []
            offset = 0
            while True:
                chunk = os.pread(self._fd, 64 * 1024, offset)
                if not chunk:
                    break
                chunks.append(chunk)
                offset += len(chunk)
            after = os.fstat(self._fd)
            _verify_stat(after, self._identity)
            content = b"".join(chunks).decode("utf-8", errors="replace")
            return content[:max_chars], len(content) > max_chars, len(content)
        except FileTargetError:
            raise
        except OSError:
            raise FileTargetError("target.file_read_failed") from None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            os.close(self._fd)

    def __enter__(self):
        if self._closed:
            raise FileTargetError("target.file_handle_closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __repr__(self) -> str:
        return "VerifiedFileHandle(<open>)" if not self._closed else "VerifiedFileHandle(<closed>)"


_HANDLE_ISSUER = object()
_DIRECTORY_FLAGS = (
    getattr(os, "O_PATH", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def open_verified_file(identity: Mapping[str, Any]) -> VerifiedFileHandle:
    """Pin each canonical directory component, then open and verify the leaf.

    The approved root is the target's immediate canonical parent. Opening from
    `/` with `dir_fd` and `O_NOFOLLOW` avoids re-traversing replaceable names
    after their directory handles have been acquired.
    """
    if (
        not isinstance(identity, Mapping)
        or identity.get("target_type") != "filesystem.file"
        or identity.get("symlink_policy") != "reject-all-components-v1"
        or identity.get("symlink") is not False
        or not isinstance(identity.get("canonical_location"), str)
        or not isinstance(identity.get("approved_root"), str)
    ):
        raise FileTargetError("target.file_descriptor_invalid")
    if not sys_platform_linux() or not hasattr(os, "O_PATH") or not hasattr(os, "O_NOFOLLOW"):
        raise FileTargetError("target.file_open_unsupported")
    target = Path(identity["canonical_location"])
    root = Path(identity["approved_root"])
    if (
        not target.is_absolute()
        or not root.is_absolute()
        or target.parent != root
        or target.name in {"", ".", ".."}
        or ".." in target.parts
        or root != root.resolve(strict=False)
    ):
        raise FileTargetError("target.file_containment_invalid")

    file_fd, parent_info = _open_canonical(target, root)
    try:
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or int(parent_info.st_dev) != identity.get("parent_device")
            or int(parent_info.st_ino) != identity.get("parent_inode")
        ):
            raise FileTargetError("target.file_parent_changed")
        opened = os.fstat(file_fd)
        _verify_stat(opened, identity)
        handle = VerifiedFileHandle(_issuer=_HANDLE_ISSUER, fd=file_fd, identity=identity)
        file_fd = -1
        return handle
    except FileTargetError:
        raise
    except (OSError, ValueError, TypeError):
        raise FileTargetError("target.file_open_failed") from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)


def _open_canonical(target: Path, root: Path) -> tuple[int, os.stat_result]:
    """Open one leaf beneath a pinned parent using only no-follow components."""
    if (
        not target.is_absolute()
        or not root.is_absolute()
        or target.parent != root
        or target.name in {"", ".", ".."}
        or ".." in target.parts
        or root != root.resolve(strict=False)
    ):
        raise FileTargetError("target.file_containment_invalid")
    if not sys_platform_linux() or not hasattr(os, "O_PATH") or not hasattr(os, "O_NOFOLLOW"):
        raise FileTargetError("target.file_open_unsupported")

    directory_fd = -1
    pinned_fd = -1
    file_fd = -1
    try:
        directory_fd = os.open("/", _DIRECTORY_FLAGS)
        for component in root.parts[1:]:
            next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        parent_info = os.fstat(directory_fd)
        if not stat.S_ISDIR(parent_info.st_mode):
            raise FileTargetError("target.parent_type_unsupported")
        # O_PATH pins the leaf without triggering device/FIFO open behavior.
        # After proving it is a regular file, /proc/self/fd opens that exact
        # pinned object for reading; it does not resolve the original name.
        pinned_fd = os.open(
            target.name,
            os.O_PATH | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        pinned_stat = os.fstat(pinned_fd)
        if not stat.S_ISREG(pinned_stat.st_mode) or stat.S_ISLNK(pinned_stat.st_mode):
            raise FileTargetError("target.file_type_unsupported")
        file_fd = os.open(
            f"/proc/self/fd/{pinned_fd}",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        read_stat = os.fstat(file_fd)
        if (read_stat.st_dev, read_stat.st_ino, read_stat.st_mode) != (
            pinned_stat.st_dev, pinned_stat.st_ino, pinned_stat.st_mode
        ):
            raise FileTargetError("target.file_identity_mismatch")
        result = file_fd, parent_info
        file_fd = -1
        return result
    except FileTargetError:
        raise
    except (OSError, ValueError, TypeError):
        raise FileTargetError("target.file_open_failed") from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if pinned_fd >= 0:
            os.close(pinned_fd)
        if directory_fd >= 0:
            os.close(directory_fd)


def sys_platform_linux() -> bool:
    import sys

    return sys.platform.startswith("linux")


class FileReadExecutionTargetAdapter:
    """Trusted bridge from a pending immutable binding to one opened handle."""

    @contextmanager
    def open_target(self, invocation) -> Iterator[VerifiedFileHandle]:
        binding = invocation.target_binding
        if binding is None or binding.target_kind != "filesystem.file":
            raise FileTargetError("target.file_binding_missing")
        handle = open_verified_file(binding.identity)
        try:
            yield handle
        finally:
            handle.close()
