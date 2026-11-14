"""
M3 — Execution Isolation and Bounds (`WS-EXEC`).

Provides:
  - Read-only host isolation capability assessment (`assess_host_isolation`)
  - Fixed-argv process broker with process-tree cancellation, wall-clock timeouts,
    output byte caps, scrubbed environment, and secret redaction (`BoundedProcessBroker`)
  - Conservative ThinkPad resource lease manager (`ResourceLeaseManager`)
  - Fail-closed sandbox backend enforcement (`SandboxBackend`, `BubblewrapSandboxBackend`)
  - Isolated patch staging and credential-scrubbed artifact collection (`StagedPatchWorkspace`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import difflib
import hashlib
import os
from pathlib import Path
import platform
import re
import select
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Callable, Mapping, Sequence

try:
    import resource as _posix_resource
except ImportError:  # pragma: no cover - Windows fallback
    _posix_resource = None


_SAFE_ENV_KEYS = frozenset({"PATH", "LANG", "LC_ALL", "TERM", "TZ"})
_DEFAULT_SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
_FORBIDDEN_ENV_NAME_TOKENS = (
    "LILA_",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE",
    "API_KEY",
    "ACCESS_KEY",
    "AUTH",
    "COOKIE",
    "SESSION",
    "SSH_",
    "GPG_",
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(lila_api_token|owner_secret|api_key|client_secret|password)\s*[:=]\s*[^\s,;\"']+"),
)


def redact_sensitive_text(text: str, extra_secrets: Sequence[str] = ()) -> tuple[str, bool]:
    """Redact credential patterns and explicit secret strings from process output or artifacts."""
    redacted = text
    detected = False
    for secret in extra_secrets:
        if secret and len(secret) >= 4 and secret in redacted:
            redacted = redacted.replace(secret, "[REDACTED]")
            detected = True
    for pattern in _SECRET_PATTERNS:
        if pattern.search(redacted):
            redacted = pattern.sub("[REDACTED]", redacted)
            detected = True
    return redacted, detected


class CancellationToken:
    """Cooperative + preemptive cancellation signal for bounded process execution."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: str = "cancelled_by_caller"

    def cancel(self, reason: str = "cancelled_by_caller") -> None:
        self._reason = reason
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason


@dataclass(frozen=True)
class HostIsolationAssessment:
    """Read-only snapshot of OS isolation capabilities on the current host."""

    assessed_utc: str
    os_name: str
    kernel_release: str
    python_venv_active: bool
    venv_is_security_boundary: bool
    bwrap_available: bool
    bwrap_path: str | None
    unshare_available: bool
    user_namespaces_available: bool
    prlimit_available: bool
    posix_rlimit_available: bool
    cgroup_v2_available: bool
    git_worktree_available: bool
    owner_approved_backend: str | None
    recommended_backend: str
    notes: tuple[str, ...] = ()


def assess_host_isolation(*, owner_approved_backend: str | None = None) -> HostIsolationAssessment:
    """Perform a strictly read-only probe of host process/sandbox facilities."""
    bwrap_path = shutil.which("bwrap")
    unshare_path = shutil.which("unshare")
    prlimit_path = shutil.which("prlimit")
    git_path = shutil.which("git")
    cgroup_v2 = os.path.isfile("/sys/fs/cgroup/cgroup.controllers")

    user_ns = False
    if os.path.isfile("/proc/sys/kernel/unprivileged_userns_clone"):
        try:
            with open("/proc/sys/kernel/unprivileged_userns_clone", "r", encoding="utf-8") as fh:
                user_ns = fh.read().strip() == "1"
        except OSError:
            user_ns = False
    elif platform.system() == "Linux":
        user_ns = os.path.exists("/proc/self/ns/user")

    in_venv = bool(os.environ.get("VIRTUAL_ENV")) or (
        getattr(os.sys, "base_prefix", os.sys.prefix) != os.sys.prefix
    )

    notes: list[str] = [
        "Python virtualenv is dependency isolation only and is never treated as a security boundary."
    ]
    if bwrap_path and user_ns:
        recommended = "bubblewrap"
        notes.append("Bubblewrap (bwrap) and user namespaces are present for unprivileged sandboxing.")
    elif unshare_path and user_ns:
        recommended = "linux-namespaces"
        notes.append("Linux user/net/mount namespaces via unshare are present.")
    else:
        recommended = "none-fail-closed"
        notes.append("No unprivileged OS sandbox binary detected; sandbox-required tasks must fail closed.")

    return HostIsolationAssessment(
        assessed_utc=datetime.now(timezone.utc).isoformat(),
        os_name=platform.system(),
        kernel_release=platform.release(),
        python_venv_active=in_venv,
        venv_is_security_boundary=False,
        bwrap_available=bwrap_path is not None,
        bwrap_path=bwrap_path,
        unshare_available=unshare_path is not None,
        user_namespaces_available=user_ns,
        prlimit_available=prlimit_path is not None,
        posix_rlimit_available=_posix_resource is not None,
        cgroup_v2_available=cgroup_v2,
        git_worktree_available=git_path is not None,
        owner_approved_backend=owner_approved_backend,
        recommended_backend=recommended,
        notes=tuple(notes),
    )


class ResourceClass:
    CPU_HEAVY = "cpu_heavy"
    IO_BOUND = "io_bound"
    DEVICE_SYSTEM = "device_system"


class ResourceLeaseError(RuntimeError):
    """Raised when a task exceeds the configured concurrency budget for its resource class."""


class ResourceLeaseManager:
    """
    Conservative resource-class concurrency limiter for the ThinkPad T450s / WSL host.
    Defaults: 1 CPU-heavy job, 2 lightweight I/O jobs, 1 device/system job.
    """

    def __init__(self, limits: Mapping[str, int] | None = None) -> None:
        default_limits = {
            ResourceClass.CPU_HEAVY: 1,
            ResourceClass.IO_BOUND: 2,
            ResourceClass.DEVICE_SYSTEM: 1,
        }
        if limits:
            default_limits.update(limits)
        self._limits = dict(default_limits)
        self._active: dict[str, int] = {k: 0 for k in self._limits}
        self._lock = threading.Lock()

    def active_count(self, resource_class: str) -> int:
        with self._lock:
            return self._active.get(resource_class, 0)

    def acquire(self, resource_class: str) -> "ResourceLease":
        with self._lock:
            if resource_class not in self._limits:
                raise ResourceLeaseError(f"Unknown resource class: {resource_class!r}")
            limit = self._limits[resource_class]
            current = self._active[resource_class]
            if current >= limit:
                raise ResourceLeaseError(
                    f"Resource ceiling reached for {resource_class} ({current}/{limit} active)."
                )
            self._active[resource_class] = current + 1
        return ResourceLease(self, resource_class)

    def _release(self, resource_class: str) -> None:
        with self._lock:
            if self._active.get(resource_class, 0) > 0:
                self._active[resource_class] -= 1


class ResourceLease:
    def __init__(self, manager: ResourceLeaseManager, resource_class: str) -> None:
        self._manager = manager
        self.resource_class = resource_class
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._manager._release(self.resource_class)

    def __enter__(self) -> "ResourceLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@dataclass(frozen=True)
class ProcessExecutionSpec:
    """Validated specification for a fixed-argv isolated process execution."""

    argv: tuple[str, ...]
    cwd: str
    execution_backend: str = "trusted-local"  # trusted-local | sandbox-required
    resource_class: str = ResourceClass.IO_BOUND
    timeout_seconds: float = 15.0
    grace_period_seconds: float = 0.25
    max_output_bytes: int = 65536
    max_cpu_seconds: int = 30
    max_file_bytes: int = 16 * 1024 * 1024
    max_open_files: int = 64
    allow_network_egress: bool = False
    idempotency: str = "non-idempotent"  # idempotent | keyed | non-idempotent
    explicit_env: Mapping[str, str] = field(default_factory=dict)
    redaction_secrets: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessExecutionOutcome:
    """Immutable result of a bounded process execution."""

    state: str  # succeeded | failed | timed_out | cancelled | unknown_outcome
    reason_code: str
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    credentials_redacted: bool
    ambiguous_side_effects: bool
    started_utc: str
    finished_utc: str
    duration_seconds: float
    backend_used: str


class SandboxBackend:
    """Interface for OS-enforced sandbox execution wrappers."""

    backend_id: str = "unapproved"
    is_strong_isolation: bool = False
    supports_filesystem_isolation: bool = False
    supports_pid_isolation: bool = False
    supports_resource_limits: bool = False

    def wrap_command(self, spec: ProcessExecutionSpec) -> tuple[str, ...]:
        raise NotImplementedError


class FailClosedSandboxBackend(SandboxBackend):
    """Default backend before an owner-reviewed OS sandbox is explicitly enabled."""

    backend_id = "fail-closed"
    is_strong_isolation = False

    def wrap_command(self, spec: ProcessExecutionSpec) -> tuple[str, ...]:
        raise PermissionError("No owner-approved strong OS sandbox backend is active.")


class BubblewrapSandboxBackend(SandboxBackend):
    """
    Linux Bubblewrap (`bwrap`) sandbox backend with:
      - No home or credential directory mount
      - Network egress disabled by default (`--unshare-net`)
      - Isolated PID/IPC/UTS/user namespaces and `--die-with-parent`
      - Read-only `/usr`, `/bin`, `/lib`, `/lib64` and isolated `spec.cwd` bind
    """

    backend_id = "bubblewrap"
    is_strong_isolation = True
    supports_filesystem_isolation = True
    supports_pid_isolation = True
    supports_resource_limits = True

    def __init__(self, bwrap_executable: str | None = None) -> None:
        self._bwrap = bwrap_executable or shutil.which("bwrap")

    def is_available(self) -> bool:
        return bool(self._bwrap and os.path.isfile(self._bwrap) and os.access(self._bwrap, os.X_OK))

    def wrap_command(self, spec: ProcessExecutionSpec) -> tuple[str, ...]:
        if not self.is_available():
            raise PermissionError("Bubblewrap executable is not available on this host.")
        cmd: list[str] = [
            str(self._bwrap),
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
        ]
        if not spec.allow_network_egress:
            cmd.append("--unshare-net")
        for ro_dir in ("/usr", "/bin", "/lib", "/lib64", "/etc/alternatives"):
            if os.path.exists(ro_dir):
                cmd.extend(["--ro-bind", ro_dir, ro_dir])
        cmd.extend([
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--bind", spec.cwd, spec.cwd,
            "--chdir", spec.cwd,
            "--",
        ])
        cmd.extend(spec.argv)
        return tuple(cmd)


class LinuxNamespacesSandboxBackend(SandboxBackend):
    """
    WSL2/Linux unprivileged namespace sandbox backend via `unshare`:
      - Isolated user, mount, PID (`--mount-proc`), IPC, UTS namespaces
      - Host `/home`, `/root`, `/mnt`, `/tmp`, `/var/tmp`, `/opt`, `/srv` masked with private `tmpfs`
      - Only `spec.cwd` is bind-mounted writable (plus optional read-only `.venv` if `argv[0]` resides there)
      - Network egress disabled by default (`--net` omitted only when allow_network_egress=True)
      - Enforced `ulimit` resource caps (`NOFILE`, `FSIZE`, `AS`) before `exec`
    """

    backend_id = "linux-namespaces"
    is_strong_isolation = True
    supports_filesystem_isolation = True
    supports_pid_isolation = True
    supports_resource_limits = True

    _SANDBOX_SETUP_SH = (
        'WS="$1"; VENV="$2"; EXE="$3"; shift 3; '
        'mount -t tmpfs -o size=16m tmpfs /media || exit 126; '
        'mkdir -p /media/nr && mount -t tmpfs -o size=16m tmpfs /media/nr || exit 126; '
        'mkdir -p /media/nr/usr /media/nr/proc /media/nr/dev /media/nr/tmp /media/nr/workspace /media/nr/old_root || exit 126; '
        'mount --rbind /usr /media/nr/usr || exit 126; '
        'mount -o remount,ro,bind /media/nr/usr 2>/dev/null || true; '
        'mount --rbind /proc /media/nr/proc || exit 126; '
        'for d in null zero urandom random; do '
        '  [ -e "/dev/$d" ] && touch "/media/nr/dev/$d" && mount --bind "/dev/$d" "/media/nr/dev/$d"; '
        'done; '
        'mount --bind "$WS" /media/nr/workspace || exit 126; '
        'if [ -n "$VENV" ] && [ -d "$VENV" ]; then '
        '  mkdir -p /media/nr/venv && mount --bind "$VENV" /media/nr/venv || exit 126; '
        '  mount -o remount,ro,bind /media/nr/venv 2>/dev/null || true; '
        '  case "$EXE" in "$VENV"/*) EXE="/venv${EXE#"$VENV"}";; esac; '
        'fi; '
        'cd /media/nr || exit 126; '
        'ln -s usr/bin bin && ln -s usr/lib lib && ln -s usr/lib64 lib64; '
        'pivot_root . old_root || exit 126; '
        'cd /workspace || exit 126; '
        'umount -l /old_root && rmdir /old_root || exit 126; '
        'ulimit -n 256 2>/dev/null || true; '
        'ulimit -f 65536 2>/dev/null || true; '
        'ulimit -v 1572864 2>/dev/null || true; '
        'ulimit -p 32 2>/dev/null || ulimit -u 32 2>/dev/null || true; '
        'ulimit -t 15 2>/dev/null || true; '
        'exec "$EXE" "$@"'
    )

    def __init__(self, unshare_executable: str | None = None) -> None:
        self._unshare = unshare_executable or shutil.which("unshare")

    def is_available(self) -> bool:
        return bool(self._unshare and os.path.isfile(self._unshare) and os.access(self._unshare, os.X_OK))

    @staticmethod
    def _detect_venv_root(executable: str) -> str:
        exe_path = Path(os.path.abspath(executable))
        for parent in exe_path.parents:
            if (parent / "pyvenv.cfg").is_file() and str(parent).startswith(("/home/", "/root/", "/opt/", "/srv/", "/tmp/")):
                return str(parent)
        return ""

    def wrap_command(self, spec: ProcessExecutionSpec) -> tuple[str, ...]:
        if not self.is_available():
            raise PermissionError("unshare executable is not available on this host.")
        venv_root = self._detect_venv_root(spec.argv[0]) if spec.argv else ""
        ws_abs = os.path.abspath(spec.cwd)
        ws_prefix = ws_abs.rstrip("/") + "/"
        mapped_argv = [
            "/workspace" if arg == ws_abs else (
                "/workspace/" + arg[len(ws_prefix):] if arg.startswith(ws_prefix) else arg
            )
            for arg in spec.argv
        ]
        cmd: list[str] = [
            str(self._unshare),
            "--user",
            "--map-root-user",
            "--mount",
            "--pid",
            "--fork",
            "--mount-proc",
            "--ipc",
            "--uts",
        ]
        if not spec.allow_network_egress:
            cmd.append("--net")
        cmd.extend([
            "--",
            "/bin/sh",
            "-c",
            self._SANDBOX_SETUP_SH,
            "_",
            ws_abs,
            venv_root,
            *mapped_argv,
        ])
        return tuple(cmd)


def validate_fixed_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Validate that `argv` is a non-empty fixed argument vector (never a shell string)."""
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise ValueError("Process execution requires a sequence of argv strings; shell strings are forbidden.")
    items = tuple(argv)
    if not items:
        raise ValueError("argv must contain at least the executable path or name.")
    for index, arg in enumerate(items):
        if not isinstance(arg, str):
            raise ValueError(f"argv[{index}] must be a string.")
        if "\x00" in arg:
            raise ValueError(f"argv[{index}] contains a NUL byte.")
    executable = items[0].strip()
    if not executable:
        raise ValueError("argv[0] executable must not be empty.")
    if "/" in executable or "\\" in executable:
        resolved = os.path.abspath(executable)
        if ".." in Path(executable).parts:
            raise ValueError("Relative traversal in executable path is forbidden.")
        if not (os.path.isfile(resolved) and os.access(resolved, os.X_OK)):
            raise ValueError(f"Executable does not exist or is not executable: {executable!r}")
        return (resolved, *items[1:])
    which_resolved = shutil.which(executable, path=os.environ.get("PATH", _DEFAULT_SAFE_PATH))
    if not which_resolved:
        raise ValueError(f"Executable could not be resolved in PATH: {executable!r}")
    return (which_resolved, *items[1:])


def build_scrubbed_environment(
    cwd: str,
    explicit_env: Mapping[str, str] | None = None,
    host_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    Construct a minimal allowlisted environment stripped of inherited credentials and tokens.
    Sets HOME and TMPDIR to the confined working directory.
    """
    source = host_env if host_env is not None else os.environ
    scrubbed: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        val = source.get(key)
        if val:
            scrubbed[key] = val
    scrubbed.setdefault("PATH", _DEFAULT_SAFE_PATH)
    scrubbed["HOME"] = cwd
    scrubbed["TMPDIR"] = cwd

    if explicit_env:
        for key, value in explicit_env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Environment keys and values must be strings.")
            upper_key = key.upper()
            if any(token in upper_key for token in _FORBIDDEN_ENV_NAME_TOKENS):
                raise ValueError(f"Forbidden secret-bearing environment variable name: {key!r}")
            if "\x00" in key or "\x00" in value:
                raise ValueError("Environment variables must not contain NUL bytes.")
            scrubbed[key] = value
    return scrubbed


class BoundedProcessBroker:
    """
    Executes validated fixed-argv commands under strict resource, timeout,
    output-size, environment, and sandbox constraints.
    """

    def __init__(
        self,
        *,
        lease_manager: ResourceLeaseManager | None = None,
        sandbox_backend: SandboxBackend | None = None,
    ) -> None:
        self._leases = lease_manager or ResourceLeaseManager()
        self._sandbox = sandbox_backend or FailClosedSandboxBackend()

    def execute(
        self,
        spec: ProcessExecutionSpec,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ProcessExecutionOutcome:
        started_dt = datetime.now(timezone.utc)
        started_mono = time.monotonic()

        # 1. Fail closed before spawning if sandbox-required and no strong sandbox is approved
        if spec.execution_backend not in ("trusted-local", "sandbox-required"):
            return self._ImmediateFailure(
                started_dt,
                started_mono,
                reason_code="backend.unknown",
                error=f"Unsupported execution backend: {spec.execution_backend!r}",
            )

        if spec.execution_backend == "sandbox-required":
            if not self._sandbox.is_strong_isolation:
                return self._ImmediateFailure(
                    started_dt,
                    started_mono,
                    reason_code="sandbox.unavailable_or_unapproved",
                    error="Sandbox-required task rejected: no owner-approved strong OS sandbox backend is active.",
                )

        if spec.allow_network_egress and spec.execution_backend == "sandbox-required":
            return self._ImmediateFailure(
                started_dt,
                started_mono,
                reason_code="sandbox.network_egress_denied",
                error="Default sandbox network egress is disabled by M3 policy.",
            )

        # 2. Validate working directory and fixed argv
        try:
            cwd_path = Path(spec.cwd).resolve(strict=True)
            if not cwd_path.is_dir():
                raise ValueError(f"Working directory is not a directory: {spec.cwd!r}")
            validated_argv = validate_fixed_argv(spec.argv)
            env = build_scrubbed_environment(str(cwd_path), spec.explicit_env)
            if spec.execution_backend == "sandbox-required":
                normalized_spec = ProcessExecutionSpec(
                    argv=validated_argv,
                    cwd=str(cwd_path),
                    execution_backend=spec.execution_backend,
                    resource_class=spec.resource_class,
                    timeout_seconds=spec.timeout_seconds,
                    grace_period_seconds=spec.grace_period_seconds,
                    max_output_bytes=spec.max_output_bytes,
                    max_cpu_seconds=spec.max_cpu_seconds,
                    max_file_bytes=spec.max_file_bytes,
                    max_open_files=spec.max_open_files,
                    allow_network_egress=spec.allow_network_egress,
                    idempotency=spec.idempotency,
                    explicit_env=spec.explicit_env,
                    redaction_secrets=spec.redaction_secrets,
                )
                command_argv = self._sandbox.wrap_command(normalized_spec)
                backend_used = self._sandbox.backend_id
            else:
                command_argv = validated_argv
                backend_used = "trusted-local"
        except PermissionError as exc:
            return self._ImmediateFailure(
                started_dt,
                started_mono,
                reason_code="sandbox.unavailable_or_unapproved",
                error=str(exc),
            )
        except Exception as exc:
            return self._ImmediateFailure(
                started_dt,
                started_mono,
                reason_code="validation.invalid_spec",
                error=str(exc),
            )

        # 3. Acquire resource lease
        try:
            lease = self._leases.acquire(spec.resource_class)
        except ResourceLeaseError as exc:
            return self._ImmediateFailure(
                started_dt,
                started_mono,
                reason_code="resource.lease_exhausted",
                error=str(exc),
            )

        with lease:
            return self._run_supervised(
                command_argv=command_argv,
                cwd=str(cwd_path),
                env=env,
                spec=spec,
                backend_used=backend_used,
                started_dt=started_dt,
                started_mono=started_mono,
                cancellation_token=cancellation_token,
            )

    def _run_supervised(
        self,
        *,
        command_argv: tuple[str, ...],
        cwd: str,
        env: dict[str, str],
        spec: ProcessExecutionSpec,
        backend_used: str,
        started_dt: datetime,
        started_mono: float,
        cancellation_token: CancellationToken | None,
    ) -> ProcessExecutionOutcome:
        def _apply_limits() -> None:
            if _posix_resource is None:
                return
            try:
                cpu = max(1, int(spec.max_cpu_seconds))
                _posix_resource.setrlimit(_posix_resource.RLIMIT_CPU, (cpu, cpu))
            except (ValueError, OSError):
                pass
            try:
                fsize = max(4096, int(spec.max_file_bytes))
                _posix_resource.setrlimit(_posix_resource.RLIMIT_FSIZE, (fsize, fsize))
            except (ValueError, OSError):
                pass

        try:
            proc = subprocess.Popen(
                command_argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
                start_new_session=True,
                preexec_fn=_apply_limits if os.name == "posix" else None,
            )
        except Exception as exc:
            return self._ImmediateFailure(
                started_dt,
                started_mono,
                reason_code="process.spawn_failed",
                error=str(exc),
                backend_used=backend_used,
            )

        stdout_buf = bytearray()
        stderr_buf = bytearray()
        stdout_truncated = False
        stderr_truncated = False
        timed_out = False
        cancelled = False
        cap = max(256, int(spec.max_output_bytes))
        deadline = started_mono + max(0.05, float(spec.timeout_seconds))

        fds = [f for f in (proc.stdout, proc.stderr) if f is not None]
        while fds or proc.poll() is None:
            now = time.monotonic()
            if cancellation_token is not None and cancellation_token.is_cancelled:
                cancelled = True
                self._terminate_process_tree(proc, spec.grace_period_seconds)
                break
            if now >= deadline:
                timed_out = True
                self._terminate_process_tree(proc, spec.grace_period_seconds)
                break

            if fds:
                ready, _, _ = select.select(fds, [], [], 0.03)
                for stream in ready:
                    chunk = os.read(stream.fileno(), 4096)
                    if not chunk:
                        fds.remove(stream)
                        continue
                    if stream is proc.stdout:
                        remaining = cap - len(stdout_buf)
                        if remaining > 0:
                            stdout_buf.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            stdout_truncated = True
                    else:
                        remaining = cap - len(stderr_buf)
                        if remaining > 0:
                            stderr_buf.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            stderr_truncated = True
            else:
                time.sleep(0.02)

        # Drain any remaining bytes up to cap after process exit
        for stream, buf_ref, is_stdout in (
            (proc.stdout, stdout_buf, True),
            (proc.stderr, stderr_buf, False),
        ):
            if stream is None:
                continue
            try:
                while True:
                    ready, _, _ = select.select([stream], [], [], 0.0)
                    if not ready:
                        break
                    chunk = os.read(stream.fileno(), 4096)
                    if not chunk:
                        break
                    remaining = cap - len(buf_ref)
                    if remaining > 0:
                        buf_ref.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        if is_stdout:
                            stdout_truncated = True
                        else:
                            stderr_truncated = True
            except OSError:
                pass
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        exit_code = proc.poll()
        finished_dt = datetime.now(timezone.utc)
        duration = max(0.0, time.monotonic() - started_mono)

        raw_stdout = stdout_buf.decode("utf-8", errors="replace")
        raw_stderr = stderr_buf.decode("utf-8", errors="replace")
        clean_stdout, out_redacted = redact_sensitive_text(raw_stdout, spec.redaction_secrets)
        clean_stderr, err_redacted = redact_sensitive_text(raw_stderr, spec.redaction_secrets)

        is_non_idempotent = spec.idempotency != "idempotent"
        if cancelled:
            state = "unknown_outcome" if is_non_idempotent else "cancelled"
            reason = cancellation_token.reason if cancellation_token else "process.cancelled"
            ambiguous = is_non_idempotent
        elif timed_out:
            state = "unknown_outcome" if is_non_idempotent else "timed_out"
            reason = "process.timed_out"
            ambiguous = is_non_idempotent
        elif exit_code == 0:
            state = "succeeded"
            reason = "process.exit_zero"
            ambiguous = False
        else:
            state = "failed"
            reason = f"process.nonzero_exit.{exit_code}"
            ambiguous = False

        return ProcessExecutionOutcome(
            state=state,
            reason_code=reason,
            exit_code=exit_code,
            stdout=clean_stdout,
            stderr=clean_stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            credentials_redacted=out_redacted or err_redacted,
            ambiguous_side_effects=ambiguous,
            started_utc=started_dt.isoformat(),
            finished_utc=finished_dt.isoformat(),
            duration_seconds=duration,
            backend_used=backend_used,
        )

    @staticmethod
    def _terminate_process_tree(proc: subprocess.Popen, grace_seconds: float) -> None:
        if proc.poll() is not None:
            return
        if os.name == "posix":
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return
            except OSError:
                proc.terminate()
        else:
            proc.terminate()

        deadline = time.monotonic() + max(0.05, grace_seconds)
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.02)

        if os.name == "posix":
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                proc.kill()
        else:
            proc.kill()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass

    @staticmethod
    def _ImmediateFailure(
        started_dt: datetime,
        started_mono: float,
        *,
        reason_code: str,
        error: str,
        backend_used: str = "none",
    ) -> ProcessExecutionOutcome:
        finished_dt = datetime.now(timezone.utc)
        clean_err, redacted = redact_sensitive_text(error)
        return ProcessExecutionOutcome(
            state="failed",
            reason_code=reason_code,
            exit_code=None,
            stdout="",
            stderr=clean_err,
            stdout_truncated=False,
            stderr_truncated=False,
            credentials_redacted=redacted,
            ambiguous_side_effects=False,
            started_utc=started_dt.isoformat(),
            finished_utc=finished_dt.isoformat(),
            duration_seconds=max(0.0, time.monotonic() - started_mono),
            backend_used=backend_used,
        )


@dataclass(frozen=True)
class CollectedArtifact:
    relative_path: str
    sha256: str
    size_bytes: int
    credentials_redacted: bool


class StagedPatchWorkspace:
    """
    Isolated staging workspace for candidate edits or tool outputs.
    Keeps the owner's primary checkout completely untouched until explicit
    owner promotion (`promote_reviewed_patch`).
    """

    def __init__(self, primary_root: str | Path, *, redaction_secrets: Sequence[str] = ()) -> None:
        self.primary_root = Path(primary_root).resolve(strict=True)
        if not self.primary_root.is_dir():
            raise ValueError(f"Primary root must be an existing directory: {primary_root}")
        self._redaction_secrets = tuple(redaction_secrets)
        self._temp_dir = Path(tempfile.mkdtemp(prefix="lila-stage-"))
        self.staging_root = self._temp_dir / "workspace"
        self._baseline_digests = self._snapshot_tree(self.primary_root)
        self._copy_clean_tree(self.primary_root, self.staging_root)
        self._closed = False

    @staticmethod
    def _should_ignore(rel_parts: tuple[str, ...]) -> bool:
        ignored = {".git", "__pycache__", ".venv", "venv", "node_modules", "bin", "obj"}
        return any(part in ignored or part.startswith(".previous-") for part in rel_parts)

    def _snapshot_tree(self, root: Path) -> dict[str, bytes]:
        snap: dict[str, bytes] = {}
        for current_root, dirs, files in os.walk(root, followlinks=False):
            rel_dir = Path(current_root).relative_to(root)
            dirs[:] = [d for d in dirs if not self._should_ignore((*rel_dir.parts, d))]
            for name in sorted(files):
                rel_parts = (*rel_dir.parts, name)
                if self._should_ignore(rel_parts):
                    continue
                full = Path(current_root) / name
                if full.is_symlink() or not full.is_file():
                    continue
                rel_key = "/".join(rel_parts)
                snap[rel_key] = full.read_bytes()
        return snap

    def _copy_clean_tree(self, src: Path, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for rel_key, data in self._baseline_digests.items():
            target = dst / rel_key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def verify_primary_unchanged(self) -> bool:
        """Confirm that the owner's primary checkout has not been mutated during staging."""
        current = self._snapshot_tree(self.primary_root)
        return current == self._baseline_digests

    def collect_unified_diff(self) -> str:
        """Return a deterministic unified diff between primary baseline and staged workspace."""
        staged = self._snapshot_tree(self.staging_root)
        all_keys = sorted(set(self._baseline_digests) | set(staged))
        diffs: list[str] = []
        for key in all_keys:
            before_bytes = self._baseline_digests.get(key)
            after_bytes = staged.get(key)
            if before_bytes == after_bytes:
                continue
            before_lines = (
                before_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
                if before_bytes is not None
                else []
            )
            after_lines = (
                after_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
                if after_bytes is not None
                else []
            )
            diff = list(
                difflib.unified_diff(
                    before_lines,
                    after_lines,
                    fromfile=f"a/{key}" if before_bytes is not None else "/dev/null",
                    tofile=f"b/{key}" if after_bytes is not None else "/dev/null",
                )
            )
            if diff:
                diffs.append("".join(diff))
        raw_diff = "\n".join(diffs)
        clean_diff, _ = redact_sensitive_text(raw_diff, self._redaction_secrets)
        return clean_diff

    def collect_sanitized_artifacts(self) -> tuple[CollectedArtifact, ...]:
        """Scan staged modified/added files, redact any detected credentials in-place, and return digests."""
        staged = self._snapshot_tree(self.staging_root)
        artifacts: list[CollectedArtifact] = []
        for key in sorted(staged):
            if staged.get(key) == self._baseline_digests.get(key):
                continue
            full = self.staging_root / key
            text = full.read_text(encoding="utf-8", errors="replace")
            clean_text, redacted = redact_sensitive_text(text, self._redaction_secrets)
            if redacted:
                full.write_text(clean_text, encoding="utf-8")
            raw_bytes = full.read_bytes()
            artifacts.append(
                CollectedArtifact(
                    relative_path=key,
                    sha256=hashlib.sha256(raw_bytes).hexdigest(),
                    size_bytes=len(raw_bytes),
                    credentials_redacted=redacted,
                )
            )
        return tuple(artifacts)

    def promote_reviewed_patch(self, *, owner_approved: bool, expected_diff_sha256: str) -> tuple[str, ...]:
        """
        Apply staged modifications to the primary checkout ONLY after explicit owner approval
        and verification that the staged diff digest matches the reviewed diff.
        """
        if not owner_approved:
            raise PermissionError("Staged patch promotion requires explicit owner approval.")
        if not self.verify_primary_unchanged():
            raise RuntimeError("Primary checkout drifted since staging began; refusing automatic promotion.")

        diff_text = self.collect_unified_diff()
        actual_sha = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
        if actual_sha != expected_diff_sha256:
            raise ValueError("Reviewed diff digest mismatch; refusing patch promotion.")

        # Ensure artifacts are scrubbed before promotion
        self.collect_sanitized_artifacts()
        staged = self._snapshot_tree(self.staging_root)
        promoted: list[str] = []

        for key in sorted(set(self._baseline_digests) | set(staged)):
            if key not in staged and key in self._baseline_digests:
                target = self.primary_root / key
                if target.exists():
                    target.unlink()
                promoted.append(key)
            elif staged.get(key) != self._baseline_digests.get(key):
                target = self.primary_root / key
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(staged[key])
                promoted.append(key)

        self._baseline_digests = self._snapshot_tree(self.primary_root)
        return tuple(promoted)

    def cleanup(self) -> None:
        if not self._closed:
            self._closed = True
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __enter__(self) -> "StagedPatchWorkspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()
