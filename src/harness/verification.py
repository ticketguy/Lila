"""Milestone M4 (WS-VERIFY): Independent Observation, Verification, and Trace Store.

Separates process exit / handler return (`success == True` / `returncode == 0`) from
independent postcondition verification, records privacy-minimized correlated traces,
and formats structured subsystem failures so Lila Core can report them honestly.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(secret|token|password|credential|authorization|api[_-]?key|private[_-]?key|proof|phrase|content|payload|body|output|transcript|prompt|message)"
)


class VerificationStatus(str, Enum):
    """Independent postcondition verification outcomes."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class VerificationEvidence:
    """Immutable independent observation of a tool or subsystem execution."""

    verifier_id: str
    status: VerificationStatus
    summary: str
    checks_run: tuple[str, ...] = ()
    observed_digest: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier_id": self.verifier_id,
            "status": self.status.value,
            "summary": self.summary,
            "checks_run": list(self.checks_run),
            "observed_digest": self.observed_digest,
            "details": dict(self.details),
        }


def sanitize_trace_fields(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Strip sensitive keys and bound string lengths for privacy-minimized trace logs."""
    if not payload:
        return {}
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        key_str = str(key)
        if _SENSITIVE_KEY_RE.search(key_str):
            clean[key_str] = "[REDACTED]"
        elif isinstance(value, Mapping):
            clean[key_str] = sanitize_trace_fields(value)
        elif isinstance(value, (list, tuple)):
            clean[key_str] = [
                sanitize_trace_fields(v) if isinstance(v, Mapping) else str(v)[:120]
                for v in list(value)[:10]
            ]
        elif isinstance(value, (int, float, bool)) or value is None:
            clean[key_str] = value
        else:
            clean[key_str] = str(value)[:160]
    return clean


def format_failure_for_lila(
    subsystem: str,
    error_code: str,
    detail: str,
    *,
    tool_name: str | None = None,
    recoverable: bool = True,
) -> dict[str, Any]:
    """Structured failure payload handed to Lila Core so Lila reports failures herself."""
    clean_detail = " ".join(str(detail or "Unknown failure").split())[:260]
    return {
        "kind": "subsystem_failure",
        "subsystem": subsystem,
        "tool": tool_name,
        "error_code": error_code,
        "detail": clean_detail,
        "recoverable": bool(recoverable),
        "lila_prompt_context": (
            f"[SYSTEM FAILURE REPORT from {subsystem}"
            + (f" ({tool_name})" if tool_name else "")
            + f": code={error_code}; detail={clean_detail}; recoverable={recoverable}. "
            + "Report this failure honestly to the owner without guessing or claiming success.]"
        ),
    }


class FilePostconditionVerifier:
    """Independently verifies file writes/appends/deletes on disk + Python AST syntax."""

    verifier_id = "verifier.filesystem.postcondition.v1"

    def verify(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        reported_success: bool,
    ) -> VerificationEvidence:
        if not reported_success:
            return VerificationEvidence(
                verifier_id=self.verifier_id,
                status=VerificationStatus.FAILED,
                summary="Handler reported failure before postcondition check.",
                checks_run=("handler_reported_failure",),
            )

        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return VerificationEvidence(
                verifier_id=self.verifier_id,
                status=VerificationStatus.UNVERIFIED,
                summary="No target path available for independent file verification.",
                checks_run=("path_present",),
            )

        path = Path(raw_path).expanduser()
        if tool_name == "file_delete":
            exists = path.exists()
            return VerificationEvidence(
                verifier_id=self.verifier_id,
                status=VerificationStatus.FAILED if exists else VerificationStatus.VERIFIED,
                summary="Target still exists after delete." if exists else "Confirmed file removed from disk.",
                checks_run=("exists_after_delete",),
                details={"exists": exists},
            )

        if not path.is_file():
            return VerificationEvidence(
                verifier_id=self.verifier_id,
                status=VerificationStatus.FAILED,
                summary="Expected file does not exist on disk after write.",
                checks_run=("file_exists",),
                details={"exists": False},
            )

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            return VerificationEvidence(
                verifier_id=self.verifier_id,
                status=VerificationStatus.FAILED,
                summary=f"Could not read file for verification: {exc}",
                checks_run=("file_exists", "file_readable"),
            )

        digest = hashlib.sha256(raw_bytes).hexdigest()
        checks = ["file_exists", "sha256_digest"]
        details: dict[str, Any] = {"size_bytes": len(raw_bytes)}

        if path.suffix.lower() == ".py":
            checks.append("python_ast_parse")
            try:
                ast.parse(raw_bytes.decode("utf-8"), filename=str(path))
                details["syntax_ok"] = True
            except (SyntaxError, UnicodeDecodeError) as exc:
                details["syntax_ok"] = False
                return VerificationEvidence(
                    verifier_id=self.verifier_id,
                    status=VerificationStatus.FAILED,
                    summary=f"Python syntax check failed after write: {exc}",
                    checks_run=tuple(checks),
                    observed_digest=digest,
                    details=details,
                )

        return VerificationEvidence(
            verifier_id=self.verifier_id,
            status=VerificationStatus.VERIFIED,
            summary=f"Verified file on disk ({len(raw_bytes)} bytes).",
            checks_run=tuple(checks),
            observed_digest=digest,
            details=details,
        )


class CommandPostconditionVerifier:
    """Separates command exit code (`returncode == 0`) from independent task verification."""

    verifier_id = "verifier.process.postcondition.v1"

    def verify(
        self,
        args: Mapping[str, Any],
        reported_success: bool,
        data: Any = None,
    ) -> VerificationEvidence:
        if isinstance(data, Mapping) and data.get("ambiguous_side_effects"):
            return VerificationEvidence(
                verifier_id=self.verifier_id,
                status=VerificationStatus.AMBIGUOUS,
                summary="Process was interrupted/timed out; side effects are uncertain.",
                checks_run=("ambiguous_side_effects_check",),
            )
        if not reported_success:
            return VerificationEvidence(
                verifier_id=self.verifier_id,
                status=VerificationStatus.FAILED,
                summary="Process exited non-zero or failed execution.",
                checks_run=("exit_status",),
            )

        expected_path = args.get("expected_path")
        if isinstance(expected_path, str) and expected_path.strip():
            target = Path(expected_path).expanduser()
            if not target.exists():
                return VerificationEvidence(
                    verifier_id=self.verifier_id,
                    status=VerificationStatus.FAILED,
                    summary="Process exited 0, but expected artifact path is missing.",
                    checks_run=("exit_status", "expected_path_exists"),
                )
            return VerificationEvidence(
                verifier_id=self.verifier_id,
                status=VerificationStatus.VERIFIED,
                summary="Process exited 0 and expected artifact exists.",
                checks_run=("exit_status", "expected_path_exists"),
            )

        # Exit code 0 alone proves execution completed, NOT independent task verification.
        return VerificationEvidence(
            verifier_id=self.verifier_id,
            status=VerificationStatus.UNVERIFIED,
            summary="Process exited 0; completed without independent postcondition verifier.",
            checks_run=("exit_status",),
        )


class ServiceHealthProbeVerifier:
    """Independent health probe verifier (HTTP or custom probe)."""

    verifier_id = "verifier.service.health_probe.v1"

    def probe_http(self, url: str, timeout_seconds: float = 2.5) -> VerificationEvidence:
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
                ok = 200 <= getattr(resp, "status", 500) < 300
                return VerificationEvidence(
                    verifier_id=self.verifier_id,
                    status=VerificationStatus.VERIFIED if ok else VerificationStatus.FAILED,
                    summary=f"Health probe returned HTTP {getattr(resp, 'status', '?')}.",
                    checks_run=("http_health_probe",),
                    details={"status_code": getattr(resp, "status", None)},
                )
        except Exception as exc:
            return VerificationEvidence(
                verifier_id=self.verifier_id,
                status=VerificationStatus.FAILED,
                summary=f"Health probe failed: {exc}",
                checks_run=("http_health_probe",),
            )


class IndependentVerifierRegistry:
    """Routes tool executions to independent postcondition verifiers."""

    def __init__(self) -> None:
        self._file_verifier = FilePostconditionVerifier()
        self._cmd_verifier = CommandPostconditionVerifier()
        self._custom: dict[str, Callable[[Mapping[str, Any], bool, Any], VerificationEvidence]] = {}

    def register(
        self,
        tool_name: str,
        verifier: Callable[[Mapping[str, Any], bool, Any], VerificationEvidence],
    ) -> None:
        self._custom[tool_name] = verifier

    def verify(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        reported_success: bool,
        data: Any = None,
    ) -> VerificationEvidence:
        if tool_name in self._custom:
            return self._custom[tool_name](args, reported_success, data)

        if tool_name in {"file_write", "file_append", "file_delete"}:
            return self._file_verifier.verify(tool_name, args, reported_success)

        if tool_name in {"bash", "ssh_exec"}:
            return self._cmd_verifier.verify(args, reported_success, data)

        if not reported_success:
            return VerificationEvidence(
                verifier_id="verifier.observation.v1",
                status=VerificationStatus.FAILED,
                summary="Execution failed.",
                checks_run=("reported_status",),
            )

        if tool_name in {
            "file_read", "file_list", "file_search", "web_search", "web_fetch",
            "self_status", "island_control", "screen_present",
        }:
            return VerificationEvidence(
                verifier_id="verifier.observation.v1",
                status=VerificationStatus.VERIFIED,
                summary="Observation/UI result produced cleanly.",
                checks_run=("reported_status", "structural_observation"),
            )

        return VerificationEvidence(
            verifier_id="verifier.default.v1",
            status=VerificationStatus.UNVERIFIED,
            summary="Completed; no independent postcondition verifier registered.",
            checks_run=("reported_status",),
        )


class CorrelatedTraceStore:
    """Append-only, correlation-linked, privacy-minimized JSONL trace store."""

    def __init__(self, trace_path: Path | str) -> None:
        self._path = Path(trace_path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        *,
        trace_id: str,
        tool_name: str,
        lifecycle_stage: str,
        evidence: VerificationEvidence,
        invocation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        elapsed_ms: float | None = None,
    ) -> dict[str, Any]:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "invocation_id": invocation_id,
            "tool": tool_name,
            "lifecycle_stage": lifecycle_stage,
            "verification": evidence.to_dict(),
            "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms is not None else None,
            "metadata": sanitize_trace_fields(metadata),
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return record

    def read_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8", errors="replace").splitlines()
        out: list[dict[str, Any]] = []
        for raw in lines[-max(1, limit):]:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out
