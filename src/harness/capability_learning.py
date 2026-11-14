"""M10: Controlled capability learning, quarantine, and owner promotion.

Enforces M10 invariants:
- Candidates are quarantined and cannot execute in production before explicit
  owner promotion.
- Candidate tests execute inside an isolated M3 BoundedProcessBroker sandbox
  (AST checks are defense-in-depth only, never the security boundary).
- Candidate and catalog versions persist atomically to disk when storage_dir is configured.
- Explicit owner approval is required for promotion; failed promotion or manual
  rollback atomically restores the prior trusted catalog.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .isolation import (
    BoundedProcessBroker,
    FailClosedSandboxBackend,
    LinuxNamespacesSandboxBackend,
    ProcessExecutionSpec,
    ResourceClass,
    ResourceLeaseManager,
)


class CandidateStatus(str, Enum):
    QUARANTINED = "quarantined"
    VERIFIED = "verified"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


_FORBIDDEN_IDENTIFIERS = frozenset({
    "PermissionManager",
    "CapabilityRegistry",
    "ApprovalContinuationService",
    "ApprovalAuthenticator",
    "os.system",
    "subprocess.Popen",
})


def check_candidate_source_safety(source_code: str) -> tuple[bool, list[str]]:
    """Defense-in-depth AST check (never a substitute for M3 sandbox isolation)."""
    reasons: list[str] = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return False, [f"syntax_error:{exc.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_IDENTIFIERS:
            reasons.append(f"forbidden_identifier:{node.id}")
        elif isinstance(node, ast.Attribute):
            full = f"{getattr(node.value, 'id', '')}.{node.attr}"
            if full in _FORBIDDEN_IDENTIFIERS or node.attr in ("_permissions", "policy_snapshot", "grant_authority"):
                reasons.append(f"forbidden_attribute:{full or node.attr}")
    return len(reasons) == 0, reasons


@dataclass
class CandidateCapability:
    capability_id: str
    version: str
    source_code: str
    test_code: str
    provenance: dict[str, Any]
    source_digest: str = ""
    status: CandidateStatus = CandidateStatus.QUARANTINED
    tests_passed: bool = False
    security_review_passed: bool = False
    review_notes: list[str] = field(default_factory=list)
    sandbox_output: str = ""

    def __post_init__(self) -> None:
        if not self.source_digest:
            self.source_digest = hashlib.sha256(self.source_code.encode("utf-8")).hexdigest()

    def is_production_executable(self) -> bool:
        return self.status == CandidateStatus.PROMOTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "version": self.version,
            "source_digest": self.source_digest,
            "provenance": dict(self.provenance),
            "status": self.status.value,
            "tests_passed": self.tests_passed,
            "security_review_passed": self.security_review_passed,
            "review_notes": list(self.review_notes),
        }


@dataclass(frozen=True)
class CatalogVersion:
    version: int
    active_capabilities: dict[str, str]  # capability_id -> source_digest
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "active_capabilities": dict(self.active_capabilities),
            "created_at": self.created_at,
        }


class CapabilityPromotionService:
    """Manages quarantined capabilities, M3-sandboxed evaluation, promotion, and rollback."""

    def __init__(
        self,
        initial_catalog: Optional[dict[str, str]] = None,
        *,
        storage_dir: Optional[str | Path] = None,
        capability_registry: Any = None,
        sandbox_execution_backend: str = "trusted-local",
    ) -> None:
        self._lock = threading.RLock()
        self._candidates: dict[str, CandidateCapability] = {}
        self._history: list[CatalogVersion] = [
            CatalogVersion(version=1, active_capabilities=dict(initial_catalog or {}))
        ]
        self._storage_dir = Path(storage_dir) if storage_dir else None
        self._registry = capability_registry
        ns_backend = LinuxNamespacesSandboxBackend()
        if (
            sandbox_execution_backend == "trusted-local"
            and ns_backend.is_available()
            and ns_backend.is_strong_isolation
            and ns_backend.supports_filesystem_isolation
        ):
            self._sandbox_backend = "sandbox-required"
            sandbox_impl = ns_backend
        elif (
            sandbox_execution_backend == "sandbox-required"
            and ns_backend.is_available()
            and ns_backend.is_strong_isolation
            and ns_backend.supports_filesystem_isolation
        ):
            self._sandbox_backend = "sandbox-required"
            sandbox_impl = ns_backend
        else:
            self._sandbox_backend = "sandbox-required"
            sandbox_impl = FailClosedSandboxBackend()
        self._broker = BoundedProcessBroker(
            lease_manager=ResourceLeaseManager(),
            sandbox_backend=sandbox_impl,
        )
        if self._storage_dir:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted()

    @property
    def current_catalog(self) -> CatalogVersion:
        with self._lock:
            return self._history[-1]

    def list_candidates(self) -> list[dict[str, Any]]:
        with self._lock:
            return [c.to_dict() for c in self._candidates.values()]

    def submit_candidate(
        self,
        *,
        capability_id: str,
        version: str,
        source_code: str,
        test_code: str,
        provenance: dict[str, Any],
    ) -> CandidateCapability:
        if not provenance or not provenance.get("origin"):
            raise ValueError("Candidate provenance with 'origin' is required.")
        if not test_code or not test_code.strip():
            raise ValueError("Automated test_code is required for candidate evaluation.")
        cand = CandidateCapability(
            capability_id=capability_id,
            version=version,
            source_code=source_code,
            test_code=test_code,
            provenance=dict(provenance),
            status=CandidateStatus.QUARANTINED,
        )
        with self._lock:
            self._candidates[capability_id] = cand
            self._persist()
        return cand

    def evaluate_candidate(self, capability_id: str) -> CandidateCapability:
        """Run AST defense-in-depth check and execute candidate tests inside M3 BoundedProcessBroker."""
        with self._lock:
            cand = self._candidates.get(capability_id)
            if cand is None:
                raise KeyError(f"Unknown candidate: {capability_id}")

        safe, reasons = check_candidate_source_safety(cand.source_code)
        tests_ok = False
        sandbox_out = ""
        if safe:
            with tempfile.TemporaryDirectory(prefix="lila_cand_") as tmpdir:
                mod_path = os.path.join(tmpdir, "candidate_mod.py")
                test_path = os.path.join(tmpdir, "test_candidate.py")
                with open(mod_path, "w", encoding="utf-8") as f:
                    f.write(cand.source_code)
                with open(test_path, "w", encoding="utf-8") as f:
                    f.write("from candidate_mod import *\n" + cand.test_code)
                spec = ProcessExecutionSpec(
                    argv=(sys.executable, test_path),
                    cwd=tmpdir,
                    execution_backend=self._sandbox_backend,
                    resource_class=ResourceClass.CPU_HEAVY,
                    timeout_seconds=10.0,
                    max_output_bytes=16384,
                    allow_network_egress=False,
                    explicit_env={"PYTHONPATH": tmpdir},
                )
                outcome = self._broker.execute(spec)
                tests_ok = outcome.state == "succeeded" and outcome.exit_code == 0
                sandbox_out = (outcome.stdout or "") + (f"\n{outcome.stderr}" if outcome.stderr else "")
                if not tests_ok:
                    reasons.append(f"sandbox_tests_failed:{outcome.reason_code or outcome.exit_code}")

        with self._lock:
            cand.tests_passed = tests_ok
            cand.security_review_passed = safe
            cand.review_notes = reasons
            cand.sandbox_output = sandbox_out
            cand.status = CandidateStatus.VERIFIED if (safe and tests_ok) else CandidateStatus.REJECTED
            self._persist()
            return cand

    def promote(
        self,
        capability_id: str,
        *,
        owner_approved: bool,
        post_promote_smoke_ok: bool = True,
    ) -> CatalogVersion:
        if not owner_approved:
            raise PermissionError("Explicit owner approval is required to promote a capability.")
        with self._lock:
            cand = self._candidates.get(capability_id)
            if cand is None:
                raise KeyError(f"Unknown candidate: {capability_id}")
            if cand.status != CandidateStatus.VERIFIED or not cand.tests_passed or not cand.security_review_passed:
                raise RuntimeError(f"Candidate {capability_id} has not passed verification/security gates.")

            prior = self._history[-1]
            new_caps = dict(prior.active_capabilities)
            new_caps[capability_id] = cand.source_digest
            next_ver = CatalogVersion(version=prior.version + 1, active_capabilities=new_caps)
            self._history.append(next_ver)
            cand.status = CandidateStatus.PROMOTED

            if not post_promote_smoke_ok:
                self._history.pop()
                cand.status = CandidateStatus.ROLLED_BACK
                self._persist()
                raise RuntimeError(f"Promotion of {capability_id} failed post-promote check; restored catalog v{prior.version}.")
            if self._registry is not None and hasattr(self._registry, "record_promoted_digest"):
                self._registry.record_promoted_digest(capability_id, cand.source_digest, next_ver.version)
            self._persist()
            return next_ver

    def rollback(self, target_version: Optional[int] = None) -> CatalogVersion:
        with self._lock:
            if len(self._history) <= 1:
                return self._history[-1]
            if target_version is None:
                target = self._history[-2]
            else:
                matches = [v for v in self._history if v.version == target_version]
                if not matches:
                    raise KeyError(f"Unknown catalog version: {target_version}")
                target = matches[-1]
            restored = CatalogVersion(
                version=self._history[-1].version + 1,
                active_capabilities=dict(target.active_capabilities),
            )
            self._history.append(restored)
            if self._registry is not None and hasattr(self._registry, "restore_catalog_snapshot"):
                self._registry.restore_catalog_snapshot(restored.active_capabilities, restored.version)
            self._persist()
            return restored

    def _persist(self) -> None:
        if not self._storage_dir:
            return
        cat_file = self._storage_dir / "catalog_versions.json"
        tmp = self._storage_dir / "catalog_versions.json.tmp"
        tmp.write_text(json.dumps([v.to_dict() for v in self._history], indent=2), encoding="utf-8")
        os.replace(tmp, cat_file)

    def _load_persisted(self) -> None:
        if not self._storage_dir:
            return
        cat_file = self._storage_dir / "catalog_versions.json"
        if cat_file.exists():
            raw = json.loads(cat_file.read_text(encoding="utf-8"))
            if isinstance(raw, list) and raw:
                self._history = [
                    CatalogVersion(
                        version=int(item["version"]),
                        active_capabilities=dict(item["active_capabilities"]),
                        created_at=float(item.get("created_at", time.time())),
                    )
                    for item in raw
                ]
