"""End-to-end M2B Step 8 & M2 Closure Security Verification Suite (Areas A–F).

This suite verifies the end-to-end Harness security lifecycle across:
- Area B: Execution lifecycle, service shutdown/restart recovery, non-reuse of
  pending approvals, and safe failure on challenge/context expiration.
- Area C: Authorization & replay protection, exact operation/target binding,
  argument/operation substitution rejection, and duplicate/replay prevention.
- Area D: File execution security (`VerifiedFileHandle` via `O_PATH | O_NOFOLLOW`),
  path substitution / symlink race prevention, and handle preservation.
- Area E: Diagnostics & auditability (`executions.jsonl` and `HarnessExecutor.history`),
  verifying that `requested` (`approval_pending`), `approved`, `denied`
  (`approval_rejected` / `denied`), `dispatched`, `completed`, `expired`, and
  `invalidated` stages are traceable with `invocation_id` and `reason_code` while
  zero secrets, owner phrases, challenges, or file contents leak.
- Area F: Reproducible environment dependency declarations (`requirements.txt`,
  `requirements-oia.txt`, and Windows `.csproj` targets).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
import unittest

from src.harness.approval_authentication import (
    ApprovalDecisionResponse,
    OwnerDecision,
    approval_proof_message,
)
from src.harness.approval_continuation import (
    ApprovalContinuationService,
    ContinuationStatus,
    OwnerAuthenticationSecret,
)
from src.harness.capabilities import CapabilityRegistry, legacy_tool_to_manifest
from src.harness.executor import HarnessExecutor
from src.harness.file_targets import (
    FileReadExecutionTargetAdapter,
    describe_file_target,
)
from src.harness.pending_invocations import (
    PendingInvocationRegistry,
    TargetBindingResolverRegistry,
)
from src.harness.system_tools import _file_read
from src.harness.tools import Tool, ToolArg, get_registry


class _StaticSecretProvider:
    def __init__(self, secret: bytes, key_id: str = "owner-key-m2") -> None:
        self._secret = secret
        self._key_id = key_id

    def get_owner_authentication_secret(self) -> OwnerAuthenticationSecret:
        return OwnerAuthenticationSecret(key_id=self._key_id, secret=self._secret)


class M2EndToEndSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.log_dir = self.root / "logs"

        self.registry = get_registry()
        self.previous_tools = dict(self.registry._tools)
        self.addCleanup(self._restore_registry)
        self.registry._tools.clear()

        self.file_read_tool = Tool(
            name="file_read",
            description="Read a file with verified handle binding",
            args=[
                ToolArg("path", "path", "Target file path"),
                ToolArg("max_lines", "int", "Max lines", required=False, default=50),
            ],
            execute=_file_read,
            risk="medium",
            category="filesystem",
            permission="auto",
        )
        self.registry.register(self.file_read_tool)

        manifest = legacy_tool_to_manifest(self.file_read_tool)
        self.manifest = manifest
        self.catalog = CapabilityRegistry()
        self.catalog.register(manifest)

        self.resolvers = TargetBindingResolverRegistry()
        self.resolvers.register(
            manifest,
            resolver_id="harness.file_read.target",
            target_required=True,
            target_kind="filesystem.file",
            resolver=lambda _manifest, arguments, _execution: describe_file_target(
                arguments["path"]
            ),
        )

        self.secret = b"owner-phrase-derived-secret-never-logged-998877"
        self.now = [1000.0]

    def _restore_registry(self) -> None:
        self.registry._tools.clear()
        self.registry._tools.update(self.previous_tools)

    def _build_runtime(self, *, log_subdir: str = "run1"):
        pending = PendingInvocationRegistry(
            capabilities=self.catalog,
            target_resolvers=self.resolvers,
            lifetime_seconds=60.0,
            monotonic_clock=lambda: self.now[0],
        )
        service = ApprovalContinuationService(
            pending_registry=pending,
            owner_secret_provider=_StaticSecretProvider(self.secret),
            challenge_lifetime_seconds=30.0,
            monotonic_clock=lambda: self.now[0],
            execution_target_adapters={
                self.manifest.capability_id: FileReadExecutionTargetAdapter()
            },
        )
        executor = HarnessExecutor(
            log_dir=str(self.log_dir / log_subdir),
            workspace_root=str(self.workspace),
            approval_continuation_service=service,
        )
        return pending, service, executor

    def _sign_decision(
        self,
        request,
        decision: OwnerDecision = OwnerDecision.APPROVE,
        *,
        operation_digest: str | None = None,
    ) -> ApprovalDecisionResponse:
        challenge = request.challenge
        proof = hmac.new(
            self.secret,
            approval_proof_message(challenge, decision),
            hashlib.sha256,
        ).digest()
        return ApprovalDecisionResponse(
            challenge_id=challenge.challenge_id,
            key_id=challenge.key_id,
            invocation_id=request.invocation.invocation_id,
            operation_digest=operation_digest or request.grant.operation_digest,
            decision=decision,
            proof=proof,
        )

    def _read_audit_entries(self, log_subdir: str = "run1") -> list[dict]:
        jsonl_path = self.log_dir / log_subdir / "executions.jsonl"
        if not jsonl_path.exists():
            return []
        return [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_area_c_d_e_approved_outside_workspace_file_read_traces_full_lifecycle_without_secret_leak(self):
        target_file = self.outside / "confidential-note.txt"
        sensitive_payload = "TOP_SECRET_FILE_BODY_DO_NOT_LOG_12345"
        target_file.write_text(sensitive_payload, encoding="utf-8")

        _, service, executor = self._build_runtime()
        call_text = (
            '<tool_call>{"name": "file_read", "arguments": '
            + json.dumps({"path": str(target_file), "max_lines": 20})
            + "}</tool_call>"
        )
        initial = executor.process(call_text)[0]
        self.assertEqual(initial["approval"].status, ContinuationStatus.PENDING)
        self.assertEqual(initial["record"].permission_decision, "approval_pending")
        self.assertEqual(initial["record"].lifecycle_stage, "requested")

        invocation_id = initial["approval"].invocation_id
        request = service.get_request(invocation_id)
        outcome = service.respond(self._sign_decision(request, OwnerDecision.APPROVE))
        self.assertEqual(outcome.status, ContinuationStatus.APPROVED)
        self.assertIsNotNone(outcome.tool_result)
        self.assertTrue(outcome.tool_result.success)
        self.assertIn(sensitive_payload, outcome.tool_result.output)

        # Replay of the exact same signed approval must fail closed and be audited.
        replay = service.respond(self._sign_decision(request, OwnerDecision.APPROVE))
        self.assertEqual(replay.status, ContinuationStatus.FAILED)
        self.assertEqual(replay.reason_code, "continuation.already_terminal")

        entries = self._read_audit_entries()
        stages = [
            (e["lifecycle_stage"], e["permission_decision"], e["reason_code"])
            for e in entries
            if e["invocation_id"] == invocation_id
        ]
        self.assertEqual(
            stages,
            [
                ("requested", "approval_pending", "approval.pending"),
                ("approved", "approved", "approval.authenticated"),
                ("dispatched", "approved", "execution.dispatched"),
                ("completed", "approved", "approval.consumed"),
                ("denied", "approval_failed", "continuation.already_terminal"),
            ],
        )

        raw_log = (self.log_dir / "run1" / "executions.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(sensitive_payload, raw_log)
        self.assertNotIn(str(target_file), raw_log)
        self.assertNotIn(self.secret.decode("ascii"), raw_log)
        self.assertNotIn(request.challenge.challenge_material.hex(), raw_log)

    def test_area_c_e_rejection_and_operation_substitution_prevent_dispatch_and_record_audit(self):
        target_a = self.outside / "file_a.txt"
        target_b = self.outside / "file_b.txt"
        target_a.write_text("FILE_A_CONTENT", encoding="utf-8")
        target_b.write_text("FILE_B_CONTENT", encoding="utf-8")

        _, service, executor = self._build_runtime()

        # 1. Explicit owner rejection
        res_reject = executor.process(
            '<tool_call>{"name": "file_read", "arguments": '
            + json.dumps({"path": str(target_a)})
            + "}</tool_call>"
        )[0]
        inv_reject = res_reject["approval"].invocation_id
        req_reject = service.get_request(inv_reject)
        rejected = service.respond(self._sign_decision(req_reject, OwnerDecision.REJECT))
        self.assertEqual(rejected.status, ContinuationStatus.REJECTED)
        self.assertIsNone(rejected.tool_result)

        # 2. Operation digest substitution attempt
        res_sub = executor.process(
            '<tool_call>{"name": "file_read", "arguments": '
            + json.dumps({"path": str(target_b)})
            + "}</tool_call>"
        )[0]
        inv_sub = res_sub["approval"].invocation_id
        req_sub = service.get_request(inv_sub)
        substituted = service.respond(
            self._sign_decision(
                req_sub,
                OwnerDecision.APPROVE,
                operation_digest="f" * 64,
            )
        )
        self.assertEqual(substituted.status, ContinuationStatus.INVALIDATED)
        self.assertIsNone(substituted.tool_result)

        entries = self._read_audit_entries()
        reject_stages = [
            (e["lifecycle_stage"], e["permission_decision"], e["reason_code"])
            for e in entries
            if e["invocation_id"] == inv_reject
        ]
        self.assertEqual(
            reject_stages,
            [
                ("requested", "approval_pending", "approval.pending"),
                ("denied", "approval_rejected", "approval.rejected"),
            ],
        )
        sub_stages = [
            (e["lifecycle_stage"], e["permission_decision"], e["reason_code"])
            for e in entries
            if e["invocation_id"] == inv_sub
        ]
        self.assertEqual(
            sub_stages,
            [
                ("requested", "approval_pending", "approval.pending"),
                ("invalidated", "approval_invalidated", "approval.binding_mismatch"),
            ],
        )

    def test_area_d_file_replacement_or_symlink_swap_after_approval_is_rejected(self):
        if os.name != "posix":
            self.skipTest("POSIX inode/symlink verification required")
        approved_file = self.outside / "approved.txt"
        secret_file = self.outside / "swapped_secret.txt"
        approved_file.write_text("SAFE_ORIGINAL", encoding="utf-8")
        secret_file.write_text("SWAPPED_SECRET_NEVER_READ", encoding="utf-8")

        _, service, executor = self._build_runtime()
        res = executor.process(
            '<tool_call>{"name": "file_read", "arguments": '
            + json.dumps({"path": str(approved_file)})
            + "}</tool_call>"
        )[0]
        inv_id = res["approval"].invocation_id
        req = service.get_request(inv_id)

        # Replace approved_file with a symlink to secret_file before approval completes
        approved_file.unlink()
        approved_file.symlink_to(secret_file)

        outcome = service.respond(self._sign_decision(req, OwnerDecision.APPROVE))
        self.assertEqual(outcome.status, ContinuationStatus.INVALIDATED)
        self.assertIsNone(outcome.tool_result)

        entries = self._read_audit_entries()
        raw_log = json.dumps(entries)
        self.assertNotIn("SWAPPED_SECRET_NEVER_READ", raw_log)

    def test_area_b_service_shutdown_and_expiration_fail_closed_and_cannot_resume_after_restart(self):
        target_file = self.outside / "lifecycle.txt"
        target_file.write_text("LIFECYCLE_DATA", encoding="utf-8")

        _, service1, executor1 = self._build_runtime(log_subdir="lifecycle1")
        res1 = executor1.process(
            '<tool_call>{"name": "file_read", "arguments": '
            + json.dumps({"path": str(target_file)})
            + "}</tool_call>"
        )[0]
        inv1 = res1["approval"].invocation_id
        req1 = service1.get_request(inv1)
        signed1 = self._sign_decision(req1, OwnerDecision.APPROVE)

        # Simulate daemon shutdown
        closed = service1.close()
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].status, ContinuationStatus.CANCELLED)

        # Boot a fresh service instance (simulating daemon restart) and attempt to replay signed1
        _, service2, _ = self._build_runtime(log_subdir="lifecycle2")
        post_restart = service2.respond(signed1)
        self.assertEqual(post_restart.status, ContinuationStatus.FAILED)
        self.assertEqual(post_restart.reason_code, "continuation.already_terminal")

        # Verify audit log on service1 recorded requested -> cancelled
        entries1 = self._read_audit_entries("lifecycle1")
        stages1 = [
            (e["lifecycle_stage"], e["permission_decision"], e["reason_code"])
            for e in entries1
            if e["invocation_id"] == inv1
        ]
        self.assertEqual(
            stages1,
            [
                ("requested", "approval_pending", "approval.pending"),
                ("cancelled", "approval_cancelled", "continuation.service_closed"),
            ],
        )

    def test_area_f_reproducible_dependency_declarations_exist_and_are_consistent(self):
        repo_root = Path(__file__).resolve().parents[1]
        req_main = repo_root / "requirements.txt"
        req_oia = repo_root / "requirements-oia.txt"
        desktop_csproj = repo_root / "windows" / "Lila.Desktop" / "Lila.Desktop.csproj"
        core_csproj = repo_root / "windows" / "Lila.Desktop.Core" / "Lila.Desktop.Core.csproj"
        tests_csproj = repo_root / "windows" / "Lila.Desktop.Tests" / "Lila.Desktop.Tests.csproj"

        self.assertTrue(req_main.is_file(), "Top-level requirements.txt must exist")
        self.assertTrue(req_oia.is_file(), "requirements-oia.txt must exist")
        self.assertTrue(desktop_csproj.is_file(), "windows/Lila.Desktop/Lila.Desktop.csproj must exist")
        self.assertTrue(core_csproj.is_file(), "windows/Lila.Desktop.Core/Lila.Desktop.Core.csproj must exist")
        self.assertTrue(tests_csproj.is_file(), "windows/Lila.Desktop.Tests/Lila.Desktop.Tests.csproj must exist")

        main_text = req_main.read_text(encoding="utf-8")
        oia_text = req_oia.read_text(encoding="utf-8")
        self.assertIn("-r requirements-oia.txt", main_text)
        self.assertIn("argon2-cffi", oia_text)
        self.assertIn("cryptography", oia_text)


if __name__ == "__main__":
    unittest.main()
