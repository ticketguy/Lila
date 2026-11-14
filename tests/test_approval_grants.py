import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.harness.approval_grants import ApprovalGrant, ApprovalGrantError
from src.harness.capabilities import CapabilityRegistry, legacy_tool_to_manifest
from src.harness.pending_invocations import (
    PendingInvocationError,
    PendingInvocationRegistry,
    PendingInvocationState,
    TargetBindingResolverRegistry,
)
from src.harness.permissions import (
    PermissionManager,
    PolicyDisposition,
    PolicyReasonCode,
    _policy_result,
)
from src.harness.tools import Tool, ToolArg, ToolResult


class ApprovalGrantTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.handler_calls = []
        self.now = [100.0]
        self.id_count = [0]

        def handler(**kwargs):
            self.handler_calls.append(kwargs)
            return ToolResult(True, "unexpected execution")

        self.tool = Tool(
            "file_write", "write a file",
            [ToolArg("path", "path", "target"), ToolArg("content", "string", "body")],
            handler,
        )
        self.manifest = legacy_tool_to_manifest(self.tool)
        self.capabilities = CapabilityRegistry()
        self.capabilities.register(self.manifest)
        self.resolved_target = [str(Path(self.temp_dir.name) / "target.txt")]
        self.resolvers = TargetBindingResolverRegistry()
        self.resolvers.register(
            self.manifest,
            resolver_id="files.write.path.v1",
            target_required=True,
            target_kind="filesystem.path",
            resolver=lambda _manifest, args, _execution: {
                "canonical_path": str(Path(self.resolved_target[0]).resolve(strict=False)),
                "requested_path": args["path"],
            },
        )
        self.arguments = {
            "path": self.resolved_target[0],
            "content": "exact body",
        }
        self.policy_result = PermissionManager(
            workspace_root=self.temp_dir.name
        ).evaluate_policy(self.tool, self.arguments)
        self.assertEqual(self.policy_result.disposition, PolicyDisposition.REQUIRE_APPROVAL)
        self.pending = self.make_pending_registry()

    def make_pending_registry(self, *, invocation_prefix="pending"):
        def id_factory():
            self.id_count[0] += 1
            return f"{invocation_prefix}-{self.id_count[0]}"

        return PendingInvocationRegistry(
            capabilities=self.capabilities,
            target_resolvers=self.resolvers,
            lifetime_seconds=30.0,
            capacity=20,
            monotonic_clock=lambda: self.now[0],
            invocation_id_factory=id_factory,
        )

    def create_pending(self, result=None):
        return self.pending.create_pending(self.policy_result if result is None else result)

    def issue(self, pending):
        return self.pending._issue_approval_grant(pending.invocation_id)

    def policy_with_revision(self, revision):
        return _policy_result(
            disposition=self.policy_result.disposition,
            reason_code=self.policy_result.reason_code,
            detail=self.policy_result.detail,
            risk=self.policy_result.risk,
            capability_id=self.policy_result.capability_id,
            contract_version=self.policy_result.contract_version,
            manifest_digest=self.policy_result.manifest_digest,
            argument_digest=self.policy_result.argument_digest,
            arguments=self.policy_result.arguments,
            execution_argument_digest=self.policy_result.execution_argument_digest,
            execution_arguments=self.policy_result.execution_arguments,
            policy_revision=revision,
        )

    def test_grant_binds_exact_pending_identity_operation_arguments_target_and_lifetime(self):
        pending = self.create_pending()
        grant = self.issue(pending)

        self.assertIsInstance(grant, ApprovalGrant)
        self.assertEqual(grant.invocation_id, pending.invocation_id)
        self.assertEqual(grant.capability_id, self.manifest.capability_id)
        self.assertEqual(grant.contract_version, self.manifest.contract_version)
        self.assertEqual(grant.manifest_digest, self.manifest.manifest_digest)
        self.assertEqual(grant.validated_arguments, pending.validated_arguments)
        self.assertEqual(grant.validated_argument_digest, pending.validated_argument_digest)
        self.assertEqual(grant.execution_arguments, pending.execution_arguments)
        self.assertEqual(grant.execution_argument_digest, pending.execution_argument_digest)
        self.assertEqual(grant.target_binding, pending.target_binding)
        self.assertEqual(grant.created_at_monotonic, self.now[0])
        self.assertEqual(grant.pending_created_at_monotonic, pending.created_at_monotonic)
        self.assertEqual(grant.expires_at_monotonic, pending.expires_at_monotonic)
        self.assertRegex(grant.operation_digest, r"^[0-9a-f]{64}$")
        self.assertTrue(grant.is_bound_to(pending, self.now[0]))
        self.assertEqual(self.handler_calls, [])

    def test_grant_is_deeply_immutable_and_integrity_checks_argument_or_identity_tampering(self):
        pending = self.create_pending()
        grant = self.issue(pending)
        with self.assertRaises(FrozenInstanceError):
            grant.capability_id = "other"
        with self.assertRaises(TypeError):
            grant.validated_arguments["content"] = "changed"
        with self.assertRaises(TypeError):
            grant.target_binding.identity["canonical_path"] = "/other"
        self.assertNotIn("exact body", repr(grant))

        for field, changed in (
            ("invocation_id", "other-invocation"),
            ("capability_id", "org.other.capability"),
            ("contract_version", "99.0"),
            ("manifest_digest", "0" * 64),
            ("validated_argument_digest", "0" * 64),
            ("execution_argument_digest", "0" * 64),
            ("created_at_monotonic", 101.0),
            ("expires_at_monotonic", 500.0),
            ("policy_revalidation_required", False),
        ):
            original = getattr(grant, field)
            object.__setattr__(grant, field, changed)
            with self.subTest(field=field), self.assertRaises(ApprovalGrantError):
                grant.verify_integrity()
            object.__setattr__(grant, field, original)
            grant.verify_integrity()

    def test_grant_cannot_be_substituted_for_another_invocation_or_operation(self):
        first = self.create_pending()
        grant = self.issue(first)
        same_operation = self.create_pending()
        self.assertNotEqual(first.invocation_id, same_operation.invocation_id)
        self.assertFalse(grant.is_bound_to(same_operation, self.now[0]))

        changed_arguments = dict(self.arguments, content="different body")
        changed_result = PermissionManager(
            workspace_root=self.temp_dir.name
        ).evaluate_policy(self.tool, changed_arguments)
        changed_pending = self.create_pending(changed_result)
        self.assertFalse(grant.is_bound_to(changed_pending, self.now[0]))

        other_tool = Tool(
            "file_read", "read a file", [ToolArg("path", "path", "target")],
            self.tool.execute,
        )
        other_manifest = legacy_tool_to_manifest(other_tool)
        self.capabilities.register(other_manifest)
        self.resolvers.register(
            other_manifest,
            resolver_id="files.read.path.v1",
            target_required=True,
            target_kind="filesystem.path",
            resolver=lambda _m, args, _e: {"path": args["path"]},
        )
        other_result = PermissionManager().evaluate_policy(other_tool, {"path": "/tmp/other"})
        other_pending = self.create_pending(other_result)
        self.assertFalse(grant.is_bound_to(other_pending, self.now[0]))
        self.assertEqual(self.handler_calls, [])

    def test_target_mismatch_expiry_cancellation_and_invalidation_fail_closed(self):
        pending = self.create_pending()
        grant = self.issue(pending)
        self.resolved_target[0] = str(Path(self.temp_dir.name) / "replacement.txt")
        replacement_pending = self.create_pending()
        self.assertFalse(grant.is_bound_to(replacement_pending, self.now[0]))
        self.assertFalse(grant.is_bound_to(pending, grant.expires_at_monotonic))

        stale = self.create_pending()
        self.resolved_target[0] = str(Path(self.temp_dir.name) / "changed-after-pending.txt")
        with self.assertRaisesRegex(PendingInvocationError, "no longer current"):
            self.issue(stale)
        self.assertEqual(
            self.pending.get(stale.invocation_id).state,
            PendingInvocationState.INVALIDATED,
        )

        cancelled = self.create_pending()
        self.pending.cancel(cancelled.invocation_id)
        with self.assertRaisesRegex(PendingInvocationError, "already terminal"):
            self.issue(cancelled)
        self.assertFalse(grant.is_bound_to(self.pending.get(cancelled.invocation_id), self.now[0]))

        invalidated = self.create_pending()
        self.pending.invalidate(invalidated.invocation_id)
        with self.assertRaisesRegex(PendingInvocationError, "already terminal"):
            self.issue(invalidated)

        expired = self.create_pending()
        self.now[0] = expired.expires_at_monotonic
        with self.assertRaisesRegex(PendingInvocationError, "expired"):
            self.issue(expired)
        self.assertEqual(self.pending.get(expired.invocation_id).state, PendingInvocationState.EXPIRED)

    def test_policy_revision_is_preserved_and_missing_revision_requires_recheck(self):
        pending = self.create_pending()
        grant = self.issue(pending)
        self.assertIsNone(grant.policy_revision)
        self.assertTrue(grant.policy_revalidation_required)
        self.assertTrue(grant.validate_current_policy(self.policy_result))

        changed_result = PermissionManager(
            workspace_root=self.temp_dir.name
        ).evaluate_policy(self.tool, dict(self.arguments, content="changed"))
        self.assertFalse(grant.validate_current_policy(changed_result))

        changed_risk = _policy_result(
            disposition=self.policy_result.disposition,
            reason_code=self.policy_result.reason_code,
            detail=self.policy_result.detail,
            risk="high",
            capability_id=self.policy_result.capability_id,
            contract_version=self.policy_result.contract_version,
            manifest_digest=self.policy_result.manifest_digest,
            argument_digest=self.policy_result.argument_digest,
            arguments=self.policy_result.arguments,
            execution_argument_digest=self.policy_result.execution_argument_digest,
            execution_arguments=self.policy_result.execution_arguments,
            policy_revision=None,
        )
        self.assertFalse(grant.validate_current_policy(changed_risk))

        revision_result = self.policy_with_revision("policy-epoch-7")
        revision_pending = self.create_pending(revision_result)
        revision_grant = self.pending._issue_approval_grant(
            revision_pending.invocation_id,
            current_policy_revision="policy-epoch-7",
        )
        self.assertEqual(revision_grant.policy_revision, "policy-epoch-7")
        self.assertFalse(revision_grant.policy_revalidation_required)
        self.assertTrue(revision_grant.validate_current_policy(revision_result))
        self.assertFalse(
            revision_grant.validate_current_policy(self.policy_with_revision("policy-epoch-8"))
        )

        with self.assertRaisesRegex(PendingInvocationError, "no longer current"):
            self.pending._issue_approval_grant(
                self.create_pending(revision_result).invocation_id,
                current_policy_revision="policy-epoch-8",
            )

    def test_untrusted_construction_and_issuance_surfaces_are_not_public(self):
        pending = self.create_pending()
        with self.assertRaises(TypeError):
            ApprovalGrant(invocation_id=pending.invocation_id)
        with self.assertRaises(TypeError):
            ApprovalGrant._from_pending(pending, created_at_monotonic=self.now[0])
        self.assertFalse(hasattr(self.pending, "issue_approval_grant"))
        self.assertFalse(hasattr(self.pending, "consume_grant"))
        self.assertFalse(hasattr(self.pending, "dispatch_grant"))
        self.assertEqual(self.handler_calls, [])

    def test_concurrent_issuance_is_atomic_and_creates_at_most_one_grant(self):
        pending = self.create_pending()
        start = threading.Barrier(8)

        def issue_once():
            start.wait()
            try:
                return self.issue(pending)
            except PendingInvocationError:
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _index: issue_once(), range(8)))
        grants = [item for item in results if item is not None]
        self.assertEqual(len(grants), 1)
        self.assertTrue(grants[0].is_bound_to(pending, self.now[0]))
        self.assertEqual(self.handler_calls, [])


if __name__ == "__main__":
    unittest.main()
