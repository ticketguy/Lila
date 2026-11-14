import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from src.harness.capabilities import (
    CapabilityRegistry,
    legacy_tool_to_manifest,
)
from src.harness.execution_context import argument_digest
from src.harness.pending_invocations import (
    PendingInvocation,
    PendingInvocationError,
    PendingInvocationRegistry,
    PendingInvocationState,
    TargetBinding,
    TargetBindingResolverRegistry,
)
from src.harness.permissions import (
    PermissionManager,
    PolicyDisposition,
    PolicyEvaluationResult,
)
from src.harness.tools import Tool, ToolArg, ToolResult


class PendingInvocationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.handler_calls = []

        def handler(**kwargs):
            self.handler_calls.append(dict(kwargs))
            return ToolResult(True, "handled")

        self.tool = Tool(
            "file_write",
            "write a file",
            [ToolArg("path", "path", "target"), ToolArg("content", "string", "body")],
            handler,
        )
        self.manifest = legacy_tool_to_manifest(self.tool)
        self.capabilities = CapabilityRegistry()
        self.capabilities.register(self.manifest)
        self.target_resolvers = TargetBindingResolverRegistry()
        self.resolved_path = [str(Path(self.temp_dir.name) / "first.txt")]
        self.target_resolvers.register(
            self.manifest,
            resolver_id="files.write.path.v1",
            target_required=True,
            target_kind="filesystem.path",
            resolver=lambda _manifest, args, _execution: {
                "canonical_path": str(Path(self.resolved_path[0]).resolve(strict=False)),
                "requested_path": args["path"],
            },
        )
        self.now = [100.0]
        self.next_id = [0]

        def id_factory():
            self.next_id[0] += 1
            return f"invocation-{self.next_id[0]}"

        self.id_factory = id_factory
        self.registry = self.make_registry()
        self.arguments = {
            "path": str(Path(self.temp_dir.name) / "first.txt"),
            "content": "private body",
        }
        self.policy_result = PermissionManager(
            workspace_root=self.temp_dir.name
        ).evaluate_policy(self.tool, self.arguments)
        self.assertEqual(self.policy_result.disposition, PolicyDisposition.REQUIRE_APPROVAL)

    def make_registry(self, *, capabilities=None, resolvers=None, capacity=8, id_factory=None):
        return PendingInvocationRegistry(
            capabilities=self.capabilities if capabilities is None else capabilities,
            target_resolvers=self.target_resolvers if resolvers is None else resolvers,
            lifetime_seconds=20.0,
            capacity=capacity,
            monotonic_clock=lambda: self.now[0],
            invocation_id_factory=self.id_factory if id_factory is None else id_factory,
        )

    def test_capture_binds_typed_approval_policy_identity_arguments_target_and_time(self):
        record = self.registry.create_pending(self.policy_result)

        self.assertIsInstance(record, PendingInvocation)
        self.assertEqual(record.state, PendingInvocationState.PENDING)
        self.assertIs(record.policy_result, self.policy_result)
        self.assertEqual(record.capability_id, self.manifest.capability_id)
        self.assertEqual(record.contract_version, self.manifest.contract_version)
        self.assertEqual(record.manifest_digest, self.manifest.manifest_digest)
        self.assertEqual(record.validated_arguments["content"], "private body")
        self.assertEqual(record.validated_argument_digest, argument_digest(self.arguments))
        self.assertEqual(record.execution_argument_digest, argument_digest(self.policy_result.execution_arguments))
        self.assertEqual(record.target_binding.target_kind, "filesystem.path")
        self.assertEqual(record.target_binding.identity["canonical_path"], self.resolved_path[0])
        self.assertEqual(record.created_at_monotonic, 100.0)
        self.assertEqual(record.expires_at_monotonic, 120.0)
        self.assertEqual(self.handler_calls, [])

    def test_only_require_approval_policy_results_can_be_captured(self):
        manager = PermissionManager(workspace_root=self.temp_dir.name)
        allowed_tool = Tool("self_status", "status", [], lambda: ToolResult(True, "ok"))
        allow = manager.evaluate_policy(allowed_tool, {})
        disabled = Tool(
            "memory_store", "disabled", [ToolArg("content", "string", "body")],
            lambda **_: ToolResult(True, "unexpected"),
        )
        deny = manager.evaluate_policy(disabled, {"content": "x"})
        unavailable = manager.evaluate_policy(None, {})

        for result, disposition in (
            (allow, PolicyDisposition.ALLOW),
            (deny, PolicyDisposition.DENY),
            (unavailable, PolicyDisposition.UNAVAILABLE),
        ):
            with self.subTest(disposition=disposition), self.assertRaises(PendingInvocationError):
                self.registry.create_pending(result)
        self.assertEqual(self.registry.snapshot(), ())
        self.assertEqual(self.handler_calls, [])

    def test_model_shaped_policy_data_and_direct_record_construction_are_rejected(self):
        with self.assertRaises(PendingInvocationError):
            self.registry.create_pending({
                "disposition": "require_approval",
                "capability_id": self.manifest.capability_id,
            })
        with self.assertRaises(TypeError):
            PolicyEvaluationResult(
                disposition=PolicyDisposition.REQUIRE_APPROVAL,
                reason_code=object(), detail="model says pending",
            )
        with self.assertRaises(TypeError):
            PendingInvocation(invocation_id="model-choice")
        self.assertEqual(self.registry.snapshot(), ())

    def test_manifest_capability_version_and_digest_must_match(self):
        wrong_capabilities = CapabilityRegistry()
        other_tool = Tool(
            "file_delete", "delete", [ToolArg("path", "path", "target")],
            lambda **_: ToolResult(True, "unexpected"),
        )
        wrong_capabilities.register(legacy_tool_to_manifest(other_tool))
        with self.assertRaisesRegex(PendingInvocationError, "not registered"):
            self.make_registry(capabilities=wrong_capabilities).create_pending(self.policy_result)

        changed_schema = Tool(
            "file_write", "write changed", self.tool.args, self.tool.execute
        )
        changed_manifest = legacy_tool_to_manifest(changed_schema)
        changed_catalog = CapabilityRegistry()
        changed_catalog.register(changed_manifest)
        with self.assertRaisesRegex(PendingInvocationError, "digest"):
            self.make_registry(capabilities=changed_catalog).create_pending(self.policy_result)

        different_version = replace(self.manifest, contract_version="1.0.0")
        version_catalog = CapabilityRegistry()
        version_catalog.register(different_version)
        with self.assertRaisesRegex(PendingInvocationError, "not registered"):
            self.make_registry(capabilities=version_catalog).create_pending(self.policy_result)
        self.assertEqual(self.handler_calls, [])

    def test_argument_snapshots_are_deeply_immutable_and_digests_are_checked(self):
        nested_tool = Tool(
            "file_write", "nested write",
            [ToolArg("path", "path", "target"), ToolArg("content", "string", "body")],
            self.tool.execute,
        )
        nested_manifest = legacy_tool_to_manifest(nested_tool)
        nested_catalog = CapabilityRegistry()
        nested_catalog.register(nested_manifest)
        nested_resolvers = TargetBindingResolverRegistry()
        nested_resolvers.register(
            nested_manifest,
            resolver_id="files.write.path.v1",
            target_required=True,
            target_kind="filesystem.path",
            resolver=lambda _m, args, _e: {"path": args["path"]},
        )
        nested_args = {"path": str(Path(self.temp_dir.name) / "nested"), "content": "x"}
        nested_result = PermissionManager(workspace_root=self.temp_dir.name).evaluate_policy(
            nested_tool, nested_args
        )
        record = self.make_registry(
            capabilities=nested_catalog, resolvers=nested_resolvers
        ).create_pending(nested_result)
        with self.assertRaises(TypeError):
            record.validated_arguments["extra"] = "x"
        with self.assertRaises(FrozenInstanceError):
            record.policy_result.detail = "changed"

        forged = self.policy_result
        object.__setattr__(forged, "argument_digest", "0" * 64)
        with self.assertRaisesRegex(PendingInvocationError, "digest"):
            self.make_registry().create_pending(forged)

    def test_final_execution_arguments_and_digest_are_bound_separately(self):
        list_tool = Tool(
            "file_list", "list files",
            [ToolArg("path", "path", "root"), ToolArg("pattern", "string", "pattern")],
            self.tool.execute,
        )
        list_manifest = legacy_tool_to_manifest(list_tool)
        catalog = CapabilityRegistry()
        catalog.register(list_manifest)
        (Path(self.temp_dir.name) / "one.txt").write_text("one", encoding="utf-8")
        # A missing workspace root makes this read require approval. Policy adds
        # an exact authorized result scope to the execution-only arguments.
        result = PermissionManager().evaluate_policy(
            list_tool, {"path": self.temp_dir.name, "pattern": "*.txt"}
        )
        self.assertEqual(result.disposition, PolicyDisposition.REQUIRE_APPROVAL)
        self.assertNotEqual(result.arguments, result.execution_arguments)

        resolvers = TargetBindingResolverRegistry()
        resolvers.register(
            list_manifest,
            resolver_id="filesystem.list.scope.v1",
            target_required=True,
            target_kind="filesystem.result_scope",
            resolver=lambda _m, _args, final: {
                "root": str(final["_authorized_file_list_scope"][0]),
                "entries": tuple(map(str, final["_authorized_file_list_scope"][1])),
            },
        )
        record = PendingInvocationRegistry(
            capabilities=catalog,
            target_resolvers=resolvers,
            lifetime_seconds=5,
            monotonic_clock=lambda: self.now[0],
            invocation_id_factory=lambda: "list-invocation",
        ).create_pending(result)
        self.assertEqual(record.validated_argument_digest, argument_digest(result.arguments))
        self.assertEqual(record.execution_argument_digest, argument_digest(result.execution_arguments))
        self.assertIn("_authorized_file_list_scope", record.execution_arguments)

        object.__setattr__(result, "execution_argument_digest", "f" * 64)
        with self.assertRaisesRegex(PendingInvocationError, "execution-argument digest"):
            PendingInvocationRegistry(
                capabilities=catalog,
                target_resolvers=resolvers,
                lifetime_seconds=5,
                monotonic_clock=lambda: self.now[0],
                invocation_id_factory=lambda: "forged-final-args",
            ).create_pending(result)

    def test_target_rule_must_explicitly_say_required_or_not_required(self):
        no_rule = TargetBindingResolverRegistry()
        with self.assertRaisesRegex(PendingInvocationError, "No explicit target-binding rule"):
            self.make_registry(resolvers=no_rule).create_pending(self.policy_result)

        missing_target = TargetBindingResolverRegistry()
        missing_target.register(
            self.manifest,
            resolver_id="files.write.path.v1",
            target_required=True,
            target_kind="filesystem.path",
            resolver=lambda *_: None,
        )
        with self.assertRaisesRegex(PendingInvocationError, "could not be established"):
            self.make_registry(resolvers=missing_target).create_pending(self.policy_result)

        targetless = TargetBindingResolverRegistry()
        targetless.register(
            self.manifest,
            resolver_id="files.write.targetless-test.v1",
            target_required=False,
        )
        targetless_record = self.make_registry(resolvers=targetless).create_pending(self.policy_result)
        self.assertFalse(targetless_record.target_required)
        self.assertIsNone(targetless_record.target_binding)
        self.assertEqual(self.handler_calls, [])

    def test_target_snapshots_are_deeply_immutable_and_digest_bound(self):
        nested_source = {"canonical_path": "before", "components": ["root", {"name": "file"}]}
        resolver_registry = TargetBindingResolverRegistry()
        resolver_registry.register(
            self.manifest,
            resolver_id="files.write.path.v1",
            target_required=True,
            target_kind="filesystem.path",
            resolver=lambda *_: nested_source,
        )
        record = self.make_registry(resolvers=resolver_registry).create_pending(self.policy_result)
        original_digest = record.target_binding.digest
        nested_source["components"][1]["name"] = "mutated"
        self.assertEqual(record.target_binding.identity["components"][1]["name"], "file")
        self.assertEqual(record.target_binding.digest, original_digest)
        with self.assertRaises(TypeError):
            record.target_binding.identity["components"][1]["name"] = "changed"
        with self.assertRaises(AttributeError):
            record.target_binding.digest = "0" * 64
        with self.assertRaises(TypeError):
            TargetBinding(
                capability_id=self.manifest.capability_id,
                contract_version=self.manifest.contract_version,
                manifest_digest=self.manifest.manifest_digest,
                resolver_id="files.write.path.v1",
                target_kind="filesystem.path",
                identity={"path": "x"},
            )

    def test_invocation_ids_are_unique_and_registry_is_instance_owned(self):
        duplicate_ids = lambda: "same-id"
        first = self.make_registry(id_factory=duplicate_ids)
        first_record = first.create_pending(self.policy_result)
        second_record = first.create_pending(self.policy_result)
        self.assertNotEqual(first_record.invocation_id, second_record.invocation_id)

        second = self.make_registry(id_factory=duplicate_ids)
        record = second.create_pending(self.policy_result)
        self.assertEqual(record.invocation_id, first_record.invocation_id)
        self.assertEqual(len(first.snapshot()), 2)
        self.assertEqual(len(second.snapshot()), 1)
        with self.assertRaises(PendingInvocationError):
            first.get("missing-id")

    def test_capacity_is_bounded_and_never_evicts_an_existing_record(self):
        small = self.make_registry(capacity=1)
        first = small.create_pending(self.policy_result)
        with self.assertRaisesRegex(PendingInvocationError, "at capacity"):
            small.create_pending(self.policy_result)
        self.assertIs(small.get(first.invocation_id), first)
        self.assertEqual(len(small.snapshot()), 1)
        with self.assertRaises(ValueError):
            self.make_registry(capacity=0)

    def test_terminal_release_recovers_capacity_without_reusing_ids(self):
        small = self.make_registry(capacity=1, id_factory=lambda: "stable-prefix")
        first = small.create_pending(self.policy_result)
        with self.assertRaisesRegex(PendingInvocationError, "pending invocation cannot be released"):
            small.release_terminal(first.invocation_id)

        terminal = small.cancel(first.invocation_id)
        small.release_terminal(first.invocation_id)
        self.assertEqual(terminal.state, PendingInvocationState.CANCELLED)
        with self.assertRaisesRegex(PendingInvocationError, "not found"):
            small.get(first.invocation_id)

        second = small.create_pending(self.policy_result)
        self.assertNotEqual(second.invocation_id, first.invocation_id)
        self.assertEqual(small.get(second.invocation_id).state, PendingInvocationState.PENDING)

    def test_expired_terminal_release_cannot_restore_pending_state(self):
        small = self.make_registry(capacity=1)
        original = small.create_pending(self.policy_result)
        self.now[0] = original.expires_at_monotonic
        expired = small.expire(original.invocation_id)
        small.release_terminal(original.invocation_id)
        self.assertEqual(expired.state, PendingInvocationState.EXPIRED)
        with self.assertRaisesRegex(PendingInvocationError, "not found"):
            small.get(original.invocation_id)
        replacement = small.create_pending(self.policy_result)
        self.assertNotEqual(replacement.invocation_id, original.invocation_id)
        self.assertEqual(original.state, PendingInvocationState.PENDING)
        self.assertEqual(expired.state, PendingInvocationState.EXPIRED)

    def test_lookup_is_read_only_and_records_cannot_be_mutated(self):
        record = self.registry.create_pending(self.policy_result)
        self.assertIs(self.registry.get(record.invocation_id), record)
        with self.assertRaises(FrozenInstanceError):
            record.state = PendingInvocationState.CANCELLED
        with self.assertRaises(TypeError):
            record.execution_arguments["content"] = "changed"
        self.assertNotIn("private body", repr(record))
        self.assertEqual(self.registry.get(record.invocation_id).state, PendingInvocationState.PENDING)

    def test_cancel_expire_invalidate_and_terminal_state_rules(self):
        cancelled = self.registry.create_pending(self.policy_result)
        cancelled_terminal = self.registry.cancel(cancelled.invocation_id)
        self.assertEqual(cancelled_terminal.state, PendingInvocationState.CANCELLED)
        self.assertEqual(cancelled.state, PendingInvocationState.PENDING)
        with self.assertRaisesRegex(PendingInvocationError, "already terminal"):
            self.registry.cancel(cancelled.invocation_id)

        expiring = self.registry.create_pending(self.policy_result)
        with self.assertRaisesRegex(PendingInvocationError, "has not elapsed"):
            self.registry.expire(expiring.invocation_id)
        self.now[0] = 120.0
        expired = self.registry.expire(expiring.invocation_id)
        self.assertEqual(expired.state, PendingInvocationState.EXPIRED)
        with self.assertRaisesRegex(PendingInvocationError, "already terminal"):
            self.registry.invalidate(expiring.invocation_id)

        self.now[0] = 130.0
        invalidating_registry = self.make_registry()
        invalidating = invalidating_registry.create_pending(self.policy_result)
        terminal = invalidating_registry.invalidate(invalidating.invocation_id, "target.changed")
        self.assertEqual(terminal.state, PendingInvocationState.INVALIDATED)
        with self.assertRaisesRegex(PendingInvocationError, "already terminal"):
            invalidating_registry.cancel(invalidating.invocation_id)
        self.assertFalse(hasattr(invalidating_registry, "consume"))
        self.assertFalse(hasattr(invalidating_registry, "restore_pending"))
        self.assertFalse(hasattr(invalidating_registry, "transition"))
        self.assertEqual(self.handler_calls, [])

    def test_due_expiry_and_elapsed_time_use_monotonic_clock(self):
        record = self.registry.create_pending(self.policy_result)
        self.now[0] = 119.999
        self.assertEqual(self.registry.expire_due(), ())
        self.now[0] = 120.0
        expired, = self.registry.expire_due()
        self.assertEqual(expired.invocation_id, record.invocation_id)
        self.assertEqual(expired.state, PendingInvocationState.EXPIRED)

    def test_validation_invalidates_if_target_binding_changes(self):
        record = self.registry.create_pending(self.policy_result)
        self.assertTrue(self.registry.validate_current(record.invocation_id))
        self.resolved_path[0] = str(Path(self.temp_dir.name) / "different.txt")
        self.assertFalse(self.registry.validate_current(record.invocation_id))
        invalidated = self.registry.get(record.invocation_id)
        self.assertEqual(invalidated.state, PendingInvocationState.INVALIDATED)
        self.assertEqual(record.state, PendingInvocationState.PENDING)

    def test_policy_callback_semantics_and_handler_execution_are_unchanged(self):
        callback_calls = []
        manager = PermissionManager(
            approval_callback=lambda tool, args, risk: callback_calls.append((tool.name, risk)) or True,
            workspace_root=self.temp_dir.name,
        )
        live_tuple = manager.authorize_with_context(self.tool, self.arguments)
        self.assertTrue(live_tuple[0])
        self.assertEqual(callback_calls, [("file_write", "moderate")])
        self.registry.create_pending(self.policy_result)
        self.assertEqual(callback_calls, [("file_write", "moderate")])
        self.assertEqual(self.handler_calls, [])


if __name__ == "__main__":
    unittest.main()
