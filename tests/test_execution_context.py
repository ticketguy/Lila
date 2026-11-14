import math
import tempfile
import unittest
from pathlib import Path

from src.harness.capabilities import legacy_tool_to_manifest
from src.harness.execution_context import (
    ExecutionContextError,
    ExecutionContextIssuer,
    argument_digest,
    typed_value_bytes,
)
from src.harness.permissions import PermissionManager
from src.harness.tools import Tool, ToolArg, ToolResult


def make_manifest(name="self_status"):
    tool = Tool(
        name=name,
        description="test capability",
        args=[],
        execute=lambda: ToolResult(success=True, output="unused"),
        risk="low",
        permission="auto",
    )
    return legacy_tool_to_manifest(tool)


class ExecutionContextTests(unittest.TestCase):
    def setUp(self):
        self.manifest = make_manifest()
        self.now = [100.0]
        self.issuer = ExecutionContextIssuer(
            lifetime_seconds=5.0, monotonic_clock=lambda: self.now[0]
        )

    def issue(self, arguments=None, result=None):
        arguments = {} if arguments is None else arguments
        result = (True, "Automatically allowed by policy.", dict(arguments)) if result is None else result
        return self.issuer.issue(self.manifest, arguments, result)

    def test_context_fields_and_nested_arguments_are_immutable_snapshots(self):
        supplied = {"nested": {"items": [1, True, 1.25]}, "path": Path("a/b")}
        execution_args = dict(supplied)
        context = self.issue(supplied, (True, "allowed", execution_args))
        supplied["nested"]["items"].append(4)
        supplied["nested"]["extra"] = "later"

        self.assertEqual(context.arguments["nested"]["items"], (1, True, 1.25))
        self.assertNotIn("extra", context.arguments["nested"])
        with self.assertRaises(TypeError):
            context.arguments["new"] = "value"
        with self.assertRaises(TypeError):
            context.arguments["nested"]["new"] = "value"
        with self.assertRaises((AttributeError, TypeError)):
            context.capability_id = "changed"
        self.assertEqual(context.capability_id, self.manifest.capability_id)
        self.assertEqual(context.contract_version, self.manifest.contract_version)
        self.assertEqual(context.manifest_digest, self.manifest.manifest_digest)

    def test_type_tagging_distinguishes_int_float_and_bool(self):
        self.assertNotEqual(argument_digest({"v": 1}), argument_digest({"v": 1.0}))
        self.assertNotEqual(argument_digest({"v": 1}), argument_digest({"v": True}))
        self.assertNotEqual(argument_digest({"v": 0.0}), argument_digest({"v": -0.0}))
        self.assertIn(b'"float64"', typed_value_bytes({"v": 0.1}))
        self.assertIn(b"0x1.999999999999ap-4", typed_value_bytes({"v": 0.1}))

    def test_mapping_order_does_not_change_digest_but_container_type_does(self):
        self.assertEqual(
            argument_digest({"a": 1, "b": [2]}),
            argument_digest({"b": [2], "a": 1}),
        )
        self.assertNotEqual(argument_digest({"v": [1]}), argument_digest({"v": (1,)}))

    def test_context_ids_are_unique(self):
        first = self.issue()
        second = self.issue()
        self.assertNotEqual(first.invocation_id, second.invocation_id)

    def test_authorization_evidence_copies_exact_permission_result_once(self):
        callback_calls = []

        def approval(tool, args, risk):
            callback_calls.append((tool.name, dict(args), risk))
            return True

        manager = PermissionManager(approval_callback=approval)
        tool = Tool("file_write", "write", [ToolArg("path", "path", "", True)], lambda **_: None)
        with tempfile.TemporaryDirectory() as temp_dir:
            args = {"path": str(Path(temp_dir) / "file.txt")}
            validated = {"path": args["path"]}
            result = manager.authorize_with_context(tool, validated)
        context = self.issuer.issue(self.manifest, validated, result)
        self.assertEqual(len(callback_calls), 1)
        self.assertTrue(context.authorization.allowed)
        self.assertEqual(context.authorization.reason, "Approved for this exact call.")
        self.assertEqual(dict(context.authorization.execution_arguments), validated)

    def test_file_list_scope_is_preserved_from_permission_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = Tool("file_list", "list", [], lambda **_: None)
            args = {"path": temp_dir, "pattern": "*"}
            result = PermissionManager(workspace_root=temp_dir).authorize_with_context(tool, args)
            context = self.issuer.issue(self.manifest, args, result)
        self.assertTrue(context.authorization.allowed)
        self.assertEqual(context.authorization.authorized_file_list_scope[0], Path(temp_dir).resolve())

    def test_issuer_rejects_wrong_permission_result_shape(self):
        with self.assertRaises(ExecutionContextError):
            self.issuer.issue(self.manifest, {}, (True, "only two"))
        with self.assertRaises(ExecutionContextError):
            self.issuer.issue(self.manifest, {}, (1, "not bool", {}))

    def test_permission_execution_arguments_must_match_validated_arguments(self):
        with self.assertRaises(ExecutionContextError):
            self.issue({"a": 1}, (True, "allowed", {"a": True}))

    def test_denied_context_cannot_validate(self):
        context = self.issue({}, (False, "denied", {}))
        with self.assertRaisesRegex(ExecutionContextError, "denied"):
            self.issuer.validate_for_invocation(context, self.manifest, {})

    def test_context_expires_at_explicit_monotonic_deadline(self):
        context = self.issue()
        self.now[0] = context.expires_at_monotonic
        with self.assertRaisesRegex(ExecutionContextError, "expired"):
            self.issuer.validate_for_invocation(context, self.manifest, {})

    def test_context_rejects_changed_identity_arguments_and_signature(self):
        context = self.issue({"count": 3})
        with self.assertRaisesRegex(ExecutionContextError, "arguments"):
            self.issuer.validate_for_invocation(context, self.manifest, {"count": 4})
        other_manifest = make_manifest("dns_lookup")
        with self.assertRaisesRegex(ExecutionContextError, "identity"):
            self.issuer.validate_for_invocation(context, other_manifest, {"count": 3})

        from dataclasses import replace
        altered = replace(context, manifest_digest="0" * 64)
        with self.assertRaisesRegex(ExecutionContextError, "integrity"):
            self.issuer.validate_for_invocation(altered, self.manifest, {"count": 3})

    def test_invalid_nonfinite_and_unsupported_values_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ExecutionContextError):
                self.issue({"value": value})
        with self.assertRaises(ExecutionContextError):
            self.issue({"value": object()})
        with self.assertRaises(ExecutionContextError):
            self.issue({1: "non-string key"})
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaises(ExecutionContextError):
            self.issue({"cycle": cyclic})

    def test_explicit_lifetime_is_required_and_validated(self):
        with self.assertRaises(TypeError):
            ExecutionContextIssuer()
        for invalid in (0, -1, math.inf, math.nan, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ExecutionContextIssuer(lifetime_seconds=invalid)


if __name__ == "__main__":
    unittest.main()
