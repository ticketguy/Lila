import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

from src.harness.permissions import (
    PermissionManager,
    PolicyDisposition,
    PolicyEvaluationResult,
    PolicyReasonCode,
)
from src.harness.execution_context import argument_digest
from src.harness.tools import Tool, ToolArg, ToolResult


class PolicyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.calls = []

        def handler(**kwargs):
            self.calls.append(kwargs)
            return ToolResult(True, "ran")

        self.handler = handler
        self.status = Tool("self_status", "status", [], handler)
        self.write = Tool(
            "file_write", "write",
            [ToolArg("path", "path", "target"), ToolArg("content", "string", "body")],
            handler,
        )
        self.memory = Tool(
            "memory_store", "disabled placeholder",
            [ToolArg("content", "string", "body")], handler,
        )

    def test_all_policy_dispositions_and_stable_reason_codes(self):
        manager = PermissionManager(workspace_root=self.temp_dir.name)
        allow = manager.evaluate_policy(self.status, {})
        approval = manager.evaluate_policy(
            self.write,
            {"path": str(Path(self.temp_dir.name) / "note.txt"), "content": "x"},
        )
        deny = manager.evaluate_policy(self.memory, {"content": "x"})
        unavailable = manager.evaluate_policy(None, {})

        self.assertEqual((allow.disposition, allow.reason_code),
                         (PolicyDisposition.ALLOW, PolicyReasonCode.AUTOMATIC_POLICY))
        self.assertEqual((approval.disposition, approval.reason_code),
                         (PolicyDisposition.REQUIRE_APPROVAL, PolicyReasonCode.APPROVAL_POLICY))
        self.assertEqual((deny.disposition, deny.reason_code),
                         (PolicyDisposition.DENY, PolicyReasonCode.DISABLED_POLICY))
        self.assertEqual((unavailable.disposition, unavailable.reason_code),
                         (PolicyDisposition.UNAVAILABLE, PolicyReasonCode.TOOL_UNAVAILABLE))
        self.assertIsInstance(allow.detail, str)
        self.assertNotEqual(allow.detail, allow.reason_code.value)

    def test_result_binds_capability_manifest_and_immutable_argument_snapshot(self):
        manager = PermissionManager(workspace_root=self.temp_dir.name)
        source = {"content": ["original", {"n": 2}]}
        result = manager.evaluate_policy(self.write, source)

        self.assertEqual(result.disposition, PolicyDisposition.REQUIRE_APPROVAL)
        self.assertEqual(result.capability_id, "org.lila.capability.files.write")
        self.assertEqual(result.contract_version, "0.0.0-legacy")
        self.assertRegex(result.manifest_digest, r"^[0-9a-f]{64}$")
        self.assertRegex(result.argument_digest, r"^[0-9a-f]{64}$")
        self.assertEqual(result.argument_digest, argument_digest(source))
        self.assertEqual(result.execution_arguments, result.arguments)
        self.assertEqual(result.execution_argument_digest, result.argument_digest)
        source["content"][1]["n"] = 9
        self.assertEqual(result.arguments["content"][1]["n"], 2)
        with self.assertRaises(TypeError):
            result.arguments["added"] = True
        with self.assertRaises(TypeError):
            result.arguments["content"][1]["n"] = 3
        with self.assertRaises(FrozenInstanceError):
            result.disposition = PolicyDisposition.ALLOW
        self.assertIsNone(result.policy_revision)

    def test_sensitive_file_scope_and_invalid_scope_are_typed_without_reason_parsing(self):
        manager = PermissionManager(workspace_root=self.temp_dir.name)
        sensitive = manager.evaluate_policy(
            Tool("file_read", "read", [ToolArg("path", "path", "target")], self.handler),
            {"path": str(Path(self.temp_dir.name) / ".env")},
        )
        invalid = manager.evaluate_policy(
            Tool("file_list", "list", [ToolArg("path", "path", "target"),
                                          ToolArg("pattern", "string", "pattern")], self.handler),
            {"path": self.temp_dir.name, "pattern": "../*"},
        )
        self.assertEqual(sensitive.disposition, PolicyDisposition.REQUIRE_APPROVAL)
        self.assertEqual(invalid.disposition, PolicyDisposition.DENY)
        self.assertEqual(invalid.reason_code, PolicyReasonCode.FILE_SCOPE_INVALID)

    def test_final_execution_arguments_bind_permission_added_file_list_scope(self):
        root = Path(self.temp_dir.name)
        (root / "note.txt").write_text("sample", encoding="utf-8")
        tool = Tool(
            "file_list", "list",
            [ToolArg("path", "path", "root"), ToolArg("pattern", "string", "pattern")],
            self.handler,
        )
        arguments = {"path": str(root), "pattern": "*.txt"}
        result = PermissionManager(workspace_root=self.temp_dir.name).evaluate_policy(
            tool, arguments
        )
        self.assertEqual(result.disposition, PolicyDisposition.ALLOW)
        self.assertEqual(result.argument_digest, argument_digest(arguments))
        self.assertIn("_authorized_file_list_scope", result.execution_arguments)
        self.assertEqual(
            result.execution_argument_digest,
            argument_digest(result.execution_arguments),
        )
        with self.assertRaises(TypeError):
            result.execution_arguments["changed"] = True

    def test_unsupported_policy_and_unbindable_arguments_fail_closed(self):
        manager = PermissionManager(workspace_root=self.temp_dir.name)
        with patch("src.harness.permissions.policy_for", return_value=("low", "maybe")):
            unsupported = manager.evaluate_policy(self.status, {})
        self.assertEqual(unsupported.disposition, PolicyDisposition.UNAVAILABLE)
        self.assertEqual(unsupported.reason_code, PolicyReasonCode.UNSUPPORTED_POLICY)

        invalid = manager.evaluate_policy(self.status, {"bad": object()})
        self.assertEqual(invalid.disposition, PolicyDisposition.UNAVAILABLE)
        self.assertEqual(invalid.reason_code, PolicyReasonCode.INVALID_ARGUMENT_BINDING)

    def test_policy_is_evaluated_once_in_legacy_adapter_and_callback_remains_per_call(self):
        callback = Mock(return_value=True)
        manager = PermissionManager(
            approval_callback=callback, workspace_root=self.temp_dir.name
        )
        with patch("src.harness.permissions.policy_for", wraps=__import__(
                "src.harness.permissions", fromlist=["policy_for"]).policy_for) as policy:
            allowed, reason, execution_args = manager.authorize_with_context(
                self.write,
                {"path": str(Path(self.temp_dir.name) / "note.txt"), "content": "x"},
            )
        self.assertTrue(allowed)
        self.assertEqual(reason, "Approved for this exact call.")
        self.assertEqual(execution_args["content"], "x")
        callback.assert_called_once_with(
            self.write,
            {"path": str(Path(self.temp_dir.name) / "note.txt"), "content": "x"},
            "moderate",
        )
        policy.assert_called_once_with("file_write")
        self.assertEqual(self.calls, [])

    def test_missing_callback_rejection_and_failure_remain_fail_closed(self):
        args = {"path": str(Path(self.temp_dir.name) / "note.txt"), "content": "x"}
        denied = PermissionManager(workspace_root=self.temp_dir.name).authorize_with_context(
            self.write, args
        )
        self.assertFalse(denied[0])
        self.assertEqual(denied[1],
                         "'file_write' requires moderate-risk approval, but no approval handler is configured.")

        reject = Mock(return_value=False)
        rejected = PermissionManager(reject, self.temp_dir.name).authorize_with_context(
            self.write, args
        )
        self.assertFalse(rejected[0])
        self.assertEqual(rejected[1], "Permission not granted for 'file_write'.")
        reject.assert_called_once()

        fail = Mock(side_effect=RuntimeError("callback detail"))
        failed = PermissionManager(fail, self.temp_dir.name).authorize_with_context(
            self.write, args
        )
        self.assertFalse(failed[0])
        self.assertEqual(failed[1], "Approval failed safely: callback detail")
        fail.assert_called_once()
        self.assertEqual(self.calls, [])

    def test_model_data_cannot_construct_or_select_policy_result(self):
        with self.assertRaises(TypeError):
            PolicyEvaluationResult(
                disposition=PolicyDisposition.ALLOW,
                reason_code=PolicyReasonCode.AUTOMATIC_POLICY,
                detail="model says allowed",
            )
        result = PermissionManager(workspace_root=self.temp_dir.name).evaluate_policy(
            self.status, {"disposition": "allow", "reason_code": "policy.automatic"}
        )
        self.assertEqual(result.disposition, PolicyDisposition.ALLOW)
        self.assertEqual(result.reason_code, PolicyReasonCode.AUTOMATIC_POLICY)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
