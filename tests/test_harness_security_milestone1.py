import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.harness.executor import HarnessExecutor
from src.harness.permissions import PermissionManager
from src.harness.system_tools import _file_list
from src.harness.tools import (
    DuplicateToolRegistrationError,
    Tool,
    ToolArg,
    ToolRegistry,
    ToolResult,
    get_registry,
)


def tool_call(name, arguments):
    import json

    return "<tool_call>" + json.dumps(
        {"name": name, "arguments": arguments}
    ) + "</tool_call>"


class FileListScopeSecurityTests(unittest.TestCase):
    def setUp(self):
        self.registry = get_registry()
        self.previous_tools = dict(self.registry._tools)
        self.registry._tools.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.restore_registry)
        self.root = Path(self.temp_dir.name) / "workspace"
        self.root.mkdir()
        self.registry.register(Tool(
            "file_list",
            "List files",
            [
                ToolArg("path", "path", "Directory", required=False, default="."),
                ToolArg("pattern", "string", "Glob", required=False, default="*"),
            ],
            _file_list,
        ))
        self.executor = HarnessExecutor(
            log_dir=self.temp_dir.name, workspace_root=str(self.root)
        )

    def restore_registry(self):
        self.registry._tools.clear()
        self.registry._tools.update(self.previous_tools)

    def run_list(self, path, pattern):
        return self.executor.process(tool_call(
            "file_list", {"path": str(path), "pattern": pattern}
        ))[0]["result"]

    def test_rejects_parent_traversal_pattern(self):
        outside = Path(self.temp_dir.name) / "outside.txt"
        outside.write_text("private", encoding="utf-8")
        result = self.run_list(self.root, "../*")
        self.assertFalse(result.success)
        self.assertIn("Traversal", result.error)
        self.assertNotIn(str(outside), result.output)

    def test_rejects_absolute_pattern(self):
        result = self.run_list(self.root, str(self.root / "*.txt"))
        self.assertFalse(result.success)
        self.assertIn("Absolute", result.error)

    def test_rejects_symlink_escape(self):
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("private", encoding="utf-8")
        link = self.root / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Symlinks are unavailable in this environment")
        result = self.run_list(self.root, "escape/*")
        self.assertFalse(result.success)
        self.assertIn("outside its directory", result.error)
        self.assertNotIn("secret.txt", result.output)

    def test_allows_recursive_patterns_and_uses_authorized_snapshot(self):
        nested = self.root / "nested"
        nested.mkdir()
        expected = nested / "allowed.txt"
        expected.write_text("ok", encoding="utf-8")
        with patch(
            "src.harness.permissions.resolve_file_list_scope",
            wraps=__import__(
                "src.harness.permissions", fromlist=["resolve_file_list_scope"]
            ).resolve_file_list_scope,
        ) as resolve_scope:
            result = self.run_list(self.root, "**/*.txt")
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output.splitlines(), [str(expected.resolve())])
        self.assertEqual(resolve_scope.call_count, 1)

    def test_allows_normal_single_directory_pattern(self):
        expected = self.root / "one.txt"
        expected.write_text("ok", encoding="utf-8")
        result = self.run_list(self.root, "*.txt")
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output, str(expected.resolve()))


class DuplicateToolRegistrationSecurityTests(unittest.TestCase):
    def test_duplicate_registration_rejects_replacement_and_preserves_policy(self):
        registry = ToolRegistry()
        original_execute = lambda **kwargs: ToolResult(True, "original")
        replacement_execute = lambda **kwargs: ToolResult(True, "replacement")
        original = Tool("self_status", "original", [], original_execute)
        replacement = Tool("self_status", "replacement", [], replacement_execute)

        registry.register(original)
        with self.assertRaisesRegex(
            DuplicateToolRegistrationError, "already registered"
        ):
            registry.register(replacement)

        self.assertIs(registry.get("self_status"), original)
        self.assertIs(registry.get("self_status").execute, original_execute)
        self.assertEqual((original.risk, original.permission), ("low", "auto"))
        self.assertEqual((replacement.risk, replacement.permission), ("high", "approval"))
        allowed, _ = PermissionManager().authorize(original, {})
        self.assertTrue(allowed)

    def test_unknown_tool_does_not_inherit_automatic_permission(self):
        registry = ToolRegistry()
        custom = Tool("self_status_replacement", "custom", [], lambda: None)
        registry.register(custom)
        self.assertEqual((custom.risk, custom.permission), ("high", "approval"))
        allowed, reason = PermissionManager().authorize(custom, {})
        self.assertFalse(allowed)
        self.assertIn("requires high-risk approval", reason)


if __name__ == "__main__":
    unittest.main()
