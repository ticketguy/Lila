import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.daemon.oia_harness import build_approval_service
from src.harness.capabilities import legacy_tool_to_manifest
from src.harness.pending_invocations import PendingInvocationError
from src.harness.tools import Tool, ToolArg, ToolResult
from src.oia import OwnerProfile


class ResolverReviewTests(unittest.TestCase):
    def test_default_file_resolver_binds_real_object_and_changes_on_target_change(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "reviewed.txt"
            target.write_text("first")
            tool = Tool("file_read", "read", [ToolArg("path", "path", "target")],
                        lambda **_: ToolResult(True, "unused"))

            class Owner:
                def owner_profile(self):
                    return OwnerProfile("owner", "key-1", 1)

            service = build_approval_service(Owner(), SimpleNamespace(all=lambda: [tool]))
            manifest = legacy_tool_to_manifest(tool)
            resolvers = service._pending._target_resolvers
            args = {"path": str(target)}
            _, required, first = resolvers._resolve(manifest, args, args)
            self.assertTrue(required)
            self.assertEqual(first.target_kind, "filesystem.file")
            self.assertEqual(first.identity["canonical_location"], str(target.resolve()))
            self.assertTrue(first.identity["display"]["reviewed"])

            target.write_text("changed size and mtime")
            _, _, second = resolvers._resolve(manifest, args, args)
            self.assertNotEqual(first.digest, second.digest)

    def test_file_list_descriptor_binds_the_resolved_result_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "one.txt").write_text("one")
            tool = Tool("file_list", "list", [
                ToolArg("path", "path", "target", required=False, default="."),
                ToolArg("pattern", "string", "pattern", required=False, default="*.txt"),
            ], lambda **_: ToolResult(True, "unused"))

            class Owner:
                def owner_profile(self):
                    return OwnerProfile("owner", "key-3", 1)

            service = build_approval_service(Owner(), SimpleNamespace(all=lambda: [tool]))
            manifest = legacy_tool_to_manifest(tool)
            resolvers = service._pending._target_resolvers
            args = {"path": str(root), "pattern": "*.txt"}
            _, _, first = resolvers._resolve(manifest, args, args)
            self.assertEqual(first.identity["display"]["values"]["matched_locations"],
                             (str(root / "one.txt"),))
            (root / "two.txt").write_text("two")
            _, _, second = resolvers._resolve(manifest, args, args)
            self.assertNotEqual(first.digest, second.digest)

    def test_capability_without_reviewed_resolver_fails_closed(self):
        tool = Tool("power_action", "power", [ToolArg("action", "string", "action")],
                    lambda **_: ToolResult(True, "unused"))

        class Owner:
            def owner_profile(self):
                return OwnerProfile("owner", "key-2", 1)

        service = build_approval_service(Owner(), SimpleNamespace(all=lambda: [tool]))
        manifest = legacy_tool_to_manifest(tool)
        with self.assertRaises(PendingInvocationError):
            service._pending._target_resolvers._resolve(manifest, {"action": "shutdown"},
                                                        {"action": "shutdown"})


if __name__ == "__main__":
    unittest.main()
