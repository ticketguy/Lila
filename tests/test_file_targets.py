import os
import json
from unittest.mock import patch
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from src.harness.file_targets import (
    FileTargetError,
    describe_file_target,
    open_verified_file,
)
from src.harness.system_tools import _file_read
from src.harness.executor import HarnessExecutor
from src.harness.tools import Tool, ToolArg, get_registry, register_tool


@unittest.skipUnless(os.name == "posix" and hasattr(os, "O_PATH"),
                     "The current safe-open adapter requires Linux O_PATH/dir_fd support")
class VerifiedFileTargetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_unchanged_approved_file_is_read_and_handle_closes(self):
        path = self.root / "approved.txt"
        path.write_text("approved contents")
        descriptor = describe_file_target(str(path))
        handle = open_verified_file(descriptor)
        result = _file_read(str(path), _verified_file=handle)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "approved contents")
        handle.close()
        self.assertTrue(handle._closed)

    def test_automatic_harness_file_read_also_uses_verified_handle(self):
        registry = get_registry()
        previous = dict(registry._tools)
        registry._tools.clear()
        path = self.root / "automatic.txt"
        path.write_text("automatic verified read")
        register_tool(Tool("file_read", "read", [ToolArg("path", "path", "target")], _file_read))
        try:
            executor = HarnessExecutor(log_dir=str(self.root / "logs"),
                                       workspace_root=str(self.root))
            call = json.dumps({"name": "file_read", "arguments": {"path": str(path)}})
            result = executor.process("<tool_call>" + call + "</tool_call>")[0]["result"]
            self.assertTrue(result.success)
            self.assertEqual(result.output, "automatic verified read")
        finally:
            registry._tools.clear()
            registry._tools.update(previous)

    def test_handler_reads_verified_descriptor_without_opening_pathname(self):
        path = self.root / "approved.txt"
        path.write_text("approved object")
        descriptor = describe_file_target(str(path))
        with open_verified_file(descriptor) as handle:
            with patch("builtins.open", side_effect=AssertionError("pathname reopen attempted")):
                result = _file_read("this pathname is deliberately irrelevant", _verified_file=handle)
            self.assertTrue(result.success)
            self.assertEqual(result.output, "approved object")
        self.assertTrue(handle._closed)

    def test_replacement_before_execution_open_is_rejected(self):
        path = self.root / "approved.txt"
        replacement = self.root / "replacement.txt"
        path.write_text("approved object")
        replacement.write_text("unapproved object")
        descriptor = describe_file_target(str(path))
        os.replace(replacement, path)
        descriptors_before = len(os.listdir("/proc/self/fd"))
        with self.assertRaises(FileTargetError):
            open_verified_file(descriptor)
        self.assertEqual(len(os.listdir("/proc/self/fd")), descriptors_before)

    def test_parent_directory_replacement_cannot_redirect_open(self):
        parent = self.root / "parent"
        parent.mkdir()
        target = parent / "item.txt"
        target.write_text("approved")
        descriptor = describe_file_target(str(target))
        old_parent = self.root / "old-parent"
        parent.rename(old_parent)
        parent.mkdir()
        (parent / "item.txt").write_text("replacement")
        with self.assertRaises(FileTargetError):
            open_verified_file(descriptor)

    def test_final_and_intermediate_symlink_substitution_are_rejected(self):
        parent = self.root / "parent"
        parent.mkdir()
        target = parent / "item.txt"
        target.write_text("approved")
        descriptor = describe_file_target(str(target))
        alternate = self.root / "alternate.txt"
        alternate.write_text("other")
        source_link = self.root / "source-link.txt"
        source_link.symlink_to(alternate)
        with self.assertRaises(FileTargetError):
            describe_file_target(str(source_link))
        target.unlink()
        target.symlink_to(alternate)
        with self.assertRaises(FileTargetError):
            open_verified_file(descriptor)

        target.unlink()
        target.write_text("approved again")
        descriptor = describe_file_target(str(target))
        moved = self.root / "moved-parent"
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(FileTargetError):
            open_verified_file(descriptor)

    def test_parent_traversal_is_rejected_before_open(self):
        target = self.root / "item.txt"
        target.write_text("contents")
        with self.assertRaises(FileTargetError):
            describe_file_target(str(self.root / "nested" / ".." / "item.txt"))

    def test_changed_identity_escaped_root_and_unsupported_type_fail_closed(self):
        path = self.root / "file.txt"
        path.write_text("before")
        descriptor = describe_file_target(str(path))
        path.write_text("changed with different size")
        with self.assertRaises(FileTargetError):
            open_verified_file(descriptor)

        descriptor = describe_file_target(str(path))
        descriptor["approved_root"] = str(self.root.parent)
        with self.assertRaises(FileTargetError):
            open_verified_file(descriptor)

        with self.assertRaises(FileTargetError):
            describe_file_target(str(self.root))

    def test_exception_path_closes_and_parallel_handles_do_not_cross(self):
        paths = []
        for index in range(2):
            path = self.root / f"{index}.txt"
            path.write_text(f"content-{index}")
            paths.append((path, describe_file_target(str(path))))

        handles = []
        try:
            with open_verified_file(paths[0][1]) as handle:
                handles.append(handle)
                raise RuntimeError("controlled failure")
        except RuntimeError:
            pass
        self.assertTrue(handles[0]._closed)

        barrier = threading.Barrier(2)

        def read(item):
            _path, descriptor = item
            with open_verified_file(descriptor) as handle:
                barrier.wait(timeout=2)
                return handle.read_text()[0]

        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(read, paths))
        self.assertEqual(results, ["content-0", "content-1"])


if __name__ == "__main__":
    unittest.main()
