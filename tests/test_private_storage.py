import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.lilacore import LilaCore
from src.core.private_storage import (
    open_private_text,
    secure_private_directory,
    secure_private_file,
)
from src.harness.executor import ExecutionRecord, HarnessExecutor


def permission_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class PrivateStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        if os.name != "posix":
            self.skipTest("POSIX permission modes are unavailable.")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_securing_directory_does_not_change_files_inside_it(self):
        directory = self.root / "private"
        directory.mkdir(mode=0o755)
        os.chmod(directory, 0o755)
        data_file = directory / "existing.txt"
        data_file.write_text("preserve this data", encoding="utf-8")
        unrelated_file = directory / "benchmark-output.txt"
        unrelated_file.write_text("research output", encoding="utf-8")
        os.chmod(data_file, 0o640)
        os.chmod(unrelated_file, 0o644)

        secure_private_directory(directory)

        self.assertEqual(permission_mode(directory), 0o700)
        self.assertEqual(permission_mode(data_file), 0o640)
        self.assertEqual(permission_mode(unrelated_file), 0o644)
        self.assertEqual(data_file.read_text(encoding="utf-8"), "preserve this data")
        self.assertEqual(unrelated_file.read_text(encoding="utf-8"), "research output")

    def test_explicit_recording_protection_changes_only_the_recording(self):
        recording = self.root / "lila-recording.wav"
        unrelated = self.root / "benchmark.wav"
        recording.write_bytes(b"recording bytes")
        unrelated.write_bytes(b"benchmark bytes")
        os.chmod(recording, 0o644)
        os.chmod(unrelated, 0o644)

        secure_private_file(recording)

        self.assertEqual(permission_mode(recording), 0o600)
        self.assertEqual(permission_mode(unrelated), 0o644)
        self.assertEqual(recording.read_bytes(), b"recording bytes")
        self.assertEqual(unrelated.read_bytes(), b"benchmark bytes")

    def test_explicit_transcript_protection_changes_only_the_transcript(self):
        transcript = self.root / "lila-recording.transcript.txt"
        unrelated = self.root / "asr-benchmark.txt"
        transcript.write_text("original transcript", encoding="utf-8")
        unrelated.write_text("benchmark output", encoding="utf-8")
        os.chmod(transcript, 0o644)
        os.chmod(unrelated, 0o644)

        secure_private_file(transcript)

        self.assertEqual(permission_mode(transcript), 0o600)
        self.assertEqual(permission_mode(unrelated), 0o644)
        self.assertEqual(transcript.read_text(encoding="utf-8"), "original transcript")
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "benchmark output")

    def test_private_text_file_is_created_with_restrictive_modes_and_umask_restored(self):
        directory = self.root / "new-private"
        output = directory / "interactions.jsonl"
        old_umask = os.umask(0o022)
        os.umask(old_umask)

        with open_private_text(output, "a") as stream:
            stream.write('{"private":true}\n')

        self.assertEqual(permission_mode(directory), 0o700)
        self.assertEqual(permission_mode(output), 0o600)
        self.assertEqual(os.umask(old_umask), old_umask)
        os.umask(old_umask)
        self.assertEqual(output.read_text(encoding="utf-8"), '{"private":true}\n')

    def test_private_directory_rejects_symlink(self):
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(PermissionError, "must not be a symlink"):
            secure_private_directory(link)

    def test_unsupported_permission_enforcement_fails_closed(self):
        directory = self.root / "unsupported"
        directory.mkdir(mode=0o755)
        os.chmod(directory, 0o755)

        with patch("src.core.private_storage.os.chmod"):
            with self.assertRaisesRegex(PermissionError, "Unable to secure private directory"):
                secure_private_directory(directory)

        self.assertEqual(permission_mode(directory), 0o755)

    def test_interaction_log_is_private_and_remains_readable_by_owner(self):
        log_dir = self.root / "logs"
        core = LilaCore.__new__(LilaCore)
        core.log_dir = log_dir
        core._training_log = []

        core._log_interaction("private user text", "private response")

        path = log_dir / "interactions.jsonl"
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(permission_mode(log_dir), 0o700)
        self.assertEqual(permission_mode(path), 0o600)
        self.assertEqual(record["user"], "private user text")

    def test_harness_execution_log_is_private(self):
        log_dir = self.root / "harness"
        log_dir.mkdir(mode=0o755)
        os.chmod(log_dir, 0o755)
        existing_log = log_dir / "executions.jsonl"
        existing_log.write_text("existing metadata\n", encoding="utf-8")
        os.chmod(existing_log, 0o644)

        executor = HarnessExecutor(log_dir=str(log_dir))
        executor._log(
            ExecutionRecord(
                timestamp="2026-01-01T00:00:00+00:00",
                tool_name="file_read",
                args={},
                success=True,
                output="",
                error=None,
                duration_ms=1.0,
                permission_decision="automatic",
            )
        )

        self.assertEqual(permission_mode(log_dir), 0o700)
        self.assertEqual(permission_mode(existing_log), 0o600)
        self.assertTrue(existing_log.read_text(encoding="utf-8").startswith("existing metadata"))


if __name__ == "__main__":
    unittest.main()
