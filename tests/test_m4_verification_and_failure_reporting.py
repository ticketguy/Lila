"""Milestone M4 (WS-VERIFY) tests: independent postcondition verification,
correlated privacy-minimized trace store, and failure-to-Lila reporting.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.harness.executor import HarnessExecutor
from src.harness.tools import ToolResult, get_registry
from src.harness.verification import (
    CommandPostconditionVerifier,
    CorrelatedTraceStore,
    FilePostconditionVerifier,
    VerificationStatus,
    format_failure_for_lila,
    sanitize_trace_fields,
)
from src.vtalks.contracts import TranscriptionError


class M4VerificationTests(unittest.TestCase):
    def test_file_postcondition_verifier_checks_disk_digest_and_python_ast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            py_file = Path(tmp) / "sample.py"
            py_file.write_text("def ok():\n    return 42\n", encoding="utf-8")

            verifier = FilePostconditionVerifier()
            ev_ok = verifier.verify("file_write", {"path": str(py_file)}, reported_success=True)
            self.assertEqual(ev_ok.status, VerificationStatus.VERIFIED)
            self.assertIsNotNone(ev_ok.observed_digest)
            self.assertIn("python_ast_parse", ev_ok.checks_run)

            # Corrupt syntax: handler claimed success, independent verifier catches failure
            py_file.write_text("def broken(:\n", encoding="utf-8")
            ev_bad = verifier.verify("file_write", {"path": str(py_file)}, reported_success=True)
            self.assertEqual(ev_bad.status, VerificationStatus.FAILED)

    def test_command_verifier_separates_exit_zero_from_verified(self) -> None:
        verifier = CommandPostconditionVerifier()
        ev = verifier.verify({}, reported_success=True)
        self.assertEqual(ev.status, VerificationStatus.UNVERIFIED)

        ev_amb = verifier.verify({}, reported_success=False, data={"ambiguous_side_effects": True})
        self.assertEqual(ev_amb.status, VerificationStatus.AMBIGUOUS)

    def test_correlated_trace_store_redacts_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "traces.jsonl"
            store = CorrelatedTraceStore(trace_path)
            verifier = CommandPostconditionVerifier()
            ev = verifier.verify({}, reported_success=True)
            record = store.append(
                trace_id="tr-123",
                tool_name="bash",
                lifecycle_stage="completed",
                evidence=ev,
                invocation_id="inv-1",
                metadata={"token": "super-secret-value", "safe": "ok"},
                elapsed_ms=12.3,
            )
            self.assertEqual(record["metadata"]["token"], "[REDACTED]")
            self.assertEqual(record["metadata"]["safe"], "ok")
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertNotIn("super-secret-value", lines[0])

    def test_executor_records_verification_and_failure_report_for_lila(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            executor = HarnessExecutor(log_dir=tmp, workspace_root=tmp)
            # Unknown tool fails closed and emits structured lila_failure_report
            items = executor.process('<tool_call>{"name": "nonexistent_tool", "arguments": {"arg": "1"}}</tool_call>')
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["verification"].status, VerificationStatus.FAILED)
            self.assertIn("lila_failure_report", items[0])
            self.assertEqual(items[0]["lila_failure_report"]["subsystem"], "harness")
            self.assertTrue(Path(tmp, "traces.jsonl").is_file())

    def test_daemon_stop_voice_session_reports_transcription_failure_to_lila(self) -> None:
        from src.daemon.service import LilaDaemon

        daemon = LilaDaemon.__new__(LilaDaemon)
        daemon.voice_enabled = True
        daemon._voice_request_lock = __import__("threading").Lock()
        daemon._voice = MagicMock()
        daemon._voice.stop_session.side_effect = TranscriptionError("parakeet model timed out")
        daemon._voice.get_session_status.return_value = {"state": "error", "error": "parakeet model timed out"}
        daemon._core = MagicMock()
        daemon.process_interaction = MagicMock(
            return_value=MagicMock(text="Sorry, voice transcription failed because the ASR model timed out.")
        )

        res = daemon.stop_voice_session("sess-1")
        self.assertEqual(res["transcript"], "")
        self.assertIn("timed out", res["response"])
        self.assertEqual(res["failure"]["subsystem"], "vtalks.voice")
        daemon.process_interaction.assert_called_once()


if __name__ == "__main__":
    unittest.main()
