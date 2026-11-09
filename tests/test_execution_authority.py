import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.core.lilacore import LilaResponse
from src.core.lilacore import LilaCore
from src.daemon.service import LilaDaemon, _make_api_handler
from src.harness.executor import HarnessExecutor
from src.harness.tools import Tool, ToolArg, ToolResult, get_registry, register_tool


def json_tool_call(name: str, arguments: dict) -> str:
    return "<tool_call>" + json.dumps(
        {"name": name, "arguments": arguments}
    ) + "</tool_call>"


class ExecutionAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = get_registry()
        self.previous_tools = dict(self.registry._tools)
        self.registry._tools.clear()
        self.calls = []

        def result_tool(value="ok", **kwargs):
            self.calls.append((value, kwargs))
            return ToolResult(success=True, output=value)

        register_tool(Tool("self_status", "status", [], result_tool))
        register_tool(Tool(
            "bash",
            "sensitive operation",
            [ToolArg("command", "string", "command")],
            result_tool,
        ))
        register_tool(Tool(
            "memory_store",
            "disabled memory",
            [ToolArg("content", "string", "content")],
            result_tool,
        ))
        register_tool(Tool(
            "test_schema",
            "schema validation",
            [
                ToolArg("count", "int", "count", minimum=1, maximum=3),
                ToolArg("mode", "string", "mode", choices=["safe", "strict"]),
                ToolArg("optional", "bool", "optional", required=False, default=False),
            ],
            result_tool,
        ))
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self._restore_registry)

    def _restore_registry(self) -> None:
        self.registry._tools.clear()
        self.registry._tools.update(self.previous_tools)

    def executor(self, approval_callback=None) -> HarnessExecutor:
        return HarnessExecutor(
            log_dir=self.temp_dir.name,
            approval_callback=approval_callback,
            workspace_root=self.temp_dir.name,
        )

    def test_low_risk_operation_runs_automatically(self) -> None:
        results = self.executor().process(json_tool_call("self_status", {}))
        self.assertTrue(results[0]["result"].success)
        self.assertEqual(results[0]["record"].permission_decision, "automatic")
        self.assertEqual(len(self.calls), 1)

    def test_approval_required_operation_runs_only_when_approved(self) -> None:
        denied = self.executor().process(
            json_tool_call("bash", {"command": "sensitive"})
        )[0]
        self.assertFalse(denied["result"].success)
        self.assertEqual(denied["record"].permission_decision, "denied")

        approval = Mock(return_value=True)
        allowed = self.executor(approval).process(
            json_tool_call("bash", {"command": "sensitive"})
        )[0]
        self.assertTrue(allowed["result"].success)
        self.assertEqual(allowed["record"].permission_decision, "approved")
        approval.assert_called_once()

    def test_explicit_denial_blocks_operation(self) -> None:
        result = self.executor(Mock(return_value=False)).process(
            json_tool_call("bash", {"command": "sensitive"})
        )[0]
        self.assertFalse(result["result"].success)
        self.assertEqual(result["record"].permission_decision, "denied")
        self.assertEqual(self.calls, [])

    def test_missing_approval_callback_fails_closed(self) -> None:
        result = self.executor().process(
            json_tool_call("bash", {"command": "sensitive"})
        )[0]
        self.assertFalse(result["result"].success)
        self.assertIn("no approval handler", result["result"].error)
        self.assertEqual(self.calls, [])

    def test_invalid_arguments_are_rejected_before_execution(self) -> None:
        invalid_calls = (
            json_tool_call("test_schema", {"count": 1}),
            json_tool_call(
                "test_schema", {"count": 1, "mode": "safe", "unknown": "x"}
            ),
            json_tool_call("test_schema", {"count": True, "mode": "safe"}),
            json_tool_call("test_schema", {"count": 4, "mode": "safe"}),
            json_tool_call("test_schema", {"count": 1, "mode": "unsafe"}),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                result = self.executor().process(call)[0]
                self.assertFalse(result["result"].success)
                self.assertEqual(
                    result["record"].permission_decision, "invalid_arguments"
                )
        self.assertEqual(self.calls, [])

    def test_bash_timeout_must_be_within_registered_range(self) -> None:
        from src.harness.system_tools import register_all_system_tools

        register_all_system_tools()
        result = self.executor(Mock(return_value=True)).process(
            json_tool_call("bash", {"command": "echo safe", "timeout": 0})
        )[0]
        self.assertFalse(result["result"].success)
        self.assertEqual(result["record"].permission_decision, "invalid_arguments")
        self.assertIn("below its minimum", result["result"].error)
        self.assertEqual(self.calls, [])

    def test_malformed_argument_syntax_is_rejected(self) -> None:
        result = self.executor().process(
            '<|tool_call|>bash(command="safe"; extra="second")<|/tool_call|>'
        )[0]
        self.assertFalse(result["result"].success)
        self.assertEqual(
            result["record"].permission_decision, "invalid_arguments"
        )
        self.assertEqual(self.calls, [])

    def test_duplicate_json_arguments_are_rejected(self) -> None:
        result = self.executor().process(
            '<tool_call>{"name":"bash","arguments":'
            '{"command":"first","command":"second"}}</tool_call>'
        )[0]
        self.assertFalse(result["result"].success)
        self.assertEqual(
            result["record"].permission_decision, "invalid_arguments"
        )
        self.assertEqual(self.calls, [])

    def test_optional_defaults_are_applied(self) -> None:
        result = self.executor(Mock(return_value=True)).process(
            json_tool_call("test_schema", {"count": 1, "mode": "SAFE"})
        )[0]
        self.assertTrue(result["result"].success)
        self.assertEqual(self.calls[0][1]["optional"], False)
        self.assertEqual(self.calls[0][1]["mode"], "safe")

    def test_unknown_tools_are_denied(self) -> None:
        result = self.executor().process(
            json_tool_call("unregistered", {})
        )[0]
        self.assertFalse(result["result"].success)
        self.assertIn("Unknown tool", result["result"].error)
        self.assertEqual(self.calls, [])

    def test_memory_tools_remain_disabled(self) -> None:
        result = self.executor(Mock(return_value=True)).process(
            json_tool_call("memory_store", {"content": "private"})
        )[0]
        self.assertFalse(result["result"].success)
        self.assertIn("disabled", result["result"].error)
        self.assertEqual(self.calls, [])

    def test_unresolvable_read_path_fails_closed(self) -> None:
        link = Path(self.temp_dir.name) / "loop"
        try:
            link.symlink_to(link)
        except (NotImplementedError, OSError):
            self.skipTest("Symlinks are unavailable in this environment")
        register_tool(Tool(
            "file_read",
            "read file",
            [ToolArg("path", "path", "path")],
            lambda **kwargs: ToolResult(success=True, output="unexpected"),
        ))
        result = self.executor().process(
            json_tool_call("file_read", {"path": str(link)})
        )[0]
        self.assertFalse(result["result"].success)
        self.assertEqual(result["record"].permission_decision, "denied")

    def test_execution_logs_store_metadata_only(self) -> None:
        secret_input = "PASSWORD=sample-private-value"
        secret_output = "clipboard sample-private-value and private file contents"

        def sensitive_result(**kwargs):
            return ToolResult(success=True, output=secret_output)

        register_tool(Tool("clipboard_get", "clipboard", [], sensitive_result))
        register_tool(Tool(
            "clipboard_set",
            "clipboard write",
            [ToolArg("content", "string", "content")],
            sensitive_result,
        ))
        executor = self.executor(Mock(return_value=True))
        executor.process(json_tool_call("clipboard_get", {}))
        executor.process(json_tool_call(
            "clipboard_set", {"content": secret_input}
        ))
        log_text = (os.path.join(self.temp_dir.name, "executions.jsonl"))
        with open(log_text, encoding="utf-8") as stream:
            logged = stream.read()
        self.assertNotIn(secret_input, logged)
        self.assertNotIn("sample-private-value", logged)
        entries = [json.loads(line) for line in logged.splitlines()]
        entry = entries[0]
        self.assertEqual(entry["tool"], "clipboard_get")
        self.assertIn(entry["permission_decision"], {"approved", "automatic"})
        self.assertEqual(entry["output"], "[REDACTED]")
        self.assertEqual(entries[1]["tool"], "clipboard_set")
        self.assertEqual(entries[1]["args"]["content"], "[REDACTED]")

    def test_daemon_delegates_to_lilacore_only(self) -> None:
        daemon = LilaDaemon(voice=False)
        core = SimpleNamespace(
            think=Mock(return_value=LilaResponse(
                text="central result",
                memory_ops=[],
                actions=[],
                confidence=1.0,
            ))
        )
        daemon.lila = core
        self.assertEqual(daemon.process_input("hello"), "central result")
        core.think.assert_called_once_with("hello")
        self.assertFalse(hasattr(daemon, "executor"))
        import inspect
        self.assertNotIn("execute_tool", inspect.getsource(LilaDaemon))

    def test_daemon_voice_transcript_uses_lilacore(self) -> None:
        daemon = LilaDaemon(voice=True)
        daemon._running = True
        daemon._activated = True
        core = SimpleNamespace(
            think=Mock(return_value=LilaResponse(
                text="voice response",
                memory_ops=[],
                actions=[],
                confidence=1.0,
            ))
        )
        daemon.lila = core

        class InjectedVoice:
            is_listening = False

            def __init__(self):
                self.spoken = []

            def start_listening(self, on_input):
                on_input("injected transcript")

            def speak(self, text):
                self.spoken.append(text)

            def stop_listening(self):
                self.is_listening = False

        voice = InjectedVoice()
        with patch("src.core.voice.LilaVoice", return_value=voice):
            daemon._voice_loop()

        core.think.assert_called_once_with("injected transcript")
        self.assertEqual(voice.spoken, ["voice response"])

    def test_lilacore_uses_harness_for_transformer_tool_calls(self) -> None:
        core = LilaCore()
        core._booted = True
        core._model = object()
        core._harness = self.executor()
        core.log_dir = Path(self.temp_dir.name)
        core._log_interaction = Mock()
        core._generate_transformers = Mock(side_effect=[
            json_tool_call("self_status", {}),
            "final answer",
        ])

        response = core.think("check status")

        self.assertEqual(response.text, "final answer")
        self.assertEqual(response.actions[0]["tool"], "self_status")
        self.assertTrue(response.actions[0]["success"])
        self.assertEqual(len(self.calls), 1)

    def test_daemon_api_requires_token_and_binds_localhost(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires LILA_API_TOKEN"):
                LilaDaemon(voice=False, port=7777)
        daemon = LilaDaemon(
            voice=False,
            port=7777,
            api_token="x" * 32,
        )
        self.assertEqual(daemon.api_host, "127.0.0.1")

    def test_daemon_module_startup_has_no_runpy_import_warning(self) -> None:
        source_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-B", "-m", "src.daemon.service", "--help"],
            cwd=source_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("RuntimeWarning", result.stderr)

    def test_jarvis_legacy_runtime_is_removed(self) -> None:
        source_root = Path(__file__).resolve().parents[1]
        self.assertFalse((source_root / "jarvis.py").exists())

    def test_api_rejects_unauthenticated_access(self) -> None:
        token = "t" * 32
        daemon = LilaDaemon(voice=False, api_token=token)
        daemon.process_input = Mock(return_value="should not execute")
        server = HTTPServer(
            ("127.0.0.1", 0), _make_api_handler(daemon, token)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        base_url = f"http://127.0.0.1:{server.server_port}"

        with self.assertRaises(HTTPError) as response:
            urlopen(base_url + "/status", timeout=3)
        self.assertEqual(response.exception.code, 401)

        request = Request(
            base_url + "/chat",
            data=json.dumps({"text": "hello"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as response:
            urlopen(request, timeout=3)
        self.assertEqual(response.exception.code, 401)
        daemon.process_input.assert_not_called()

    def test_authenticated_api_routes_through_daemon_core(self) -> None:
        token = "t" * 32
        daemon = LilaDaemon(voice=False, api_token=token)
        daemon.process_input = Mock(return_value="central result")
        server = HTTPServer(
            ("127.0.0.1", 0), _make_api_handler(daemon, token)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)

        request = Request(
            f"http://127.0.0.1:{server.server_port}/chat",
            data=json.dumps({"text": "hello"}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload["response"], "central result")
        daemon.process_input.assert_called_once_with("hello")


if __name__ == "__main__":
    unittest.main()
