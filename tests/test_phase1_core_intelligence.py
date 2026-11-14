"""Phase 1 Core Intelligence unit tests."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.core.lilacore import LilaCore
from src.core.personality import LilaIdentity
from src.harness.tools import Tool, ToolArg, ToolRegistry, ToolResult
from src.harness.visual_perception import CaptureScope, VisualPerceptionBroker


class _StubHarness:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def get_tool_prompt(self):
        return 'RUNTIME CAPABILITIES\n<tool_call>\n{"name": "file_read", "arguments": {"path": "lila.py"}}\n</tool_call>'

    def process(self, candidate):
        self.calls.append(candidate)
        return list(self.results)

    @staticmethod
    def format_results(results):
        return "\n".join(
            f"{item['name']}: {item['result'].output or item['result'].error}"
            for item in results
        )

    @staticmethod
    def has_pending_tools(text):
        return "<tool_call>" in text


class Phase1CoreIntelligenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_personality_builds_unified_system_prompt_without_divergence(self) -> None:
        mem_path = self.root / "lila_memory.md"
        core = LilaCore(api_mode=True, memory_md_path=str(mem_path))
        core.remember("family", "Sammie prefers concise answers.")
        core._harness = _StubHarness()

        msg_system = core._build_messages("hi", None)[0]["content"]
        txt_prompt = core._build_prompt("hi", None)

        self.assertIn("Sammie prefers concise answers.", msg_system)
        self.assertIn("Sammie prefers concise answers.", txt_prompt)
        self.assertIn("Conversational temperament & rules:", msg_system)
        self.assertTrue(txt_prompt.startswith(msg_system))

    def test_tool_format_standardized_and_legacy_tags_cleaned(self) -> None:
        reg = ToolRegistry()
        reg.register(
            Tool(
                name="self_status",
                description="Status check",
                args=[ToolArg("detail", "string", "Detail", required=False)],
                execute=lambda **_: ToolResult(True, "ok"),
                category="system",
            )
        )
        block = reg.prompt_block()
        self.assertIn("<tool_call>", block)
        self.assertNotIn("<|tool_call|>", block)

        core = LilaCore(api_mode=True)
        cleaned = core._clean_response(
            'Hello <|tool_call|>self_status()<|/tool_call|> <tool_call>{"name":"self_status","arguments":{}}</tool_call> <think>secret'
        )
        self.assertEqual(cleaned, "Hello")

    def test_harness_cycle_strips_unclosed_think_before_parsing(self) -> None:
        core = LilaCore(api_mode=True)
        core._booted = True
        core._llm = object()
        core._harness = _StubHarness()
        # Unclosed <think> wraps a bogus tool call; stripping it leaves no tool calls.
        raw = 'Done.<think>internal <tool_call>{"name":"self_status","arguments":{}}</tool_call>'
        out, actions, raw_results = core._run_harness_cycle(
            [{"role": "user", "content": "hi"}], raw
        )
        self.assertEqual(actions, [])
        self.assertEqual(raw_results, [])
        self.assertEqual(core._clean_response(out), "Done.")

    def test_gguf_max_tokens_and_env_config(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LILA_MAX_TOKENS": "512",
                "LILA_N_CTX": "2048",
                "LILA_N_BATCH": "64",
                "LILA_N_THREADS": "4",
                "LILA_N_GPU_LAYERS": "2",
            },
        ):
            core = LilaCore(api_mode=True)

        self.assertEqual(core.max_tokens, 512)
        self.assertEqual(core.n_ctx, 2048)
        self.assertEqual(core.n_batch, 64)
        self.assertEqual(core.n_threads, 4)
        self.assertEqual(core.n_gpu_layers, 2)

        captured_kwargs = {}

        def fake_chat(**kwargs):
            captured_kwargs.update(kwargs)
            return iter([{"choices": [{"delta": {"content": "Hi"}}]}])

        core._llm = SimpleNamespace(create_chat_completion=fake_chat)
        text = core._generate_gguf([{"role": "user", "content": "hi"}])
        self.assertEqual(text, "Hi")
        self.assertEqual(captured_kwargs["max_tokens"], 512)

    def test_orchestrator_runs_independent_calls_and_preserves_invalid_diagnostic(self) -> None:
        core = LilaCore(api_mode=True)
        core._harness = Mock()

        def process_side_effect(payload_text):
            if "bad_tool" in payload_text:
                return [{"name": "bad_tool", "result": ToolResult(False, "", error="Malformed tool argument list.")}]
            return [{"name": "good_tool", "result": ToolResult(True, "sibling ok")}]

        core._harness.process.side_effect = process_side_effect

        results = core._execute_calls_via_orchestrator([
            {"name": "bad_tool", "args": {"__invalid_tool_call__": "Malformed tool argument list."}},
            {"name": "good_tool", "args": {}},
        ])
        self.assertEqual(len(results), 2)
        by_name = {item["name"]: item["result"] for item in results}
        self.assertFalse(by_name["bad_tool"].success)
        self.assertIn("Malformed tool argument list.", by_name["bad_tool"].error)
        self.assertTrue(by_name["good_tool"].success)
        self.assertEqual(by_name["good_tool"].output, "sibling ok")

    def test_gguf_vlm_boot_fallback_and_model_router_lease(self) -> None:
        gguf = self.root / "Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf"
        mmproj = self.root / "mmproj-Qwen3.5-0.8B-heretic-Q8_0.gguf"
        gguf.write_bytes(b"gguf")
        mmproj.write_bytes(b"mmproj")

        fake_llm_instance = object()
        fake_llama_cls = Mock(return_value=fake_llm_instance)
        fake_handler_cls = Mock(side_effect=RuntimeError("unsupported projector"))
        fake_llama_cpp = SimpleNamespace(Llama=fake_llama_cls)
        fake_chat_format = SimpleNamespace(Qwen25VLChatHandler=fake_handler_cls)

        core = LilaCore(gguf_path=str(gguf), mmproj_path=str(mmproj))
        with patch.dict(
            sys.modules,
            {"llama_cpp": fake_llama_cpp, "llama_cpp.llama_chat_format": fake_chat_format},
        ):
            core._boot_gguf()

        self.assertIs(core._llm, fake_llm_instance)
        self.assertFalse(core._vision_enabled)
        self.assertEqual(core.model_router.active_lease_count, 1)
        core.shutdown()
        self.assertEqual(core.model_router.active_lease_count, 0)

    def test_vision_messages_session_reset_history_trimming_and_error_confidence(self) -> None:
        img_path = self.root / "sample.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        broker = VisualPerceptionBroker()
        token = broker.issue_owner_gesture_token(CaptureScope.WINDOW)
        obs = broker.capture_explicit(
            initiator="owner_gesture",
            owner_gesture_id=token,
            scope=CaptureScope.WINDOW,
            target_label="desktop",
            image_bytes=b"\x89PNG\r\n\x1a\n",
            width=10,
            height=10,
        )

        core = LilaCore(api_mode=True, visual_broker=broker, memory_md_path=str(self.root / "mem.md"))
        core._booted = True
        core._llm = object()

        msgs = core._build_messages(
            "What is in this image?",
            None,
            images=core._resolve_image_urls(
                image_paths=[str(img_path)],
                context={"capture_id": obs.capture_id},
            ),
        )
        user_content = msgs[-1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(len(user_content), 3)
        self.assertEqual(user_content[0]["type"], "image_url")
        self.assertTrue(user_content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(user_content[2], {"type": "text", "text": "What is in this image?"})

        # Token-budgeted trimming respects n_ctx.
        core.n_ctx = 600
        core.max_tokens = 128
        core._conversation_history = [
            {"role": "user", "content": "x" * 1600},
            {"role": "assistant", "content": "y" * 1600},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        trimmed = core._trimmed_history("sys", "now")
        self.assertLess(len(trimmed), 4)

        # Honest error handling sets confidence=0.0.
        core._generate_gguf = Mock(side_effect=RuntimeError("context window exceeded"))
        err_resp = core.think("hello")
        self.assertEqual(err_resp.confidence, 0.0)
        self.assertIn("context", err_resp.text.lower())

        old_session = core.session_id
        new_session = core.reset_session()
        self.assertNotEqual(old_session, new_session)
        self.assertEqual(core._conversation_history, [])
        self.assertEqual(core.what_do_i_know()["session_id"], new_session)


if __name__ == "__main__":
    unittest.main()
