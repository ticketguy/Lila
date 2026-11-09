"""Loopback API tests using a fake capture/transcriber session."""

import json
import http.client
import threading
import time
from types import SimpleNamespace
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock, patch

from src.daemon.service import LilaDaemon, _make_api_handler
from src.core.voice import MicrophoneError


TOKEN = "t" * 48
TRANSCRIPT = "  exact transcript, punctuation!\n"


class FakeVoice:
    def __init__(self):
        self.state = {"state": "idle", "session_id": None, "error": None}

    def start_session(self):
        self.state = {"state": "recording", "session_id": "session-1", "error": None}
        return dict(self.state)

    def stop_session(self, session_id):
        if session_id != "session-1" or self.state["state"] != "recording":
            raise RuntimeError("Voice session is not active.")
        self.state = {"state": "completed", "session_id": session_id, "error": None}
        return TRANSCRIPT

    def cancel_session(self, session_id):
        if session_id != "session-1":
            raise RuntimeError("Voice session ID is missing or no longer current.")
        self.state = {"state": "cancelled", "session_id": session_id, "error": None}
        return dict(self.state)

    def get_session_status(self):
        return dict(self.state)


class EmptyRecordingVoice(FakeVoice):
    def stop_session(self, session_id):
        raise MicrophoneError(
            "Microphone recording is empty or has an invalid format: /private/path.wav"
        )


class ServiceVoiceApiTests(unittest.TestCase):
    def setUp(self):
        self.daemon = LilaDaemon(voice=True, api_token=TOKEN)
        self.daemon._service_state = "running"
        self.daemon._running = True
        self.daemon.lila = SimpleNamespace(gguf_path="mock-model")
        self.daemon._voice = FakeVoice()
        self.daemon.process_input = Mock(return_value="Lila response")
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), _make_api_handler(self.daemon, TOKEN)
        )
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, payload=None, method=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self.base_url + path,
            data=data,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
            },
            method=method or ("POST" if data is not None else "GET"),
        )
        with urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read())

    def request_raw(self, path, body, content_type="application/json"):
        request = Request(
            self.base_url + path,
            data=body,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": content_type,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_chunked_body_is_rejected_with_specific_contract_error(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.putrequest("POST", "/chat")
            connection.putheader("Authorization", f"Bearer {TOKEN}")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Transfer-Encoding", "chunked")
            connection.endheaders()
            chunk = b'{"text":"hey"}'
            connection.send(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n0\r\n\r\n")
            response = connection.getresponse()
            payload = json.loads(response.read())
        finally:
            connection.close()

        self.assertEqual(response.status, 400)
        self.assertEqual(payload["code"], "chunked_body_unsupported")
        self.assertIn("Content-Length", payload["error"])
        self.daemon.process_input.assert_not_called()

    def test_invalid_json_has_a_specific_error(self):
        status, payload = self.request_raw("/chat", b"{")
        self.assertEqual(status, 400)
        self.assertEqual(payload["code"], "invalid_json")

    def test_empty_request_body_has_a_specific_error(self):
        status, payload = self.request_raw("/chat", b"")
        self.assertEqual(status, 400)
        self.assertEqual(payload["code"], "empty_request_body")

    def test_missing_or_invalid_text_is_distinguished(self):
        status, missing = self.request_raw("/chat", b"{}")
        self.assertEqual(status, 400)
        self.assertEqual(missing["code"], "missing_field")

        status, invalid = self.request_raw("/chat", b'{"text":7}')
        self.assertEqual(status, 400)
        self.assertEqual(invalid["code"], "invalid_field_value")

        status, empty = self.request_raw("/chat", b'{"text":"  "}')
        self.assertEqual(status, 400)
        self.assertEqual(empty["code"], "empty_field_value")
        self.daemon.process_input.assert_not_called()

    def test_voice_start_requires_an_empty_json_object(self):
        status, payload = self.request_raw("/voice/start", b'{"unexpected":true}')
        self.assertEqual(status, 400)
        self.assertEqual(payload["code"], "unexpected_fields")
        self.assertEqual(self.daemon._voice.state["state"], "idle")

    def test_empty_recording_error_is_safe_and_never_reaches_lilacore(self):
        self.daemon._voice = EmptyRecordingVoice()
        _, started = self.request("/voice/start", {})
        status, payload = self.request_raw(
            "/voice/stop", json.dumps({"session_id": started["session_id"]}).encode()
        )
        self.assertEqual(status, 502)
        self.assertEqual(payload["code"], "empty_or_invalid_recording")
        self.assertNotIn("/private/path.wav", payload["error"])
        self.daemon.process_input.assert_not_called()

    def test_invalid_voice_transition_has_a_safe_conflict_error(self):
        status, payload = self.request_raw(
            "/voice/stop", b'{"session_id":"stale-session"}'
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload["code"], "voice_session_conflict")
        self.assertNotIn("stale-session", payload["error"])

    def test_content_type_is_validated(self):
        status, payload = self.request_raw("/chat", b'{"text":"hey"}', "text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(payload["code"], "unsupported_media_type")
        self.daemon.process_input.assert_not_called()

    def test_authenticated_health_reports_readiness(self):
        status, payload = self.request("/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["status"], "running")

    def test_voice_start_stop_preserves_transcript_to_core(self):
        _, started = self.request("/voice/start", {})
        self.assertEqual(started["session_id"], "session-1")
        _, completed = self.request("/voice/stop", {"session_id": "session-1"})
        self.assertEqual(completed["transcript"], TRANSCRIPT)
        self.daemon.process_input.assert_called_once_with(TRANSCRIPT)
        self.assertEqual(completed["response"], "Lila response")

    def test_voice_cancel_does_not_deliver_transcript(self):
        self.request("/voice/start", {})
        _, result = self.request("/voice/cancel", {"session_id": "session-1"})
        self.assertEqual(result["state"], "cancelled")
        self.daemon.process_input.assert_not_called()

    def test_stop_requires_session_id(self):
        with self.assertRaises(HTTPError) as raised:
            self.request("/voice/stop", {})
        self.assertEqual(raised.exception.code, 400)
        self.daemon.process_input.assert_not_called()

    def test_concurrent_text_calls_are_serialized_through_one_core(self):
        del self.daemon.process_input
        state_lock = threading.Lock()
        active = 0
        maximum_active = 0

        def think(text):
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1
            return SimpleNamespace(text=text)

        self.daemon.lila = SimpleNamespace(think=think)
        workers = [
            threading.Thread(target=self.daemon.process_input, args=(str(i),))
            for i in range(4)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=2)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(maximum_active, 1)

    def test_headless_lifecycle_boots_one_core_and_shuts_down(self):
        daemon = LilaDaemon(
            voice=False, headless=True, port=8765, api_token=TOKEN
        )
        daemon._shutdown_event.set()
        core = Mock()
        with patch("src.daemon.service.LilaCore", return_value=core) as factory:
            with patch.object(daemon, "_acquire_instance_lock"):
                with patch.object(daemon, "_start_api"):
                    with patch.object(daemon, "_stop_api"):
                        daemon.start()
        factory.assert_called_once()
        core.boot.assert_called_once()
        self.assertEqual(daemon.service_status()["status"], "stopped")

    def test_external_gguf_model_path_uses_existing_gguf_loader(self):
        model = "/tmp/lila-replaceable/model.gguf"
        daemon = LilaDaemon(model_path=model, voice=False, headless=True,
                           port=8765, api_token=TOKEN)
        daemon._shutdown_event.set()
        core = Mock()
        with patch("src.daemon.service.LilaCore", return_value=core) as factory:
            with patch.object(daemon, "_acquire_instance_lock"):
                with patch.object(daemon, "_start_api"):
                    with patch.object(daemon, "_stop_api"):
                        daemon.start()

        factory.assert_called_once_with(
            gguf_path=model, approval_callback=None, allow_remote_model=False
        )
        core.boot.assert_called_once()

    def test_model_startup_failure_is_reported_and_cleaned_up(self):
        daemon = LilaDaemon(
            voice=False, headless=True, port=8765, api_token=TOKEN
        )
        core = Mock()
        core.boot.side_effect = RuntimeError("model unavailable")
        with patch("src.daemon.service.LilaCore", return_value=core):
            with patch.object(daemon, "_acquire_instance_lock"):
                with patch.object(daemon, "_start_api"):
                    with patch.object(daemon, "_stop_api"):
                        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                            daemon.start()
        self.assertEqual(daemon.service_status()["status"], "failed")
        self.assertIn("model unavailable", daemon.service_status()["error"])


if __name__ == "__main__":
    unittest.main()
