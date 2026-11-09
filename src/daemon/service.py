"""Lila's foreground daemon and authenticated loopback service.

Foreground voice uses terminal-driven push-to-talk; headless voice is controlled
through authenticated start/stop requests. There is no continuous listening.
Tool calls route through LilaCore and HarnessExecutor.

Usage:
    python -m src.daemon.service             # Start daemon
    python -m src.daemon.service --no-voice  # Text-only daemon
    python -m src.daemon.service --headless --port 8765
"""

import hmac
import json
import os
import signal
import sys
import time
import threading
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.lilacore import LilaCore
from src.core.runtime_lock import acquire_runtime_lock, release_runtime_lock


class DaemonAlreadyRunningError(RuntimeError):
    """Raised when another Lila daemon already owns the workspace lock."""


class LilaDaemon:
    """One LilaCore instance with terminal or authenticated HTTP interfaces."""
    
    def __init__(self, asi_path: Optional[str] = None, model_path: Optional[str] = None,
                 voice: bool = True, port: int = 0,
                 api_token: Optional[str] = None,
                 allow_remote_model: bool = False,
                 headless: bool = False):
        self.asi_path = asi_path
        self.model_path = model_path
        self.allow_remote_model = allow_remote_model
        self.headless = headless
        self.voice_enabled = voice
        self.port = port
        self.api_host = "127.0.0.1"
        self.api_token = api_token or os.environ.get("LILA_API_TOKEN")
        if port < 0 or port > 65535:
            raise ValueError("API port must be between 0 and 65535.")
        if port and (not self.api_token or len(self.api_token) < 32):
            raise ValueError(
                "HTTP API requires LILA_API_TOKEN (at least 32 characters)."
            )
        if headless and not port:
            raise ValueError("Headless mode requires an authenticated API port.")
        
        self.lila: Optional[LilaCore] = None
        self._voice = None
        self._running = False
        self._activated = False
        self._service_state = "stopped"
        self._service_error: str | None = None
        self._shutdown_event = threading.Event()
        self._interaction_lock = threading.RLock()
        self._voice_request_lock = threading.Lock()
        self._instance_lock_fd: int | None = None
        self._voice_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._api_thread: Optional[threading.Thread] = None
        self._api_server = None
        
    
    def start(self):
        """Run until stopped; headless mode never reads from terminal stdin."""
        self._acquire_instance_lock()
        self._running = True
        self._activated = self.headless
        self._service_state = "starting"
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._shutdown)
            signal.signal(signal.SIGTERM, self._shutdown)

        try:
            # Bind early so clients can poll authenticated health while the
            # single language model is loading.
            if self.port and self.headless:
                self._start_api()
            print("\n🌸 Loading Lila...")
            self.lila = self._create_lila_core()
            self.lila.boot()
            if self.voice_enabled:
                from src.core.voice import LilaVoice
                self._voice = LilaVoice()
            self._service_state = "running"
            print("🌸 Lila loaded.\n")

            if self.headless:
                self._shutdown_event.wait()
            elif self.voice_enabled:
                self._voice_thread = threading.Thread(
                    target=self._voice_loop, daemon=True
                )
                self._voice_thread.start()
                while self._running:
                    self._shutdown_event.wait(0.25)
            else:
                self._text_loop()
        except Exception as exc:
            self._service_state = "failed"
            self._service_error = f"{type(exc).__name__}: {exc}"[:500]
            print(f"Lila startup/runtime failure: {self._service_error}", file=sys.stderr)
            raise
        finally:
            self._cleanup()

    def _acquire_instance_lock(self) -> None:
        try:
            self._instance_lock_fd = acquire_runtime_lock()
        except RuntimeError as exc:
            raise DaemonAlreadyRunningError(
                str(exc)
            ) from exc

    def _create_lila_core(self):
        """Route local GGUF files through the existing GGUF loader."""
        model_argument = {"model_path": self.model_path}
        if self.model_path and Path(self.model_path).suffix.casefold() == ".gguf":
            model_argument = {"gguf_path": self.model_path}
        return LilaCore(
            **model_argument,
            approval_callback=None,
            allow_remote_model=self.allow_remote_model,
        )

    def _release_instance_lock(self) -> None:
        descriptor = self._instance_lock_fd
        self._instance_lock_fd = None
        release_runtime_lock(descriptor)
    
    def _shutdown(self, signum=None, frame=None):
        """Request graceful shutdown; cleanup runs in the service thread."""
        self._running = False
        self._service_state = "stopping"
        self._shutdown_event.set()

    def stop(self) -> None:
        """Public in-process shutdown hook used by supervisors and tests."""
        self._shutdown()

    def _cleanup(self) -> None:
        self._running = False
        if self._service_state != "failed":
            self._service_state = "stopping"
        if self._voice is not None:
            try:
                status = self._voice.get_session_status()
                if status["state"] in {"recording", "transcribing", "cancelling", "timed_out"}:
                    self._voice.cancel_session(status["session_id"])
            except Exception as exc:
                print(f"Voice shutdown warning: {type(exc).__name__}", file=sys.stderr)
        self._stop_api()
        if self._voice_thread is not None and self._voice_thread.is_alive():
            self._voice_thread.join(timeout=2)
        self.lila = None
        self._release_instance_lock()
        if self._service_state != "failed":
            self._service_state = "stopped"

    def _start_api(self) -> None:
        from http.server import ThreadingHTTPServer

        if not self.api_token:
            raise RuntimeError("HTTP API cannot start without authentication.")
        server = ThreadingHTTPServer(
            (self.api_host, self.port), _make_api_handler(self, self.api_token)
        )
        server.daemon_threads = True
        server.request_queue_size = 16
        self.port = int(server.server_address[1])
        self._api_server = server
        self._api_thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.25},
            name="lila-http-api",
            daemon=True,
        )
        self._api_thread.start()

    def _stop_api(self) -> None:
        server = self._api_server
        self._api_server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if self._api_thread is not None and self._api_thread.is_alive():
            self._api_thread.join(timeout=3)
        self._api_thread = None
    
    # These phrases are checked only after typed input or a completed recording.
    ACTIVATION_PHRASES = ["lila", "hey lila", "wake up", "initialize", "begin", "start"]
    
    def _is_activation_phrase(self, text: str) -> bool:
        """Check an explicitly submitted string for an activation phrase."""
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in self.ACTIVATION_PHRASES)
    
    def _activate(self):
        """
        Activate the conversation after an explicit text/transcript trigger.
        """
        if self._activated:
            return
        
        print("\n🌸 Activation triggered. Lila is initializing herself...\n")
        self._activated = True
        
        # Start background services now that she's active
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        if self.port > 0 and self._api_server is None:
            self._start_api()
        
        init_prompt = (
            "You have just been activated on a new system. "
            "Initialize yourself: detect the OS and hardware, "
            "report what you find. Sensitive actions require owner approval."
        )
        
        response = self.process_input(init_prompt)
        print(f"Lila: {response}\n")
    
    def process_input(self, text: str) -> str:
        """Route voice, text, and authenticated HTTP input through LilaCore."""
        with self._interaction_lock:
            if not self.lila:
                raise RuntimeError("Lila model is not ready.")
            return self.lila.think(text).text

    def service_status(self) -> dict:
        return {
            "status": self._service_state,
            "ready": self._service_state == "running" and self.lila is not None,
            "voice_enabled": self.voice_enabled,
            "model": self.model_path or (self.lila.gguf_path if self.lila else "default"),
            "error": self._service_error,
        }

    def start_voice_session(self) -> dict:
        if self._service_state != "running":
            raise RuntimeError("Lila service is not ready.")
        if not self.voice_enabled or self._voice is None:
            raise RuntimeError("Voice sessions are disabled.")
        return self._voice.start_session()

    def stop_voice_session(self, session_id: str | None) -> dict:
        if self._voice is None or not self.voice_enabled:
            raise RuntimeError("Voice sessions are disabled.")
        # One stop request owns a completed transcript. Duplicate requests cannot
        # submit the same utterance to the conversation twice.
        with self._voice_request_lock:
            transcript = self._voice.stop_session(session_id)
            response = self.process_input(transcript)
            return {
                "session_id": session_id,
                "transcript": transcript,
                "response": response,
                "status": self._voice.get_session_status(),
            }

    def cancel_voice_session(self, session_id: str | None) -> dict:
        if self._voice is None or not self.voice_enabled:
            raise RuntimeError("Voice sessions are disabled.")
        return self._voice.cancel_session(session_id)

    def voice_status(self) -> dict:
        if self._voice is None or not self.voice_enabled:
            return {"state": "disabled", "session_id": None, "error": None}
        return self._voice.get_session_status()
    
    def _text_loop(self):
        """Interactive text input loop (when voice is disabled)."""
        while self._running:
            try:
                user_input = input("You: " if self._activated else "> ")
                if not user_input.strip():
                    continue
                if user_input.lower() in ("quit", "exit"):
                    self._shutdown()
                    break
                
                # Check the typed line for a phrase; no audio is monitored here.
                if not self._activated:
                    if self._is_activation_phrase(user_input):
                        self._activate()
                    else:
                        print("  (not active — enter an activation phrase)\n")
                    continue
                
                # Active — process through Lila
                response = self.process_input(user_input)
                print(f"Lila: {response}\n")
            except (KeyboardInterrupt, EOFError):
                self._shutdown()
                break
    
    def _voice_loop(self):
        """Run terminal-driven push-to-talk and forward captured transcripts."""
        try:
            from src.core.voice import LilaVoice
            voice = self._voice or LilaVoice()
            self._voice = voice
            voice.start_listening(
                on_input=lambda transcript: self._handle_voice_transcript(
                    voice, transcript
                )
            )
            while self._running and voice.is_listening:
                time.sleep(0.1)
        except ImportError:
            print("   Voice: unavailable; verify the configured capture and ASR tools.")
            self._text_loop()
        except Exception as e:
            print(f"   Voice error: {type(e).__name__}: {e}")
            self._text_loop()

    def _handle_voice_transcript(self, voice, transcript: str):
        """Route recognized speech through the same LilaCore path as text."""
        if not transcript or not self._running:
            return
        if not self._activated:
            if self._is_activation_phrase(transcript):
                self._activate()
                voice.speak("I'm awake.")
            return
        response = self.process_input(transcript)
        voice.speak(response)
    
    def _monitor_loop(self):
        """
        Poll the schedule file and optional .asi path for change detection.

        Schedule event handling and actual .asi hot-reload are not implemented;
        this loop does not monitor general filesystem or network activity.
        """
        schedule_file = os.path.expanduser("~/.lila/schedule.json")
        last_check = 0
        
        while self._running:
            time.sleep(10)  # Check every 10 seconds
            
            # Check schedule
            if os.path.exists(schedule_file):
                try:
                    with open(schedule_file) as f:
                        schedule = json.load(f)
                    
                    now = time.time()
                    for event in schedule:
                        if event.get("done"):
                            continue
                        # Simple time-based trigger (could be smarter)
                        # TODO: parse event["time"] properly
                except Exception:
                    pass
            
            # Detect an .asi update; reloading is a future implementation.
            if self.asi_path and os.path.exists(self.asi_path):
                mtime = os.path.getmtime(self.asi_path)
                if mtime > last_check and last_check > 0:
                    print("\n🌸 [Hot-reload] New .asi detected, reloading adapters...")
                    # TODO: call asi_reload_fabric via the C engine
                last_check = mtime
    
def _make_api_handler(daemon: LilaDaemon, expected_token: str):
    """Build an authenticated handler used only on the localhost API."""
    from http.server import BaseHTTPRequestHandler
    from src.core.voice import (
        MicrophoneError,
        TranscriptionError,
        VoiceInputError,
        VoiceSessionBusyError,
        VoiceSessionCancelled,
        VoiceSessionError,
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = "LilaLocalAPI"
        MAX_BODY_BYTES = 1_000_000

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def _send_json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            value = self.headers.get("Authorization", "")
            scheme, separator, supplied_token = value.partition(" ")
            valid = (
                separator == " "
                and scheme.casefold() == "bearer"
                and hmac.compare_digest(
                    supplied_token.encode("utf-8"),
                    expected_token.encode("utf-8"),
                )
            )
            if not valid:
                self._send_json(401, {"error": "Authentication required."})
            return valid

        def do_GET(self):
            if not self._authorized():
                return
            if self.path not in {"/status", "/health", "/voice/status"}:
                self._send_json(404, {"error": "Not found."})
                return
            if self.path == "/voice/status":
                self._send_json(200, daemon.voice_status())
            else:
                status = daemon.service_status()
                status["running"] = daemon._running
                status["voice"] = daemon.voice_enabled
                self._send_json(200, status)

        def do_POST(self):
            if not self._authorized():
                return
            if self.path not in {
                "/chat", "/voice/start", "/voice/stop", "/voice/cancel"
            }:
                self._send_json(404, {"error": "Not found."})
                return
            if self.headers.get("Transfer-Encoding"):
                self._send_json(400, {
                    "error": "Chunked request bodies are not supported; send Content-Length.",
                    "code": "chunked_body_unsupported",
                })
                return
            if self.headers.get_content_type().casefold() != "application/json":
                self._send_json(415, {
                    "error": "Content-Type must be application/json.",
                    "code": "unsupported_media_type",
                })
                return
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                self._send_json(411, {
                    "error": "Content-Length is required.",
                    "code": "content_length_required",
                })
                return
            if not raw_length.isdecimal():
                self._send_json(400, {
                    "error": "Content-Length must be a non-negative integer.",
                    "code": "invalid_content_length",
                })
                return
            length = int(raw_length)
            if length > self.MAX_BODY_BYTES:
                self._send_json(413, {
                    "error": "Request body exceeds the 1 MB limit.",
                    "code": "request_too_large",
                })
                return
            if length == 0:
                self._send_json(400, {
                    "error": "Request body is empty; send a JSON object.",
                    "code": "empty_request_body",
                })
                return
            raw_body = self.rfile.read(length)
            if len(raw_body) != length:
                self._send_json(400, {
                    "error": "Request body ended before Content-Length bytes arrived.",
                    "code": "incomplete_request_body",
                })
                return
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {
                    "error": "Request body must contain valid UTF-8 JSON.",
                    "code": "invalid_json",
                })
                return
            if not isinstance(payload, dict):
                self._send_json(400, {
                    "error": "JSON request body must be an object.",
                    "code": "invalid_json_value",
                })
                return
            if self.path == "/chat":
                if "text" not in payload:
                    self._send_json(400, {
                        "error": "Required field 'text' is missing.",
                        "code": "missing_field",
                    })
                    return
                if not isinstance(payload["text"], str):
                    self._send_json(400, {
                        "error": "Field 'text' must be a string.",
                        "code": "invalid_field_value",
                    })
                    return
                if not payload["text"].strip():
                    self._send_json(400, {
                        "error": "Field 'text' must not be empty.",
                        "code": "empty_field_value",
                    })
                    return
                operation = lambda: {"response": daemon.process_input(payload["text"])}
            elif self.path == "/voice/start":
                if payload != {}:
                    self._send_json(400, {
                        "error": "Voice start accepts an empty JSON object only.",
                        "code": "unexpected_fields",
                    })
                    return
                operation = daemon.start_voice_session
            else:
                if "session_id" not in payload:
                    self._send_json(400, {
                        "error": "Required field 'session_id' is missing.",
                        "code": "missing_field",
                    })
                    return
                session_id = payload["session_id"]
                if not isinstance(session_id, str) or not session_id.strip():
                    self._send_json(400, {
                        "error": "Field 'session_id' must be a non-empty string.",
                        "code": "invalid_field_value",
                    })
                    return
                if self.path == "/voice/stop":
                    operation = lambda: daemon.stop_voice_session(session_id)
                else:
                    operation = lambda: daemon.cancel_voice_session(session_id)
            try:
                result = operation()
            except VoiceSessionBusyError:
                self._send_json(409, {
                    "error": "Voice session transition is not allowed in the current state.",
                    "code": "voice_session_conflict",
                })
                return
            except VoiceSessionCancelled:
                self._send_json(409, {
                    "error": "Voice session was cancelled.",
                    "code": "voice_session_cancelled",
                })
                return
            except VoiceSessionError:
                self._send_json(409, {
                    "error": "Voice session is no longer active.",
                    "code": "voice_session_conflict",
                })
                return
            except VoiceInputError as exc:
                detail = str(exc).casefold()
                if isinstance(exc, MicrophoneError) and (
                    "empty" in detail or "invalid format" in detail
                ):
                    message = "Recording is empty or has an invalid audio format."
                    code = "empty_or_invalid_recording"
                elif isinstance(exc, MicrophoneError):
                    message = "Microphone capture failed. Check the WSL audio input."
                    code = "microphone_failure"
                elif isinstance(exc, TranscriptionError):
                    message = "Speech recognition failed to transcribe the recording."
                    code = "transcription_failure"
                else:
                    message = "Voice input processing failed."
                    code = "voice_input_failure"
                self._send_json(502, {"error": message, "code": code})
                return
            except RuntimeError as exc:
                detail = str(exc).casefold()
                if "not ready" in detail:
                    self._send_json(503, {
                        "error": "Lila service is not ready.",
                        "code": "service_not_ready",
                    })
                elif "disabled" in detail:
                    self._send_json(503, {
                        "error": "Voice sessions are disabled.",
                        "code": "voice_disabled",
                    })
                else:
                    self._send_json(409, {
                        "error": "Voice session transition is not allowed in the current state.",
                        "code": "voice_session_conflict",
                    })
                return
            except Exception as exc:
                self._send_json(500, {"error": f"Request failed ({type(exc).__name__})."})
                return
            self._send_json(200, result if isinstance(result, dict) else {"response": result})

        def log_message(self, format, *args):
            return

    return Handler


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lila foreground runtime")
    parser.add_argument("--asi", default=None, help="Path to .asi file")
    parser.add_argument(
        "--model",
        default=os.environ.get("LILA_MODEL_PATH"),
        help="Local model path (or configured LILA_MODEL_PATH); remote IDs require --allow-remote-model",
    )
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="Allow downloading the explicitly configured --model identifier",
    )
    parser.add_argument("--no-voice", action="store_true", help="Disable voice (text only)")
    parser.add_argument("--port", type=int, default=0, help="HTTP API port (0=disabled)")
    parser.add_argument("--headless", action="store_true", help="Run without terminal input")
    args = parser.parse_args()
    
    daemon = LilaDaemon(
        asi_path=args.asi,
        model_path=args.model,
        allow_remote_model=args.allow_remote_model,
        voice=not args.no_voice,
        port=args.port,
        headless=args.headless,
    )
    daemon.start()


if __name__ == "__main__":
    main()
