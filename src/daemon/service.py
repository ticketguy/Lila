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
import hashlib
import json
import os
import re
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
from src.core.utf8 import configure_standard_streams


class DaemonAlreadyRunningError(RuntimeError):
    """Raised when another Lila daemon already owns the workspace lock."""


class LilaDaemon:
    """One LilaCore instance with terminal or authenticated HTTP interfaces."""
    
    def __init__(self, asi_path: Optional[str] = None, model_path: Optional[str] = None,
                 voice: bool = True, port: int = 0,
                 api_token: Optional[str] = None,
                 allow_remote_model: bool = False,
                 headless: bool = False, approval_continuation_service=None):
        self.asi_path = asi_path
        self.model_path = model_path
        self.allow_remote_model = allow_remote_model
        self.headless = headless
        self.approval_continuation_service = approval_continuation_service
        self.owner_authentication_service = None
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
        self._oia_voice_session_id: str | None = None
        self._oia_voice_invocation_id: str | None = None
        self._oia_spoken_enroll_session_id: str | None = None
        self._instance_lock_fd: int | None = None
        self._voice_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._api_thread: Optional[threading.Thread] = None
        self._api_server = None

        from src.daemon.events import DaemonTaskCenter, VersionedEventStream
        from src.harness.capability_learning import CapabilityPromotionService
        from src.harness.orchestration import BoundedTaskOrchestrator, DurableCheckpointStore
        from src.harness.visual_perception import VisualPerceptionBroker
        from src.sentinel.triage import SentinelTriageEngine
        self.event_stream = VersionedEventStream()
        self.task_center = DaemonTaskCenter(self.event_stream)
        self.sentinel_engine = SentinelTriageEngine()
        self.visual_perception = VisualPerceptionBroker()
        self.checkpoint_store = DurableCheckpointStore(
            os.path.join(os.path.expanduser("~"), ".local", "share", "lila", "harness_checkpoints.json")
        )
        self.orchestrator = BoundedTaskOrchestrator(checkpoint_store=self.checkpoint_store, max_workers=2)
        self.capability_learning = CapabilityPromotionService(
            storage_dir=os.path.join(os.path.expanduser("~"), ".local", "share", "lila", "capabilities")
        )
        _, recovered_tasks = self.checkpoint_store.recover_on_restart()
        self.recovered_uncertain_tasks = [
            node.to_dict() for node in recovered_tasks.values()
            if node.state.value == "uncertain"
        ]
        for report in self.recovered_uncertain_tasks:
            self.event_stream.publish("task.uncertain", report)
        
    
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
            self._wire_approval_events()
            if self.voice_enabled:
                from src.core.voice import LilaVoice
                self._voice = LilaVoice()
            if self.owner_authentication_service is not None:
                self.owner_authentication_service.set_spoken_provider_available(
                    bool(self._voice and self._voice.private_asr_available)
                )
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
        from src.oia.cli import default_profile_path
        from src.oia.service import OwnerAuthenticationService, OwnerProfileStore
        from src.daemon.oia_harness import build_approval_service

        owner_store = OwnerProfileStore(default_profile_path())
        if not owner_store.enrolled and not self.headless and sys.stdin.isatty():
            from src.oia.cli import enroll_local
            try:
                enroll_local(owner_store)
            except Exception as exc:
                print(
                    "Owner enrollment unavailable "
                    f"(reason_code={getattr(exc, 'reason_code', 'owner_enrollment.failed')}).",
                    file=sys.stderr,
                )
        elif not owner_store.enrolled:
            print("Owner profile absent; run `python -m src.oia.cli` in a trusted local terminal before approvals.")
        owner_service = OwnerAuthenticationService(owner_store)
        self.owner_authentication_service = owner_service
        model_argument = {"model_path": self.model_path}
        if self.model_path and Path(self.model_path).suffix.casefold() == ".gguf":
            model_argument = {"gguf_path": self.model_path}
        core_options = {
            **model_argument,
            "approval_callback": None,
            "allow_remote_model": self.allow_remote_model,
        }
        if self.approval_continuation_service is not None:
            core_options["approval_continuation_service"] = self.approval_continuation_service
        else:
            core_options["approval_service_factory"] = (
                lambda tool_registry: build_approval_service(owner_service, tool_registry)
            )
        core = LilaCore(**core_options)
        if hasattr(core, "orchestrator"):
            core.orchestrator = self.orchestrator
        return core

    def _wire_approval_events(self) -> None:
        svc = (getattr(self.lila, "approval_continuation_service", None)
               if self.lila is not None else None) or self.approval_continuation_service
        if svc is None or not hasattr(svc, "register_audit_observer"):
            return

        def _on_audit(ev) -> None:
            stage = str(getattr(ev, "lifecycle_stage", ""))
            ev_type = "approval.requested" if stage == "requested" else "approval.resolved"
            self.event_stream.publish(ev_type, {
                "invocation_id": getattr(ev, "invocation_id", ""),
                "capability_id": getattr(ev, "capability_id", ""),
                "lifecycle_stage": stage,
                "reason_code": getattr(ev, "reason_code", ""),
            })

        svc.register_audit_observer(_on_audit)

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
        approval_service = (
            getattr(self.lila, "approval_continuation_service", None)
            if self.lila is not None else None
        ) or self.approval_continuation_service
        if approval_service is not None:
            try:
                approval_service.close()
            except Exception as exc:
                print(f"Approval shutdown warning: {type(exc).__name__}", file=sys.stderr)
        if self.lila is not None and hasattr(self.lila, "shutdown"):
            try:
                self.lila.shutdown()
            except Exception as exc:
                print(f"Lila shutdown warning: {type(exc).__name__}", file=sys.stderr)
        self.lila = None
        self._release_instance_lock()
        if self._service_state != "failed":
            self._service_state = "stopped"

    def _start_api(self) -> None:
        from http.server import ThreadingHTTPServer
        from src.daemon.transport import allowed_client_fingerprints, build_mutual_tls_context

        if not self.api_token:
            raise RuntimeError("HTTP API cannot start without authentication.")
        server = ThreadingHTTPServer(
            (self.api_host, self.port),
            _make_api_handler(self, self.api_token,
                              allowed_client_fingerprints=allowed_client_fingerprints())
        )
        try:
            tls_context = build_mutual_tls_context()
            server.socket = tls_context.wrap_socket(server.socket, server_side=True)
        except Exception:
            server.server_close()
            raise
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
    
    def process_interaction(
        self,
        text: str,
        images: list[str] | None = None,
        image_paths: list[str] | None = None,
    ):
        """Return the full backward-compatible LilaResponse for one input."""
        with self._interaction_lock:
            if not self.lila:
                raise RuntimeError("Lila model is not ready.")
            if images is None and image_paths is None:
                return self.lila.think(text)
            return self.lila.think(text, images=images, image_paths=image_paths)

    def process_input(self, text: str) -> str:
        """Legacy text-only interface retained for terminal and callers."""
        return self.process_interaction(text).text

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

        def _on_live_transcript(chunk_text: str, accumulated: str, is_final: bool) -> None:
            self.event_stream.publish(
                "voice.transcript",
                {
                    "session_id": self._voice.get_session_status().get("session_id"),
                    "text": chunk_text,
                    "accumulated": accumulated,
                    "is_final": is_final,
                },
            )

        try:
            result = self._voice.start_session(on_transcript=_on_live_transcript)
        except TypeError:
            result = self._voice.start_session()
        self.event_stream.publish("voice.state", {"state": "recording", "session_id": result.get("session_id")})
        return result

    def stop_voice_session(self, session_id: str | None) -> dict:
        if self._voice is None or not self.voice_enabled:
            raise RuntimeError("Voice sessions are disabled.")
        # One stop request owns a completed transcript. Duplicate requests cannot
        # submit the same utterance to the conversation twice.
        with self._voice_request_lock:
            from src.vtalks.contracts import TranscriptionError

            try:
                transcript = self._voice.stop_session(session_id)
            except TranscriptionError as exc:
                from src.harness.verification import format_failure_for_lila

                failure = format_failure_for_lila(
                    "vtalks.voice",
                    type(exc).__name__,
                    str(exc),
                )
                lila_text = f"I couldn't complete voice transcription ({failure['error_code']}): {failure['detail']}"
                try:
                    if self._core is not None:
                        lila_reply = self.process_interaction(failure["lila_prompt_context"])
                        if lila_reply.text:
                            lila_text = lila_reply.text
                except Exception:
                    pass
                return {
                    "session_id": session_id,
                    "transcript": "",
                    "response": lila_text,
                    "tool_results": [],
                    "tool_results_omitted": 0,
                    "failure": failure,
                    "status": self._voice.get_session_status(),
                }
            response = self.process_interaction(transcript)
            return {
                "session_id": session_id,
                "transcript": transcript,
                "response": response.text,
                "tool_results": response.tool_results,
                "tool_results_omitted": response.tool_results_omitted,
                "status": self._voice.get_session_status(),
            }

    def cancel_voice_session(self, session_id: str | None) -> dict:
        if self._voice is None or not self.voice_enabled:
            raise RuntimeError("Voice sessions are disabled.")
        result = self._voice.cancel_session(session_id)
        self.event_stream.publish("voice.state", {"state": "cancelled", "session_id": session_id})
        return result

    def voice_status(self) -> dict:
        if self._voice is None or not self.voice_enabled:
            return {"state": "disabled", "session_id": None, "error": None}
        return self._voice.get_session_status()

    def start_owner_voice_session(self, invocation_id: str) -> dict:
        """Start captured speech for one currently pending OIA decision."""
        owner = self.owner_authentication_service
        service = (getattr(self.lila, "approval_continuation_service", None)
                   if self.lila is not None else self.approval_continuation_service)
        if owner is None or service is None or not owner.has_owner():
            raise RuntimeError("Owner authentication is unavailable.")
        from src.daemon.approvals import ApprovalApi
        ApprovalApi(service).require_reviewable(invocation_id)
        if not any(
            provider.provider_id == "spoken_owner_phrase"
            and provider.available and provider.enrolled
            for provider in owner.providers()
        ):
            raise RuntimeError("Spoken owner authentication is not verified on this installation.")
        request = service.get_request(invocation_id)
        if self._voice is None or not self.voice_enabled:
            raise RuntimeError("Voice capture is disabled.")
        if self._oia_voice_session_id is not None:
            raise RuntimeError("Owner voice authentication is already active.")
        started = self._voice.start_private_session()
        self._oia_voice_session_id = started["session_id"]
        self._oia_voice_invocation_id = invocation_id
        return {
            "session_id": started["session_id"],
            "state": started["state"],
            "voice_challenge_code": f"{int.from_bytes(request.challenge.challenge_material[:4], 'big') % 1_000_000:06d}",
        }

    def stop_owner_voice_session(self, invocation_id: str, session_id: str) -> dict:
        owner = self.owner_authentication_service
        service = (getattr(self.lila, "approval_continuation_service", None)
                   if self.lila is not None else self.approval_continuation_service)
        if (owner is None or service is None or self._voice is None
                or session_id != self._oia_voice_session_id
                or invocation_id != self._oia_voice_invocation_id):
            raise RuntimeError("Owner voice session is unavailable.")
        transcript = ""
        try:
            transcript = self._voice.stop_private_session(session_id)
            from src.daemon.oia_harness import authenticate_spoken_decision

            result = authenticate_spoken_decision(
                owner, service, invocation_id, transcript
            )
            return {
                "invocation_id": result.invocation_id,
                "status": result.status.value,
                "reason_code": result.reason_code,
            }
        finally:
            transcript = ""
            self._oia_voice_session_id = None
            self._oia_voice_invocation_id = None

    def cancel_owner_voice_session(self, invocation_id: str, session_id: str) -> dict:
        if (self._voice is None or session_id != self._oia_voice_session_id
                or invocation_id != self._oia_voice_invocation_id):
            raise RuntimeError("Owner voice session is unavailable.")
        try:
            return self._voice.cancel_session(session_id)
        finally:
            self._oia_voice_session_id = None
            self._oia_voice_invocation_id = None

    def start_spoken_enrollment_session(self) -> dict:
        """Start private ASR capture to verify and enroll spoken_owner_phrase."""
        owner = self.owner_authentication_service
        if owner is None or not owner.has_owner():
            raise RuntimeError("Owner authentication is not enrolled locally.")
        if self._voice is None or not self.voice_enabled or not self._voice.private_asr_available:
            raise RuntimeError("Private ASR is unavailable for spoken owner enrollment.")
        if self._oia_spoken_enroll_session_id is not None:
            raise RuntimeError("Spoken owner enrollment session is already active.")
        started = self._voice.start_private_session()
        self._oia_spoken_enroll_session_id = started["session_id"]
        return {"session_id": started["session_id"], "state": started["state"]}

    def stop_spoken_enrollment_session(self, session_id: str) -> dict:
        owner = self.owner_authentication_service
        if (owner is None or self._voice is None
                or session_id != self._oia_spoken_enroll_session_id):
            raise RuntimeError("Spoken owner enrollment session is unavailable.")
        transcript = ""
        try:
            transcript = self._voice.stop_private_session(session_id)
            provider = owner.enroll_spoken_phrase(
                transcript, asr_available=self._voice.private_asr_available
            )
            return {
                "enrolled": provider.enrolled,
                "available": provider.available,
                "reason_code": provider.reason_code,
                "providers": [p.__dict__ for p in owner.providers()],
            }
        finally:
            transcript = ""
            self._oia_spoken_enroll_session_id = None

    def cancel_spoken_enrollment_session(self, session_id: str) -> dict:
        if self._voice is None or session_id != self._oia_spoken_enroll_session_id:
            raise RuntimeError("Spoken owner enrollment session is unavailable.")
        try:
            return self._voice.cancel_session(session_id)
        finally:
            self._oia_spoken_enroll_session_id = None
    
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
                    with open(schedule_file, encoding="utf-8") as f:
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
    
def _chat_payload(
    daemon: LilaDaemon,
    text: str,
    images: list[str] | None = None,
    image_paths: list[str] | None = None,
) -> dict:
    """Preserve the historical response string and add structured results."""
    task_center = getattr(daemon, "task_center", None)
    rec = task_center.create_task("chat", text[:80]) if task_center is not None else None
    if rec is not None:
        task_center.update_task(rec.task_id, status="running", progress=0.25)
    try:
        if images is None and image_paths is None:
            response = daemon.process_interaction(text)
        else:
            response = daemon.process_interaction(text, images=images, image_paths=image_paths)
        if rec is not None:
            task_center.update_task(
                rec.task_id,
                status="completed",
                progress=1.0,
                result={"tool_results_count": len(response.tool_results)},
            )
        payload = {
            "response": response.text,
            "tool_results": response.tool_results,
            "tool_results_omitted": response.tool_results_omitted,
        }
        if rec is not None:
            payload["task_id"] = rec.task_id
        return payload
    except Exception as exc:
        if rec is not None:
            task_center.update_task(rec.task_id, status="failed", error=type(exc).__name__)
        raise


def encode_json_utf8(payload: dict) -> bytes:
    """Serialize API payloads as unescaped Unicode encoded explicitly as UTF-8."""
    return json.dumps(payload, ensure_ascii=False).encode("utf-8", errors="strict")


def _make_api_handler(daemon: LilaDaemon, expected_token: str, *,
                      allowed_client_fingerprints: frozenset[str] | None = None):
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
            body = encode_json_utf8(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if allowed_client_fingerprints is not None:
                peer = b""
                try:
                    peer = self.connection.getpeercert(binary_form=True)
                    fingerprint = hashlib.sha256(peer).hexdigest()
                except Exception:
                    fingerprint = ""
                if (not peer or not any(hmac.compare_digest(fingerprint, expected)
                                        for expected in allowed_client_fingerprints)):
                    self._send_json(401, {"error": "Client certificate authentication required."})
                    return False
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
            from src.daemon.approvals import ApprovalApi, ApprovalApiErrorCode
            approval_api = ApprovalApi(
                getattr(daemon.lila, "approval_continuation_service", None)
                if daemon.lila is not None else daemon.approval_continuation_service
            )
            if self.path == "/approvals":
                try:
                    self._send_json(200, approval_api.list_pending().to_dict())
                except RuntimeError:
                    self._send_json(503, {"error": "Approval service is unavailable.", "code": ApprovalApiErrorCode.SERVICE_UNAVAILABLE.value})
                return
            approval_match = re.fullmatch(r"/approvals/([A-Za-z0-9][A-Za-z0-9._-]{0,127})(?:/(status))?", self.path)
            if approval_match:
                invocation_id, status_suffix = approval_match.groups()
                try:
                    result = (approval_api.get_status(invocation_id) if status_suffix
                              else approval_api.get_pending(invocation_id))
                    self._send_json(200, result.to_dict())
                except ValueError:
                    self._send_json(400, {"error": "Invalid approval identifier.", "code": ApprovalApiErrorCode.INVALID_IDENTIFIER.value})
                except LookupError:
                    self._send_json(404, {"error": "Approval request was not found.", "code": ApprovalApiErrorCode.NOT_FOUND.value})
                except RuntimeError:
                    self._send_json(503, {"error": "Approval service is unavailable.", "code": ApprovalApiErrorCode.SERVICE_UNAVAILABLE.value})
                return
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(self.path)
            path_only = parsed.path
            qs = parse_qs(parsed.query)
            if path_only == "/events":
                try:
                    since = int(qs.get("since", ["0"])[0])
                    limit = int(qs.get("limit", ["100"])[0])
                except ValueError:
                    self._send_json(400, {"error": "Invalid cursor or limit.", "code": "invalid_cursor"})
                    return
                self._send_json(200, daemon.event_stream.poll_since(since_cursor=since, limit=limit))
                return
            if path_only == "/events/stream":
                try:
                    raw_since = self.headers.get("Last-Event-ID") or qs.get("since", ["0"])[0]
                    cursor = max(0, int(raw_since))
                except ValueError:
                    self._send_json(400, {"error": "Invalid Last-Event-ID or since cursor.", "code": "invalid_cursor"})
                    return
                once_mode = qs.get("once", ["0"])[0] == "1"
                from src.daemon.events import StreamEvent, format_sse_event
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close" if once_mode else "keep-alive")
                self.end_headers()
                try:
                    while not daemon._shutdown_event.is_set():
                        batch = daemon.event_stream.wait_for_events(since_cursor=cursor, timeout=1.0 if once_mode else 5.0, limit=50)
                        if batch["events"]:
                            for ev_dict in batch["events"]:
                                ev = StreamEvent(
                                    event_id=ev_dict["event_id"],
                                    event_type=ev_dict["event_type"],
                                    payload=ev_dict["payload"],
                                    timestamp=ev_dict["timestamp"],
                                    schema_version=ev_dict["schema_version"],
                                )
                                self.wfile.write(format_sse_event(ev))
                            cursor = batch["next_cursor"]
                        else:
                            self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        if once_mode:
                            break
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                return
            if path_only == "/traces":
                try:
                    limit = max(1, min(int(qs.get("limit", ["50"])[0]), 200))
                except ValueError:
                    self._send_json(400, {"error": "Invalid limit.", "code": "invalid_limit"})
                    return
                trace_path = os.path.join(os.path.expanduser("~"), ".local", "share", "lila", "harness_traces.jsonl")
                records = []
                if os.path.exists(trace_path):
                    with open(trace_path, "r", encoding="utf-8", errors="replace") as tf:
                        lines = [ln.strip() for ln in tf if ln.strip()][-limit:]
                    for ln in reversed(lines):
                        try:
                            records.append(json.loads(ln))
                        except Exception:
                            pass
                self._send_json(200, {"traces": records})
                return
            if path_only == "/capabilities/candidates":
                self._send_json(200, {
                    "catalog_version": daemon.capability_learning.current_catalog.to_dict(),
                    "candidates": daemon.capability_learning.list_candidates(),
                })
                return
            if path_only == "/tasks":
                self._send_json(200, {"tasks": daemon.task_center.list_tasks()})
                return
            task_match = re.fullmatch(r"/tasks/([A-Za-z0-9][A-Za-z0-9._-]{0,127})", path_only)
            if task_match:
                task = daemon.task_center.get_task(task_match.group(1))
                if task is None:
                    self._send_json(404, {"error": "Task not found.", "code": "task_not_found"})
                else:
                    self._send_json(200, task)
                return
            if path_only == "/sentinel/triage":
                sources_raw = qs.get("sources", ["system_resources,defender,firewall"])[0]
                opted_in = [s.strip() for s in sources_raw.split(",") if s.strip()]
                report = daemon.sentinel_engine.run_triage(opted_in_sources=opted_in)
                self._send_json(200, report.to_dict())
                return
            if self.path not in {"/status", "/health", "/voice/status", "/owner/status"}:
                self._send_json(404, {"error": "Not found."})
                return
            if self.path == "/voice/status":
                self._send_json(200, daemon.voice_status())
            elif self.path == "/owner/status":
                owner = (getattr(daemon.lila, "owner_authentication_service", None)
                         if daemon.lila is not None else None) or daemon.owner_authentication_service
                self._send_json(200, {
                    "enrolled": bool(owner and owner.has_owner()),
                    "session_valid": bool(owner and owner.session_valid()),
                    "providers": [p.__dict__ for p in owner.providers()] if owner else [],
                })
            else:
                status = daemon.service_status()
                status["running"] = daemon._running
                status["voice"] = daemon.voice_enabled
                self._send_json(200, status)

        def do_POST(self):
            if not self._authorized():
                return
            approval_match = re.fullmatch(r"/approvals/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/decision", self.path)
            owner_decision_match = re.fullmatch(
                r"/owner/approvals/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/decision", self.path
            )
            owner_voice_match = re.fullmatch(
                r"/owner/approvals/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/voice/(start|stop|cancel)", self.path
            )
            owner_spoken_enroll_match = re.fullmatch(
                r"/owner/spoken/(start|stop|cancel)", self.path
            )
            task_cancel_match = re.fullmatch(r"/tasks/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/cancel", self.path)
            cap_promote_match = re.fullmatch(r"/capabilities/candidates/([A-Za-z0-9][A-Za-z0-9._-]{0,127})/promote", self.path)
            if self.path not in {
                "/chat", "/voice/start", "/voice/stop", "/voice/cancel",
                "/sentinel/triage",
                "/perception/gesture", "/perception/capture", "/perception/observe",
                "/perception/ocr", "/perception/pause", "/perception/resume",
                "/perception/discard", "/perception/stop",
                "/capabilities/rollback",
            } and approval_match is None and owner_decision_match is None and owner_voice_match is None and owner_spoken_enroll_match is None and task_cancel_match is None and cap_promote_match is None:
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
            if length > (4096 if owner_decision_match or owner_voice_match else self.MAX_BODY_BYTES):
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
            raw_body = bytearray(self.rfile.read(length))
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
            if owner_decision_match:
                phrase = ""
                try:
                    if set(payload) != {"phrase", "decision"}:
                        raise ValueError("owner_authentication.request_invalid")
                    phrase_value = payload["phrase"]
                    decision_text = payload["decision"]
                    if not isinstance(phrase_value, str) or not 1 <= len(phrase_value) <= 512:
                        raise ValueError("owner_authentication.request_invalid")
                    if decision_text not in ("approve", "reject"):
                        raise ValueError("owner_authentication.request_invalid")
                    owner = (getattr(daemon.lila, "owner_authentication_service", None)
                             if daemon.lila is not None else None) or daemon.owner_authentication_service
                    service = (getattr(daemon.lila, "approval_continuation_service", None)
                               if daemon.lila is not None else daemon.approval_continuation_service)
                    if owner is None or service is None:
                        self._send_json(503, {"code": "owner_authentication.unavailable"})
                    else:
                        from src.daemon.approvals import ApprovalApi
                        ApprovalApi(service).require_reviewable(owner_decision_match.group(1))
                        from src.daemon.oia_harness import authenticate_typed_decision
                        phrase = phrase_value
                        result = authenticate_typed_decision(
                            owner,
                            service,
                            owner_decision_match.group(1),
                            phrase,
                            decision_text,
                        )
                        self._send_json(200, {
                            "invocation_id": result.invocation_id,
                            "status": result.status.value,
                            "reason_code": result.reason_code,
                        })
                    return
                except Exception as exc:
                    code = getattr(exc, "reason_code", "owner_authentication.failed")
                    self._send_json(403, {"code": code})
                    return
                finally:
                    phrase = ""
                    payload["phrase"] = ""
                    payload.clear()
                    for index in range(len(raw_body)):
                        raw_body[index] = 0
            if owner_voice_match:
                try:
                    invocation_id, action = owner_voice_match.groups()
                    if action == "start":
                        if payload:
                            raise ValueError("owner_authentication.request_invalid")
                        result = daemon.start_owner_voice_session(invocation_id)
                    else:
                        if set(payload) != {"session_id"} or not isinstance(payload["session_id"], str):
                            raise ValueError("owner_authentication.request_invalid")
                        if action == "stop":
                            result = daemon.stop_owner_voice_session(invocation_id, payload["session_id"])
                        else:
                            result = daemon.cancel_owner_voice_session(invocation_id, payload["session_id"])
                    self._send_json(200, result)
                except Exception as exc:
                    code = getattr(exc, "reason_code", "owner_authentication.voice_unavailable")
                    self._send_json(503, {"code": code})
                finally:
                    payload.clear()
                    for index in range(len(raw_body)):
                        raw_body[index] = 0
                return
            if owner_spoken_enroll_match:
                try:
                    action = owner_spoken_enroll_match.group(1)
                    if action == "start":
                        if payload:
                            raise ValueError("owner_authentication.request_invalid")
                        result = daemon.start_spoken_enrollment_session()
                    else:
                        if set(payload) != {"session_id"} or not isinstance(payload["session_id"], str):
                            raise ValueError("owner_authentication.request_invalid")
                        if action == "stop":
                            result = daemon.stop_spoken_enrollment_session(payload["session_id"])
                        else:
                            result = daemon.cancel_spoken_enrollment_session(payload["session_id"])
                    self._send_json(200, result)
                except Exception as exc:
                    code = getattr(exc, "reason_code", "owner_authentication.spoken_enrollment_failed")
                    status = 403 if hasattr(exc, "reason_code") else 503
                    self._send_json(status, {"code": code, "error": str(exc)})
                finally:
                    payload.clear()
                    for index in range(len(raw_body)):
                        raw_body[index] = 0
                return
            if approval_match:
                from src.daemon.approvals import ApprovalApi, ApprovalApiErrorCode, ApprovalDecisionRequest
                try:
                    decision = ApprovalDecisionRequest.parse(approval_match.group(1), payload)
                except ValueError as exc:
                    code = str(exc)
                    status = 403 if code == ApprovalApiErrorCode.AUTHENTICATION_REQUIRED.value else 400
                    message = "A separate owner decision proof is required." if status == 403 else "Invalid approval decision request."
                    self._send_json(status, {"error": message, "code": code})
                    return
                service = (getattr(daemon.lila, "approval_continuation_service", None)
                           if daemon.lila is not None else daemon.approval_continuation_service)
                try:
                    result = ApprovalApi(service).decide(decision).to_dict()
                except RuntimeError:
                    self._send_json(503, {"error": "Approval service is unavailable.", "code": ApprovalApiErrorCode.SERVICE_UNAVAILABLE.value})
                    return
                self._send_json(200, result)
                return
            if task_cancel_match:
                task_id = task_cancel_match.group(1)
                reason = str(payload.get("reason") or "owner_cancelled")
                try:
                    rec = daemon.task_center.cancel_task(task_id, reason=reason)
                    self._send_json(200, rec.to_dict())
                except KeyError:
                    self._send_json(404, {"error": "Task not found.", "code": "task_not_found"})
                return
            if self.path == "/sentinel/triage":
                from src.sentinel.contracts import TelemetryFact
                from src.sentinel.triage import collect_local_readonly_facts
                sources = payload.get("sources") or ["system_resources", "defender", "firewall"]
                raw_host_facts = payload.get("host_facts") or []
                if not isinstance(sources, list) or not isinstance(raw_host_facts, list):
                    self._send_json(400, {"error": "Invalid Sentinel payload.", "code": "invalid_sentinel_payload"})
                    return
                validated_host_facts = []
                available_host_sources = set()
                try:
                    for hf in raw_host_facts:
                        if not isinstance(hf, dict) or hf.get("read_only") is not True:
                            raise ValueError("Host facts must strictly set read_only=true.")
                        src_name = str(hf.get("source") or "")
                        metric = str(hf.get("metric") or "")
                        if src_name not in ("defender", "firewall", "windows_security_events") or not metric:
                            raise ValueError("Invalid host fact source or metric.")
                        validated_host_facts.append(TelemetryFact(source=src_name, metric=metric, value=hf.get("value"), read_only=True))
                        available_host_sources.add(src_name)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc), "code": "invalid_sentinel_fact"})
                    return

                def _combined_collector(allowed_sources):
                    wsl_sources = [s for s in allowed_sources if s not in available_host_sources]
                    facts, avail, gaps = collect_local_readonly_facts(wsl_sources)
                    facts.extend(validated_host_facts)
                    avail.extend(sorted(available_host_sources))
                    return facts, avail, gaps

                report = daemon.sentinel_engine.run_triage(
                    opted_in_sources=[str(s) for s in sources],
                    collector_fn=_combined_collector,
                )
                self._send_json(200, report.to_dict())
                return
            if self.path.startswith("/perception/"):
                import base64
                from src.harness.visual_perception import CaptureScope
                try:
                    if self.path == "/perception/gesture":
                        scope = CaptureScope(str(payload.get("scope") or "region"))
                        tok = daemon.visual_perception.issue_owner_gesture_token(scope)
                        self._send_json(200, {"owner_gesture_token": tok, "scope": scope.value})
                    elif self.path == "/perception/capture":
                        scope = CaptureScope(str(payload.get("scope") or "region"))
                        img_b64 = str(payload.get("image_base64") or "")
                        obs = daemon.visual_perception.capture_explicit(
                            initiator=str(payload.get("initiator") or ""),
                            owner_gesture_id=payload.get("owner_gesture_token"),
                            scope=scope,
                            target_label=str(payload.get("target_label") or "Selection"),
                            image_bytes=base64.b64decode(img_b64) if img_b64 else b"",
                            width=int(payload.get("width") or 1),
                            height=int(payload.get("height") or 1),
                        )
                        self._send_json(200, obs.to_public_dict())
                    elif self.path == "/perception/observe":
                        res = daemon.visual_perception.observe_vision_metadata(str(payload.get("capture_id") or ""))
                        self._send_json(200, res)
                    elif self.path == "/perception/ocr":
                        res = daemon.visual_perception.extract_ocr_uia_text(
                            str(payload.get("capture_id") or ""),
                            uia_text=str(payload.get("uia_text") or ""),
                        )
                        self._send_json(200, res)
                    elif self.path == "/perception/pause":
                        daemon.visual_perception.pause()
                        self._send_json(200, {"paused": True})
                    elif self.path == "/perception/resume":
                        daemon.visual_perception.resume()
                        self._send_json(200, {"paused": False})
                    elif self.path == "/perception/discard":
                        ok = daemon.visual_perception.discard(str(payload.get("capture_id") or ""))
                        self._send_json(200, {"discarded": ok})
                    elif self.path == "/perception/stop":
                        cnt = daemon.visual_perception.stop_all()
                        self._send_json(200, {"stopped": cnt})
                except PermissionError as exc:
                    self._send_json(403, {"error": str(exc), "code": "perception_denied"})
                except (ValueError, KeyError, LookupError) as exc:
                    self._send_json(400, {"error": str(exc), "code": "invalid_perception_request"})
                return
            if cap_promote_match:
                cid = cap_promote_match.group(1)
                owner_approved = payload.get("owner_approved") is True
                try:
                    ver = daemon.capability_learning.promote(cid, owner_approved=owner_approved)
                    self._send_json(200, ver.to_dict())
                except PermissionError as exc:
                    self._send_json(403, {"error": str(exc), "code": "owner_approval_required"})
                except (KeyError, RuntimeError) as exc:
                    self._send_json(400, {"error": str(exc), "code": "promotion_failed"})
                return
            if self.path == "/capabilities/rollback":
                target_v = payload.get("target_version")
                try:
                    ver = daemon.capability_learning.rollback(int(target_v) if target_v is not None else None)
                    self._send_json(200, ver.to_dict())
                except (KeyError, ValueError) as exc:
                    self._send_json(400, {"error": str(exc), "code": "rollback_failed"})
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
                images = payload.get("images")
                image_paths = payload.get("image_paths")
                if images is not None and (not isinstance(images, list) or not all(isinstance(x, str) for x in images)):
                    self._send_json(400, {
                        "error": "Field 'images' must be a list of strings.",
                        "code": "invalid_field_value",
                    })
                    return
                if image_paths is not None and (not isinstance(image_paths, list) or not all(isinstance(x, str) for x in image_paths)):
                    self._send_json(400, {
                        "error": "Field 'image_paths' must be a list of strings.",
                        "code": "invalid_field_value",
                    })
                    return
                operation = lambda: _chat_payload(daemon, payload["text"], images=images, image_paths=image_paths)
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
                from src.harness.verification import format_failure_for_lila
                failure = format_failure_for_lila("vtalks.voice", code, message)
                daemon.event_stream.publish("voice.state", {"state": "error", "code": code, "message": message})
                self._send_json(502, {"error": message, "code": code, "failure": failure})
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
    configure_standard_streams()
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
