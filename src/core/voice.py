"""Push-to-talk speech input for Lila using an external ASR backend."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from src.core.private_storage import (
    open_private_text,
    restrictive_umask,
    secure_private_directory,
    secure_private_file,
)


class VoiceInputError(RuntimeError):
    """Base exception for microphone and speech-recognition failures."""


class MicrophoneError(VoiceInputError):
    """Raised when audio capture cannot produce a usable recording."""


class AudioCaptureTimeoutError(MicrophoneError):
    """Raised when a recording reaches its configured duration limit."""


class TranscriptionError(VoiceInputError):
    """Raised when the configured ASR backend cannot return a transcript."""


class VoiceSessionError(VoiceInputError):
    """Raised when a remote voice session cannot be started or completed."""


class VoiceSessionBusyError(VoiceSessionError):
    """Raised when a voice session is already active."""


class VoiceSessionCancelled(VoiceSessionError):
    """Raised when a voice session is cancelled before transcript delivery."""


class Transcriber(Protocol):
    """Replaceable interface implemented by speech-recognition backends."""

    def transcribe(self, audio_path: Path) -> str:
        """Return the backend's original transcript without rewriting it."""


def _voice_experiment_root() -> Path:
    configured = os.environ.get("LILA_VOICE_HOME")
    if configured:
        return Path(configured).expanduser()
    project_root = Path(__file__).resolve().parents[2]
    portable_root = project_root / "experiments" / "voice"
    if portable_root.exists():
        return portable_root
    # Continue using the pre-portability checkout's external voice assets.
    legacy_root = Path(__file__).resolve().parents[3] / "experiments" / "voice"
    if (legacy_root / "whisper.cpp" / "models").is_dir():
        return legacy_root
    return portable_root


def _voice_data_root() -> Path:
    configured = os.environ.get("LILA_RECORDINGS_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home).expanduser() if xdg_data_home else Path.home() / ".local" / "share"
    return base / "lila" / "recordings"


def _voice_asset_path(environment_name: str, *parts: str) -> Path:
    configured = os.environ.get(environment_name)
    return Path(configured).expanduser() if configured else _voice_experiment_root().joinpath(*parts)


@dataclass
class VoiceConfig:
    """Voice runtime settings; the unused voice behavior fields are reserved."""

    # Reserved for a future wake-word feature; current input is push-to-talk.
    wake_words: list[str] = field(
        default_factory=lambda: ["lila", "hey lila", "lila,"]
    )
    # Reserved for future TTS; speak() currently prints text to the console.
    tts_model: str = "default"
    # Reserved model label; the active backend loads the configured asr_model path.
    stt_model: str = "parakeet-tdt-0.6b-v3-q4_k"
    # Reserved for future TTS voice selection; no speech synthesis is active.
    voice_style: str = "warm"
    capture_device: str | None = field(
        default_factory=lambda: os.environ.get("LILA_VOICE_CAPTURE_DEVICE") or None
    )
    capture_command: str = field(
        default_factory=lambda: os.environ.get("LILA_VOICE_CAPTURE_COMMAND", "parecord")
    )
    sample_rate: int = 16_000
    max_recording_seconds: int = 120
    transcription_timeout_seconds: int = 180
    transcription_threads: int = 2
    recordings_dir: Path = field(
        default_factory=_voice_data_root
    )
    asr_executable: Path = field(
        default_factory=lambda: _voice_asset_path(
            "LILA_VOICE_ASR_EXECUTABLE", "whisper.cpp", "build", "bin", "parakeet-cli"
        )
    )
    asr_model: Path = field(
        default_factory=lambda: _voice_asset_path(
            "LILA_VOICE_ASR_MODEL", "whisper.cpp", "models",
            "ggml-parakeet-tdt-0.6b-v3-q4_k.bin"
        )
    )

    def __post_init__(self) -> None:
        if self.sample_rate != 16_000:
            raise ValueError("Voice capture must use a 16 kHz sample rate.")
        if self.max_recording_seconds <= 0:
            raise ValueError("Maximum recording duration must be positive.")
        if self.transcription_timeout_seconds <= 0:
            raise ValueError("Transcription timeout must be positive.")
        if self.transcription_threads <= 0:
            raise ValueError("Transcription thread count must be positive.")


class ParakeetTranscriber:
    """Run the existing CPU-capable Parakeet CLI without loading another model."""

    def __init__(
        self,
        executable: Path,
        model: Path,
        timeout_seconds: int = 180,
        threads: int = 2,
    ) -> None:
        self.executable = Path(executable)
        self.model = Path(model)
        self.timeout_seconds = timeout_seconds
        self.threads = threads
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen | None = None
        self._cancel_generation = 0

    def transcribe(self, audio_path: Path) -> str:
        audio_path = Path(audio_path)
        for label, path in (
            ("Parakeet executable", self.executable),
            ("Parakeet model", self.model),
            ("Audio recording", audio_path),
        ):
            if not path.is_file():
                raise TranscriptionError(f"{label} not found: {path}")

        command = [
            str(self.executable),
            "--no-gpu",
            "--threads",
            str(self.threads),
            "--model",
            str(self.model),
            "--no-prints",
            str(audio_path),
        ]
        with self._process_lock:
            generation = self._cancel_generation
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            raise TranscriptionError(
                f"Unable to start Parakeet ({type(exc).__name__})."
            ) from exc
        with self._process_lock:
            self._active_process = process
            # Cancellation can race process creation. A generation counter
            # ensures a request made just before registration still kills it.
            if generation != self._cancel_generation and process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.communicate()
            raise TranscriptionError(
                f"Parakeet timed out after {self.timeout_seconds} seconds."
            ) from exc
        finally:
            with self._process_lock:
                cancelled = generation != self._cancel_generation
                if self._active_process is process:
                    self._active_process = None

        if cancelled:
            raise VoiceSessionCancelled("Voice transcription was cancelled.")
        if process.returncode != 0:
            detail = stderr.strip()[:500]
            suffix = f" {detail}" if detail else ""
            raise TranscriptionError(
                f"Parakeet exited with status {process.returncode}.{suffix}"
            )
        if not stdout.strip():
            raise TranscriptionError("Parakeet returned an empty transcript.")
        return stdout

    def cancel_active(self) -> None:
        """Terminate the current ASR process when a session is cancelled."""
        with self._process_lock:
            self._cancel_generation += 1
            process = self._active_process
            if process is None:
                return
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass


class LilaVoice:
    """Capture push-to-talk utterances and deliver raw transcripts to a callback."""

    def __init__(
        self,
        config: VoiceConfig | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self.config = config or VoiceConfig()
        self._transcriber = transcriber or ParakeetTranscriber(
            executable=self.config.asr_executable,
            model=self.config.asr_model,
            timeout_seconds=self.config.transcription_timeout_seconds,
            threads=self.config.transcription_threads,
        )
        self._listening = False
        self._on_input: Callable[[str], None] | None = None
        self._capture_process: subprocess.Popen | None = None
        self._capture_lock = threading.Lock()
        self._session_lock = threading.RLock()
        self._session_state = "idle"
        self._session_id: str | None = None
        self._session_started_at: float | None = None
        self._session_audio_path: Path | None = None
        self._session_timer: threading.Timer | None = None
        self._session_error: str | None = None
        self._session_cancel_requested = False

    def speak(self, text: str) -> None:
        """Print the response; speech synthesis is not implemented."""
        print(f"🌸 Lila: {text}")

    def start_listening(self, on_input: Callable[[str], None]) -> None:
        """Run a push-to-talk loop and pass each raw transcript to ``on_input``."""
        self._on_input = on_input
        self._listening = True
        print("🌸 Push-to-talk ready. Press Enter to record; type q to quit.")
        try:
            while self._listening:
                try:
                    command = input("Voice input (Enter to record, q to quit): ")
                except EOFError:
                    self.stop_listening()
                    break
                except KeyboardInterrupt:
                    self.stop_listening()
                    break

                if command.strip().casefold() in {"q", "quit", "exit"}:
                    self.stop_listening()
                    break
                if command.strip():
                    print("Press Enter to record, or type q to quit.")
                    continue

                transcript = self.listen_once()
                if not self._listening:
                    break
                if self._on_input is not None:
                    self._on_input(transcript)
        except Exception:
            self.stop_listening()
            raise

    def listen_once(self) -> str:
        """Capture one utterance, save its audio and raw transcript, then return it."""
        audio_path = self._capture_audio()
        transcript = self._transcriber.transcribe(audio_path)
        if not transcript.strip():
            raise TranscriptionError("Speech recognition returned no speech.")

        transcript_path = audio_path.with_name(
            f"{audio_path.stem}.transcript.txt"
        )
        try:
            with open_private_text(transcript_path, "w") as stream:
                stream.write(transcript)
        except OSError as exc:
            raise VoiceInputError(
                f"Unable to preserve transcript at {transcript_path}."
            ) from exc
        return transcript

    def start_session(self) -> dict:
        """Start one API-controlled recording without reading terminal input."""
        with self._session_lock:
            if self._session_state in {"recording", "transcribing", "cancelling", "timed_out"}:
                raise VoiceSessionBusyError("A voice session is already active.")

            recordings_dir = Path(self.config.recordings_dir)
            try:
                secure_private_directory(recordings_dir)
            except OSError as exc:
                raise MicrophoneError(
                    f"Unable to create voice recordings directory: {recordings_dir}"
                ) from exc

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            audio_path = recordings_dir / f"lila-{timestamp}-{uuid.uuid4().hex[:8]}.wav"
            command = self._capture_command(audio_path)
            try:
                with restrictive_umask():
                    process = subprocess.Popen(command)
            except OSError as exc:
                self._session_state = "failed"
                self._session_error = f"Unable to start microphone capture ({type(exc).__name__})."
                raise MicrophoneError(self._session_error) from exc

            time.sleep(0.1)
            if process.poll() is not None:
                process.wait()
                if audio_path.exists():
                    secure_private_file(audio_path)
                self._session_state = "failed"
                self._session_error = (
                    f"Microphone capture exited before recording (status {process.returncode})."
                )
                raise MicrophoneError(self._session_error)

            session_id = uuid.uuid4().hex
            with self._capture_lock:
                self._capture_process = process
            self._session_id = session_id
            self._session_started_at = time.monotonic()
            self._session_audio_path = audio_path
            self._session_state = "recording"
            self._session_error = None
            self._session_cancel_requested = False
            timer = threading.Timer(
                self.config.max_recording_seconds,
                self._expire_session,
                args=(session_id,),
            )
            timer.daemon = True
            self._session_timer = timer
            timer.start()
            return self.get_session_status()

    def stop_session(self, session_id: str | None = None) -> str:
        """Stop capture, run Parakeet, persist its raw output and return it."""
        with self._session_lock:
            self._require_session(session_id)
            if self._session_state == "failed" and self._session_error:
                raise VoiceSessionError(self._session_error)
            if self._session_state != "recording":
                raise VoiceSessionBusyError(
                    f"Voice session cannot be stopped while {self._session_state}."
                )
            self._session_state = "transcribing"
            timer = self._session_timer
            process = self._capture_process
            audio_path = self._session_audio_path
            active_id = self._session_id
            if timer is not None:
                timer.cancel()

        assert process is not None and audio_path is not None and active_id is not None
        try:
            self._stop_capture_process(process)
            secure_private_file(audio_path)
            self._validate_audio_file(audio_path)
            transcript = self._transcriber.transcribe(audio_path)
            if self._cancel_was_requested(active_id):
                raise VoiceSessionCancelled("Voice session was cancelled.")
            if not transcript.strip():
                raise TranscriptionError("Speech recognition returned no speech.")
            transcript_path = audio_path.with_name(
                f"{audio_path.stem}.transcript.txt"
            )
            self._persist_session_transcript(active_id, transcript, transcript_path)
            return transcript
        except Exception as exc:
            with self._session_lock:
                if self._session_id == active_id:
                    if self._session_cancel_requested:
                        self._session_state = "cancelled"
                        self._session_error = None
                    else:
                        self._session_state = "failed"
                        self._session_error = str(exc)[:300]
            raise
        finally:
            with self._capture_lock:
                if self._capture_process is process:
                    self._capture_process = None

    def _persist_session_transcript(
        self, session_id: str, transcript: str, transcript_path: Path
    ) -> None:
        """Write privately, then atomically publish only if cancellation has not won."""
        temporary_path = transcript_path.with_name(
            f"{transcript_path.name}.tmp-{uuid.uuid4().hex}"
        )
        try:
            with open_private_text(temporary_path, "w") as stream:
                stream.write(transcript)

            # The final check and atomic rename share the session lock with
            # cancel_session. Cancellation can proceed while the private
            # temporary file is written, but cannot slip between this check
            # and publication. The lock is held only for the local rename and
            # state update, not for ASR or file-content I/O.
            with self._session_lock:
                if self._session_id != session_id or self._session_cancel_requested:
                    raise VoiceSessionCancelled("Voice session was cancelled.")
                os.replace(temporary_path, transcript_path)
                self._session_state = "completed"
                self._session_error = None
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def cancel_session(self, session_id: str | None = None) -> dict:
        """Cancel recording or an in-flight ASR call without delivering text."""
        with self._session_lock:
            self._require_session(session_id)
            if self._session_state not in {"recording", "transcribing", "timed_out"}:
                raise VoiceSessionBusyError(
                    f"Voice session cannot be cancelled while {self._session_state}."
                )
            previous_state = self._session_state
            active_id = self._session_id
            process = self._capture_process
            timer = self._session_timer
            self._session_cancel_requested = True
            self._session_state = "cancelling"
            if timer is not None:
                timer.cancel()

        try:
            if previous_state in {"recording", "timed_out"} and process is not None:
                self._stop_capture_process(process)
                audio_path = self._session_audio_path
                if audio_path is not None and audio_path.is_file():
                    secure_private_file(audio_path)
            cancel_asr = getattr(self._transcriber, "cancel_active", None)
            if previous_state == "transcribing" and callable(cancel_asr):
                cancel_asr()
        except Exception as exc:
            with self._session_lock:
                self._session_state = "failed"
                self._session_error = f"Cancellation failed ({type(exc).__name__})."
            raise VoiceSessionError(self._session_error) from exc

        with self._session_lock:
            if self._session_id == active_id and previous_state != "transcribing":
                self._session_state = "cancelled"
                self._session_error = None
            status = self.get_session_status()
        if process is not None and previous_state != "transcribing":
            with self._capture_lock:
                if self._capture_process is process:
                    self._capture_process = None
        return status

    def get_session_status(self) -> dict:
        """Return bounded state metadata without exposing private transcript text."""
        with self._session_lock:
            elapsed = None
            if self._session_started_at is not None and self._session_state in {
                "recording", "transcribing", "cancelling", "timed_out"
            }:
                elapsed = round(time.monotonic() - self._session_started_at, 2)
            return {
                "state": self._session_state,
                "session_id": self._session_id,
                "elapsed_seconds": elapsed,
                "error": self._session_error,
            }

    def _require_session(self, session_id: str | None) -> None:
        if self._session_id is None or (
            session_id is not None and session_id != self._session_id
        ):
            raise VoiceSessionError("Voice session ID is missing or no longer current.")

    def _cancel_was_requested(self, session_id: str) -> bool:
        with self._session_lock:
            return self._session_id != session_id or self._session_cancel_requested

    def _expire_session(self, session_id: str) -> None:
        with self._session_lock:
            if self._session_id != session_id or self._session_state != "recording":
                return
            self._session_state = "timed_out"
            self._session_error = (
                f"Recording reached {self.config.max_recording_seconds} seconds."
            )
            process = self._capture_process
        if process is not None:
            self._stop_capture_process(process)
        audio_path = self._session_audio_path
        if audio_path is not None and audio_path.is_file():
            try:
                secure_private_file(audio_path)
            except OSError:
                pass
        with self._session_lock:
            if self._session_id == session_id and self._session_state == "timed_out":
                self._session_state = "failed"

    def _validate_audio_file(self, audio_path: Path) -> None:
        try:
            with wave.open(str(audio_path), "rb") as recording:
                valid_audio = (
                    recording.getnchannels() == 1
                    and recording.getframerate() == self.config.sample_rate
                    and recording.getnframes() > 0
                )
        except (OSError, EOFError, wave.Error) as exc:
            raise MicrophoneError(
                f"Microphone produced an invalid WAV recording: {audio_path}"
            ) from exc
        if not valid_audio:
            raise MicrophoneError(
                f"Microphone recording is empty or has an invalid format: {audio_path}"
            )

    def _capture_audio(self) -> Path:
        recordings_dir = Path(self.config.recordings_dir)
        try:
            secure_private_directory(recordings_dir)
        except OSError as exc:
            raise MicrophoneError(
                f"Unable to create voice recordings directory: {recordings_dir}"
            ) from exc

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        audio_path = recordings_dir / f"lila-{timestamp}-{uuid.uuid4().hex[:8]}.wav"
        command = self._capture_command(audio_path)
        try:
            # parecord creates the WAV, so give the child a restrictive umask.
            with restrictive_umask():
                process = subprocess.Popen(command)
        except OSError as exc:
            raise MicrophoneError(
                f"Unable to start microphone capture ({type(exc).__name__})."
            ) from exc
        with self._capture_lock:
            self._capture_process = process
        time.sleep(0.1)
        if process.poll() is not None:
            process.wait()
            self._clear_capture_process(process)
            raise MicrophoneError(
                f"Microphone capture exited before recording "
                f"(status {process.returncode})."
            )

        duration_reached = threading.Event()

        def stop_at_duration_limit() -> None:
            duration_reached.set()
            try:
                process.send_signal(signal.SIGINT)
            except OSError:
                pass

        timer = threading.Timer(
            self.config.max_recording_seconds, stop_at_duration_limit
        )
        timer.daemon = True
        timer.start()
        try:
            input("Recording. Speak now, then press Enter to finish: ")
        except (EOFError, KeyboardInterrupt) as exc:
            raise MicrophoneError("Audio capture was cancelled.") from exc
        finally:
            timer.cancel()
            self._stop_capture_process(process)
            self._clear_capture_process(process)

        try:
            secure_private_file(audio_path)
        except (OSError, PermissionError) as exc:
            raise MicrophoneError(
                f"Unable to secure captured audio at {audio_path}."
            ) from exc

        if duration_reached.is_set():
            raise AudioCaptureTimeoutError(
                f"Recording reached {self.config.max_recording_seconds} seconds; "
                f"audio retained at {audio_path}."
            )

        if process.returncode not in (0, -signal.SIGINT):
            raise MicrophoneError(
                f"Microphone capture exited with status {process.returncode}."
            )
        try:
            with wave.open(str(audio_path), "rb") as recording:
                valid_audio = (
                    recording.getnchannels() == 1
                    and recording.getframerate() == self.config.sample_rate
                    and recording.getnframes() > 0
                )
        except (OSError, EOFError, wave.Error) as exc:
            raise MicrophoneError(
                f"Microphone produced an invalid WAV recording: {audio_path}"
            ) from exc
        if not valid_audio:
            raise MicrophoneError(
                f"Microphone recording is empty or has an invalid format: "
                f"{audio_path}"
            )
        return audio_path

    def _capture_command(self, audio_path: Path) -> list[str]:
        command = [self.config.capture_command]
        if self.config.capture_device:
            command.append(f"--device={self.config.capture_device}")
        command.extend([
            f"--rate={self.config.sample_rate}",
            "--channels=1",
            "--format=s16le",
            "--file-format=wav",
            str(audio_path),
        ])
        return command

    @staticmethod
    def _stop_capture_process(process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def stop_listening(self) -> None:
        self._listening = False
        with self._capture_lock:
            process = self._capture_process
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except OSError:
                pass

    def _clear_capture_process(self, process: subprocess.Popen) -> None:
        with self._capture_lock:
            if self._capture_process is process:
                self._capture_process = None

    @property
    def is_listening(self) -> bool:
        return self._listening
