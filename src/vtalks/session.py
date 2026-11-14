"""vTalks push-to-talk and private voice-session orchestration."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
import uuid
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .asr import ParakeetTranscriber
from .capture import AudioCapture
from .contracts import (
    AudioCaptureTimeoutError, MicrophoneError, Transcriber, TranscriptionError,
    VoiceConfig, VoiceInputError, VoiceSessionBusyError, VoiceSessionCancelled,
    VoiceSessionError,
)
from .storage import (
    open_private_text, restrictive_umask, secure_private_directory,
    secure_private_file,
)


class _LiveVoiceStream:
    """Tee raw 16 kHz s16le PCM to a WAV file and run chunked live ASR."""

    _FRAME_SECONDS = 0.1
    _PRE_ROLL_FRAMES = 3

    def __init__(
        self,
        *,
        process: subprocess.Popen,
        audio_path: Path,
        transcriber: object,
        sample_rate: int,
        chunk_seconds: float,
        silence_seconds: float,
        rms_threshold: float,
        on_transcript: Callable[[str, str, bool], None] | None,
    ) -> None:
        self._process = process
        self._audio_path = audio_path
        self._transcriber = transcriber
        self._sample_rate = sample_rate
        self._chunk_frames = max(5, int(round(chunk_seconds / self._FRAME_SECONDS)))
        self._silence_frames_limit = max(2, int(round(silence_seconds / self._FRAME_SECONDS)))
        self._rms_threshold = rms_threshold
        self._on_transcript = on_transcript
        self._frame_bytes = int(sample_rate * self._FRAME_SECONDS) * 2
        self._asr_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._cancelled = threading.Event()
        self._segments: list[str] = []
        self._segments_lock = threading.Lock()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="vtalks-pcm-reader", daemon=True
        )
        self._asr_thread = threading.Thread(
            target=self._asr_loop, name="vtalks-live-asr", daemon=True
        )

    def start(self) -> None:
        self._reader_thread.start()
        self._asr_thread.start()

    def _reader_loop(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            self._asr_queue.put(None)
            return

        pre_roll: deque[np.ndarray] = deque(maxlen=self._PRE_ROLL_FRAMES)
        speech_frames: list[np.ndarray] = []
        in_speech = False
        silence_frames = 0

        try:
            with restrictive_umask(), wave.open(str(self._audio_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self._sample_rate)
                while not self._cancelled.is_set():
                    raw = stdout.read(self._frame_bytes)
                    if not raw:
                        break
                    even_len = len(raw) - (len(raw) % 2)
                    if even_len <= 0:
                        continue
                    pcm_bytes = raw[:even_len]
                    wav.writeframes(pcm_bytes)
                    samples = (
                        np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    rms = float(np.sqrt(np.mean(samples * samples)))
                    if rms >= self._rms_threshold:
                        if not in_speech:
                            speech_frames.extend(pre_roll)
                            pre_roll.clear()
                            in_speech = True
                        speech_frames.append(samples)
                        silence_frames = 0
                    elif in_speech:
                        speech_frames.append(samples)
                        silence_frames += 1
                        if silence_frames >= self._silence_frames_limit:
                            self._asr_queue.put(np.concatenate(speech_frames))
                            speech_frames.clear()
                            in_speech = False
                            silence_frames = 0
                    else:
                        pre_roll.append(samples)

                    if in_speech and len(speech_frames) >= self._chunk_frames:
                        self._asr_queue.put(np.concatenate(speech_frames))
                        speech_frames.clear()
                        silence_frames = 0
        finally:
            if self._audio_path.exists():
                secure_private_file(self._audio_path)
            if speech_frames and not self._cancelled.is_set():
                tail = np.concatenate(speech_frames)
                if len(tail) >= int(self._sample_rate * 0.2):
                    self._asr_queue.put(tail)
            self._asr_queue.put(None)

    def _asr_loop(self) -> None:
        transcribe_pcm = getattr(self._transcriber, "transcribe_pcm")
        while True:
            chunk = self._asr_queue.get()
            if chunk is None or self._cancelled.is_set():
                break
            try:
                text = transcribe_pcm(chunk).strip()
            except TranscriptionError:
                continue
            if not text or self._cancelled.is_set():
                continue
            with self._segments_lock:
                self._segments.append(text)
                accumulated = " ".join(self._segments)
            if self._on_transcript is not None:
                try:
                    self._on_transcript(text, accumulated, False)
                except Exception:
                    pass

    def finish(self, timeout_seconds: float = 30.0) -> str:
        self._reader_thread.join(timeout=5.0)
        self._asr_thread.join(timeout=timeout_seconds)
        with self._segments_lock:
            return " ".join(self._segments).strip()

    def cancel(self) -> None:
        self._cancelled.set()
        while not self._asr_queue.empty():
            try:
                self._asr_queue.get_nowait()
            except queue.Empty:
                break
        self._asr_queue.put(None)
        cancel_asr = getattr(self._transcriber, "cancel_active", None)
        if callable(cancel_asr):
            cancel_asr()


class LilaVoice:
    """Capture push-to-talk utterances and deliver raw transcripts to a callback."""

    def __init__(
        self,
        config: VoiceConfig | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self.config = config or VoiceConfig()
        self._capture = AudioCapture(self.config)
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
        self._session_private = False
        self._live_stream: _LiveVoiceStream | None = None
        self._live_on_transcript: Callable[[str, str, bool], None] | None = None

    def _open_private_text(self, path: Path, mode: str):
        return open_private_text(path, mode)

    def speak(self, text: str) -> None:
        """Print the response; speech synthesis is not implemented."""
        print(f"🌸 Lila: {text}")

    @property
    def private_asr_available(self) -> bool:
        """True only when the configured existing Parakeet assets are present."""
        return (isinstance(self._transcriber, ParakeetTranscriber)
                and self._transcriber.executable.is_file()
                and self._transcriber.model.is_file())

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
            with self._open_private_text(transcript_path, "w") as stream:
                stream.write(transcript)
        except OSError as exc:
            raise VoiceInputError(
                f"Unable to preserve transcript at {transcript_path}."
            ) from exc
        return transcript

    def start_session(
        self,
        on_transcript: Callable[[str, str, bool], None] | None = None,
    ) -> dict:
        """Start one API-controlled recording without reading terminal input."""
        return self._start_session(private=False, on_transcript=on_transcript)

    def start_private_session(self) -> dict:
        """Start a capture reserved for a private caller-managed workflow."""
        return self._start_session(private=True)

    def _start_session(
        self,
        *,
        private: bool,
        on_transcript: Callable[[str, str, bool], None] | None = None,
    ) -> dict:
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
            use_streaming = (
                not private
                and getattr(self.config, "live_transcription", False)
                and callable(getattr(self._transcriber, "transcribe_pcm", None))
                and self._capture.can_stream_raw()
                and getattr(self._capture.start, "__func__", None) is AudioCapture.start
            )
            command = (
                self._capture.raw_command()
                if use_streaming
                else self._capture_command(audio_path)
            )
            try:
                with restrictive_umask():
                    process = (
                        self._capture.start_raw(command)
                        if use_streaming
                        else self._capture.start(command)
                    )
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
            self._session_private = private
            self._live_on_transcript = on_transcript if use_streaming else None
            if use_streaming:
                live_stream = _LiveVoiceStream(
                    process=process,
                    audio_path=audio_path,
                    transcriber=self._transcriber,
                    sample_rate=self.config.sample_rate,
                    chunk_seconds=getattr(self.config, "live_chunk_seconds", 4.0),
                    silence_seconds=getattr(self.config, "live_silence_seconds", 0.6),
                    rms_threshold=getattr(self.config, "live_rms_threshold", 0.015),
                    on_transcript=on_transcript,
                )
                self._live_stream = live_stream
                live_stream.start()
            else:
                self._live_stream = None
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
        return self._stop_session(session_id, private=False)

    def stop_private_session(self, session_id: str | None = None) -> str:
        """Transcribe one private utterance without transcript-file retention.

        Uses the configured existing ParakeetTranscriber and the same capture
        session. The caller must keep the result in a private authentication
        path and must not forward it to Core or conversation history.
        """
        return self._stop_session(session_id, private=True)

    def _stop_session(self, session_id: str | None, *, private: bool) -> str:
        with self._session_lock:
            self._require_session(session_id)
            if private is not self._session_private:
                raise VoiceSessionError("Voice session type does not match the requested route.")
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
            live_stream = self._live_stream
            live_on_transcript = self._live_on_transcript
            self._live_stream = None
            self._live_on_transcript = None
            if timer is not None:
                timer.cancel()

        assert process is not None and audio_path is not None and active_id is not None
        try:
            self._stop_capture_process(process)
            streamed_transcript = ""
            if live_stream is not None:
                streamed_transcript = live_stream.finish(
                    timeout_seconds=float(self.config.transcription_timeout_seconds)
                )
            secure_private_file(audio_path)
            self._validate_audio_file(audio_path)
            transcript = (
                streamed_transcript
                if streamed_transcript.strip()
                else self._transcriber.transcribe(audio_path)
            )
            if self._cancel_was_requested(active_id):
                raise VoiceSessionCancelled("Voice session was cancelled.")
            if not transcript.strip():
                raise TranscriptionError("Speech recognition returned no speech.")
            if live_on_transcript is not None:
                try:
                    live_on_transcript("", transcript, True)
                except Exception:
                    pass
            if private:
                with self._session_lock:
                    if self._session_id != active_id or self._session_cancel_requested:
                        raise VoiceSessionCancelled("Voice session was cancelled.")
                    self._session_state = "completed"
                    self._session_error = None
            else:
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
                        self._session_error = (
                            "private transcription failed"
                            if private else str(exc)[:300]
                        )
            raise
        finally:
            with self._capture_lock:
                if self._capture_process is process:
                    self._capture_process = None
            if private:
                # Best-effort cleanup of captured auth material. Secure erasure
                # cannot be guaranteed on copy-on-write or journaled filesystems.
                try:
                    audio_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _persist_session_transcript(
        self, session_id: str, transcript: str, transcript_path: Path
    ) -> None:
        """Write privately, then atomically publish only if cancellation has not won."""
        temporary_path = transcript_path.with_name(
            f"{transcript_path.name}.tmp-{uuid.uuid4().hex}"
        )
        try:
            with self._open_private_text(temporary_path, "w") as stream:
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
            live_stream = self._live_stream
            self._live_stream = None
            self._live_on_transcript = None
            self._session_cancel_requested = True
            self._session_state = "cancelling"
            if timer is not None:
                timer.cancel()

        try:
            if live_stream is not None:
                live_stream.cancel()
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
                process = AudioCapture.start(command)
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
        self._validate_audio_file(audio_path)
        return audio_path

    def _capture_command(self, audio_path: Path) -> list[str]:
        return self._capture.command(audio_path)

    def _validate_audio_file(self, audio_path: Path) -> None:
        self._capture.validate(audio_path)

    @staticmethod
    def _stop_capture_process(process: subprocess.Popen) -> None:
        AudioCapture.stop(process)

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
