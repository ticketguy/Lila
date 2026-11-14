"""ASR backend implementations used by vTalks."""

from __future__ import annotations

import ctypes
import subprocess
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .contracts import TranscriptionError, Transcriber, VoiceSessionCancelled

_PARAKEET_SAMPLE_RATE = 16000
_PARAKEET_SAMPLING_GREEDY = 0
_GGML_ABORT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)
_GGML_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


class _ParakeetContextParams(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("gpu_device", ctypes.c_int),
    ]


class _ParakeetFullParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("no_context", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        ("new_segment_callback", ctypes.c_void_p),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("new_token_callback", ctypes.c_void_p),
        ("new_token_callback_user_data", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", ctypes.c_void_p),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_user_data", ctypes.c_void_p),
    ]


def _load_wav_mono_f32(audio_path: Path) -> np.ndarray | None:
    """Read a 16-bit 16 kHz mono PCM WAV into float32 [-1, 1], or None if unsupported."""
    try:
        with wave.open(str(audio_path), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != _PARAKEET_SAMPLE_RATE:
                return None
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, OSError):
        return None
    if not raw:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class _WarmParakeetEngine:
    """In-process libparakeet.so wrapper keeping the ASR model warm in memory."""

    def __init__(self, library_path: Path, model_path: Path) -> None:
        self.library_path = Path(library_path)
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self._lib: ctypes.CDLL | None = None
        self._ctx: ctypes.c_void_p | None = None
        self._log_cb = _GGML_LOG_CALLBACK(lambda _level, _text, _user: None)

    def _ensure_loaded(self) -> tuple[ctypes.CDLL, ctypes.c_void_p]:
        if self._lib is not None and self._ctx is not None:
            return self._lib, self._ctx
        lib = ctypes.CDLL(str(self.library_path))
        lib.parakeet_log_set.argtypes = [_GGML_LOG_CALLBACK, ctypes.c_void_p]
        lib.parakeet_log_set.restype = None
        lib.parakeet_context_default_params.argtypes = []
        lib.parakeet_context_default_params.restype = _ParakeetContextParams
        lib.parakeet_init_from_file_with_params.argtypes = [ctypes.c_char_p, _ParakeetContextParams]
        lib.parakeet_init_from_file_with_params.restype = ctypes.c_void_p
        lib.parakeet_full_default_params_by_ref.argtypes = [ctypes.c_int]
        lib.parakeet_full_default_params_by_ref.restype = ctypes.POINTER(_ParakeetFullParams)
        lib.parakeet_free_params.argtypes = [ctypes.POINTER(_ParakeetFullParams)]
        lib.parakeet_free_params.restype = None
        lib.parakeet_full.argtypes = [
            ctypes.c_void_p,
            _ParakeetFullParams,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.parakeet_full.restype = ctypes.c_int
        lib.parakeet_full_n_segments.argtypes = [ctypes.c_void_p]
        lib.parakeet_full_n_segments.restype = ctypes.c_int
        lib.parakeet_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.parakeet_full_get_segment_text.restype = ctypes.c_char_p
        lib.parakeet_free.argtypes = [ctypes.c_void_p]
        lib.parakeet_free.restype = None

        lib.parakeet_log_set(self._log_cb, None)
        ctx_params = lib.parakeet_context_default_params()
        ctx_params.use_gpu = False
        ctx = lib.parakeet_init_from_file_with_params(str(self.model_path).encode("utf-8"), ctx_params)
        if not ctx:
            raise TranscriptionError(f"Failed to load Parakeet model: {self.model_path}")
        self._lib = lib
        self._ctx = ctx
        return lib, ctx

    def warm_up(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def transcribe_pcm(
        self,
        samples: np.ndarray,
        *,
        threads: int,
        should_abort: Callable[[], bool],
    ) -> str:
        pcm = np.ascontiguousarray(samples, dtype=np.float32)
        if pcm.size == 0:
            raise TranscriptionError("Parakeet returned an empty transcript.")
        with self._lock:
            lib, ctx = self._ensure_loaded()
            params_ptr = lib.parakeet_full_default_params_by_ref(_PARAKEET_SAMPLING_GREEDY)
            if not params_ptr:
                raise TranscriptionError("Unable to allocate Parakeet inference parameters.")
            try:
                params = params_ptr.contents
                params.n_threads = threads
                abort_cb = _GGML_ABORT_CALLBACK(lambda _user_data: bool(should_abort()))
                params.abort_callback = ctypes.cast(abort_cb, ctypes.c_void_p)
                rc = lib.parakeet_full(
                    ctx,
                    params,
                    pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    int(pcm.size),
                )
            finally:
                lib.parakeet_free_params(params_ptr)

            if should_abort():
                raise VoiceSessionCancelled("Voice transcription was cancelled.")
            if rc != 0:
                raise TranscriptionError(f"Parakeet inference failed with status {rc}.")
            n_seg = lib.parakeet_full_n_segments(ctx)
            parts: list[str] = []
            for i in range(n_seg):
                seg = lib.parakeet_full_get_segment_text(ctx, i)
                if seg:
                    parts.append(seg.decode("utf-8", errors="replace"))
        text = "".join(parts).strip()
        if not text:
            raise TranscriptionError("Parakeet returned an empty transcript.")
        return text

    def close(self) -> None:
        with self._lock:
            if self._lib is not None and self._ctx is not None:
                self._lib.parakeet_free(self._ctx)
                self._ctx = None


class ParakeetTranscriber:
    """Run the existing CPU-capable Parakeet CLI without loading another model."""
    """Run CPU-capable Parakeet ASR, keeping libparakeet.so warm when available."""

    def __init__(
        self,
        executable: Path,
        model: Path,
        timeout_seconds: int = 180,
        threads: int = 2,
        library_path: Path | None = None,
    ) -> None:
        self.executable = Path(executable)
        self.model = Path(model)
        self.timeout_seconds = timeout_seconds
        self.threads = threads
        self.library_path = Path(library_path) if library_path is not None else self.executable.parent / "libparakeet.so"
        self._process_lock = threading.Lock()
        self._active_process: subprocess.Popen | None = None
        self._cancel_generation = 0
        self._warm_engine: _WarmParakeetEngine | None = None

    def warm_up(self) -> bool:
        """Preload libparakeet.so into memory when available; returns True if warmed."""
        if not (self.library_path.is_file() and self.model.is_file()):
            return False
        if self._warm_engine is None:
            self._warm_engine = _WarmParakeetEngine(self.library_path, self.model)
        self._warm_engine.warm_up()
        return True

    def transcribe_pcm(self, samples: np.ndarray) -> str:
        """Transcribe an in-memory 16 kHz mono float32 PCM buffer."""
        if not (self.library_path.is_file() and self.model.is_file()):
            raise TranscriptionError("Warm libparakeet.so backend is unavailable.")
        if self._warm_engine is None:
            self._warm_engine = _WarmParakeetEngine(self.library_path, self.model)
        with self._process_lock:
            generation = self._cancel_generation
        return self._warm_engine.transcribe_pcm(
            samples,
            threads=self.threads,
            should_abort=lambda: generation != self._cancel_generation,
        )

    def close(self) -> None:
        if self._warm_engine is not None:
            self._warm_engine.close()

    def transcribe(self, audio_path: Path) -> str:
        audio_path = Path(audio_path)
        for label, path in (
            ("Parakeet executable", self.executable),
            ("Parakeet model", self.model),
            ("Audio recording", audio_path),
        ):
            if not path.is_file():
                raise TranscriptionError(f"{label} not found: {path}")

        if self.library_path.is_file():
            pcm = _load_wav_mono_f32(audio_path)
            if pcm is not None:
                return self.transcribe_pcm(pcm)

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
