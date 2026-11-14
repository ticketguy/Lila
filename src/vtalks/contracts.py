"""Public vTalks configuration and session contracts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

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
    live_transcription: bool = True
    live_chunk_seconds: float = 4.0
    live_silence_seconds: float = 0.6
    live_rms_threshold: float = 0.015
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
