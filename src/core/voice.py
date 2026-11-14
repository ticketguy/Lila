"""Backward-compatible imports for the vTalks subsystem.

New integrations should import from :mod:`src.vtalks`. This adapter retains
the historical patch seams used by Lila's tests and external callers.
"""
from __future__ import annotations

# These module objects are intentionally exposed: patching subprocess.Popen,
# threading.Timer, time.sleep, or signal through this module continues to patch
# the same standard-library module object used by vTalks.
import signal
import subprocess
import threading
import time

from pathlib import Path

from src.vtalks import (
    AudioCaptureTimeoutError, MicrophoneError, ParakeetTranscriber, Transcriber,
    TranscriptionError, VoiceConfig, VoiceInputError, VoiceSessionBusyError,
    VoiceSessionCancelled, VoiceSessionError,
)
from src.vtalks.session import LilaVoice as _LilaVoice
from src.vtalks.storage import (
    open_private_text, restrictive_umask, secure_private_directory,
    secure_private_file,
)


class LilaVoice(_LilaVoice):
    """Legacy subclass that preserves the old storage monkeypatch seam."""

    def _open_private_text(self, path: Path, mode: str):
        return open_private_text(path, mode)


__all__ = [
    "AudioCaptureTimeoutError", "LilaVoice", "MicrophoneError",
    "ParakeetTranscriber", "Transcriber", "TranscriptionError",
    "VoiceConfig", "VoiceInputError", "VoiceSessionBusyError",
    "VoiceSessionCancelled", "VoiceSessionError",
]
