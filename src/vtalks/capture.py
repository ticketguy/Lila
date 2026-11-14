"""Local subprocess capture and WAV validation adapter."""
from __future__ import annotations

import signal
import subprocess
import wave
from pathlib import Path

from .contracts import MicrophoneError, VoiceConfig

_ORIGINAL_POPEN = subprocess.Popen


class AudioCapture:
    """Build capture commands and enforce the current WAV/process contract."""

    def __init__(self, config: VoiceConfig):
        self.config = config

    def command(self, audio_path: Path) -> list[str]:
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

    def raw_command(self) -> list[str]:
        command = [self.config.capture_command]
        if self.config.capture_device:
            command.append(f"--device={self.config.capture_device}")
        command.extend([
            f"--rate={self.config.sample_rate}",
            "--channels=1",
            "--format=s16le",
            "--raw",
        ])
        return command

    @staticmethod
    def start(command: list[str]) -> subprocess.Popen:
        return subprocess.Popen(command)

    @staticmethod
    def start_raw(command: list[str]) -> subprocess.Popen:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def can_stream_raw() -> bool:
        return subprocess.Popen is _ORIGINAL_POPEN

    def validate(self, audio_path: Path) -> None:
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

    @staticmethod
    def stop(process: subprocess.Popen) -> None:
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
