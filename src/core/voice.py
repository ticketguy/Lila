"""
Lila Voice — Speech I/O

Lila speaks and listens. This handles:
- Speech-to-text (listening for input)
- Text-to-speech (speaking responses)
- Wake word detection ("Lila", "Hey Lila")
"""

from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class VoiceConfig:
    wake_words: list = None
    tts_model: str = "default"
    stt_model: str = "default"
    voice_style: str = "warm"  # Lila's voice character
    
    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["lila", "hey lila", "lila,"]


class LilaVoice:
    """
    Lila's voice interface.
    
    Usage:
        voice = LilaVoice()
        voice.start_listening(on_wake=handle_wake)
        voice.speak("Hello Sammie")
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._listening = False
        self._on_input: Optional[Callable] = None
    
    def speak(self, text: str):
        """Convert text to speech and play."""
        # TODO: Wire TTS (e.g., Bark, XTTS, or system TTS)
        print(f"🌸 Lila: {text}")
    
    def start_listening(self, on_input: Callable[[str], None]):
        """Start listening for voice input."""
        self._on_input = on_input
        self._listening = True
        # TODO: Wire STT (e.g., Whisper)
        print("🌸 Lila is listening...")
    
    def stop_listening(self):
        self._listening = False
    
    @property
    def is_listening(self) -> bool:
        return self._listening
