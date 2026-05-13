#!/usr/bin/env python3
"""
LILA — Private Family ASI Assistant
Start her up. Talk to her. She remembers. She grows.

Usage:
    python lila.py --gguf path/to/model.gguf    # Load from GGUF (recommended)
    python lila.py --model google/gemma-3-4b-it  # Load from HuggingFace
    python lila.py --voice --gguf model.gguf     # Voice mode
"""

import argparse
import os
from src.core.lilacore import LilaCore


def main():
    parser = argparse.ArgumentParser(description="Lila — Private Family ASI")
    parser.add_argument("--voice", action="store_true", help="Enable voice I/O")
    parser.add_argument("--gguf", default=None, help="Path to GGUF model file")
    parser.add_argument("--model", default=None, help="HuggingFace model path")
    args = parser.parse_args()
    
    # Boot with whatever model source is provided
    lila = LilaCore(model_path=args.model, gguf_path=args.gguf)
    lila.boot()
    
    if args.voice:
        try:
            from src.core.voice import LilaVoice
            voice = LilaVoice()
            voice.start_listening(on_input=lambda text: _handle(lila, voice, text))
        except ImportError:
            print("Voice dependencies not installed. Falling back to text mode.")
            _text_loop(lila)
    else:
        _text_loop(lila)


def _text_loop(lila):
    """Interactive text mode."""
    print("\n🌸 Lila is ready. Type to talk. Ctrl+C to exit.\n")
    while True:
        try:
            user_input = input("Sammie: ")
            if not user_input.strip():
                continue
            if user_input.lower() in ("quit", "exit"):
                break
            response = lila.think(user_input)
            print(f"Lila: {response.text}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n🌸 Lila is resting. Goodbye.")
            break


def _handle(lila, voice, text):
    response = lila.think(text)
    voice.speak(response.text)


if __name__ == "__main__":
    main()
