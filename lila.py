#!/usr/bin/env python3
"""
LILA — Private Family ASI Assistant
Start her up. Talk to her. She remembers. She grows.

Usage:
    python lila.py              # Interactive mode
    python lila.py --voice      # Voice mode
"""

import argparse
from src.core.lilacore import LilaCore
from src.core.voice import LilaVoice, VoiceConfig


def main():
    parser = argparse.ArgumentParser(description="Lila — Private Family ASI")
    parser.add_argument("--voice", action="store_true", help="Enable voice I/O")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Model path")
    args = parser.parse_args()
    
    # Boot
    lila = LilaCore(model_path=args.model)
    lila.boot()
    
    if args.voice:
        voice = LilaVoice()
        voice.start_listening(on_input=lambda text: _handle(lila, voice, text))
    else:
        # Text mode
        print("\n🌸 Lila is ready. Type to talk. Ctrl+C to exit.\n")
        while True:
            try:
                user_input = input("Sammie: ")
                if not user_input.strip():
                    continue
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
