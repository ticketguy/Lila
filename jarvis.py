#!/usr/bin/env python3
"""
Lila JARVIS Mode — Always on. Instant response. Full control.

Like JARVIS from Iron Man:
  - No dormant state. No wake words. She's ON.
  - You speak, she responds immediately.
  - She's proactive — monitors, alerts, acts without being asked.
  - Full system control via harness.
  - Voice-first. Terminal is fallback.

Requires:
  pip install llama-cpp-python
  (optional) pip install speech_recognition pyttsx3 pyaudio

Usage:
  python jarvis.py --model gemma-4b-Q4_K_M.gguf
  python jarvis.py --asi lila.asi
  python jarvis.py --model gemma-4b-Q4_K_M.gguf --voice
"""

import os
import sys
import json
import time
import struct
import tempfile
import threading
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class LilaJarvis:
    """
    JARVIS-mode Lila. Always on. No dormant state.
    Uses llama-cpp-python for inference (it actually works).
    """

    def __init__(self, model_path: str = None, asi_path: str = None,
                 n_ctx: int = 4096, n_threads: int = None, voice: bool = False):
        self.model_path = model_path
        self.asi_path = asi_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count()
        self.voice_enabled = voice
        self._llm = None
        self._running = False
        self._history = []
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """You are Lila, Sammie's private AI assistant. You run on his local machine.
You have FULL system access. You can execute commands, control hardware, manage networks.

When you need to act, output tool calls like this:
<|tool_call|>bash(command="ls -la")<|/tool_call|>

You are always on. You respond immediately. You are proactive — if you notice
something that needs attention, you speak up. You are warm but direct.

You can: run shell commands, read/write files, make network requests, SSH into servers,
control WiFi, GPIO, I2C, serial ports, set volume, send notifications, manage processes.

You are not a chatbot. You are an intelligence that controls this system."""

    def boot(self):
        """Load the model and be ready immediately. No dormant state."""
        print("\n🌸 Lila booting (JARVIS mode)...")

        gguf_path = self._resolve_model_path()
        if not gguf_path:
            print("ERROR: No model file. Provide --model or --asi")
            sys.exit(1)

        print(f"   Loading: {gguf_path}")
        print(f"   Threads: {self.n_threads}")
        print(f"   Context: {self.n_ctx}")

        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=gguf_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,  # CPU only (GPU via n_gpu_layers=-1)
                verbose=False,
                chat_format="gemma",  # Adjust per model
            )
            print("   Engine: llama.cpp ✓")
        except ImportError:
            print("ERROR: llama-cpp-python not installed.")
            print("  pip install llama-cpp-python")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR loading model: {e}")
            sys.exit(1)

        print("\n🌸 Lila is ON. Always listening.\n")

    def _resolve_model_path(self) -> str:
        """Get the GGUF path — either direct or extracted from .asi."""
        if self.model_path and os.path.exists(self.model_path):
            return self.model_path

        if self.asi_path and os.path.exists(self.asi_path):
            return self._extract_gguf_from_asi(self.asi_path)

        # Check common locations
        for path in ["lila.gguf", "model.gguf", os.path.expanduser("~/.lila/model.gguf")]:
            if os.path.exists(path):
                return path

        return None

    def _extract_gguf_from_asi(self, asi_path: str) -> str:
        """Extract the GGUF blob from a .asi v2 file to a temp location."""
        print(f"   Extracting GGUF from .asi...")

        with open(asi_path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != 0x41534921:
                print(f"   Not a valid .asi file")
                return None

            f.seek(0)
            header = f.read(64)
            _, version, flags, n_sections, total_size, table_offset = \
                struct.unpack_from('<IIIIQq', header, 0)

            # Parse section table to find GGUF blob
            f.seek(table_offset)
            gguf_offset = 0
            gguf_size = 0
            for _ in range(n_sections):
                stype, sflags, soffset, ssize, _ = struct.unpack('<IIQQQ', f.read(32))
                if stype == 0x02:  # GGUF_BLOB
                    gguf_offset = soffset
                    gguf_size = ssize
                    break

            if gguf_size == 0:
                print("   No GGUF blob found in .asi")
                return None

            # Extract to temp file
            tmp_dir = os.path.expanduser("~/.lila/cache/")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, "model.gguf")

            # Only extract if not already cached (or size changed)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) == gguf_size:
                print(f"   Using cached GGUF: {tmp_path}")
                return tmp_path

            print(f"   Extracting {gguf_size/1e9:.2f} GB to {tmp_path}...")
            f.seek(gguf_offset)
            with open(tmp_path, 'wb') as out:
                remaining = gguf_size
                while remaining > 0:
                    chunk_size = min(64 * 1024 * 1024, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)

            print(f"   Extracted.")
            return tmp_path

    def think(self, user_input: str) -> str:
        """Generate a response. This is the core inference call."""
        if not self._llm:
            return "[Model not loaded]"

        # Build messages for chat completion
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add recent history (last 10 turns)
        for msg in self._history[-20:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})

        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stop=["<end_of_turn>", "Sammie:", "\n\nSammie:"],
            )
            text = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            text = f"[Inference error: {e}]"

        # Store in history
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": text})

        return text

    def process(self, user_input: str) -> str:
        """
        Full pipeline: think → detect tool calls → execute → feed back → respond.
        """
        response = self.think(user_input)

        # Check for tool calls and execute them
        max_rounds = 5
        for _ in range(max_rounds):
            if "<|tool_call|>" not in response:
                break

            # Extract and execute tool calls
            results = self._execute_tools(response)
            if not results:
                break

            # Feed results back
            result_text = "\n".join(results)
            response = self.think(f"[Tool results]\n{result_text}\n[Continue your response to Sammie]")

        return response

    def _execute_tools(self, text: str) -> list:
        """Extract and execute tool calls from Lila's output."""
        import re
        pattern = r'<\|tool_call\|>\s*(\w+)\(([^)]*)\)\s*<\|/tool_call\|>'
        results = []

        for match in re.finditer(pattern, text):
            tool_name = match.group(1)
            args_str = match.group(2)

            # Parse args
            args = {}
            for arg_match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
                args[arg_match.group(1)] = arg_match.group(2)
            for arg_match in re.finditer(r'(\w+)\s*=\s*(\d+)', args_str):
                if arg_match.group(1) not in args:
                    args[arg_match.group(1)] = int(arg_match.group(2))

            # Execute
            result = self._run_tool(tool_name, args)
            results.append(f"[{tool_name}]: {result}")

        return results

    def _run_tool(self, name: str, args: dict) -> str:
        """Execute a single tool."""
        import subprocess

        if name == "bash":
            try:
                r = subprocess.run(args.get("command", "echo hi"),
                                   shell=True, capture_output=True, text=True, timeout=30)
                return (r.stdout + r.stderr)[:2000]
            except Exception as e:
                return f"Error: {e}"

        elif name == "file_read":
            try:
                with open(args.get("path", ""), 'r', errors='replace') as f:
                    return f.read(10000)
            except Exception as e:
                return f"Error: {e}"

        elif name == "file_write":
            try:
                path = args.get("path", "")
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                with open(path, 'w') as f:
                    f.write(args.get("content", ""))
                return f"Written to {path}"
            except Exception as e:
                return f"Error: {e}"

        # Fallback: try to run as bash
        return self._run_tool("bash", {"command": f"{name} {' '.join(str(v) for v in args.values())}"})

    def run(self):
        """Main loop. Always on. No dormant state."""
        self.boot()
        self._running = True
        signal.signal(signal.SIGINT, lambda *_: setattr(self, '_running', False))

        if self.voice_enabled:
            self._run_voice()
        else:
            self._run_text()

    def _run_text(self):
        """Text interaction loop."""
        while self._running:
            try:
                user_input = input("Sammie: ")
                if not user_input.strip():
                    continue
                if user_input.lower() in ("quit", "exit"):
                    break

                response = self.process(user_input)
                print(f"Lila: {response}\n")
            except (KeyboardInterrupt, EOFError):
                break

        print("\n🌸 Lila signing off.")

    def _run_voice(self):
        """Voice loop — always listening, instant response."""
        try:
            import speech_recognition as sr
            import pyttsx3

            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)

            print("🌸 Voice active. Speak anytime.\n")

            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)

            while self._running:
                try:
                    with mic as source:
                        audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
                    text = recognizer.recognize_google(audio)
                    if not text.strip():
                        continue

                    print(f"[heard] {text}")
                    response = self.process(text)
                    print(f"Lila: {response}\n")

                    engine.say(response)
                    engine.runAndWait()
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"[STT error: {e}]")
                except Exception as e:
                    print(f"[voice error: {e}]")

        except ImportError:
            print("Voice unavailable. Install: pip install SpeechRecognition pyttsx3 pyaudio")
            self._run_text()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Lila JARVIS Mode")
    p.add_argument("--model", help="Path to .gguf model file")
    p.add_argument("--asi", help="Path to .asi file (extracts GGUF)")
    p.add_argument("--voice", action="store_true", help="Enable voice I/O")
    p.add_argument("--ctx", type=int, default=4096, help="Context window")
    p.add_argument("--threads", type=int, default=None, help="CPU threads")
    args = p.parse_args()

    lila = LilaJarvis(
        model_path=args.model,
        asi_path=args.asi,
        n_ctx=args.ctx,
        n_threads=args.threads,
        voice=args.voice,
    )
    lila.run()
