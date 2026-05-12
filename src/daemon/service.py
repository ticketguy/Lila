"""
Lila Daemon — Always-on background service.

Runs Lila as a persistent process that:
  - Listens for voice input continuously
  - Monitors the system (events, schedules, sensors)
  - Executes tool calls autonomously
  - Hot-reloads .asi when training completes
  - Manages the voice pipeline (mic → STT → Lila → TTS → speaker)

Usage:
    python -m src.daemon.service             # Start daemon
    python -m src.daemon.service --no-voice  # Text-only daemon
    python -m src.daemon.service --port 7777 # With HTTP API
"""

import os
import sys
import time
import json
import signal
import threading
from pathlib import Path
from typing import Optional, Callable

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.lilacore import LilaCore, LilaResponse
from src.harness.executor import HarnessExecutor
from src.harness.system_tools import register_all_system_tools
from src.harness.extended_tools import register_extended_tools
from src.harness.tools import parse_tool_calls, execute_tool, format_tool_result


class LilaDaemon:
    """
    The always-on Lila service.
    
    She boots, she listens, she acts. Continuously.
    No human in the loop for execution — full autonomy.
    """
    
    def __init__(self, asi_path: Optional[str] = None, model_path: Optional[str] = None,
                 voice: bool = True, port: int = 0):
        self.asi_path = asi_path
        self.model_path = model_path
        self.voice_enabled = voice
        self.port = port
        
        self.lila: Optional[LilaCore] = None
        self.executor = HarnessExecutor()
        self._running = False
        self._voice_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._api_thread: Optional[threading.Thread] = None
        
        # Register all tools
        register_all_system_tools()
        register_extended_tools()
    
    def start(self):
        """Boot Lila and start all background loops."""
        print("\n🌸 Lila Daemon starting...")
        
        # Boot the model
        self.lila = LilaCore(model_path=self.model_path)
        self.lila.boot()
        
        self._running = True
        
        # Start voice loop (if enabled)
        if self.voice_enabled:
            self._voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
            self._voice_thread.start()
            print("   Voice: listening")
        
        # Start system monitor
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("   Monitor: active")
        
        # Start HTTP API (if port specified)
        if self.port > 0:
            self._api_thread = threading.Thread(target=self._api_loop, daemon=True)
            self._api_thread.start()
            print(f"   API: http://localhost:{self.port}")
        
        print("\n🌸 Lila is awake and listening.\n")
        
        # Handle shutdown gracefully
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        
        # Main loop — process text input if no voice
        if not self.voice_enabled:
            self._text_loop()
        else:
            # Keep main thread alive
            while self._running:
                time.sleep(1)
    
    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown."""
        print("\n🌸 Lila is resting. Goodbye.")
        self._running = False
    
    def process_input(self, text: str) -> str:
        """
        Process user input through the full pipeline:
          1. Lila thinks (inference)
          2. Check for tool calls in output
          3. Execute any tool calls
          4. If tools were called, feed results back and think again
          5. Return final response
        """
        if not self.lila:
            return "[Lila not booted]"
        
        # Initial inference
        response = self.lila.think(text)
        output = response.text
        
        # Tool execution loop (Lila can chain multiple calls)
        max_iterations = 5
        for _ in range(max_iterations):
            calls = parse_tool_calls(output)
            if not calls:
                break
            
            # Execute all tool calls
            results = []
            for call in calls:
                result = execute_tool(call["name"], **call["args"])
                results.append(format_tool_result(call["name"], result))
            
            # Feed results back and let Lila continue
            result_context = "\n".join(results)
            continuation = self.lila.think(
                f"[Tool results]\n{result_context}\n[Continue your response]",
                context={"mode": "continuation"}
            )
            output = continuation.text
        
        return output
    
    def _text_loop(self):
        """Interactive text input loop (when voice is disabled)."""
        while self._running:
            try:
                user_input = input("Sammie: ")
                if not user_input.strip():
                    continue
                if user_input.lower() in ("quit", "exit"):
                    self._shutdown()
                    break
                
                response = self.process_input(user_input)
                print(f"Lila: {response}\n")
            except (KeyboardInterrupt, EOFError):
                self._shutdown()
                break
    
    def _voice_loop(self):
        """
        Continuous voice listening loop.
        Mic → Speech-to-Text → Lila → Text-to-Speech → Speaker
        """
        try:
            from src.core.voice import LilaVoice
            voice = LilaVoice()
            
            while self._running:
                # Listen for speech
                text = voice.listen(timeout=None)  # Block until speech detected
                if not text or not self._running:
                    continue
                
                print(f"[heard] {text}")
                
                # Process through Lila
                response = self.process_input(text)
                
                # Speak the response
                voice.speak(response)
                print(f"[spoke] {response}")
                
        except ImportError:
            print("   Voice: unavailable (install speech_recognition, pyttsx3)")
            # Fall through to text mode
            self._text_loop()
        except Exception as e:
            print(f"   Voice error: {e}")
            self._text_loop()
    
    def _monitor_loop(self):
        """
        System monitoring loop.
        Checks for events that Lila should react to:
          - Schedule events (time-based triggers)
          - File system changes (watched directories)
          - Network events
          - Hot-reload requests (new .asi available)
        """
        schedule_file = os.path.expanduser("~/.lila/schedule.json")
        last_check = 0
        
        while self._running:
            time.sleep(10)  # Check every 10 seconds
            
            # Check schedule
            if os.path.exists(schedule_file):
                try:
                    with open(schedule_file) as f:
                        schedule = json.load(f)
                    
                    now = time.time()
                    for event in schedule:
                        if event.get("done"):
                            continue
                        # Simple time-based trigger (could be smarter)
                        # TODO: parse event["time"] properly
                except Exception:
                    pass
            
            # Check for hot-reload (.asi update)
            if self.asi_path and os.path.exists(self.asi_path):
                mtime = os.path.getmtime(self.asi_path)
                if mtime > last_check and last_check > 0:
                    print("\n🌸 [Hot-reload] New .asi detected, reloading adapters...")
                    # TODO: call asi_reload_fabric via the C engine
                last_check = mtime
    
    def _api_loop(self):
        """
        Simple HTTP API for external integrations.
        POST /chat {"text": "..."} → {"response": "..."}
        GET /status → {"running": true, ...}
        """
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        daemon = self
        
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/chat":
                    length = int(self.headers.get('Content-Length', 0))
                    body = json.loads(self.rfile.read(length))
                    text = body.get("text", "")
                    
                    response = daemon.process_input(text)
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"response": response}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_GET(self):
                if self.path == "/status":
                    status = {
                        "running": daemon._running,
                        "voice": daemon.voice_enabled,
                        "model": daemon.model_path or "default",
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(status).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress access logs
        
        server = HTTPServer(('0.0.0.0', self.port), Handler)
        while self._running:
            server.handle_request()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lila Daemon — Always-on intelligence")
    parser.add_argument("--asi", default=None, help="Path to .asi file")
    parser.add_argument("--model", default=None, help="HF model path")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice (text only)")
    parser.add_argument("--port", type=int, default=0, help="HTTP API port (0=disabled)")
    args = parser.parse_args()
    
    daemon = LilaDaemon(
        asi_path=args.asi,
        model_path=args.model,
        voice=not args.no_voice,
        port=args.port,
    )
    daemon.start()


if __name__ == "__main__":
    main()
