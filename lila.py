#!/usr/bin/env python3
"""Lila — private family AI assistant."""

import argparse
import json
from src.core.lilacore import LilaCore
from src.core.runtime_lock import acquire_runtime_lock, release_runtime_lock


def _show_activity(message):
    """Display safe progress messages without showing private reasoning."""
    print(f"\\n🌸 {message}", flush=True)


def _approve_tool(tool, args, risk):
    """Ask the local owner to approve one exact tool call."""
    sensitive_words = (
        "password", "passwd", "secret", "token",
        "api_key", "private_key", "authorization", "cookie",
    )

    visible_args = {}
    for key, value in args.items():
        if any(word in key.lower() for word in sensitive_words):
            visible_args[key] = "[REDACTED]"
        else:
            visible_args[key] = value

    print("\\n" + "=" * 55)
    print("LILA PERMISSION REQUEST")
    print(f"Tool: {tool.name}")
    print(f"Risk: {risk.upper()}")
    print("Arguments:")
    print(json.dumps(visible_args, indent=2, ensure_ascii=False))
    print("=" * 55)

    try:
        answer = input("Approve this exact operation? Type YES: ")
    except (EOFError, KeyboardInterrupt):
        print("\\nApproval denied.")
        return False

    return answer.strip() == "YES"


def main():
    parser = argparse.ArgumentParser(description="Lila — Private Family AI")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable push-to-talk microphone input",
    )
    parser.add_argument("--gguf", default=None, help="Path to a GGUF model")
    parser.add_argument(
        "--model",
        default=None,
        help="Local model directory; remote IDs require --allow-remote-model",
    )
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="Allow downloading the explicitly configured --model identifier",
    )
    parser.add_argument("--check", action="store_true", help="Run startup check and exit")
    args = parser.parse_args()

    lock_fd = acquire_runtime_lock()
    try:
        lila = LilaCore(
            model_path=args.model,
            gguf_path=args.gguf,
            allow_remote_model=args.allow_remote_model,
            approval_callback=_approve_tool,
            activity_callback=_show_activity,
        )
        lila.boot()

        status = lila.what_do_i_know()
        print(f"Workspace: {lila.workspace_root}")
        print(f"Inference backend: {status['engine']}")
        print(f"Harness tools: {len(lila.available_tools)} available")

        if args.check:
            print("Startup check complete.")
            return

        if not lila.has_inference_backend:
            print("No AI model is connected. Exiting safely.")
            return

        if args.voice:
            from src.core.voice import LilaVoice, VoiceInputError
            voice = LilaVoice()
            try:
                voice.start_listening(
                    on_input=lambda text: _handle(lila, voice, text)
                )
            except VoiceInputError as exc:
                print(f"Voice input stopped: {exc}")
        else:
            _text_loop(lila)
    finally:
        release_runtime_lock(lock_fd)


def _text_loop(lila):
    print("\n🌸 Lila is ready. Type to talk. Type 'exit' to leave.\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("🌸 Lila is resting. Goodbye.")
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
