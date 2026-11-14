import signal
import stat
import subprocess
import tempfile
import threading
import unittest
import wave
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.lilacore import LilaResponse
from src.core.voice import (
    AudioCaptureTimeoutError,
    LilaVoice,
    MicrophoneError,
    ParakeetTranscriber,
    TranscriptionError,
    VoiceConfig,
    VoiceSessionBusyError,
    VoiceSessionCancelled,
    open_private_text,
)
from lila import _handle


RAW_TRANSCRIPT = "  Original ASR transcript, unchanged. \n"


def write_wav(path: Path, frames: bytes = b"\x00\x00" * 160) -> None:
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16_000)
        recording.writeframes(frames)


class FakeCaptureProcess:
    def __init__(self, command, frames: bytes = b"\x00\x00" * 160):
        self.command = command
        self.returncode = None
        write_wav(Path(command[-1]), frames)

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.returncode = -signum

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.returncode = -signal.SIGKILL


class VoiceInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.recordings_dir = self.root / "recordings"
        self.config = VoiceConfig(
            recordings_dir=self.recordings_dir,
            asr_executable=self.root / "parakeet-cli",
            asr_model=self.root / "parakeet-model.bin",
            max_recording_seconds=30,
            transcription_timeout_seconds=7,
            transcription_threads=2,
        )

    def voice(self, transcriber=None) -> LilaVoice:
        return LilaVoice(
            config=self.config,
            transcriber=transcriber or Mock(transcribe=Mock(return_value=RAW_TRANSCRIPT)),
        )

    def test_microphone_capture_saves_raw_audio_and_delivers_raw_transcript(self):
        transcriber = Mock(transcribe=Mock(return_value=RAW_TRANSCRIPT))
        voice = self.voice(transcriber)

        with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess) as popen:
            with patch("builtins.input", return_value=""):
                transcript = voice.listen_once()

        audio_path = transcriber.transcribe.call_args.args[0]
        transcript_path = audio_path.with_name(
            f"{audio_path.stem}.transcript.txt"
        )
        self.assertEqual(transcript, RAW_TRANSCRIPT)
        self.assertTrue(audio_path.is_file())
        self.assertEqual(transcript_path.read_text(encoding="utf-8"), RAW_TRANSCRIPT)
        self.assertEqual(stat.S_IMODE(self.recordings_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(audio_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(transcript_path.stat().st_mode), 0o600)
        self.assertEqual(
            popen.call_args.args[0][1:],
            [
                "--rate=16000",
                "--channels=1",
                "--format=s16le",
                "--file-format=wav",
                str(audio_path),
            ],
        )

    def test_microphone_capture_does_not_change_unrelated_research_file(self):
        self.recordings_dir.mkdir(parents=True)
        benchmark = self.recordings_dir / "asr-benchmark-output.txt"
        benchmark.write_text("unchanged research output", encoding="utf-8")
        benchmark.chmod(0o644)
        voice = self.voice()

        with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess):
            with patch("builtins.input", return_value=""):
                audio_path = voice._capture_audio()

        self.assertEqual(benchmark.stat().st_mode & 0o777, 0o644)
        self.assertEqual(
            benchmark.read_text(encoding="utf-8"), "unchanged research output"
        )
        self.assertEqual(audio_path.stat().st_mode & 0o777, 0o600)

    def test_explicit_capture_device_is_applied_without_becoming_a_default(self):
        config = VoiceConfig(
            recordings_dir=self.recordings_dir,
            asr_executable=self.root / "parakeet-cli",
            asr_model=self.root / "parakeet-model.bin",
            capture_device="usb_microphone.monitor",
        )
        voice = LilaVoice(config=config)
        with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess) as popen:
            with patch("builtins.input", return_value=""):
                audio_path = voice._capture_audio()

        self.assertEqual(popen.call_args.args[0][1], "--device=usb_microphone.monitor")
        self.assertEqual(popen.call_args.args[0][-1], str(audio_path))

    def test_empty_audio_is_rejected_without_calling_asr(self):
        transcriber = Mock()
        voice = self.voice(transcriber)

        with patch(
            "src.core.voice.subprocess.Popen",
            side_effect=lambda command: FakeCaptureProcess(command, frames=b""),
        ):
            with patch("builtins.input", return_value=""):
                with self.assertRaisesRegex(MicrophoneError, "empty"):
                    voice.listen_once()

        transcriber.transcribe.assert_not_called()

    def test_microphone_start_failure_is_reported(self):
        voice = self.voice()
        with patch(
            "src.core.voice.subprocess.Popen",
            side_effect=FileNotFoundError("parecord"),
        ):
            with self.assertRaisesRegex(MicrophoneError, "Unable to start"):
                voice._capture_audio()

    def test_microphone_capture_that_exits_immediately_is_reported(self):
        process = Mock(returncode=1)
        process.poll.return_value = 1
        process.wait.return_value = 1
        voice = self.voice()
        with patch("src.core.voice.subprocess.Popen", return_value=process):
            with patch("src.core.voice.time.sleep"):
                with self.assertRaisesRegex(
                    MicrophoneError, "exited before recording"
                ):
                    voice._capture_audio()

    def test_recording_duration_limit_is_reported_and_audio_retained(self):
        class ImmediateTimer:
            def __init__(self, _interval, callback):
                self.callback = callback

            def start(self):
                self.callback()

            def cancel(self):
                pass

        voice = self.voice()
        with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess):
            with patch("src.core.voice.threading.Timer", ImmediateTimer):
                with patch("builtins.input", return_value=""):
                    with self.assertRaises(AudioCaptureTimeoutError):
                        voice._capture_audio()

        self.assertEqual(len(list(self.recordings_dir.glob("*.wav"))), 1)

    def test_stop_listening_stops_active_microphone_capture(self):
        voice = self.voice()
        capture_started = threading.Event()
        release_prompt = threading.Event()
        processes = []
        capture_result = []

        def create_process(command):
            process = FakeCaptureProcess(command)
            processes.append(process)
            return process

        def wait_for_finish(_prompt):
            capture_started.set()
            release_prompt.wait(timeout=5)
            return ""

        with patch("src.core.voice.subprocess.Popen", side_effect=create_process):
            with patch("builtins.input", side_effect=wait_for_finish):
                capture_thread = threading.Thread(
                    target=lambda: capture_result.append(voice._capture_audio())
                )
                capture_thread.start()
                self.assertTrue(capture_started.wait(timeout=5))
                voice.stop_listening()
                self.assertEqual(processes[0].returncode, -signal.SIGINT)
                release_prompt.set()
                capture_thread.join(timeout=5)

        self.assertFalse(capture_thread.is_alive())
        self.assertEqual(len(capture_result), 1)
        self.assertIsNone(voice._capture_process)

    def test_asr_failure_preserves_captured_raw_audio(self):
        voice = self.voice(
            Mock(transcribe=Mock(side_effect=TranscriptionError("ASR failed")))
        )
        with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess):
            with patch("builtins.input", return_value=""):
                with self.assertRaisesRegex(TranscriptionError, "ASR failed"):
                    voice.listen_once()

        self.assertEqual(len(list(self.recordings_dir.glob("*.wav"))), 1)

    def test_transcription_timeout_is_reported(self):
        audio_path = self.root / "speech.wav"
        executable = self.root / "parakeet-cli"
        model = self.root / "parakeet.bin"
        for path in (audio_path, executable, model):
            path.touch()
        transcriber = ParakeetTranscriber(
            executable, model, timeout_seconds=7, threads=2
        )

        process = Mock()
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["parakeet-cli"], 7), ("", "")
        ]
        process.returncode = -9
        with patch("src.core.voice.subprocess.Popen", return_value=process) as popen:
            with self.assertRaisesRegex(TranscriptionError, "timed out after 7"):
                transcriber.transcribe(audio_path)

        process.kill.assert_called_once()
        process.communicate.assert_called()
        self.assertIn("--no-gpu", popen.call_args.args[0])

    def test_parakeet_transcriber_returns_unmodified_stdout(self):
        audio_path = self.root / "speech.wav"
        executable = self.root / "parakeet-cli"
        model = self.root / "parakeet.bin"
        for path in (audio_path, executable, model):
            path.touch()
        transcriber = ParakeetTranscriber(executable, model, threads=2)

        process = Mock()
        process.communicate.return_value = (RAW_TRANSCRIPT, "")
        process.returncode = 0
        with patch("src.core.voice.subprocess.Popen", return_value=process) as popen:
            transcript = transcriber.transcribe(audio_path)

        self.assertEqual(transcript, RAW_TRANSCRIPT)
        self.assertIn("--no-prints", popen.call_args.args[0])
        self.assertIn(str(model), popen.call_args.args[0])
        self.assertIn(str(audio_path), popen.call_args.args[0])

    def test_parakeet_transcriber_reports_missing_runtime_assets(self):
        transcriber = ParakeetTranscriber(
            self.root / "missing-cli",
            self.root / "missing-model",
        )
        with self.assertRaisesRegex(TranscriptionError, "not found"):
            transcriber.transcribe(self.root / "missing-audio.wav")

    def test_parakeet_active_process_can_be_cancelled(self):
        audio_path = self.root / "speech.wav"
        executable = self.root / "parakeet-cli"
        model = self.root / "parakeet.bin"
        for path in (audio_path, executable, model):
            path.touch()
        started = threading.Event()
        finished = threading.Event()

        class BlockingProcess:
            returncode = None

            def communicate(self, timeout=None):
                started.set()
                finished.wait(2)
                return "", ""

            def poll(self):
                return self.returncode

            def kill(self):
                self.returncode = -signal.SIGKILL
                finished.set()

        process = BlockingProcess()
        transcriber = ParakeetTranscriber(executable, model)
        errors = []

        def transcribe():
            try:
                transcriber.transcribe(audio_path)
            except Exception as exc:
                errors.append(exc)

        with patch("src.core.voice.subprocess.Popen", return_value=process):
            worker = threading.Thread(target=transcribe)
            worker.start()
            self.assertTrue(started.wait(1))
            transcriber.cancel_active()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], VoiceSessionCancelled)

    def test_empty_transcript_is_rejected(self):
        voice = self.voice(Mock(transcribe=Mock(return_value=" \n")))
        with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess):
            with patch("builtins.input", return_value=""):
                with self.assertRaisesRegex(TranscriptionError, "no speech"):
                    voice.listen_once()

    def test_api_session_is_single_and_delivers_original_transcript(self):
        transcriber = Mock(transcribe=Mock(return_value=RAW_TRANSCRIPT))
        voice = self.voice(transcriber)
        with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess):
            with patch("src.core.voice.time.sleep"):
                started = voice.start_session()
                self.assertEqual(started["state"], "recording")
                with self.assertRaises(VoiceSessionBusyError):
                    voice.start_session()
                transcript = voice.stop_session(started["session_id"])

        self.assertEqual(transcript, RAW_TRANSCRIPT)
        self.assertEqual(voice.get_session_status()["state"], "completed")
        transcriber.transcribe.assert_called_once()

    def test_api_session_cancellation_does_not_transcribe(self):
        transcriber = Mock(transcribe=Mock(return_value=RAW_TRANSCRIPT))
        voice = self.voice(transcriber)
        with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess):
            with patch("src.core.voice.time.sleep"):
                started = voice.start_session()
                cancelled = voice.cancel_session(started["session_id"])

        self.assertEqual(cancelled["state"], "cancelled")
        transcriber.transcribe.assert_not_called()

    def test_cancellation_during_transcript_staging_never_publishes_transcript(self):
        transcriber = Mock(transcribe=Mock(return_value=RAW_TRANSCRIPT))
        voice = self.voice(transcriber)
        staged = threading.Event()
        allow_publish_attempt = threading.Event()
        errors = []
        @contextmanager
        def pause_after_staging(path, mode="a"):
            with open_private_text(path, mode) as stream:
                yield stream
                if ".tmp-" in Path(path).name:
                    staged.set()
                    if not allow_publish_attempt.wait(timeout=3):
                        raise TimeoutError("test did not release staged transcript")

        with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess):
            with patch("src.core.voice.time.sleep"):
                started = voice.start_session()
            audio_path = voice._session_audio_path
            self.assertIsNotNone(audio_path)

            def stop_session():
                try:
                    voice.stop_session(started["session_id"])
                except Exception as exc:
                    errors.append(exc)

            with patch("src.core.voice.open_private_text", side_effect=pause_after_staging):
                worker = threading.Thread(target=stop_session)
                worker.start()
                self.assertTrue(staged.wait(timeout=2), "transcript was not staged")
                cancelled = voice.cancel_session(started["session_id"])
                self.assertIn(cancelled["state"], {"cancelling", "cancelled"})
                allow_publish_attempt.set()
                worker.join(timeout=2)

        self.assertFalse(worker.is_alive(), "transcription worker did not finish")
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], VoiceSessionCancelled)
        transcript_path = audio_path.with_name(f"{audio_path.stem}.transcript.txt")
        self.assertFalse(transcript_path.exists())
        self.assertFalse(list(audio_path.parent.glob(transcript_path.name + ".tmp-*")))
        self.assertEqual(voice.get_session_status()["state"], "cancelled")

    def test_push_to_talk_callback_receives_exact_transcript(self):
        voice = self.voice()
        voice.listen_once = Mock(return_value=RAW_TRANSCRIPT)
        received = []

        with patch("builtins.input", side_effect=["", "q"]):
            voice.start_listening(received.append)

        self.assertEqual(received, [RAW_TRANSCRIPT])
        self.assertFalse(voice.is_listening)

    def test_cli_handler_passes_transcript_to_lilacore(self):
        response = LilaResponse(
            text="Lila response",
            memory_ops=[],
            actions=[],
            confidence=1.0,
        )
        core = Mock(think=Mock(return_value=response))
        voice = Mock()

        _handle(core, voice, RAW_TRANSCRIPT)

        core.think.assert_called_once_with(RAW_TRANSCRIPT)
        voice.speak.assert_called_once_with("Lila response")

    def test_private_auth_transcription_uses_existing_asr_and_deletes_auth_audio(self):
        audio_path = self.root / "private-auth.wav"
        write_wav(audio_path)
        transcript = "private phrase must not persist"
        transcriber = Mock(transcribe=Mock(return_value=transcript))
        voice = self.voice(transcriber)
        process = Mock()
        voice._session_id = "private-session"
        voice._session_state = "recording"
        voice._session_private = True
        voice._session_audio_path = audio_path
        voice._capture_process = process
        with patch.object(voice, "_stop_capture_process"), patch.object(voice, "_validate_audio_file"):
            result = voice.stop_private_session("private-session")

        self.assertEqual(result, transcript)
        transcriber.transcribe.assert_called_once_with(audio_path)
        self.assertFalse(audio_path.exists())
        self.assertFalse(audio_path.with_name("private-auth.transcript.txt").exists())
        self.assertEqual(voice.get_session_status()["state"], "completed")

    def test_default_voice_assets_and_recordings_stay_outside_git_checkout(self):
        config = VoiceConfig()
        repo_root = Path(__file__).resolve().parents[1]

        self.assertNotIn(repo_root, config.recordings_dir.parents)
        self.assertIn("experiments", config.asr_executable.parts)
        self.assertIn("experiments", config.asr_model.parts)
        self.assertIsNone(config.capture_device)
        self.assertEqual(config.sample_rate, 16_000)
        self.assertEqual(config.recordings_dir.parts[-2:], ("lila", "recordings"))

    def test_voice_assets_and_capture_source_can_be_overridden_per_machine(self):
        with patch.dict(os.environ, {
            "LILA_VOICE_HOME": str(self.root / "voice assets"),
            "LILA_VOICE_ASR_EXECUTABLE": str(self.root / "bin" / "parakeet"),
            "LILA_VOICE_ASR_MODEL": str(self.root / "models" / "custom.bin"),
            "LILA_VOICE_CAPTURE_DEVICE": "custom.source",
            "LILA_RECORDINGS_DIR": str(self.root / "private-recordings"),
        }):
            config = VoiceConfig()

        self.assertEqual(config.asr_executable, self.root / "bin" / "parakeet")
        self.assertEqual(config.asr_model, self.root / "models" / "custom.bin")
        self.assertEqual(config.capture_device, "custom.source")
        self.assertEqual(config.recordings_dir, self.root / "private-recordings")

    def test_live_voice_stream_chunks_speech_emits_callbacks_and_writes_wav(self):
        import io
        import wave
        import numpy as np
        from src.vtalks.session import _LiveVoiceStream

        sr = 16_000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        speech = (0.25 * np.sin(2 * np.pi * 440.0 * t) * 32767.0).astype(np.int16)
        silence = np.zeros(int(sr * 0.7), dtype=np.int16)
        pcm_stream = io.BytesIO(speech.tobytes() + silence.tobytes())

        fake_proc = Mock()
        fake_proc.stdout = pcm_stream

        class PcmTranscriber:
            def __init__(self):
                self.chunks = []

            def transcribe(self, audio_path):
                raise AssertionError("full-file transcribe should not be needed")

            def transcribe_pcm(self, samples):
                self.chunks.append(len(samples))
                return "hello lila"

        transcriber = PcmTranscriber()
        events = []
        audio_path = self.root / "streamed.wav"
        stream = _LiveVoiceStream(
            process=fake_proc,
            audio_path=audio_path,
            transcriber=transcriber,
            sample_rate=sr,
            chunk_seconds=4.0,
            silence_seconds=0.6,
            rms_threshold=0.015,
            on_transcript=lambda chunk, acc, final: events.append((chunk, acc, final)),
        )
        stream.start()
        combined = stream.finish(timeout_seconds=5.0)

        self.assertEqual(combined, "hello lila")
        self.assertEqual(len(transcriber.chunks), 1)
        self.assertEqual(events, [("hello lila", "hello lila", False)])
        with wave.open(str(audio_path), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), sr)
            self.assertGreater(wav.getnframes(), 0)

    def test_private_session_never_streams_live_transcripts(self):
        class DualTranscriber:
            def transcribe(self, audio_path):
                return "secret passphrase"

            def transcribe_pcm(self, samples):
                raise AssertionError("private session must never call transcribe_pcm")

        voice = self.voice(transcriber=DualTranscriber())
        voice.config.live_transcription = True
        with patch.object(voice._capture, "start_raw", side_effect=AssertionError("must not stream raw")):
            with patch("src.core.voice.subprocess.Popen", side_effect=FakeCaptureProcess):
                with patch("src.core.voice.time.sleep"):
                    started = voice.start_private_session()
                    result = voice.stop_private_session(started["session_id"])

        self.assertEqual(result, "secret passphrase")


if __name__ == "__main__":
    unittest.main()
