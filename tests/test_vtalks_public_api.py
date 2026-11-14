import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.core.voice import LilaVoice as LegacyLilaVoice
from src.vtalks import (
    AudioCapture,
    LilaVoice,
    ParakeetTranscriber,
    VoiceConfig,
)


class VTalksPublicApiTests(unittest.TestCase):
    def test_public_classes_are_exported_from_the_subsystem(self):
        self.assertEqual(LilaVoice.__module__, "src.vtalks.session")
        self.assertEqual(ParakeetTranscriber.__module__, "src.vtalks.asr")
        self.assertEqual(VoiceConfig.__module__, "src.vtalks.contracts")

    def test_legacy_voice_adapter_subclasses_public_implementation(self):
        self.assertTrue(issubclass(LegacyLilaVoice, LilaVoice))

    def test_capture_adapter_builds_the_configured_command(self):
        with TemporaryDirectory() as directory:
            config = VoiceConfig(recordings_dir=Path(directory), capture_device="test-source")
            command = AudioCapture(config).command(Path(directory) / "capture.wav")

        self.assertEqual(command[0], "parecord")
        self.assertIn("--device=test-source", command)
        self.assertEqual(command[-1], str(Path(directory) / "capture.wav"))


if __name__ == "__main__":
    unittest.main()
