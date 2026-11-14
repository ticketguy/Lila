"""Exercise the per-user Linux activation script with isolated fake services."""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "linux" / "activate-voice-input.sh"


class VoiceActivationScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        self.command_log = self.root / "systemctl.log"
        for name in ("systemctl", "parecord"):
            executable = self.fake_bin / name
            executable.write_text(
                "#!/bin/sh\n"
                + ("printf '%s\\n' \"$*\" >> \"$COMMAND_LOG\"\n" if name == "systemctl" else "")
                + "exit 0\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
        self.voice_home = self.root / "voice-assets"
        self.asr = self.voice_home / "whisper.cpp" / "build" / "bin" / "parakeet-cli"
        self.model = self.voice_home / "whisper.cpp" / "models" / "parakeet.bin"
        self.asr.parent.mkdir(parents=True)
        self.model.parent.mkdir(parents=True)
        self.asr.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.asr.chmod(0o755)
        self.model.write_bytes(b"test model placeholder")

    def run_script(self, *arguments):
        environment = os.environ.copy()
        environment.update({
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
            "LILA_VOICE_HOME": str(self.voice_home),
            "LILA_VOICE_ASR_EXECUTABLE": str(self.asr),
            "LILA_VOICE_ASR_MODEL": str(self.model),
            "COMMAND_LOG": str(self.command_log),
            "PATH": str(self.fake_bin) + os.pathsep + environment.get("PATH", ""),
        })
        return subprocess.run(
            ["bash", str(SCRIPT), "--python", sys.executable, *arguments],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_activation_is_idempotent_and_deactivation_preserves_private_data(self):
        first = self.run_script()
        self.assertEqual(first.returncode, 0, first.stderr)
        unit = self.home / ".config" / "systemd" / "user" / "lila.service"
        service_env = self.home / ".config" / "lila" / "service.env"
        marker = self.home / ".config" / "lila" / "voice-input-install.json"
        self.assertTrue(unit.is_file())
        self.assertTrue(marker.is_file())
        self.assertEqual(stat.S_IMODE(service_env.stat().st_mode), 0o600)
        token_line = next(line for line in service_env.read_text().splitlines()
                          if line.startswith("LILA_API_TOKEN="))
        self.assertGreaterEqual(len(token_line.partition("=")[2]), 32)
        self.assertNotIn(token_line.partition("=")[2], first.stdout + first.stderr)

        second = self.run_script()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("already activated", second.stdout)
        original_token = service_env.read_text()

        removed = self.run_script("--deactivate")
        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertFalse(unit.exists())
        self.assertFalse(marker.exists())
        self.assertEqual(service_env.read_text(), original_token)
        self.assertTrue(self.model.is_file())

    def test_existing_unmarked_unit_is_never_overwritten(self):
        unit = self.home / ".config" / "systemd" / "user" / "lila.service"
        unit.parent.mkdir(parents=True)
        unit.write_text("user-owned unit\n", encoding="utf-8")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(unit.read_text(encoding="utf-8"), "user-owned unit\n")

    def test_invalid_external_model_path_is_rejected_before_unit_install(self):
        config = self.home / ".config" / "lila"
        config.mkdir(parents=True)
        service_env = config / "service.env"
        service_env.write_text(
            "LILA_API_TOKEN=" + "t" * 48 + "\n"
            "LILA_MODEL_PATH=/missing/local/model.gguf\n",
            encoding="utf-8",
        )
        unit = self.home / ".config" / "systemd" / "user" / "lila.service"

        result = self.run_script()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LILA_MODEL_PATH does not exist", result.stderr)
        self.assertFalse(unit.exists())
        self.assertIn("LILA_MODEL_PATH=/missing/local/model.gguf", service_env.read_text())


if __name__ == "__main__":
    unittest.main()
