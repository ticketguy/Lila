import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVISION_SCRIPT = ROOT / "deploy" / "linux" / "provision-mtls.sh"
TOKEN_READER = ROOT / "deploy" / "linux" / "read-service-token.py"
TOKEN_ROTATOR = ROOT / "deploy" / "linux" / "rotate-api-token.py"


@unittest.skipUnless(shutil.which("openssl"), "OpenSSL command line is required")
class MutualTlsProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config_home = self.root / "config"
        self.lila_config = self.config_home / "lila"
        self.lila_config.mkdir(parents=True)
        self.service_env = self.lila_config / "service.env"
        self.service_env.write_text(
            "LILA_API_TOKEN=test-only-api-token-that-is-long-enough\n"
            "EXISTING_SETTING=preserve-me\n",
            encoding="utf-8",
        )
        self.service_env.chmod(0o600)
        self.environment = os.environ.copy()
        self.environment["XDG_CONFIG_HOME"] = str(self.config_home)
        self.environment["HOME"] = str(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_provisioner(self):
        return subprocess.run(
            ["bash", str(PROVISION_SCRIPT)],
            cwd=ROOT,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_creates_private_material_and_preserves_service_configuration(self):
        result = self.run_provisioner()

        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(result.stdout)
        tls_dir = self.lila_config / "tls"
        material = tls_dir / "material"
        self.assertEqual((self.lila_config.stat().st_mode & 0o777), 0o700)
        self.assertEqual((tls_dir.stat().st_mode & 0o777), 0o700)
        for name in ("ca.key", "server.key", "client.key", "ca.pem", "server.pem", "client.pem"):
            self.assertEqual((material / name).stat().st_mode & 0o777, 0o600)
        self.assertEqual((tls_dir / "island-client.pfx").stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.service_env.stat().st_mode & 0o777, 0o600)

        contents = self.service_env.read_text(encoding="utf-8")
        self.assertIn("LILA_API_TOKEN=test-only-api-token-that-is-long-enough", contents)
        self.assertIn("EXISTING_SETTING=preserve-me", contents)
        self.assertIn(f'LILA_TLS_SERVER_CERT="{material}/server.pem"', contents)
        self.assertIn(f'LILA_TLS_SERVER_KEY="{material}/server.key"', contents)
        self.assertIn(f'LILA_TLS_CLIENT_CA="{material}/ca.pem"', contents)
        self.assertIn(f'LILA_TLS_ALLOWED_CLIENT_SHA256="{manifest["client_sha256"]}"', contents)
        self.assertEqual(len(manifest["client_sha256"]), 64)
        self.assertEqual(len(manifest["server_sha256"]), 64)
        self.assertNotIn("test-only-api-token-that-is-long-enough", result.stdout)
        self.assertNotIn("test-only-api-token-that-is-long-enough", result.stderr)

        for certificate, purpose in (("server.pem", "sslserver"), ("client.pem", "sslclient")):
            verified = subprocess.run(
                [
                    "openssl", "verify", "-purpose", purpose,
                    "-CAfile", str(material / "ca.pem"), str(material / certificate),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_repeat_setup_is_idempotent_and_does_not_rotate_pins(self):
        first = self.run_provisioner()
        self.assertEqual(first.returncode, 0, first.stderr)
        first_manifest = json.loads(first.stdout)
        original_env = self.service_env.read_bytes()

        second = self.run_provisioner()

        self.assertEqual(second.returncode, 0, second.stderr)
        second_manifest = json.loads(second.stdout)
        self.assertEqual(first_manifest["client_sha256"], second_manifest["client_sha256"])
        self.assertEqual(first_manifest["server_sha256"], second_manifest["server_sha256"])
        self.assertEqual(original_env, self.service_env.read_bytes())

    def test_partial_certificate_state_is_preserved_and_rejected(self):
        material = self.lila_config / "tls" / "material"
        material.mkdir(parents=True)
        material.chmod(0o700)
        sentinel = material / "ca.key"
        sentinel.write_text("preexisting fixture", encoding="utf-8")

        result = self.run_provisioner()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete", result.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preexisting fixture")
        self.assertNotIn("LILA_TLS_SERVER_CERT=", self.service_env.read_text(encoding="utf-8"))

    def test_material_symlink_is_rejected_without_touching_its_target(self):
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel"
        sentinel.write_text("unchanged", encoding="utf-8")
        tls_dir = self.lila_config / "tls"
        tls_dir.mkdir()
        (tls_dir / "material").symlink_to(outside, target_is_directory=True)

        result = self.run_provisioner()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not be a symlink", result.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
        self.assertFalse((outside / "server.key").exists())

    def test_existing_mismatched_mtls_values_are_not_overwritten(self):
        original = self.service_env.read_text(encoding="utf-8")
        self.service_env.write_text(
            original + 'LILA_TLS_SERVER_CERT="/existing/custom/server.pem"\n',
            encoding="utf-8",
        )
        self.service_env.chmod(0o600)
        before = self.service_env.read_bytes()

        result = self.run_provisioner()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("partial mTLS configuration", result.stderr)
        self.assertEqual(before, self.service_env.read_bytes())

    def test_windows_setup_token_reader_emits_only_the_valid_configured_token(self):
        result = subprocess.run(
            ["python3", str(TOKEN_READER)],
            cwd=ROOT,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "test-only-api-token-that-is-long-enough\n")

    def test_token_reader_rejects_broad_file_permissions_without_revealing_token(self):
        self.service_env.chmod(0o644)

        result = subprocess.run(
            ["python3", str(TOKEN_READER)],
            cwd=ROOT,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mode 0600", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("test-only-api-token-that-is-long-enough", result.stderr)

    def test_explicit_token_rotation_preserves_other_settings_and_never_prints_secret(self):
        result = subprocess.run(
            ["python3", str(TOKEN_ROTATOR)],
            cwd=ROOT,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        contents = self.service_env.read_text(encoding="utf-8")
        self.assertIn("EXISTING_SETTING=preserve-me", contents)
        self.assertNotIn("test-only-api-token-that-is-long-enough", contents)
        self.assertEqual(self.service_env.stat().st_mode & 0o777, 0o600)
        new_token = next(
            line.partition("=")[2]
            for line in contents.splitlines()
            if line.startswith("LILA_API_TOKEN=")
        )
        self.assertGreaterEqual(len(new_token), 32)


if __name__ == "__main__":
    unittest.main()
