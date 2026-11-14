import hashlib
import hmac
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from src.oia import (
    AuthenticationChallenge,
    AuthenticationEvidence,
    AuthenticationRequest,
    OIAConfig,
    OIAError,
    OwnerAuthenticationService,
    OwnerProfile,
)


class _Store:
    enrolled = True

    def profile(self):
        return OwnerProfile("owner-1", "key-1", 0)

    def credential(self):
        raise AssertionError("Tests supply an isolated phrase unlock.")


class _Encoder:
    def encode(self, challenge, response):
        return f"{challenge.challenge_id}|{challenge.context_id}|{response}".encode()


class OIAPublicApiTests(unittest.TestCase):
    def test_importing_oia_does_not_import_harness_implementation(self):
        code = (
            "import sys; import src.oia; "
            "assert not any(name.startswith('src.harness.') for name in sys.modules)"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_config_preserves_default_path_and_allows_app_identity_override(self):
        with patch.dict(
            os.environ,
            {"XDG_DATA_HOME": "/tmp/oia-data", "LILA_APPLICATION_ID": "sample"},
            clear=True,
        ):
            config = OIAConfig.from_environment()
        self.assertEqual(config.application_id, "sample")
        self.assertEqual(
            config.credential_path,
            Path("/tmp/oia-data/sample/owner/credential.json"),
        )

    def test_authentication_returns_bound_evidence_and_wipes_handoff_key(self):
        service = OwnerAuthenticationService(_Store())
        owner_key = bytearray(b"oia-test-verifier-secret-32-bytes!!")

        def unlock(phrase, *, source):
            if phrase != "correct owner phrase":
                raise OIAError("owner_authentication.failed")
            return bytearray(owner_key)

        service._unlock = unlock
        challenge = AuthenticationChallenge(
            challenge_id="challenge-1",
            key_id="key-1",
            context_id="request-1",
            context_digest="a" * 64,
            challenge_material=b"fresh nonce",
            issued_at_monotonic=10.0,
            expires_at_monotonic=30.0,
            purpose="test.response",
        )
        request = AuthenticationRequest(challenge, ("yes", "no"))
        evidence = service.authenticate(
            request,
            "correct owner phrase",
            response="yes",
            proof_encoder=_Encoder(),
        )

        self.assertIsInstance(evidence, AuthenticationEvidence)
        self.assertEqual(evidence.context_id, "request-1")
        self.assertEqual(evidence.context_digest, "a" * 64)
        self.assertEqual(evidence.response, "yes")
        expected = hmac.new(
            bytes(owner_key), b"challenge-1|request-1|yes", hashlib.sha256
        ).digest()
        self.assertEqual(evidence.proof, expected)

        captured = []

        def consume(authenticated, verifier_key):
            captured.append(verifier_key)
            return authenticated

        service.authenticate(
            request,
            "correct owner phrase",
            response="no",
            proof_encoder=_Encoder(),
            consumer=consume,
        )
        self.assertEqual(captured[0], bytearray(len(owner_key)))


if __name__ == "__main__":
    unittest.main()
