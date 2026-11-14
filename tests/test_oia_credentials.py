import json
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.oia.cli import enroll_local
from src.oia.credentials import (
    CredentialError,
    normalize_owner_phrase,
    unlock_owner_secret,
    validate_new_phrase,
)
from src.oia.service import OIAError, OwnerAuthenticationService, OwnerProfileStore


PHRASE = "violet orchard window copper lantern river"


class OwnerCredentialTests(unittest.TestCase):
    def test_single_character_phrase_is_allowed(self):
        self.assertEqual(validate_new_phrase("x"), "x")

    def test_empty_phrase_is_rejected(self):
        for phrase in ("", "   "):
            with self.subTest(phrase=phrase):
                with self.assertRaisesRegex(CredentialError, "owner_phrase.empty"):
                    validate_new_phrase(phrase)

    def test_phrase_over_maximum_length_is_rejected(self):
        with self.assertRaisesRegex(CredentialError, "owner_phrase.too_long"):
            validate_new_phrase("x" * 513)

    def test_enrollment_prompt_explains_phrase_rules(self):
        store = Mock(enrolled=False)
        store.create.return_value = SimpleNamespace(owner_id="test-owner")
        output = StringIO()

        with patch("src.oia.cli.getpass.getpass", side_effect=[PHRASE, PHRASE]):
            with redirect_stdout(output):
                enroll_local(store)

        self.assertIn("any non-empty private phrase, in any language", output.getvalue())
        self.assertIn("maximum is 512 characters", output.getvalue())

    def test_local_enrollment_protects_random_secret_and_exact_phrase_unlocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owner" / "credential.json"
            store = OwnerProfileStore(path)
            self.assertIsNone(store.profile())
            profile = store.create(PHRASE, PHRASE)
            self.assertEqual(profile.key_id, store.profile().key_id)
            raw = path.read_text()
            self.assertNotIn(PHRASE, raw)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            credential = store.credential()
            secret = unlock_owner_secret(PHRASE, credential)
            self.assertEqual(len(secret), 32)
            self.assertNotEqual(secret.decode("latin1"), PHRASE)
            for index in range(len(secret)):
                secret[index] = 0

    def test_wrong_phrase_tampering_and_repeat_bootstrap_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owner.json"
            store = OwnerProfileStore(path)
            store.create(PHRASE, PHRASE)
            with self.assertRaises(CredentialError):
                unlock_owner_secret("violet orchard window brass lantern river", store.credential())
            with self.assertRaises(OIAError):
                store.create(PHRASE, PHRASE)
            payload = json.loads(path.read_text())
            payload["owner_id"] = "tampered-owner"
            path.write_text(json.dumps(payload))
            with self.assertRaises(CredentialError):
                unlock_owner_secret(PHRASE, store.credential())
            payload = json.loads(path.read_text())
            payload["ciphertext"] = "A" * len(payload["ciphertext"])
            path.write_text(json.dumps(payload))
            with self.assertRaises(CredentialError):
                unlock_owner_secret(PHRASE, store.credential())

    def test_normalization_is_exact_except_nfkc_case_and_whitespace(self):
        self.assertEqual(
            normalize_owner_phrase("Violet   ORCHARD window copper lantern river"),
            normalize_owner_phrase(PHRASE),
        )
        self.assertNotEqual(
            normalize_owner_phrase("violet, orchard window copper lantern river"),
            normalize_owner_phrase(PHRASE),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = OwnerProfileStore(Path(directory) / "owner.json")
            with self.assertRaises(OIAError):
                store.create("violet, orchard window copper lantern river", PHRASE)

    def test_owner_presence_session_is_bounded_and_locks_on_restart(self):
        now = [100.0]
        with tempfile.TemporaryDirectory() as directory:
            store = OwnerProfileStore(Path(directory) / "owner.json")
            store.create(PHRASE, PHRASE)
            service = OwnerAuthenticationService(
                store, clock=lambda: now[0], session_seconds=10.0
            )
            service.establish_presence(PHRASE, source="typed")
            self.assertTrue(service.session_valid())
            now[0] = 110.0
            self.assertFalse(service.session_valid())
            service.establish_presence(PHRASE, source="typed")
            service.lock()
            self.assertFalse(service.session_valid())
            restarted = OwnerAuthenticationService(store, clock=lambda: now[0])
            self.assertFalse(restarted.session_valid())

    def test_spoken_provider_stays_unavailable_without_live_owner_phrase_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = OwnerProfileStore(Path(directory) / "owner.json")
            store.create(PHRASE, PHRASE)
            service = OwnerAuthenticationService(store)
            service.set_spoken_provider_available(True)
            spoken = next(p for p in service.providers() if p.provider_id == "spoken_owner_phrase")
            self.assertFalse(spoken.available)
            self.assertEqual(spoken.reason_code, "provider.live_owner_asr_unverified")

    def test_enroll_spoken_phrase_persists_across_restart_and_rejects_wrong_phrase(self):
        with tempfile.TemporaryDirectory() as directory:
            cred_path = Path(directory) / "owner.json"
            store = OwnerProfileStore(cred_path)
            store.create(PHRASE, PHRASE)
            service = OwnerAuthenticationService(store)

            with self.assertRaises(OIAError):
                service.enroll_spoken_phrase("wrong spoken phrase", asr_available=True)
            spoken = next(p for p in service.providers() if p.provider_id == "spoken_owner_phrase")
            self.assertFalse(spoken.enrolled)
            self.assertFalse(spoken.available)

            enrolled_provider = service.enroll_spoken_phrase(PHRASE, asr_available=True)
            self.assertTrue(enrolled_provider.enrolled)
            self.assertTrue(enrolled_provider.available)
            self.assertEqual(enrolled_provider.reason_code, "provider.enrolled")

            # Restarting the service with ASR available preserves verified spoken enrollment.
            restarted = OwnerAuthenticationService(OwnerProfileStore(cred_path))
            restarted.set_spoken_provider_available(True)
            spoken_after = next(p for p in restarted.providers() if p.provider_id == "spoken_owner_phrase")
            self.assertTrue(spoken_after.enrolled)
            self.assertTrue(spoken_after.available)
            self.assertEqual(spoken_after.reason_code, "provider.enrolled")


if __name__ == "__main__":
    unittest.main()
