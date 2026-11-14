"""Owner profile enrollment, exact phrase verification and decision proofing."""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .contracts import (
    AuthenticationChallenge, AuthenticationEvidence, AuthenticationRequest,
    EnrollmentResult, EvidenceConsumerCallable, OIAError, OwnerPresenceSession,
    OwnerProfile, ProofMessageEncoder,
)
from .credentials import (
    CredentialError, CredentialStorage, CredentialUnavailable, OwnerCredentialFile,
    ProtectedOwnerSecret, normalize_owner_phrase, protect_owner_secret,
    unlock_owner_secret, validate_new_phrase, _wipe,
)
from .providers import ProviderAvailability, discover_providers


class OwnerProfileStore:
    """Local store for one primary owner. No remote bootstrap operation exists."""

    def __init__(
        self, path: Path | None = None, *, storage: CredentialStorage | None = None
    ):
        if storage is None and path is None:
            raise ValueError("A credential path or storage provider is required.")
        self._file: CredentialStorage = (
            storage if storage is not None else OwnerCredentialFile(path)
        )
        self._spoken_state_path: Path | None = (
            self._file.path.with_name("owner_spoken_verified.json")
            if isinstance(self._file, OwnerCredentialFile)
            else None
        )
        self._spoken_verified_mem: tuple[str, str] | None = None

    @property
    def enrolled(self) -> bool:
        return self._file.exists()

    def profile(self) -> OwnerProfile | None:
        if not self.enrolled:
            return None
        credential = self._file.read()
        # The timestamp is deliberately not needed for proof or authority.
        return OwnerProfile(credential.owner_id, credential.key_id, 0)

    def create(self, phrase: str, confirmation: str) -> OwnerProfile:
        if self.enrolled:
            raise OIAError("owner_profile.already_exists")
        try:
            canonical = validate_new_phrase(phrase)
            if not hmac.compare_digest(canonical, validate_new_phrase(confirmation)):
                raise OIAError("owner_enrollment.confirmation_mismatch")
            owner_id = str(uuid.uuid4())
            key_id = "owner-" + secrets.token_hex(12)
            credential = protect_owner_secret(canonical, owner_id, key_id)
            self._file.write_new(credential)
            return OwnerProfile(owner_id, key_id, int(time.time()))
        except CredentialUnavailable:
            raise OIAError("owner_authentication.credential_provider_unavailable") from None
        except CredentialError as exc:
            raise OIAError(str(exc)) from None

    def enroll(self, phrase: str, confirmation: str) -> EnrollmentResult:
        """Create a profile and return the explicit enrollment result contract."""
        return EnrollmentResult(self.create(phrase, confirmation))

    @property
    def spoken_verified(self) -> bool:
        profile = self.profile()
        if profile is None:
            return False
        if self._spoken_state_path is None:
            return self._spoken_verified_mem == (profile.owner_id, profile.key_id)
        try:
            data = json.loads(self._spoken_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            isinstance(data, dict)
            and data.get("verified") is True
            and data.get("owner_id") == profile.owner_id
            and data.get("key_id") == profile.key_id
        )

    def mark_spoken_verified(self, profile: OwnerProfile) -> None:
        if self._spoken_state_path is None:
            self._spoken_verified_mem = (profile.owner_id, profile.key_id)
            return
        payload = json.dumps(
            {
                "owner_id": profile.owner_id,
                "key_id": profile.key_id,
                "verified": True,
            }
        )
        self._spoken_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._spoken_state_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        os.replace(tmp, self._spoken_state_path)
        try:
            os.chmod(self._spoken_state_path, 0o600)
        except OSError:
            pass

    def credential(self) -> ProtectedOwnerSecret:
        try:
            return self._file.read()
        except CredentialError as exc:
            raise OIAError(str(exc)) from None


class OwnerAuthenticationService:
    """Authenticates exact owner intent; it does not make Harness policy."""

    def __init__(self, store: OwnerProfileStore, *,
                 clock: Callable[[], float] = time.monotonic,
                 session_seconds: float = 300.0,
                 failure_limit: int = 5, lockout_seconds: float = 60.0):
        self.store = store
        self._clock = clock
        self._session_seconds = session_seconds
        self._failure_limit = failure_limit
        self._lockout_seconds = lockout_seconds
        self._lock = threading.RLock()
        self._failures = 0
        self._locked_until = 0.0
        self._session_expires = 0.0
        self._presence_session_id: str | None = None
        self._spoken_verified = bool(
            store.enrolled and getattr(store, "spoken_verified", False)
        )
        self._spoken_provider_available = False

    def providers(self) -> tuple[ProviderAvailability, ...]:
        items = list(discover_providers())
        enrolled = self.has_owner()
        spoken_enrolled = enrolled and self._spoken_verified
        spoken_available = spoken_enrolled and self._spoken_provider_available
        return tuple(
            replace(item, enrolled=enrolled,
                    reason_code="provider.enrolled" if enrolled else item.reason_code)
            if item.provider_id == "typed_owner_phrase" else
            replace(
                item,
                available=spoken_available,
                enrolled=spoken_enrolled,
                reason_code=(
                    "provider.enrolled"
                    if spoken_enrolled and spoken_available
                    else ("provider.available" if self._spoken_provider_available
                          else "provider.live_owner_asr_unverified")
                ),
            )
            if item.provider_id == "spoken_owner_phrase" else item
            for item in items
        )

    def set_spoken_provider_available(
        self, available: bool, *, owner_phrase_verified: bool = False
    ) -> None:
        # Assets alone do not prove that this installation recognizes the
        # enrolled owner's phrase. Only live spoken phrase verification (now or
        # persisted for the active owner profile) enables spoken approval.
        if owner_phrase_verified:
            self._spoken_verified = True
            self._spoken_provider_available = bool(available)
        else:
            self._spoken_provider_available = bool(available and self._spoken_verified)

    def enroll_spoken_phrase(
        self, phrase: str, *, asr_available: bool = True
    ) -> ProviderAvailability:
        """Verify a live spoken phrase transcript against the enrolled owner secret."""
        if not asr_available:
            raise OIAError("spoken_enrollment.asr_unavailable")
        secret = self._unlock(phrase, source="spoken")
        _wipe(secret)
        profile = self.owner_profile()
        if profile is None:
            raise OIAError("owner_authentication.not_enrolled")
        if hasattr(self.store, "mark_spoken_verified"):
            self.store.mark_spoken_verified(profile)
        self.set_spoken_provider_available(True, owner_phrase_verified=True)
        return next(p for p in self.providers() if p.provider_id == "spoken_owner_phrase")

    def has_owner(self) -> bool:
        return self.store.enrolled

    def owner_profile(self) -> OwnerProfile | None:
        return self.store.profile()

    def establish_presence(self, phrase: str, *, source: str) -> OwnerPresenceSession:
        """Verify phrase and open bounded presence; phrase input is never logged."""
        secret = self._unlock(phrase, source=source)
        _wipe(secret)
        with self._lock:
            self._session_expires = self._clock() + self._session_seconds
            self._presence_session_id = uuid.uuid4().hex
            return OwnerPresenceSession(
                self._presence_session_id, self._session_expires
            )

    def session_valid(self) -> bool:
        with self._lock:
            return self._session_expires > self._clock()

    def lock(self) -> None:
        with self._lock:
            self._session_expires = 0.0
            self._presence_session_id = None

    def authenticate(
        self,
        request: AuthenticationRequest,
        phrase: str,
        *,
        response: str,
        proof_encoder: ProofMessageEncoder,
        source: str = "typed",
        consumer: EvidenceConsumerCallable | None = None,
    ) -> AuthenticationEvidence | object:
        """Authenticate an owner response and return challenge-bound evidence.

        An optional trusted, in-process consumer may independently verify the
        evidence while the unwrapped owner key is available. OIA does not
        authorize, dispatch, or decide whether the evidence is sufficient.
        """
        if not isinstance(request, AuthenticationRequest):
            raise OIAError("owner_authentication.request_invalid")
        context = request.challenge
        if (
            type(request.challenge) is not AuthenticationChallenge
            or type(request.allowed_responses) is not tuple
            or not request.allowed_responses
            or any(not isinstance(item, str) or not item for item in request.allowed_responses)
            or not isinstance(response, str)
            or response not in request.allowed_responses
            or not callable(getattr(proof_encoder, "encode", None))
        ):
            raise OIAError("owner_authentication.response_invalid")
        profile = self.store.profile()
        if profile is None or profile.key_id != context.key_id:
            raise OIAError("owner_authentication.identity_binding_mismatch")

        secret = self._unlock(phrase, source=source)
        try:
            try:
                message = proof_encoder.encode(context, response)
            except Exception:
                raise OIAError("owner_authentication.proof_encoding_failed") from None
            if not isinstance(message, bytes) or not message:
                raise OIAError("owner_authentication.proof_encoding_failed")
            evidence = AuthenticationEvidence(
                challenge_id=context.challenge_id,
                key_id=context.key_id,
                context_id=context.context_id,
                context_digest=context.context_digest,
                response=response,
                proof=hmac.new(bytes(secret), message, "sha256").digest(),
                authenticated_at_monotonic=self._clock(),
            )
            if consumer is None:
                return evidence
            if not callable(consumer):
                raise OIAError("owner_authentication.consumer_invalid")
            return consumer(evidence, secret)
        finally:
            _wipe(secret)

    def verify_spoken_phrase(self, transcript: str) -> bool:
        """Private exact phrase verifier; never call from Core/conversation paths."""
        secret = self._unlock(transcript, source="spoken")
        _wipe(secret)
        return True

    def authenticate_spoken(
        self,
        request: AuthenticationRequest,
        transcript: str,
        *,
        proof_encoder: ProofMessageEncoder,
        consumer: EvidenceConsumerCallable | None = None,
    ) -> AuthenticationEvidence | object:
        """Authenticate an exact spoken response plus the fresh challenge code."""
        if not isinstance(request, AuthenticationRequest):
            raise OIAError("owner_authentication.request_invalid")
        if (
            type(request.challenge) is not AuthenticationChallenge
            or type(request.allowed_responses) is not tuple
            or not request.allowed_responses
            or any(not isinstance(item, str) or not item for item in request.allowed_responses)
            or type(request.challenge.challenge_material) is not bytes
            or len(request.challenge.challenge_material) < 4
        ):
            raise OIAError("owner_authentication.request_invalid")
        if not isinstance(transcript, str) or len(transcript) > 2048:
            raise OIAError("owner_authentication.transcript_invalid")
        normalized = normalize_owner_phrase(transcript)
        match = re.fullmatch(r"(.+)\s+(approve|reject)\s+([0-9]{6})", normalized)
        if match is None:
            raise OIAError("owner_authentication.transcript_ambiguous")
        phrase, decision_text, response_code = match.groups()
        if decision_text not in request.allowed_responses:
            raise OIAError("owner_authentication.response_invalid")
        expected_code = (
            f"{int.from_bytes(request.challenge.challenge_material[:4], 'big') % 1_000_000:06d}"
        )
        if not hmac.compare_digest(response_code, expected_code):
            raise OIAError("owner_authentication.challenge_response_invalid")
        return self.authenticate(
            request,
            phrase,
            response=decision_text,
            proof_encoder=proof_encoder,
            source="spoken",
            consumer=consumer,
        )

    def _unlock(self, phrase: str, *, source: str) -> bytearray:
        if source not in {"typed", "spoken", "local_setup"}:
            raise OIAError("owner_authentication.source_untrusted")
        now = self._clock()
        with self._lock:
            if now < self._locked_until:
                raise OIAError("owner_authentication.rate_limited")
        try:
            credential = self.store.credential()
            secret = unlock_owner_secret(phrase, credential)
        except CredentialUnavailable:
            self._record_failure(now)
            raise OIAError("owner_authentication.credential_provider_unavailable") from None
        except (CredentialError, OIAError):
            self._record_failure(now)
            raise OIAError("owner_authentication.failed") from None
        with self._lock:
            self._failures = 0
            self._locked_until = 0.0
        return secret

    def _record_failure(self, now: float) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._failure_limit:
                self._locked_until = now + self._lockout_seconds
                self._failures = 0
