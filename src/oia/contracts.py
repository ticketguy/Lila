"""Public identity and authentication contracts for OIA.

These values intentionally describe identity and challenge-bound evidence in
application-neutral terms. Integrations own challenge creation, proof encoding,
and independent verification.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

_APPLICATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class OIAError(RuntimeError):
    """Typed OIA failure carrying a stable, non-secret reason code."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, repr=False)
class OwnerProfile:
    """A stable local owner identity, without credential material."""

    owner_id: str
    key_id: str
    enrolled_at_unix: int

    def __repr__(self) -> str:
        return f"OwnerProfile(owner_id={self.owner_id!r}, key_id={self.key_id!r})"


@dataclass(frozen=True)
class EnrollmentResult:
    """Result of creating the first local owner profile."""

    profile: OwnerProfile
    created: bool = True


@dataclass(frozen=True)
class OIAConfig:
    """Explicit local profile configuration with backward-compatible defaults."""

    credential_path: Path
    application_id: str = "lila"

    def __post_init__(self) -> None:
        if not isinstance(self.credential_path, Path):
            raise TypeError("credential_path must be a pathlib.Path.")
        if not isinstance(self.application_id, str) or not _APPLICATION_ID.fullmatch(
            self.application_id
        ):
            raise ValueError("application_id must be a short path-safe identifier.")

    @classmethod
    def from_environment(cls) -> "OIAConfig":
        """Resolve the legacy profile override and XDG-compatible default."""
        application_id = os.environ.get("LILA_APPLICATION_ID", "lila").strip() or "lila"
        configured = os.environ.get("LILA_OWNER_PROFILE_PATH")
        if configured:
            path = Path(configured).expanduser()
        else:
            data_home = Path(
                os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
            ).expanduser()
            path = data_home / application_id / "owner" / "credential.json"
        return cls(credential_path=path, application_id=application_id)


@dataclass(frozen=True, repr=False)
class AuthenticationChallenge:
    """Challenge context supplied by a trusted consuming integration.

    `context_id` and `context_digest` bind evidence to the integration's
    request. OIA treats `challenge_material` as opaque nonce material and
    leaves expiry and one-use enforcement to the challenge issuer/verifier.
    """

    challenge_id: str
    key_id: str
    context_id: str
    context_digest: str
    challenge_material: bytes
    issued_at_monotonic: float
    expires_at_monotonic: float
    purpose: str

    def __repr__(self) -> str:
        return (
            "AuthenticationChallenge(challenge_id={!r}, context_id={!r}, "
            "purpose={!r}, challenge_material=<redacted>)"
        ).format(self.challenge_id, self.context_id, self.purpose)


@dataclass(frozen=True)
class AuthenticationRequest:
    """Owner-authentication request scoped to one integration challenge."""

    challenge: AuthenticationChallenge
    allowed_responses: tuple[str, ...]


@dataclass(frozen=True, repr=False)
class AuthenticationEvidence:
    """Proof of an authenticated owner response, not an authorization grant."""

    challenge_id: str
    key_id: str
    context_id: str
    context_digest: str
    response: str
    proof: bytes
    authenticated_at_monotonic: float

    def __repr__(self) -> str:
        return (
            "AuthenticationEvidence(challenge_id={!r}, context_id={!r}, "
            "response={!r}, proof=<redacted>)"
        ).format(self.challenge_id, self.context_id, self.response)


class ProofMessageEncoder(Protocol):
    """Trusted integration codec for its established proof-verification rule."""

    def encode(self, challenge: AuthenticationChallenge, response: str) -> bytes:
        """Return exactly the bytes that the consuming verifier expects."""


class EvidenceConsumer(Protocol):
    """In-process consumer that independently verifies evidence.

    The temporary key buffer exists only for Lila's current in-process Harness
    verifier, which receives its unwrapped owner key per response. Consumers
    must not retain it; OIA wipes it when this callback returns.
    """

    def __call__(
        self, evidence: AuthenticationEvidence, verifier_key: bytearray
    ) -> object: ...


@dataclass(frozen=True)
class OwnerPresenceSession:
    """Bounded in-memory owner-presence session metadata."""

    session_id: str
    expires_at_monotonic: float


CredentialPath = Path
EvidenceConsumerCallable = Callable[[AuthenticationEvidence, bytearray], object]
