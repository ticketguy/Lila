"""M2B Step 4 owner-decision challenge authentication.

This module authenticates a decision for one exact in-memory approval grant.
It does not authorize execution, consume a grant, persist state, or provide a
UI/daemon workflow. The owner secret must be provisioned independently by a
trusted owner-controlled channel; it must not be the ordinary daemon API token.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import base64
import hashlib
import hmac
import json
import math
import re
import secrets
import threading
import time
from typing import Callable

from .approval_grants import ApprovalGrant
from .pending_invocations import PendingInvocationRegistry, PendingInvocationState


_CHALLENGE_ISSUER = object()
_AUTHENTICATED_DECISION_ISSUER = object()
_PROOF_DOMAIN = b"lila-owner-approval-decision\x00v1\x00"
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class ApprovalAuthenticationError(ValueError):
    """A challenge or owner-decision proof failed closed."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class OwnerDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


@dataclass(frozen=True, slots=True, init=False, repr=False)
class ApprovalChallenge:
    """Fresh challenge bound to one invocation and operation digest.

    Challenge material is returned only to the trusted owner-facing adapter.
    It is omitted from repr and must never be logged. The value is a nonce,
    not an owner authentication secret.
    """

    challenge_id: str
    key_id: str
    invocation_id: str
    operation_digest: str
    challenge_material: bytes
    issued_at_monotonic: float
    expires_at_monotonic: float

    def __init__(self, *, _issuer=None, **values) -> None:
        if _issuer is not _CHALLENGE_ISSUER:
            raise TypeError("Approval challenges can only be issued by trusted Harness code.")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return f"ApprovalChallenge(challenge_id={self.challenge_id!r}, key_id={self.key_id!r})"


@dataclass(frozen=True, slots=True, repr=False)
class ApprovalDecisionResponse:
    """Untrusted response envelope received from an owner-facing adapter."""

    challenge_id: object
    key_id: object
    invocation_id: object
    operation_digest: object
    decision: object
    proof: object

    def __repr__(self) -> str:
        return (
            "ApprovalDecisionResponse(challenge_id={!r}, decision={!r}, proof=<redacted>)"
        ).format(self.challenge_id, self.decision)


@dataclass(frozen=True, slots=True, init=False, repr=False)
class AuthenticatedOwnerDecision:
    """Verified statement of owner intent; deliberately not an authorization."""

    challenge_id: str
    key_id: str
    invocation_id: str
    operation_digest: str
    decision: OwnerDecision
    authenticated_at_monotonic: float

    def __init__(self, *, _issuer=None, **values) -> None:
        if _issuer is not _AUTHENTICATED_DECISION_ISSUER:
            raise TypeError("Authenticated decisions can only be issued by the verifier.")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (
            "AuthenticatedOwnerDecision(challenge_id={!r}, invocation_id={!r}, "
            "decision={!r})"
        ).format(self.challenge_id, self.invocation_id, self.decision.value)


class ApprovalAuthenticator:
    """Verify HMAC-SHA-256 owner decisions against exact pending grants.

    The `owner_secret` is a distinct, high-entropy shared secret provisioned
    through an owner-controlled channel. This verifier is an application
    boundary, not an OS isolation boundary: hostile code already executing in
    this Python process may inspect memory or invoke trusted internals.
    """

    def __init__(
        self,
        *,
        pending_registry: PendingInvocationRegistry,
        owner_secret: bytes,
        key_id: str,
        challenge_lifetime_seconds: float = 60.0,
        capacity: int = 256,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(pending_registry, PendingInvocationRegistry):
            raise TypeError("An instance-owned PendingInvocationRegistry is required.")
        if not isinstance(owner_secret, bytes) or len(owner_secret) < 32:
            raise ValueError("An owner authentication secret of at least 32 bytes is required.")
        if not isinstance(key_id, str) or not _KEY_ID_RE.fullmatch(key_id):
            raise ValueError("key_id must be a stable non-secret identifier.")
        if (
            isinstance(challenge_lifetime_seconds, bool)
            or not isinstance(challenge_lifetime_seconds, (int, float))
            or not math.isfinite(challenge_lifetime_seconds)
            or challenge_lifetime_seconds <= 0
        ):
            raise ValueError("Challenge lifetime must be positive and finite.")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Challenge capacity must be a positive integer.")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable.")
        # Copy so the caller cannot mutate a bytearray or retain an alias.
        self._owner_secret = bytes(owner_secret)
        self._pending_registry = pending_registry
        self._key_id = key_id
        self._challenge_lifetime = float(challenge_lifetime_seconds)
        self._capacity = capacity
        self._clock = monotonic_clock
        self._challenges: dict[str, tuple[ApprovalChallenge, ApprovalGrant]] = {}
        self._lock = threading.RLock()

    def issue_challenge(
        self, invocation_id: str, grant: ApprovalGrant
    ) -> ApprovalChallenge:
        """Issue unpredictable challenge material for the exact live grant."""
        if not isinstance(grant, ApprovalGrant):
            raise ApprovalAuthenticationError("approval.grant_invalid")
        try:
            pending = self._pending_registry.get(invocation_id)
            now = self._read_clock()
            if (
                pending.state is not PendingInvocationState.PENDING
                or grant.invocation_id != invocation_id
                or not grant.is_bound_to(pending, now)
                or not self._pending_registry.validate_current(invocation_id)
            ):
                raise ApprovalAuthenticationError("approval.binding_invalid")
            # Re-read after current-manifest/target validation; a terminal or
            # changed registry record cannot be used to mint a challenge.
            current = self._pending_registry.get(invocation_id)
            now = self._read_clock()
            if (
                current.state is not PendingInvocationState.PENDING
                or not grant.is_bound_to(current, now)
            ):
                raise ApprovalAuthenticationError("approval.binding_invalid")
        except ApprovalAuthenticationError:
            raise
        except Exception:
            raise ApprovalAuthenticationError("approval.binding_unavailable") from None

        with self._lock:
            self._prune_expired_locked(now)
            if len(self._challenges) >= self._capacity:
                raise ApprovalAuthenticationError("approval.challenge_capacity")
            try:
                random_material = secrets.token_bytes(48)
            except Exception:
                raise ApprovalAuthenticationError("approval.random_unavailable") from None
            if not isinstance(random_material, bytes) or len(random_material) != 48:
                raise ApprovalAuthenticationError("approval.random_unavailable")
            challenge_id = (
                "ch-"
                + base64.urlsafe_b64encode(random_material[:16])
                .rstrip(b"=")
                .decode("ascii")
            )
            nonce = random_material[16:]
            if challenge_id in self._challenges:
                raise ApprovalAuthenticationError("approval.random_collision")
            expires = min(now + self._challenge_lifetime, grant.expires_at_monotonic)
            if not math.isfinite(expires) or expires <= now:
                raise ApprovalAuthenticationError("approval.challenge_expired")
            challenge = ApprovalChallenge(
                _issuer=_CHALLENGE_ISSUER,
                challenge_id=challenge_id,
                key_id=self._key_id,
                invocation_id=invocation_id,
                operation_digest=grant.operation_digest,
                challenge_material=nonce,
                issued_at_monotonic=now,
                expires_at_monotonic=expires,
            )
            self._challenges[challenge_id] = (challenge, grant)
            return challenge

    def authenticate(
        self, response: ApprovalDecisionResponse, *, owner_secret: bytes | bytearray | None = None
    ) -> AuthenticatedOwnerDecision:
        """Authenticate explicit owner intent; never authorize or dispatch."""
        if not isinstance(response, ApprovalDecisionResponse):
            raise ApprovalAuthenticationError("approval.response_malformed")
        if (
            not isinstance(response.challenge_id, str)
            or not isinstance(response.key_id, str)
            or not isinstance(response.invocation_id, str)
            or not isinstance(response.operation_digest, str)
            or type(response.decision) is not OwnerDecision
            or not isinstance(response.proof, bytes)
            or len(response.proof) != hashlib.sha256().digest_size
        ):
            raise ApprovalAuthenticationError("approval.response_malformed")

        with self._lock:
            entry = self._challenges.get(response.challenge_id)
            if entry is None:
                raise ApprovalAuthenticationError("approval.challenge_unknown")
            challenge, grant = entry
            now = self._read_clock()
            if now >= challenge.expires_at_monotonic:
                del self._challenges[challenge.challenge_id]
                raise ApprovalAuthenticationError("approval.challenge_expired")
            if (
                response.key_id != challenge.key_id
                or response.invocation_id != challenge.invocation_id
                or response.operation_digest != challenge.operation_digest
                or challenge.invocation_id != grant.invocation_id
                or challenge.operation_digest != grant.operation_digest
            ):
                raise ApprovalAuthenticationError("approval.binding_mismatch")

            try:
                pending = self._pending_registry.get(challenge.invocation_id)
                valid = (
                    pending.state is PendingInvocationState.PENDING
                    and grant.is_bound_to(pending, now)
                    and self._pending_registry.validate_current(challenge.invocation_id)
                )
                current = self._pending_registry.get(challenge.invocation_id) if valid else None
                valid = bool(
                    valid
                    and current is not None
                    and current.state is PendingInvocationState.PENDING
                    and grant.is_bound_to(current, self._read_clock())
                )
            except Exception:
                valid = False
            if not valid:
                raise ApprovalAuthenticationError("approval.binding_invalid")

            expected = self._proof_for(challenge, response.decision, owner_secret)
            if not hmac.compare_digest(expected, response.proof):
                raise ApprovalAuthenticationError("approval.proof_invalid")
            return AuthenticatedOwnerDecision(
                _issuer=_AUTHENTICATED_DECISION_ISSUER,
                challenge_id=challenge.challenge_id,
                key_id=challenge.key_id,
                invocation_id=challenge.invocation_id,
                operation_digest=challenge.operation_digest,
                decision=response.decision,
                authenticated_at_monotonic=now,
            )

    def consume_authenticated(self, assertion: AuthenticatedOwnerDecision) -> None:
        """Burn the exact challenge after the lifecycle owner accepts its result.

        Authentication alone remains repeatable evidence; only this trusted
        lifecycle operation consumes the challenge, and a second call fails.
        """
        if type(assertion) is not AuthenticatedOwnerDecision:
            raise ApprovalAuthenticationError("approval.assertion_untrusted")
        with self._lock:
            entry = self._challenges.get(assertion.challenge_id)
            if entry is None:
                raise ApprovalAuthenticationError("approval.challenge_consumed_or_unknown")
            challenge, grant = entry
            now = self._read_clock()
            if now >= challenge.expires_at_monotonic:
                del self._challenges[assertion.challenge_id]
                raise ApprovalAuthenticationError("approval.challenge_expired")
            if (
                assertion.key_id != challenge.key_id
                or assertion.invocation_id != challenge.invocation_id
                or assertion.operation_digest != challenge.operation_digest
                or assertion.challenge_id != challenge.challenge_id
                or grant.invocation_id != assertion.invocation_id
                or grant.operation_digest != assertion.operation_digest
            ):
                raise ApprovalAuthenticationError("approval.binding_mismatch")
            del self._challenges[assertion.challenge_id]

    def invalidate_invocation(self, invocation_id: str) -> None:
        """Revoke every outstanding challenge for one terminal invocation."""
        with self._lock:
            for challenge_id, (challenge, _grant) in tuple(self._challenges.items()):
                if challenge.invocation_id == invocation_id:
                    del self._challenges[challenge_id]

    def _proof_for(
        self, challenge: ApprovalChallenge, decision: OwnerDecision,
        owner_secret: bytes | bytearray | None = None,
    ) -> bytes:
        message = approval_proof_message(challenge, decision)
        secret = self._owner_secret if owner_secret is None else owner_secret
        if not isinstance(secret, (bytes, bytearray)) or len(secret) < 32:
            raise ApprovalAuthenticationError("approval.owner_authentication_unavailable")
        return hmac.new(secret, message, hashlib.sha256).digest()

    def _read_clock(self) -> float:
        try:
            now = float(self._clock())
        except Exception:
            raise ApprovalAuthenticationError("approval.clock_unavailable") from None
        if not math.isfinite(now):
            raise ApprovalAuthenticationError("approval.clock_invalid")
        return now

    def _prune_expired_locked(self, now: float) -> None:
        expired = [
            challenge_id for challenge_id, (challenge, _grant) in self._challenges.items()
            if now >= challenge.expires_at_monotonic
        ]
        for challenge_id in expired:
            del self._challenges[challenge_id]


def approval_proof_message(challenge: ApprovalChallenge, decision: OwnerDecision) -> bytes:
    """Canonical HMAC input shared with the future owner-facing adapter."""
    if type(decision) is not OwnerDecision:
        raise ApprovalAuthenticationError("approval.decision_invalid")
    payload = {
        "domain": "lila-owner-approval-decision-v1",
        "key_id": challenge.key_id,
        "challenge_id": challenge.challenge_id,
        "invocation_id": challenge.invocation_id,
        "operation_digest": challenge.operation_digest,
        "challenge_material": base64.urlsafe_b64encode(
            challenge.challenge_material
        ).decode("ascii"),
        "issued_at_monotonic": challenge.issued_at_monotonic.hex(),
        "expires_at_monotonic": challenge.expires_at_monotonic.hex(),
        "decision": decision.value,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _PROOF_DOMAIN + encoded
