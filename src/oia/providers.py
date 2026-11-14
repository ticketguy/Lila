"""Replaceable owner-authentication provider contracts and honest discovery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ProviderAvailability:
    provider_id: str
    available: bool
    enrolled: bool
    reason_code: str


class OwnerAuthenticationProvider(Protocol):
    provider_id: str

    def availability(self) -> ProviderAvailability: ...


class SpeakerVerificationProvider(Protocol):
    """Optional local voice-identity interface; implementations need evaluation."""
    provider_id: str
    model_identity: str

    def extract_embedding(self, audio_path: Path) -> Sequence[float]: ...

    def similarity(self, enrolled_template: Sequence[float],
                   audio_path: Path) -> float: ...


class PlatformAuthenticationProvider(Protocol):
    """Replaceable OS biometric or FIDO2 challenge provider contract."""
    provider_id: str

    def availability(self) -> ProviderAvailability: ...

    def authenticate(self, challenge: bytes) -> bytes: ...


class UnsupportedProvider:
    """Truthful unavailable provider; has no method that can authenticate."""

    def __init__(self, provider_id: str, reason_code: str = "provider.unsupported"):
        self.provider_id = provider_id
        self._reason_code = reason_code

    def availability(self) -> ProviderAvailability:
        return ProviderAvailability(self.provider_id, False, False, self._reason_code)


def discover_providers() -> tuple[ProviderAvailability, ...]:
    """Report tested implementation availability, never guessed hardware."""
    from .credentials import crypto_available

    phrase_available = crypto_available()
    return (
        ProviderAvailability("typed_owner_phrase", phrase_available, False,
                             "provider.available" if phrase_available else "provider.crypto_dependencies_unavailable"),
        ProviderAvailability("spoken_owner_phrase", False, False, "provider.live_owner_asr_unverified"),
        ProviderAvailability("speaker_verification", False, False, "provider.model_not_configured"),
        UnsupportedProvider("os_biometrics").availability(),
        UnsupportedProvider("fido2_security_key").availability(),
        UnsupportedProvider("pin_fallback").availability(),
        UnsupportedProvider("recovery_credential").availability(),
    )
