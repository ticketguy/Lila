"""Public Owner Identity and Authentication (OIA) subsystem contracts."""

from .contracts import (
    AuthenticationChallenge,
    AuthenticationEvidence,
    AuthenticationRequest,
    EvidenceConsumer,
    EnrollmentResult,
    OIAConfig,
    OIAError,
    OwnerPresenceSession,
    OwnerProfile,
    ProofMessageEncoder,
)
from .credentials import CredentialStorage, ProtectedOwnerSecret
from .providers import (
    OwnerAuthenticationProvider,
    PlatformAuthenticationProvider,
    ProviderAvailability,
    SpeakerVerificationProvider,
    discover_providers,
)
from .service import OwnerAuthenticationService, OwnerProfileStore

__all__ = [
    "AuthenticationChallenge",
    "AuthenticationEvidence",
    "AuthenticationRequest",
    "CredentialStorage",
    "EvidenceConsumer",
    "EnrollmentResult",
    "OIAConfig",
    "OIAError",
    "OwnerAuthenticationProvider",
    "OwnerAuthenticationService",
    "OwnerPresenceSession",
    "OwnerProfile",
    "OwnerProfileStore",
    "PlatformAuthenticationProvider",
    "ProofMessageEncoder",
    "ProtectedOwnerSecret",
    "ProviderAvailability",
    "SpeakerVerificationProvider",
    "discover_providers",
]
