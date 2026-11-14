"""Local interactive owner enrollment. No HTTP or remote bootstrap route."""
from __future__ import annotations

import getpass
from pathlib import Path

from .contracts import OIAConfig
from .service import OwnerProfileStore


def default_profile_path() -> Path:
    return OIAConfig.from_environment().credential_path


def main(argv: list[str] | None = None) -> int:
    import sys

    args = sys.argv[1:] if argv is None else argv
    store = OwnerProfileStore(default_profile_path())
    if args and args[0] == "spoken":
        return _enroll_spoken_cli(store)
    try:
        enroll_local(store)
    except OIAEnrollmentAlreadyPresent:
        print("An owner profile already exists. No changes were made.")
        return 2
    except Exception as exc:
        print(f"Owner enrollment failed ({getattr(exc, 'reason_code', str(exc))}).")
        return 1
    return 0


def _enroll_spoken_cli(store: OwnerProfileStore) -> int:
    if not store.enrolled:
        print("Typed Owner Phrase must be enrolled first (python -m src.oia.cli).")
        return 1
    from .service import OwnerAuthenticationService
    from src.vtalks.session import LilaVoice

    voice = LilaVoice()
    if not voice.private_asr_available:
        print("Private Parakeet ASR assets are unavailable; spoken approval cannot be enrolled.")
        return 1
    service = OwnerAuthenticationService(store)
    started = voice.start_private_session()
    print("Speak your enrolled Private Owner Phrase now, then press Enter to verify...")
    input()
    transcript = voice.stop_private_session(started["session_id"])
    try:
        provider = service.enroll_spoken_phrase(transcript, asr_available=True)
        print(f"Spoken Owner Phrase verified and enrolled ({provider.reason_code}).")
        return 0
    except Exception as exc:
        print(f"Spoken Owner Phrase verification failed ({getattr(exc, 'reason_code', str(exc))}).")
        return 1
    finally:
        transcript = ""


class OIAEnrollmentAlreadyPresent(RuntimeError):
    pass


def enroll_local(store: OwnerProfileStore):
    """Prompt only the process's local terminal; no network bootstrap exists."""
    if store.enrolled:
        raise OIAEnrollmentAlreadyPresent
    print("Before we begin, let's establish your owner identity. This will allow me to recognise your instructions and verify your authorisations.")
    print(
        "Choose any non-empty private phrase, in any language. One word or several is fine; "
        "the maximum is 512 characters. Avoid personal facts or phrases used elsewhere. "
        "The words approve, reject, and challenge are reserved."
    )
    phrase = getpass.getpass("Private Owner Phrase: ")
    confirmation = getpass.getpass("Confirm Owner Phrase: ")
    try:
        profile = store.create(phrase, confirmation)
    except Exception as exc:
        raise RuntimeError(getattr(exc, "reason_code", "owner_enrollment.failed")) from None
    finally:
        phrase = ""
        confirmation = ""
    print(f"Owner profile created ({profile.owner_id}).")
    print("Typed Owner Phrase is enrolled. Voice, speaker verification, biometrics, passkeys and recovery credentials are not enrolled by this setup.")
    print("The credential file contains an encrypted random owner secret. Keep a separate recovery plan; recovery credentials are not yet implemented.")
    return profile


if __name__ == "__main__":
    raise SystemExit(main())
