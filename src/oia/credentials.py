"""Encrypted Owner Phrase credentials using Argon2id and AES-GCM.

No fallback cipher or plaintext credential format is accepted. Install the
dependencies in ``requirements-oia.txt`` before creating an owner profile.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import stat
import threading
import unicodedata
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol


_FORMAT = "lila-oia-credential-v1"
_AAD_DOMAIN = b"lila-oia-owner-secret-v1\x00"
_TIME_COST = 3
_MEMORY_COST_KIB = 65_536
_PARALLELISM = 2
_UMASK_LOCK = threading.RLock()


@contextmanager
def _restrictive_umask() -> Iterator[None]:
    """Protect credential creation without depending on Core storage code."""
    with _UMASK_LOCK:
        previous = os.umask(0o077)
        try:
            yield
        finally:
            os.umask(previous)


def _secure_private_directory(path: Path) -> Path:
    directory = Path(path)
    if directory.is_symlink():
        raise PermissionError("Owner credential directory must not be a symlink.")
    with _restrictive_umask():
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
    _require_mode(directory, 0o700, "directory")
    return directory


def _secure_private_file(path: Path) -> Path:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise PermissionError("Owner credential must be a regular file.")
    os.chmod(target, 0o600)
    _require_mode(target, 0o600, "file")
    return target


def _require_mode(path: Path, expected: int, kind: str) -> None:
    actual = stat.S_IMODE(path.stat().st_mode)
    if actual != expected:
        raise PermissionError(
            f"Unable to secure owner credential {kind}: "
            f"expected mode {expected:04o}, got {actual:04o}"
        )


class CredentialUnavailable(RuntimeError):
    """Required vetted credential-protection dependency is unavailable."""


class CredentialError(ValueError):
    """Credential could not be safely created, read or unlocked."""


@dataclass(frozen=True, repr=False)
class ProtectedOwnerSecret:
    owner_id: str
    key_id: str
    salt: bytes
    nonce: bytes
    ciphertext: bytes

    def __repr__(self) -> str:
        return f"ProtectedOwnerSecret(owner_id={self.owner_id!r}, key_id={self.key_id!r}, material=<redacted>)"


class CredentialStorage(Protocol):
    """Storage contract for protected owner credentials."""

    def exists(self) -> bool: ...

    def read(self) -> ProtectedOwnerSecret: ...

    def write_new(self, credential: ProtectedOwnerSecret) -> None: ...


def normalize_owner_phrase(value: str) -> str:
    """Exact comparison normalization: NFKC, casefold and whitespace fold only.

    Punctuation, accents and words are never removed, corrected or fuzzily
    matched. ASR punctuation differences therefore require another attempt.
    """
    if not isinstance(value, str):
        raise CredentialError("owner_phrase.invalid")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def validate_new_phrase(value: str) -> str:
    phrase = normalize_owner_phrase(value)
    words = phrase.split()
    if not phrase:
        raise CredentialError("owner_phrase.empty")
    if len(phrase) > 512:
        raise CredentialError("owner_phrase.too_long")
    if any(ord(ch) < 32 for ch in phrase):
        raise CredentialError("owner_phrase.invalid")
    if any(word in {"approve", "reject", "challenge"} for word in words):
        raise CredentialError("owner_phrase.reserved_word")
    return phrase


def _crypto():
    try:
        from argon2.low_level import Type, hash_secret_raw
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise CredentialUnavailable("owner_authentication.credential_provider_unavailable") from None
    return Type, hash_secret_raw, AESGCM


def crypto_available() -> bool:
    try:
        _crypto()
        return True
    except CredentialUnavailable:
        return False


def protect_owner_secret(phrase: str, owner_id: str, key_id: str) -> ProtectedOwnerSecret:
    Type, hash_secret_raw, AESGCM = _crypto()
    canonical = bytearray(validate_new_phrase(phrase).encode("utf-8"))
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(12)
    key = bytearray(hash_secret_raw(
        secret=bytes(canonical), salt=salt, time_cost=_TIME_COST,
        memory_cost=_MEMORY_COST_KIB, parallelism=_PARALLELISM,
        hash_len=32, type=Type.ID,
    ))
    owner_secret = bytearray(secrets.token_bytes(32))
    try:
        ciphertext = AESGCM(bytes(key)).encrypt(
            nonce, bytes(owner_secret), _aad(owner_id, key_id)
        )
        return ProtectedOwnerSecret(owner_id, key_id, salt, nonce, ciphertext)
    finally:
        _wipe(canonical)
        _wipe(key)
        _wipe(owner_secret)


def unlock_owner_secret(phrase: str, credential: ProtectedOwnerSecret) -> bytearray:
    Type, hash_secret_raw, AESGCM = _crypto()
    canonical = bytearray(normalize_owner_phrase(phrase).encode("utf-8"))
    if not canonical or len(canonical) > 2048:
        _wipe(canonical)
        raise CredentialError("owner_authentication.failed")
    key = bytearray(hash_secret_raw(
        secret=bytes(canonical), salt=credential.salt, time_cost=_TIME_COST,
        memory_cost=_MEMORY_COST_KIB, parallelism=_PARALLELISM,
        hash_len=32, type=Type.ID,
    ))
    try:
        return bytearray(AESGCM(bytes(key)).decrypt(
            credential.nonce, credential.ciphertext,
            _aad(credential.owner_id, credential.key_id),
        ))
    except Exception:
        raise CredentialError("owner_authentication.failed") from None
    finally:
        _wipe(canonical)
        _wipe(key)


def _wipe(value: bytearray) -> None:
    for index in range(len(value)):
        value[index] = 0


class OwnerCredentialFile:
    """Private JSON metadata containing only an encrypted random secret."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def write_new(self, credential: ProtectedOwnerSecret) -> None:
        payload = {
            "format": _FORMAT,
            "owner_id": credential.owner_id,
            "key_id": credential.key_id,
            "kdf": {"name": "argon2id", "time_cost": _TIME_COST,
                    "memory_cost_kib": _MEMORY_COST_KIB, "parallelism": _PARALLELISM},
            "cipher": "AES-256-GCM",
            "salt": _b64(credential.salt), "nonce": _b64(credential.nonce),
            "ciphertext": _b64(credential.ciphertext),
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        _secure_private_directory(self.path.parent)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = None
        created = False
        try:
            with _restrictive_umask():
                descriptor = os.open(self.path, flags, 0o600)
                created = True
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise PermissionError("Owner credential path is not a regular file.")
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = None
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            raise CredentialError("owner_profile.already_exists") from None
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            if created:
                try:
                    self.path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def read(self) -> ProtectedOwnerSecret:
        if not self.path.is_file() or self.path.is_symlink():
            raise CredentialError("owner_profile.not_found")
        _secure_private_file(self.path)
        try:
            if self.path.stat().st_size > 16_384:
                raise ValueError
            payload: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            if (payload.get("format") != _FORMAT or payload.get("cipher") != "AES-256-GCM"
                    or payload.get("kdf") != {"name": "argon2id", "time_cost": _TIME_COST,
                                                "memory_cost_kib": _MEMORY_COST_KIB,
                                                "parallelism": _PARALLELISM}):
                raise ValueError
            owner_id, key_id = payload["owner_id"], payload["key_id"]
            if not isinstance(owner_id, str) or not isinstance(key_id, str):
                raise ValueError
            salt, nonce, ciphertext = map(_unb64, (
                payload["salt"], payload["nonce"], payload["ciphertext"]
            ))
            if len(salt) != 16 or len(nonce) != 12 or len(ciphertext) != 48:
                raise ValueError
            return ProtectedOwnerSecret(owner_id, key_id, salt, nonce, ciphertext)
        except Exception:
            raise CredentialError("owner_profile.corrupt") from None


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _aad(owner_id: str, key_id: str) -> bytes:
    return _AAD_DOMAIN + owner_id.encode("utf-8") + b"\x00" + key_id.encode("utf-8")


def _unb64(value: object) -> bytes:
    if not isinstance(value, str):
        raise ValueError("invalid credential encoding")
    decoded = base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    if _b64(decoded) != value:
        raise ValueError("non-canonical credential encoding")
    return decoded
