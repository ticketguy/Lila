"""Fail-closed TLS configuration for the Windows Island API listener.

The WSL daemon owns its server key. Client certificates are issued separately
and validated against the configured client CA. No HTTP fallback is provided.
"""
from __future__ import annotations

import os
import re
import ssl
import stat
from pathlib import Path


class TransportConfigurationError(RuntimeError):
    pass


def allowed_client_fingerprints(environ=None) -> frozenset[str]:
    env = os.environ if environ is None else environ
    raw = env.get("LILA_TLS_ALLOWED_CLIENT_SHA256", "")
    pins = frozenset(item.strip().lower() for item in raw.split(",") if item.strip())
    if not pins or any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in pins):
        raise TransportConfigurationError("An explicit Island client certificate allowlist is required.")
    return pins


def build_mutual_tls_context(environ=None) -> ssl.SSLContext:
    env = os.environ if environ is None else environ
    names = ("LILA_TLS_SERVER_CERT", "LILA_TLS_SERVER_KEY", "LILA_TLS_CLIENT_CA")
    values = [env.get(name) for name in names]
    if any(not value for value in values):
        raise TransportConfigurationError("mTLS certificate configuration is incomplete.")
    cert_path, key_path, ca_path = map(Path, values)
    try:
        mode = stat.S_IMODE(key_path.stat().st_mode)
        if mode & 0o077:
            raise TransportConfigurationError("Daemon TLS private key permissions must be 0600 or stricter.")
        if not cert_path.is_file() or not ca_path.is_file():
            raise TransportConfigurationError("mTLS certificate or trust file is unavailable.")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        # Python/OpenSSL server TLS does not enable early data by default.
        context.options |= getattr(ssl, "OP_NO_COMPRESSION", 0)
        context.load_cert_chain(str(cert_path), str(key_path))
        context.load_verify_locations(cafile=str(ca_path))
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    except TransportConfigurationError:
        raise
    except (OSError, ssl.SSLError, ValueError):
        raise TransportConfigurationError("mTLS configuration could not be loaded.") from None
