"""Application-level execution context for shadow capability validation.

This context is an authority guard inside Lila's trusted application process.
It is not an operating-system security boundary and cannot contain malicious
code already executing in that process. The production Harness issues and
evaluates contexts in shadow mode after the single legacy authorization result
is produced. Legacy authorization remains authoritative for allow/deny and
dispatch; a context cannot authorize or execute a capability by itself.

The issuer requires an explicit lifetime policy supplied by trusted runtime
configuration; it has no implicit default timeout.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .capabilities import CapabilityManifest


ARGUMENT_DIGEST_DOMAIN = b"lila-execution-arguments\0typed-v1\0"
CONTEXT_SIGNATURE_DOMAIN = b"lila-execution-context\0hmac-sha256-v1\0"


class _FrozenList(tuple):
    """Immutable list snapshot whose type tag remains distinct from tuple."""


class ExecutionContextError(ValueError):
    """Invalid, unauthorized, expired, or mismatched execution context."""

    def __init__(self, message: str, reason_code: str = "context.invalid") -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _typed_value(value: Any, active: set[int] | None = None) -> Any:
    """Return a deterministic tagged JSON tree without coercing values."""
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        if not math.isfinite(value):
            raise ExecutionContextError("Non-finite float arguments are unsupported.")
        return ["float64", value.hex()]
    if type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ExecutionContextError("Argument strings must contain valid Unicode.") from exc
        return ["string", value]
    if isinstance(value, Path):
        path_value = os.fspath(value)
        if not isinstance(path_value, str):
            raise ExecutionContextError("Byte paths are unsupported.")
        return ["path", _typed_value(path_value)]

    if active is None:
        active = set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ExecutionContextError("Cyclic argument mappings are unsupported.")
        active.add(marker)
        try:
            pairs = []
            for key in value:
                if type(key) is not str:
                    raise ExecutionContextError("Argument object keys must be strings.")
                try:
                    key.encode("utf-8", errors="strict")
                except UnicodeEncodeError as exc:
                    raise ExecutionContextError("Argument keys must contain valid Unicode.") from exc
                pairs.append((key, value[key]))
            pairs.sort(key=lambda pair: pair[0])
            return ["mapping", [[key, _typed_value(item, active)] for key, item in pairs]]
        finally:
            active.remove(marker)
    if isinstance(value, _FrozenList):
        marker = id(value)
        if marker in active:
            raise ExecutionContextError("Cyclic argument sequences are unsupported.")
        active.add(marker)
        try:
            return ["list", [_typed_value(item, active) for item in value]]
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise ExecutionContextError("Cyclic argument sequences are unsupported.")
        active.add(marker)
        try:
            tag = "tuple" if isinstance(value, tuple) else "list"
            return [tag, [_typed_value(item, active) for item in value]]
        finally:
            active.remove(marker)
    raise ExecutionContextError(
        f"Unsupported argument value type: {type(value).__name__}."
    )


def typed_value_bytes(value: Any) -> bytes:
    """Encode supported values with explicit type tags and stable UTF-8 JSON."""
    tree = _typed_value(value)
    try:
        return json.dumps(
            tree, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ExecutionContextError("Arguments could not be encoded safely.") from exc


def argument_digest(arguments: Mapping[str, Any]) -> str:
    if not isinstance(arguments, Mapping):
        raise ExecutionContextError("Validated arguments must be an object.")
    return hashlib.sha256(
        ARGUMENT_DIGEST_DOMAIN + typed_value_bytes(arguments)
    ).hexdigest()


def _freeze(value: Any, active: set[int] | None = None) -> Any:
    """Defensively copy supported values into immutable snapshots."""
    if value is None or type(value) in (bool, int, float, str) or isinstance(value, Path):
        # Validate supported scalar values and finite floats now, at issuance.
        _typed_value(value)
        return value
    if active is None:
        active = set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ExecutionContextError("Cyclic argument mappings are unsupported.")
        active.add(marker)
        try:
            copied = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ExecutionContextError("Argument object keys must be strings.")
                _typed_value(key)
                copied[key] = _freeze(item, active)
            return MappingProxyType(copied)
        finally:
            active.remove(marker)
    if isinstance(value, _FrozenList):
        marker = id(value)
        if marker in active:
            raise ExecutionContextError("Cyclic argument sequences are unsupported.")
        active.add(marker)
        try:
            return _FrozenList(_freeze(item, active) for item in value)
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise ExecutionContextError("Cyclic argument sequences are unsupported.")
        active.add(marker)
        try:
            frozen_items = tuple(_freeze(item, active) for item in value)
            return _FrozenList(frozen_items) if isinstance(value, list) else frozen_items
        finally:
            active.remove(marker)
    raise ExecutionContextError(
        f"Unsupported argument value type: {type(value).__name__}."
    )


@dataclass(frozen=True)
class AuthorizationEvidence:
    """Immutable snapshot of a PermissionManager-shaped result tuple.

    This is application-supplied evidence, not an independently verifiable
    PermissionManager attestation. The context issuer authenticates the copied
    fields after issuance; it cannot prove who produced the input tuple.
    """

    allowed: bool
    reason: str
    execution_arguments: Mapping[str, Any]
    authorized_file_list_scope: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool) or not isinstance(self.reason, str):
            raise ExecutionContextError("Malformed permission result evidence.")
        frozen = _freeze(self.execution_arguments)
        if not isinstance(frozen, Mapping):
            raise ExecutionContextError("Permission execution arguments must be an object.")
        scope = _freeze(self.authorized_file_list_scope)
        object.__setattr__(self, "execution_arguments", frozen)
        object.__setattr__(self, "authorized_file_list_scope", scope)

    @classmethod
    def from_permission_result(
        cls, result: Any, validated_arguments: Mapping[str, Any]
    ) -> "AuthorizationEvidence":
        """Copy the exact `(allowed, reason, execution_args)` tuple supplied.

        This method does not call PermissionManager or verify the tuple's
        provenance. Its trusted caller must pass the result of the single
        authorization check for this invocation. It rejects unexpected return
        shapes and verifies that only the known file-list scope was added to
        the original argument mapping.
        """
        if not isinstance(result, tuple) or len(result) != 3:
            raise ExecutionContextError(
                "Permission result must be the existing three-item tuple."
            )
        allowed, reason, execution_args = result
        if not isinstance(allowed, bool) or not isinstance(reason, str):
            raise ExecutionContextError("Permission result has invalid decision fields.")
        if not isinstance(validated_arguments, Mapping) or not isinstance(execution_args, Mapping):
            raise ExecutionContextError("Permission arguments must be mappings.")

        original = dict(validated_arguments)
        if "_authorized_file_list_scope" in original:
            raise ExecutionContextError("Validated arguments use a reserved permission key.")
        returned = dict(execution_args)
        scope = returned.pop("_authorized_file_list_scope", None)
        if argument_digest(original) != argument_digest(returned):
            raise ExecutionContextError(
                "Permission execution arguments differ from validated arguments."
            )
        if scope is not None:
            if (
                not isinstance(scope, tuple)
                or len(scope) != 2
                or not isinstance(scope[0], Path)
                or not isinstance(scope[1], tuple)
                or any(not isinstance(item, Path) for item in scope[1])
            ):
                raise ExecutionContextError("Permission returned an unknown scope shape.")
        return cls(allowed, reason, execution_args, scope)


@dataclass(frozen=True)
class ExecutionContext:
    """HMAC-authenticated, immutable, single-invocation application data.

    The MAC verifies integrity relative to one issuer instance. It does not
    independently attest that PermissionManager produced the input result.
    """

    capability_id: str
    contract_version: str
    manifest_digest: str
    arguments: Mapping[str, Any]
    argument_digest: str
    authorization: AuthorizationEvidence
    invocation_id: str
    issuer_id: str
    issued_at_monotonic: float
    expires_at_monotonic: float
    _signature: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        arguments = _freeze(self.arguments)
        if not isinstance(arguments, Mapping):
            raise ExecutionContextError("Execution arguments must be an object.")
        object.__setattr__(self, "arguments", arguments)
        if argument_digest(arguments) != self.argument_digest:
            raise ExecutionContextError("Execution argument digest does not match its snapshot.")
        if not isinstance(self.authorization, AuthorizationEvidence):
            raise ExecutionContextError("Authorization evidence is required.")
        if not isinstance(self._signature, bytes):
            raise ExecutionContextError("Context signature is malformed.")


class ExecutionContextIssuer:
    """Trusted application component that signs contexts after authorization.

    `lifetime_seconds` is mandatory and must come from trusted configuration.
    This class has no production default and does not perform authorization.
    """

    def __init__(
        self,
        *,
        lifetime_seconds: float,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(lifetime_seconds, bool)
            or not isinstance(lifetime_seconds, (int, float))
            or not math.isfinite(lifetime_seconds)
            or lifetime_seconds <= 0
        ):
            raise ValueError("An explicit positive finite context lifetime is required.")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable.")
        self._lifetime_seconds = float(lifetime_seconds)
        self._clock = monotonic_clock
        self._issuer_id = uuid.uuid4().hex
        self._key = secrets.token_bytes(32)

    def issue(
        self,
        manifest: CapabilityManifest,
        validated_arguments: Mapping[str, Any],
        permission_result: Any,
    ) -> ExecutionContext:
        """Authenticate one caller-supplied result snapshot in a context.

        The trusted caller is responsible for supplying the result of exactly
        one PermissionManager check. The issuer validates shape and binds the
        supplied values, but cannot independently establish their provenance.
        """
        if not isinstance(manifest, CapabilityManifest):
            raise ExecutionContextError("A capability manifest is required.")
        if not isinstance(validated_arguments, Mapping):
            raise ExecutionContextError("Validated arguments must be an object.")
        arguments = _freeze(validated_arguments)
        if not isinstance(arguments, Mapping):
            raise ExecutionContextError("Validated arguments must be an object.")
        evidence = AuthorizationEvidence.from_permission_result(
            permission_result, arguments
        )
        issued = float(self._clock())
        if not math.isfinite(issued):
            raise ExecutionContextError("Monotonic clock returned an invalid value.")
        expires = issued + self._lifetime_seconds
        if not math.isfinite(expires) or expires <= issued:
            raise ExecutionContextError("Context deadline could not be represented safely.")
        args_digest = argument_digest(arguments)
        invocation_id = uuid.uuid4().hex
        unsigned = {
            "capability_id": manifest.capability_id,
            "contract_version": manifest.contract_version,
            "manifest_digest": manifest.manifest_digest,
            "arguments": arguments,
            "argument_digest": args_digest,
            "authorization": {
                "allowed": evidence.allowed,
                "reason": evidence.reason,
                "execution_arguments": evidence.execution_arguments,
                "authorized_file_list_scope": evidence.authorized_file_list_scope,
            },
            "invocation_id": invocation_id,
            "issuer_id": self._issuer_id,
            "issued_at_monotonic": issued,
            "expires_at_monotonic": expires,
        }
        signature = self._sign(unsigned)
        return ExecutionContext(
            capability_id=manifest.capability_id,
            contract_version=manifest.contract_version,
            manifest_digest=manifest.manifest_digest,
            arguments=arguments,
            argument_digest=args_digest,
            authorization=evidence,
            invocation_id=invocation_id,
            issuer_id=self._issuer_id,
            issued_at_monotonic=issued,
            expires_at_monotonic=expires,
            _signature=signature,
        )

    def _context_payload(self, context: ExecutionContext) -> bytes:
        evidence = context.authorization
        value = {
            "capability_id": context.capability_id,
            "contract_version": context.contract_version,
            "manifest_digest": context.manifest_digest,
            "arguments": context.arguments,
            "argument_digest": context.argument_digest,
            "authorization": {
                "allowed": evidence.allowed,
                "reason": evidence.reason,
                "execution_arguments": evidence.execution_arguments,
                "authorized_file_list_scope": evidence.authorized_file_list_scope,
            },
            "invocation_id": context.invocation_id,
            "issuer_id": context.issuer_id,
            "issued_at_monotonic": context.issued_at_monotonic,
            "expires_at_monotonic": context.expires_at_monotonic,
        }
        return typed_value_bytes(value)

    def _sign(self, value: Mapping[str, Any]) -> bytes:
        payload = typed_value_bytes(value)
        return hmac.new(
            self._key, CONTEXT_SIGNATURE_DOMAIN + payload, hashlib.sha256
        ).digest()

    def verify_authenticity(self, context: ExecutionContext) -> None:
        if not isinstance(context, ExecutionContext):
            raise ExecutionContextError(
                "An issued execution context is required.", "context.missing"
            )
        if context.issuer_id != self._issuer_id:
            raise ExecutionContextError(
                "Context was issued by a different authority.", "context.issuer_mismatch"
            )
        expected = hmac.new(
            self._key,
            CONTEXT_SIGNATURE_DOMAIN + self._context_payload(context),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected, context._signature):
            raise ExecutionContextError(
                "Execution context integrity check failed.", "context.signature_invalid"
            )

    def validate_for_permission_decision(
        self,
        context: ExecutionContext,
        manifest: CapabilityManifest | None,
        arguments: Mapping[str, Any],
    ) -> AuthorizationEvidence:
        """Verify signed authorization evidence without requiring it to allow.

        This supports shadow decisions for both permitted and denied calls. It
        does not invoke PermissionManager or reinterpret its reason text.
        """
        self.verify_authenticity(context)
        if not isinstance(context.authorization, AuthorizationEvidence):
            raise ExecutionContextError(
                "Authorization evidence is unavailable.", "authorization.evidence_unavailable"
            )
        if manifest is not None and not isinstance(manifest, CapabilityManifest):
            raise ExecutionContextError("A capability manifest is required.")
        if manifest is not None and (
            context.capability_id != manifest.capability_id
            or context.contract_version != manifest.contract_version
            or context.manifest_digest != manifest.manifest_digest
        ):
            raise ExecutionContextError(
                "Context capability identity or digest mismatched.",
                "context.manifest_mismatch",
            )
        if not isinstance(arguments, Mapping):
            raise ExecutionContextError(
                "Invocation arguments must be an object.", "context.arguments_invalid"
            )
        if argument_digest(arguments) != context.argument_digest:
            raise ExecutionContextError(
                "Invocation arguments do not match the context.",
                "context.arguments_mismatch",
            )
        if context.argument_digest != argument_digest(context.arguments):
            raise ExecutionContextError(
                "Context argument snapshot was altered.", "context.arguments_altered"
            )
        now = float(self._clock())
        if not math.isfinite(now) or now >= context.expires_at_monotonic:
            raise ExecutionContextError(
                "Execution context has expired.", "context.expired"
            )
        return context.authorization

    def validate_for_invocation(
        self,
        context: ExecutionContext,
        manifest: CapabilityManifest,
        arguments: Mapping[str, Any],
    ) -> None:
        evidence = self.validate_for_permission_decision(
            context, manifest, arguments
        )
        if not evidence.allowed:
            raise ExecutionContextError(
                "Permission result denied this invocation.", "authorization.denied"
            )

