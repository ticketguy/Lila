"""Trusted, in-memory records for policy results awaiting owner approval.

This module defines application-level contracts only. It is not an OS security
boundary, approval protocol, persistence layer, or execution dispatcher. A
trusted Harness caller supplies a typed PolicyEvaluationResult; model-shaped
data is never loaded into authoritative records.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import re
import threading
import time
import uuid
from typing import Any

from .capabilities import CapabilityManifest, CapabilityRegistry
from .execution_context import _freeze, argument_digest, typed_value_bytes
from .permissions import PolicyDisposition, PolicyEvaluationResult


_PENDING_ISSUER = object()
_TARGET_ISSUER = object()
_POLICY_REVISION_UNAVAILABLE = object()
_TARGET_DIGEST_DOMAIN = b"lila-pending-target-binding\0v1\0"
_RESOLVER_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,99}$")
_INVOCATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PendingInvocationError(ValueError):
    """A pending invocation could not be constructed or validated safely."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code


class PendingInvocationState(str, Enum):
    PENDING = "pending"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
    CONSUMED = "consumed"


class TargetBinding:
    """Immutable target identity produced by a registered trusted resolver.

    `identity` is an application-defined mapping. Its interpretation belongs
    to the resolver identified by `resolver_id`; paths, devices, processes, and
    remote resources therefore need not share one target representation.
    """

    __slots__ = (
        "capability_id", "contract_version", "manifest_digest", "resolver_id",
        "target_kind", "identity", "digest", "_sealed",
    )

    def __init__(
        self,
        *,
        _issuer=None,
        capability_id: str,
        contract_version: str,
        manifest_digest: str,
        resolver_id: str,
        target_kind: str,
        identity: Mapping[str, Any],
    ) -> None:
        if _issuer is not _TARGET_ISSUER:
            raise TypeError("Target bindings can only be created by a trusted resolver registry.")
        if not isinstance(identity, Mapping) or not identity:
            raise PendingInvocationError(
                "target.identity_missing", "A non-empty target identity mapping is required."
            )
        if not all(isinstance(item, str) and item for item in (
            capability_id, contract_version, manifest_digest, resolver_id, target_kind,
        )):
            raise PendingInvocationError(
                "target.identity_invalid", "Target identity metadata is invalid."
            )
        frozen = _freeze(identity)
        if not isinstance(frozen, Mapping) or not frozen:
            raise PendingInvocationError(
                "target.identity_missing", "A non-empty target identity mapping is required."
            )
        payload = {
            "capability_id": capability_id,
            "contract_version": contract_version,
            "manifest_digest": manifest_digest,
            "resolver_id": resolver_id,
            "target_kind": target_kind,
            "identity": frozen,
        }
        digest = hashlib.sha256(
            _TARGET_DIGEST_DOMAIN + typed_value_bytes(payload)
        ).hexdigest()
        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "contract_version", contract_version)
        object.__setattr__(self, "manifest_digest", manifest_digest)
        object.__setattr__(self, "resolver_id", resolver_id)
        object.__setattr__(self, "target_kind", target_kind)
        object.__setattr__(self, "identity", frozen)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("TargetBinding is immutable.")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (
            "TargetBinding(capability_id={!r}, contract_version={!r}, "
            "resolver_id={!r}, target_kind={!r}, digest={!r})"
        ).format(
            self.capability_id, self.contract_version, self.resolver_id,
            self.target_kind, self.digest,
        )


@dataclass(frozen=True, slots=True)
class _TargetRule:
    resolver_id: str
    target_required: bool
    target_kind: str | None
    resolver: Callable[[CapabilityManifest, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None] | None


class TargetBindingResolverRegistry:
    """Instance-owned, explicit target semantics for capability versions.

    A missing rule is an error, not an assumption that the operation has no
    target. Resolvers must be trusted, read-only functions; they must not call
    a capability handler or perform the operation being described.
    """

    def __init__(self) -> None:
        self._rules: dict[tuple[str, str, str], _TargetRule] = {}
        self._lock = threading.RLock()

    def register(
        self,
        manifest: CapabilityManifest,
        *,
        resolver_id: str,
        target_required: bool,
        target_kind: str | None = None,
        resolver: Callable[[CapabilityManifest, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
    ) -> None:
        if not isinstance(manifest, CapabilityManifest):
            raise TypeError("A CapabilityManifest is required.")
        if not isinstance(resolver_id, str) or not _RESOLVER_ID_RE.fullmatch(resolver_id):
            raise ValueError("resolver_id must be a stable lowercase identifier.")
        if type(target_required) is not bool:
            raise TypeError("target_required must be a bool.")
        if target_required:
            if not isinstance(target_kind, str) or not target_kind.strip():
                raise ValueError("A target-required rule needs an explicit target_kind.")
            if not callable(resolver):
                raise ValueError("A target-required rule needs a trusted resolver.")
        elif target_kind is not None or resolver is not None:
            raise ValueError("A target-not-required rule cannot have a target resolver.")

        key = (manifest.capability_id, manifest.contract_version, manifest.manifest_digest)
        rule = _TargetRule(resolver_id, target_required, target_kind, resolver)
        with self._lock:
            if key in self._rules:
                raise ValueError("Target semantics are already registered for this manifest.")
            self._rules[key] = rule

    def _resolve(
        self,
        manifest: CapabilityManifest,
        validated_arguments: Mapping[str, Any],
        execution_arguments: Mapping[str, Any],
    ) -> tuple[str, bool, TargetBinding | None]:
        key = (manifest.capability_id, manifest.contract_version, manifest.manifest_digest)
        with self._lock:
            rule = self._rules.get(key)
        if rule is None:
            raise PendingInvocationError(
                "target.binding_policy_unavailable",
                "No explicit target-binding rule exists for this capability version.",
            )
        if not rule.target_required:
            return rule.resolver_id, False, None
        try:
            identity = rule.resolver(manifest, validated_arguments, execution_arguments)
        except Exception:
            raise PendingInvocationError(
                "target.resolution_failed", "The trusted target resolver failed."
            ) from None
        if not isinstance(identity, Mapping) or not identity:
            raise PendingInvocationError(
                "target.identity_unavailable",
                "The required target could not be established reliably.",
            )
        binding = TargetBinding(
            _issuer=_TARGET_ISSUER,
            capability_id=manifest.capability_id,
            contract_version=manifest.contract_version,
            manifest_digest=manifest.manifest_digest,
            resolver_id=rule.resolver_id,
            target_kind=rule.target_kind,
            identity=identity,
        )
        return rule.resolver_id, True, binding


@dataclass(frozen=True, slots=True, init=False, repr=False)
class PendingInvocation:
    """Deeply immutable application-level record for one pending operation."""

    invocation_id: str
    capability_id: str
    contract_version: str
    manifest_digest: str
    policy_result: PolicyEvaluationResult
    validated_arguments: Mapping[str, Any]
    validated_argument_digest: str
    execution_arguments: Mapping[str, Any]
    execution_argument_digest: str
    target_resolver_id: str
    target_required: bool
    target_binding: TargetBinding | None
    created_at_monotonic: float
    expires_at_monotonic: float
    state: PendingInvocationState
    state_changed_at_monotonic: float
    state_reason_code: str

    def __init__(self, *, _issuer=None, **values) -> None:
        if _issuer is not _PENDING_ISSUER:
            raise TypeError("PendingInvocation records can only be issued by the trusted registry.")
        self._assign_validated(values)

    def _assign_validated(self, values: Mapping[str, Any]) -> None:
        policy_result = values["policy_result"]
        if type(policy_result) is not PolicyEvaluationResult:
            raise PendingInvocationError(
                "policy.result_untrusted", "A typed trusted PolicyEvaluationResult is required."
            )
        if policy_result.disposition is not PolicyDisposition.REQUIRE_APPROVAL:
            raise PendingInvocationError(
                "policy.not_approval_required",
                "Only REQUIRE_APPROVAL policy results can become pending invocations.",
            )
        if policy_result.capability_id != values["capability_id"]:
            raise PendingInvocationError("manifest.capability_mismatch", "Capability ID binding mismatch.")
        if policy_result.contract_version != values["contract_version"]:
            raise PendingInvocationError("manifest.version_mismatch", "Capability version binding mismatch.")
        if policy_result.manifest_digest != values["manifest_digest"]:
            raise PendingInvocationError("manifest.digest_mismatch", "Manifest digest binding mismatch.")

        validated = _freeze(values["validated_arguments"])
        execution = _freeze(values["execution_arguments"])
        if not isinstance(validated, Mapping) or not isinstance(execution, Mapping):
            raise PendingInvocationError("arguments.invalid", "Argument snapshots must be mappings.")
        validated_digest = argument_digest(validated)
        execution_digest = argument_digest(execution)
        if validated_digest != values["validated_argument_digest"]:
            raise PendingInvocationError("arguments.digest_mismatch", "Validated argument digest mismatch.")
        if execution_digest != values["execution_argument_digest"]:
            raise PendingInvocationError("execution_arguments.digest_mismatch", "Execution argument digest mismatch.")
        if policy_result.arguments is None or policy_result.argument_digest != validated_digest:
            raise PendingInvocationError("arguments.policy_binding_mismatch", "Arguments do not match policy evidence.")
        if argument_digest(policy_result.arguments) != validated_digest:
            raise PendingInvocationError("arguments.policy_snapshot_invalid", "Policy argument snapshot is inconsistent.")
        if policy_result.execution_arguments is None or policy_result.execution_argument_digest != execution_digest:
            raise PendingInvocationError(
                "execution_arguments.policy_binding_mismatch",
                "Execution arguments do not match policy evidence.",
            )
        if argument_digest(policy_result.execution_arguments) != execution_digest:
            raise PendingInvocationError(
                "execution_arguments.policy_snapshot_invalid",
                "Policy execution-argument snapshot is inconsistent.",
            )

        target_required = values["target_required"]
        binding = values["target_binding"]
        if type(target_required) is not bool:
            raise PendingInvocationError("target.rule_invalid", "Target requirement must be explicit.")
        if target_required != (binding is not None):
            raise PendingInvocationError(
                "target.binding_missing", "Target requirement and binding disagree."
            )
        if binding is not None and (
            not isinstance(binding, TargetBinding)
            or binding.capability_id != values["capability_id"]
            or binding.contract_version != values["contract_version"]
            or binding.manifest_digest != values["manifest_digest"]
            or binding.resolver_id != values["target_resolver_id"]
        ):
            raise PendingInvocationError("target.binding_mismatch", "Target binding identity mismatch.")

        created = values["created_at_monotonic"]
        expires = values["expires_at_monotonic"]
        state_changed = values["state_changed_at_monotonic"]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (created, expires, state_changed)
        ) or expires <= created or state_changed < created:
            raise PendingInvocationError("invocation.time_invalid", "Invocation times are invalid.")
        if not isinstance(values["state"], PendingInvocationState):
            raise PendingInvocationError("invocation.state_invalid", "Invocation state is invalid.")
        if not isinstance(values["state_reason_code"], str) or not values["state_reason_code"]:
            raise PendingInvocationError("invocation.reason_invalid", "A lifecycle reason code is required.")

        for name in (
            "invocation_id", "capability_id", "contract_version", "manifest_digest",
            "validated_argument_digest", "execution_argument_digest", "target_resolver_id",
        ):
            value = values[name]
            if not isinstance(value, str) or not value:
                raise PendingInvocationError("invocation.identity_invalid", f"{name} is required.")

        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "validated_arguments", validated)
        object.__setattr__(self, "execution_arguments", execution)

    def __repr__(self) -> str:
        """Show lifecycle and digests without exposing argument payloads."""
        return (
            "PendingInvocation(invocation_id={!r}, capability_id={!r}, "
            "contract_version={!r}, state={!r}, validated_argument_digest={!r}, "
            "execution_argument_digest={!r}, target_digest={!r})"
        ).format(
            self.invocation_id,
            self.capability_id,
            self.contract_version,
            self.state.value,
            self.validated_argument_digest,
            self.execution_argument_digest,
            self.target_binding.digest if self.target_binding is not None else None,
        )


class PendingInvocationRegistry:
    """Bounded, instance-owned registry; it never dispatches capabilities."""

    def __init__(
        self,
        *,
        capabilities: CapabilityRegistry,
        target_resolvers: TargetBindingResolverRegistry,
        lifetime_seconds: float,
        capacity: int = 128,
        monotonic_clock: Callable[[], float] = time.monotonic,
        invocation_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        if not isinstance(capabilities, CapabilityRegistry):
            raise TypeError("An instance-owned CapabilityRegistry is required.")
        if not isinstance(target_resolvers, TargetBindingResolverRegistry):
            raise TypeError("An instance-owned TargetBindingResolverRegistry is required.")
        if (
            isinstance(lifetime_seconds, bool)
            or not isinstance(lifetime_seconds, (int, float))
            or not math.isfinite(lifetime_seconds)
            or lifetime_seconds <= 0
        ):
            raise ValueError("An explicit positive finite pending lifetime is required.")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Registry capacity must be a positive integer.")
        if not callable(monotonic_clock) or not callable(invocation_id_factory):
            raise TypeError("Clock and invocation ID factory must be callable.")
        self._capabilities = capabilities
        self._target_resolvers = target_resolvers
        self._lifetime_seconds = float(lifetime_seconds)
        self._capacity = capacity
        self._clock = monotonic_clock
        self._invocation_id_factory = invocation_id_factory
        self._records: dict[str, PendingInvocation] = {}
        self._grants_issued: set[str] = set()
        # Keep an O(1)-space monotonic sequence so a terminal record can be
        # released without allowing a stale invocation ID to name a new one.
        self._next_invocation_sequence = 1
        self._lock = threading.RLock()

    def create_pending(self, policy_result: PolicyEvaluationResult) -> PendingInvocation:
        """Capture and register one typed REQUIRE_APPROVAL result; never execute it."""
        if type(policy_result) is not PolicyEvaluationResult:
            raise PendingInvocationError(
                "policy.result_untrusted", "A typed trusted PolicyEvaluationResult is required."
            )
        if policy_result.disposition is not PolicyDisposition.REQUIRE_APPROVAL:
            raise PendingInvocationError(
                "policy.not_approval_required",
                "Only REQUIRE_APPROVAL policy results can be registered.",
            )
        identity = (
            policy_result.capability_id,
            policy_result.contract_version,
            policy_result.manifest_digest,
        )
        if not all(isinstance(item, str) and item for item in identity):
            raise PendingInvocationError("manifest.identity_missing", "Policy capability identity is incomplete.")
        try:
            manifest = self._capabilities.get(identity[0], identity[1])
        except KeyError:
            raise PendingInvocationError("manifest.unknown", "Policy capability is not registered.") from None
        if manifest.manifest_digest != identity[2]:
            raise PendingInvocationError("manifest.digest_mismatch", "Policy manifest digest is not current.")
        if not isinstance(policy_result.arguments, Mapping) or not isinstance(
            policy_result.execution_arguments, Mapping
        ):
            raise PendingInvocationError("arguments.missing", "Policy argument snapshots are required.")
        validated_digest = argument_digest(policy_result.arguments)
        execution_digest = argument_digest(policy_result.execution_arguments)
        if policy_result.argument_digest != validated_digest:
            raise PendingInvocationError("arguments.digest_mismatch", "Policy argument digest mismatch.")
        if policy_result.execution_argument_digest != execution_digest:
            raise PendingInvocationError(
                "execution_arguments.digest_mismatch", "Policy execution-argument digest mismatch."
            )

        resolver_id, target_required, binding = self._target_resolvers._resolve(
            manifest, policy_result.arguments, policy_result.execution_arguments
        )
        created = self._read_clock()
        expires = created + self._lifetime_seconds
        if not math.isfinite(expires) or expires <= created:
            raise PendingInvocationError("invocation.time_invalid", "Pending deadline is invalid.")
        with self._lock:
            if len(self._records) >= self._capacity:
                raise PendingInvocationError("registry.capacity_reached", "Pending registry is at capacity.")
            try:
                id_prefix = self._invocation_id_factory()
            except Exception:
                raise PendingInvocationError(
                    "invocation.id_unavailable", "Trusted ID factory failed."
                ) from None
            sequence = self._next_invocation_sequence
            if not isinstance(id_prefix, str) or not _INVOCATION_ID_RE.fullmatch(id_prefix):
                raise PendingInvocationError("invocation.id_invalid", "Trusted ID factory returned an invalid ID.")
            invocation_id = f"{id_prefix}-{sequence}"
            if not _INVOCATION_ID_RE.fullmatch(invocation_id):
                raise PendingInvocationError("invocation.id_invalid", "Trusted ID factory returned an invalid ID.")
            record = PendingInvocation(
                _issuer=_PENDING_ISSUER,
                invocation_id=invocation_id,
                capability_id=manifest.capability_id,
                contract_version=manifest.contract_version,
                manifest_digest=manifest.manifest_digest,
                policy_result=policy_result,
                validated_arguments=policy_result.arguments,
                validated_argument_digest=validated_digest,
                execution_arguments=policy_result.execution_arguments,
                execution_argument_digest=execution_digest,
                target_resolver_id=resolver_id,
                target_required=target_required,
                target_binding=binding,
                created_at_monotonic=created,
                expires_at_monotonic=expires,
                state=PendingInvocationState.PENDING,
                state_changed_at_monotonic=created,
                state_reason_code="invocation.created",
            )
            self._next_invocation_sequence += 1
            self._records[invocation_id] = record
        return record

    def get(self, invocation_id: str) -> PendingInvocation:
        """Return the stored immutable record without changing its lifecycle."""
        with self._lock:
            try:
                return self._records[invocation_id]
            except KeyError:
                raise PendingInvocationError("invocation.unknown", "Pending invocation was not found.") from None

    def snapshot(self) -> tuple[PendingInvocation, ...]:
        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def cancel(self, invocation_id: str) -> PendingInvocation:
        return self._transition_pending(invocation_id, PendingInvocationState.CANCELLED, "invocation.cancelled")

    def invalidate(self, invocation_id: str, reason_code: str = "invocation.assumptions_invalid") -> PendingInvocation:
        if not isinstance(reason_code, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,99}", reason_code):
            raise ValueError("Invalid invalidation reason code.")
        return self._transition_pending(invocation_id, PendingInvocationState.INVALIDATED, reason_code)

    def expire(self, invocation_id: str) -> PendingInvocation:
        with self._lock:
            record = self._get_locked(invocation_id)
            self._require_pending(record)
            now = self._read_clock()
            if now < record.expires_at_monotonic:
                raise PendingInvocationError("invocation.not_expired", "Invocation lifetime has not elapsed.")
            return self._store_transition_locked(record, PendingInvocationState.EXPIRED, "invocation.expired", now)

    def expire_due(self) -> tuple[PendingInvocation, ...]:
        """Expire all elapsed pending records; terminal records are retained."""
        with self._lock:
            now = self._read_clock()
            expired = []
            for record in tuple(self._records.values()):
                if record.state is PendingInvocationState.PENDING and now >= record.expires_at_monotonic:
                    expired.append(self._store_transition_locked(
                        record, PendingInvocationState.EXPIRED, "invocation.expired", now
                    ))
            return tuple(expired)

    def release_terminal(self, invocation_id: str) -> None:
        """Explicitly release a terminal record to recover bounded capacity.

        Pending records are never evicted. The monotonic ID sequence is
        retained, so releasing a record cannot make its identifier reusable.
        Callers holding the returned immutable record still see its terminal
        state; registry lookup after release reports that it is no longer held.
        """
        with self._lock:
            record = self._get_locked(invocation_id)
            if record.state is PendingInvocationState.PENDING:
                raise PendingInvocationError(
                    "invocation.pending", "A pending invocation cannot be released."
                )
            del self._records[invocation_id]
            self._grants_issued.discard(invocation_id)

    def _issue_approval_grant(
        self,
        invocation_id: str,
        *,
        current_policy_revision: str | None | object = _POLICY_REVISION_UNAVAILABLE,
    ):
        """Trusted internal issuance path for one exact pending invocation.

        This private Harness boundary only creates a bound data record. It does
        not represent a human decision, enable dispatch, or consume the grant.
        A later authorization boundary must revalidate current policy when the
        pending result had no authoritative revision.
        """
        with self._lock:
            record = self._get_locked(invocation_id)
            self._require_pending(record)
            now = self._read_clock()
            if now >= record.expires_at_monotonic:
                self._store_transition_locked(
                    record, PendingInvocationState.EXPIRED, "invocation.expired", now
                )
                raise PendingInvocationError("invocation.expired", "Pending invocation has expired.")
            if invocation_id in self._grants_issued:
                raise PendingInvocationError(
                    "grant.already_issued", "A grant was already issued for this pending invocation."
                )
            if not self.validate_current(
                invocation_id, current_policy_revision=current_policy_revision
            ):
                raise PendingInvocationError(
                    "grant.pending_invalid", "Pending invocation bindings are no longer current."
                )
            # validate_current is called under this RLock, so cancellation or a
            # second issuance cannot race between validation and association.
            record = self._get_locked(invocation_id)
            now = self._read_clock()
            if now >= record.expires_at_monotonic:
                self._store_transition_locked(
                    record, PendingInvocationState.EXPIRED, "invocation.expired", now
                )
                raise PendingInvocationError("invocation.expired", "Pending invocation has expired.")
            from .approval_grants import ApprovalGrant, _GRANT_ISSUER

            grant = ApprovalGrant._from_pending(
                record, created_at_monotonic=now, _issuer=_GRANT_ISSUER
            )
            self._grants_issued.add(invocation_id)
            return grant

    def validate_current(
        self,
        invocation_id: str,
        *,
        current_policy_revision: str | None | object = _POLICY_REVISION_UNAVAILABLE,
    ) -> bool:
        """Recheck lifetime, manifest and target; invalidate if bindings drift.

        This foundation cannot re-evaluate current policy because the existing
        policy contract has no authoritative revision/epoch. If a future record
        carries one, the caller must provide the current revision explicitly.
        """
        with self._lock:
            record = self._get_locked(invocation_id)
            self._require_pending(record)
            now = self._read_clock()
            if now >= record.expires_at_monotonic:
                self._store_transition_locked(record, PendingInvocationState.EXPIRED, "invocation.expired", now)
                return False
            if record.policy_result.policy_revision is not None:
                if (
                    current_policy_revision is _POLICY_REVISION_UNAVAILABLE
                    or current_policy_revision != record.policy_result.policy_revision
                ):
                    self._store_transition_locked(
                        record, PendingInvocationState.INVALIDATED, "invocation.policy_revision_changed", now
                    )
                    return False

            try:
                manifest = self._capabilities.get(record.capability_id, record.contract_version)
                if manifest.manifest_digest != record.manifest_digest:
                    raise PendingInvocationError("manifest.digest_changed", "Manifest changed.")
                resolver_id, target_required, binding = self._target_resolvers._resolve(
                    manifest, record.validated_arguments, record.execution_arguments
                )
                if (
                    resolver_id != record.target_resolver_id
                    or target_required != record.target_required
                    or not self._same_target(binding, record.target_binding)
                ):
                    raise PendingInvocationError("target.binding_changed", "Target binding changed.")
            except (KeyError, PendingInvocationError):
                self._store_transition_locked(
                    record, PendingInvocationState.INVALIDATED, "invocation.binding_invalidated", now
                )
                return False
            return True

    def _consume_exact_grant(self, invocation_id, grant, fresh_policy_result, assertion):
        """Atomically consume the matching pending record under registry lock.

        This private transition is reserved for the trusted approval lifecycle;
        it does not itself authenticate a decision or dispatch a capability.
        """
        from .approval_grants import ApprovalGrant
        from .approval_authentication import AuthenticatedOwnerDecision, OwnerDecision

        with self._lock:
            record = self._get_locked(invocation_id)
            self._require_pending(record)
            now = self._read_clock()
            if now >= record.expires_at_monotonic:
                return self._store_transition_locked(
                    record, PendingInvocationState.EXPIRED, "invocation.expired", now
                )
            if (
                not isinstance(grant, ApprovalGrant)
                or type(assertion) is not AuthenticatedOwnerDecision
                or assertion.decision is not OwnerDecision.APPROVE
                or assertion.invocation_id != invocation_id
                or assertion.operation_digest != grant.operation_digest
                or invocation_id not in self._grants_issued
                or not grant.is_bound_to(record, now)
                or not grant.validate_current_policy(fresh_policy_result)
            ):
                raise PendingInvocationError(
                    "invocation.grant_binding_invalid", "Grant or fresh policy did not match this invocation."
                )
            if not self.validate_current(invocation_id):
                return self._get_locked(invocation_id)
            record = self._get_locked(invocation_id)
            now = self._read_clock()
            if now >= record.expires_at_monotonic:
                return self._store_transition_locked(
                    record, PendingInvocationState.EXPIRED, "invocation.expired", now
                )
            return self._store_transition_locked(
                record, PendingInvocationState.CONSUMED, "invocation.approval_consumed", now
            )

    def _transition_pending(
        self, invocation_id: str, state: PendingInvocationState, reason_code: str
    ) -> PendingInvocation:
        with self._lock:
            record = self._get_locked(invocation_id)
            self._require_pending(record)
            now = self._read_clock()
            if now >= record.expires_at_monotonic:
                self._store_transition_locked(record, PendingInvocationState.EXPIRED, "invocation.expired", now)
                raise PendingInvocationError("invocation.expired", "Pending invocation has expired.")
            return self._store_transition_locked(record, state, reason_code, now)

    def _store_transition_locked(
        self,
        record: PendingInvocation,
        state: PendingInvocationState,
        reason_code: str,
        now: float,
    ) -> PendingInvocation:
        if record.state is not PendingInvocationState.PENDING:
            raise PendingInvocationError("invocation.terminal", "Terminal invocation state cannot transition.")
        if state not in {
            PendingInvocationState.CANCELLED,
            PendingInvocationState.EXPIRED,
            PendingInvocationState.INVALIDATED,
            PendingInvocationState.CONSUMED,
        }:
            raise PendingInvocationError("invocation.transition_invalid", "Lifecycle transition is not available.")
        updated = PendingInvocation(
            _issuer=_PENDING_ISSUER,
            invocation_id=record.invocation_id,
            capability_id=record.capability_id,
            contract_version=record.contract_version,
            manifest_digest=record.manifest_digest,
            policy_result=record.policy_result,
            validated_arguments=record.validated_arguments,
            validated_argument_digest=record.validated_argument_digest,
            execution_arguments=record.execution_arguments,
            execution_argument_digest=record.execution_argument_digest,
            target_resolver_id=record.target_resolver_id,
            target_required=record.target_required,
            target_binding=record.target_binding,
            created_at_monotonic=record.created_at_monotonic,
            expires_at_monotonic=record.expires_at_monotonic,
            state=state,
            state_changed_at_monotonic=now,
            state_reason_code=reason_code,
        )
        self._records[record.invocation_id] = updated
        return updated

    def _get_locked(self, invocation_id: str) -> PendingInvocation:
        if not isinstance(invocation_id, str):
            raise TypeError("invocation_id must be a string.")
        try:
            return self._records[invocation_id]
        except KeyError:
            raise PendingInvocationError("invocation.unknown", "Pending invocation was not found.") from None

    @staticmethod
    def _require_pending(record: PendingInvocation) -> None:
        if record.state is not PendingInvocationState.PENDING:
            raise PendingInvocationError("invocation.terminal", "Invocation is already terminal.")

    @staticmethod
    def _same_target(left: TargetBinding | None, right: TargetBinding | None) -> bool:
        if left is None or right is None:
            return left is right
        return (
            left.capability_id == right.capability_id
            and left.contract_version == right.contract_version
            and left.manifest_digest == right.manifest_digest
            and left.resolver_id == right.resolver_id
            and left.target_kind == right.target_kind
            and left.digest == right.digest
        )

    def _read_clock(self) -> float:
        try:
            value = float(self._clock())
        except Exception:
            raise PendingInvocationError("invocation.clock_unavailable", "Monotonic clock is unavailable.") from None
        if not math.isfinite(value):
            raise PendingInvocationError("invocation.clock_invalid", "Monotonic clock returned an invalid value.")
        return value
