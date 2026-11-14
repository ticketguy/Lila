"""Typed, non-authoritative shadow decisions from existing M1B evidence.

This module does not call PermissionManager, execute tools, issue approval, or
change policy. Decisions describe the caller-supplied authorization-result
snapshot captured in an issuer-authenticated ExecutionContext. Its HMAC proves
context integrity under that issuer key, not independent PermissionManager
provenance. Decisions are always marked ``not_executed``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Any, Mapping

from .capabilities import CapabilityRegistry
from .execution_context import (
    ExecutionContext,
    ExecutionContextError,
    ExecutionContextIssuer,
)


class PermissionOutcome(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"
    INDETERMINATE = "indeterminate"


class PermissionReasonCode(str, Enum):
    AUTHORIZATION_ALLOWED = "authorization.allowed"
    AUTHORIZATION_DENIED = "authorization.denied"
    AUTHORIZATION_OUTCOME_AMBIGUOUS = "authorization.outcome_ambiguous"
    APPROVAL_REQUIRED = "approval.required"
    CAPABILITY_UNKNOWN = "capability.unknown"
    MANIFEST_MISMATCH = "manifest.mismatch"
    REQUEST_BINDING_MISMATCH = "request.binding_mismatch"
    CONTEXT_EXPIRED = "context.expired"
    CONTEXT_ARGUMENT_MISMATCH = "context.arguments_mismatch"
    AUTHORIZATION_EVIDENCE_UNAVAILABLE = "authorization.evidence_unavailable"
    POLICY_EVALUATION_UNAVAILABLE = "policy.evaluation_unavailable"


class PermissionDecisionError(ValueError):
    """No trustworthy invocation binding was available for a decision."""

    def __init__(self, reason_code: PermissionReasonCode, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.outcome = PermissionOutcome.INDETERMINATE


@dataclass(frozen=True)
class PermissionDecision:
    """Immutable shadow observation bound to one exact invocation."""

    outcome: PermissionOutcome
    reason_code: PermissionReasonCode
    capability_id: str
    contract_version: str
    manifest_digest: str
    argument_digest: str
    invocation_id: str
    detail: str
    policy_revision: str | None = None
    execution_status: str = "not_executed"

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, PermissionOutcome):
            raise TypeError("outcome must be a PermissionOutcome.")
        if not isinstance(self.reason_code, PermissionReasonCode):
            raise TypeError("reason_code must be a PermissionReasonCode.")
        for name in (
            "capability_id", "contract_version", "manifest_digest",
            "argument_digest", "invocation_id", "detail",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.policy_revision is not None and not isinstance(self.policy_revision, str):
            raise TypeError("policy_revision must be a string or None.")
        if self.execution_status != "not_executed":
            raise ValueError("Shadow decisions must remain not_executed.")


@dataclass(frozen=True)
class ApprovalRequest:
    """A request for a human decision; it is not an approval grant."""

    capability_id: str
    contract_version: str
    manifest_digest: str
    argument_digest: str
    invocation_id: str
    reason_code: PermissionReasonCode
    detail: str
    status: str = "request_only_not_authorized"

    def __post_init__(self) -> None:
        for name in (
            "capability_id", "contract_version", "manifest_digest",
            "argument_digest", "invocation_id", "detail",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.reason_code is not PermissionReasonCode.APPROVAL_REQUIRED:
            raise ValueError("Approval requests require the approval.required reason code.")
        if self.status != "request_only_not_authorized":
            raise ValueError("An approval request cannot represent an authorization grant.")


def approval_request_for(decision: PermissionDecision) -> ApprovalRequest:
    """Represent an explicit approval-required decision without granting it."""
    if not isinstance(decision, PermissionDecision):
        raise TypeError("A typed PermissionDecision is required.")
    if (
        decision.outcome is not PermissionOutcome.REQUIRE_APPROVAL
        or decision.reason_code is not PermissionReasonCode.APPROVAL_REQUIRED
    ):
        raise ValueError("The decision does not explicitly require approval.")
    return ApprovalRequest(
        capability_id=decision.capability_id,
        contract_version=decision.contract_version,
        manifest_digest=decision.manifest_digest,
        argument_digest=decision.argument_digest,
        invocation_id=decision.invocation_id,
        reason_code=decision.reason_code,
        detail=decision.detail,
    )


@dataclass(frozen=True)
class PermissionDecisionObservation:
    """Bounded, non-sensitive projection of one shadow evaluation."""

    status: str
    outcome: PermissionOutcome
    reason_code: str
    capability_id: str | None = None
    contract_version: str | None = None
    manifest_digest: str | None = None
    argument_digest: str | None = None
    invocation_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"evaluated", "unavailable"}:
            raise ValueError("Unknown shadow observation status.")
        if not isinstance(self.outcome, PermissionOutcome):
            raise TypeError("outcome must be a PermissionOutcome.")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("reason_code must be a non-empty string.")
        if len(self.reason_code) > 100:
            raise ValueError("reason_code exceeds the observation limit.")
        for value in (
            self.capability_id, self.contract_version, self.manifest_digest,
            self.argument_digest, self.invocation_id,
        ):
            if value is not None and (not isinstance(value, str) or len(value) > 256):
                raise ValueError("Observation identity fields must be bounded strings.")


class PermissionDecisionObserver:
    """Instance-owned in-memory ring buffer containing no prompt or payload."""

    def __init__(self, capacity: int = 256) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Observation capacity must be a positive integer.")
        self._records: deque[PermissionDecisionObservation] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record_decision(self, decision: PermissionDecision) -> None:
        if not isinstance(decision, PermissionDecision):
            raise TypeError("A PermissionDecision is required.")
        record = PermissionDecisionObservation(
            status="evaluated",
            outcome=decision.outcome,
            reason_code=decision.reason_code.value,
            capability_id=decision.capability_id,
            contract_version=decision.contract_version,
            manifest_digest=decision.manifest_digest,
            argument_digest=decision.argument_digest,
            invocation_id=decision.invocation_id,
        )
        with self._lock:
            self._records.append(record)

    def record_unavailable(
        self,
        reason_code: str,
        *,
        manifest=None,
        invocation_id: str | None = None,
        argument_digest: str | None = None,
    ) -> None:
        """Record only stable diagnostics and non-payload binding metadata."""
        record = PermissionDecisionObservation(
            status="unavailable",
            outcome=PermissionOutcome.INDETERMINATE,
            reason_code=reason_code,
            capability_id=(manifest.capability_id if manifest is not None else None),
            contract_version=(manifest.contract_version if manifest is not None else None),
            manifest_digest=(manifest.manifest_digest if manifest is not None else None),
            argument_digest=argument_digest,
            invocation_id=invocation_id,
        )
        with self._lock:
            self._records.append(record)

    def snapshot(self) -> tuple[PermissionDecisionObservation, ...]:
        with self._lock:
            return tuple(self._records)


class PermissionDecisionEvaluator:
    """Project issuer-authenticated M1B result snapshots in shadow mode."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        issuer: ExecutionContextIssuer,
    ) -> None:
        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("An instance-owned CapabilityRegistry is required.")
        if not isinstance(issuer, ExecutionContextIssuer):
            raise TypeError("A trusted ExecutionContextIssuer is required.")
        self._registry = registry
        self._issuer = issuer

    @staticmethod
    def _bound_decision(
        context: ExecutionContext,
        outcome: PermissionOutcome,
        reason_code: PermissionReasonCode,
        detail: str,
    ) -> PermissionDecision:
        return PermissionDecision(
            outcome=outcome,
            reason_code=reason_code,
            capability_id=context.capability_id,
            contract_version=context.contract_version,
            manifest_digest=context.manifest_digest,
            argument_digest=context.argument_digest,
            invocation_id=context.invocation_id,
            detail=detail[:1000] or reason_code.value,
            # PermissionManager has no authoritative policy revision/epoch.
            policy_revision=None,
        )

    def evaluate(
        self,
        *,
        capability_id: str,
        contract_version: str,
        manifest_digest: str,
        arguments: Mapping[str, Any],
        context: ExecutionContext | None,
    ) -> PermissionDecision:
        """Project one already-authorized/denied invocation; never re-check it."""
        if context is None:
            raise PermissionDecisionError(
                PermissionReasonCode.AUTHORIZATION_EVIDENCE_UNAVAILABLE,
                "A signed invocation context is required.",
            )
        if not isinstance(context, ExecutionContext):
            raise PermissionDecisionError(
                PermissionReasonCode.AUTHORIZATION_EVIDENCE_UNAVAILABLE,
                "Authorization evidence is not an issued execution context.",
            )

        try:
            # This verifies the context signature, argument snapshot and
            # deadline but does not require `allowed=True`.
            evidence = self._issuer.validate_for_permission_decision(
                context, None, arguments
            )
        except ExecutionContextError as exc:
            if exc.reason_code in {"context.signature_invalid", "context.issuer_mismatch"}:
                raise PermissionDecisionError(
                    PermissionReasonCode.AUTHORIZATION_EVIDENCE_UNAVAILABLE,
                    "The invocation context is not authentic.",
                ) from exc
            code = (
                PermissionReasonCode.CONTEXT_EXPIRED
                if exc.reason_code == "context.expired"
                else PermissionReasonCode.CONTEXT_ARGUMENT_MISMATCH
                if exc.reason_code.startswith("context.arguments")
                else PermissionReasonCode.AUTHORIZATION_EVIDENCE_UNAVAILABLE
            )
            # These failures occur after authenticity verification, so their
            # result remains bound to the signed invocation and cannot allow.
            try:
                self._issuer.verify_authenticity(context)
            except ExecutionContextError as invalid:
                raise PermissionDecisionError(
                    PermissionReasonCode.AUTHORIZATION_EVIDENCE_UNAVAILABLE,
                    "The invocation context is not authentic.",
                ) from invalid
            return self._bound_decision(
                context, PermissionOutcome.INDETERMINATE, code, str(exc)
            )

        if not isinstance(evidence.allowed, bool):
            return self._bound_decision(
                context,
                PermissionOutcome.INDETERMINATE,
                PermissionReasonCode.AUTHORIZATION_EVIDENCE_UNAVAILABLE,
                "Authorization evidence has no supported decision state.",
            )

        # Exact identity/version lookup only. Catalog presence is descriptive;
        # it does not independently authorize or enable a capability.
        try:
            manifest = self._registry.get(
                context.capability_id, context.contract_version
            )
        except KeyError:
            return self._bound_decision(
                context,
                PermissionOutcome.INDETERMINATE,
                PermissionReasonCode.CAPABILITY_UNKNOWN,
                "The exact capability version is absent from this catalog.",
            )

        if manifest.manifest_digest != context.manifest_digest:
            return self._bound_decision(
                context,
                PermissionOutcome.INDETERMINATE,
                PermissionReasonCode.MANIFEST_MISMATCH,
                "The catalog manifest digest differs from the signed invocation.",
            )
        if (
            capability_id != context.capability_id
            or contract_version != context.contract_version
            or manifest_digest != context.manifest_digest
        ):
            return self._bound_decision(
                context,
                PermissionOutcome.INDETERMINATE,
                PermissionReasonCode.REQUEST_BINDING_MISMATCH,
                "The requested identity or digest differs from the signed invocation.",
            )

        # Revalidate against the exact catalog manifest. This is context
        # validation only; no permission policy or callback is invoked.
        try:
            evidence = self._issuer.validate_for_permission_decision(
                context, manifest, arguments
            )
        except ExecutionContextError as exc:
            code = (
                PermissionReasonCode.CONTEXT_EXPIRED
                if exc.reason_code == "context.expired"
                else PermissionReasonCode.MANIFEST_MISMATCH
                if exc.reason_code == "context.manifest_mismatch"
                else PermissionReasonCode.CONTEXT_ARGUMENT_MISMATCH
                if exc.reason_code.startswith("context.arguments")
                else PermissionReasonCode.AUTHORIZATION_EVIDENCE_UNAVAILABLE
            )
            return self._bound_decision(
                context, PermissionOutcome.INDETERMINATE, code, str(exc)
            )

        if evidence.allowed:
            return self._bound_decision(
                context,
                PermissionOutcome.ALLOW,
                PermissionReasonCode.AUTHORIZATION_ALLOWED,
                evidence.reason,
            )
        # The existing tuple does not distinguish approval pending, unavailable
        # approval, user denial, policy denial, or callback failure. Preserve
        # that ambiguity instead of treating every false result as a denial or
        # inferring a pending approval from the human-readable reason.
        return self._bound_decision(
            context,
            PermissionOutcome.INDETERMINATE,
            PermissionReasonCode.AUTHORIZATION_OUTCOME_AMBIGUOUS,
            evidence.reason,
        )
