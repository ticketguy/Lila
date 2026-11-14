"""Exact-operation approval grant records for the M2B permission kernel.

Grants are immutable, in-memory binding records. This module does not accept
human decisions, authenticate an owner, consume a grant, or authorize dispatch.
Those operations belong to later M2B steps. A grant with no trusted policy
revision must be checked against a fresh typed policy result before any later
authorization boundary relies on it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

from .execution_context import _freeze, argument_digest, typed_value_bytes
from .pending_invocations import PendingInvocation, PendingInvocationState, TargetBinding, _TARGET_DIGEST_DOMAIN
from .permissions import PolicyDisposition, PolicyEvaluationResult, PolicyReasonCode


_GRANT_ISSUER = object()
_OPERATION_DIGEST_DOMAIN = b"lila-exact-approval-operation\0v1\0"


class ApprovalGrantError(ValueError):
    """An exact-operation grant could not be validated safely."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code


def _operation_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    reason = canonical.get("policy_reason_code")
    if isinstance(reason, PolicyReasonCode):
        canonical["policy_reason_code"] = reason.value
    return hashlib.sha256(
        _OPERATION_DIGEST_DOMAIN + typed_value_bytes(canonical)
    ).hexdigest()


def _target_payload(binding: TargetBinding | None) -> Mapping[str, Any] | None:
    if binding is None:
        return None
    return {
        "capability_id": binding.capability_id,
        "contract_version": binding.contract_version,
        "manifest_digest": binding.manifest_digest,
        "resolver_id": binding.resolver_id,
        "target_kind": binding.target_kind,
        "identity": binding.identity,
        "digest": binding.digest,
    }


@dataclass(frozen=True, slots=True, init=False, repr=False)
class ApprovalGrant:
    """An immutable grant record bound to one exact pending invocation.

    The record is not a general permission token and has no dispatch or
    consumption method. Construction is restricted to the trusted pending
    invocation registry's issuance path.
    """

    invocation_id: str
    capability_id: str
    contract_version: str
    manifest_digest: str
    operation_digest: str
    validated_arguments: Mapping[str, Any]
    validated_argument_digest: str
    execution_arguments: Mapping[str, Any]
    execution_argument_digest: str
    target_required: bool
    target_binding: TargetBinding | None
    policy_reason_code: PolicyReasonCode
    policy_risk: str | None
    policy_revision: str | None
    policy_revalidation_required: bool
    pending_created_at_monotonic: float
    created_at_monotonic: float
    expires_at_monotonic: float

    def __init__(self, *, _issuer=None, **values) -> None:
        if _issuer is not _GRANT_ISSUER:
            raise TypeError("ApprovalGrant records can only be issued by trusted Harness code.")
        self._assign_validated(values)

    @classmethod
    def _from_pending(
        cls,
        pending: PendingInvocation,
        *,
        created_at_monotonic: float,
        _issuer=None,
    ) -> "ApprovalGrant":
        if _issuer is not _GRANT_ISSUER:
            raise TypeError("ApprovalGrant issuance is restricted to trusted Harness code.")
        if not isinstance(pending, PendingInvocation):
            raise ApprovalGrantError("grant.pending_untrusted", "A trusted pending invocation is required.")
        payload = {
            "invocation_id": pending.invocation_id,
            "capability_id": pending.capability_id,
            "contract_version": pending.contract_version,
            "manifest_digest": pending.manifest_digest,
            "validated_arguments": pending.validated_arguments,
            "validated_argument_digest": pending.validated_argument_digest,
            "execution_arguments": pending.execution_arguments,
            "execution_argument_digest": pending.execution_argument_digest,
            "target_required": pending.target_required,
            "target_binding": _target_payload(pending.target_binding),
            "policy_reason_code": pending.policy_result.reason_code,
            "policy_risk": pending.policy_result.risk,
            "policy_revision": pending.policy_result.policy_revision,
            "policy_revalidation_required": pending.policy_result.policy_revision is None,
            "pending_created_at_monotonic": pending.created_at_monotonic,
            "created_at_monotonic": created_at_monotonic,
            "expires_at_monotonic": pending.expires_at_monotonic,
        }
        operation_digest = _operation_digest(payload)
        grant_values = dict(payload)
        grant_values["target_binding"] = pending.target_binding
        return cls(
            _issuer=_GRANT_ISSUER,
            **grant_values,
            operation_digest=operation_digest,
        )

    def _assign_validated(self, values: Mapping[str, Any]) -> None:
        pending_id = values.get("invocation_id")
        identity = (
            values.get("capability_id"),
            values.get("contract_version"),
            values.get("manifest_digest"),
        )
        if not isinstance(pending_id, str) or not pending_id:
            raise ApprovalGrantError("grant.identity_invalid", "Invocation ID is required.")
        if not all(isinstance(item, str) and item for item in identity):
            raise ApprovalGrantError("grant.identity_invalid", "Capability identity is incomplete.")

        validated = _freeze(values.get("validated_arguments"))
        execution = _freeze(values.get("execution_arguments"))
        if not isinstance(validated, Mapping) or not isinstance(execution, Mapping):
            raise ApprovalGrantError("grant.arguments_invalid", "Bound arguments must be mappings.")
        validated_digest = argument_digest(validated)
        execution_digest = argument_digest(execution)
        if validated_digest != values.get("validated_argument_digest"):
            raise ApprovalGrantError("grant.argument_digest_mismatch", "Validated argument digest mismatch.")
        if execution_digest != values.get("execution_argument_digest"):
            raise ApprovalGrantError("grant.execution_digest_mismatch", "Execution argument digest mismatch.")

        target_required = values.get("target_required")
        target = values.get("target_binding")
        if type(target_required) is not bool or target_required != (target is not None):
            raise ApprovalGrantError("grant.target_binding_mismatch", "Target requirement and binding disagree.")
        if target is not None:
            if not isinstance(target, TargetBinding):
                raise ApprovalGrantError("grant.target_untrusted", "Target binding is not trusted.")
            if (
                target.capability_id != identity[0]
                or target.contract_version != identity[1]
                or target.manifest_digest != identity[2]
            ):
                raise ApprovalGrantError("grant.target_binding_mismatch", "Target identity does not match the capability.")
            self._verify_target(target)

        revision = values.get("policy_revision")
        reason_code = values.get("policy_reason_code")
        risk = values.get("policy_risk")
        if not isinstance(reason_code, PolicyReasonCode):
            raise ApprovalGrantError("grant.policy_reason_invalid", "Typed policy reason code is required.")
        if risk is not None and not isinstance(risk, str):
            raise ApprovalGrantError("grant.policy_risk_invalid", "Policy risk must be a string or absent.")
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise ApprovalGrantError("grant.policy_revision_invalid", "Policy revision must be non-empty or absent.")
        if values.get("policy_revalidation_required") is not (revision is None):
            raise ApprovalGrantError("grant.policy_revision_binding_mismatch", "Policy revalidation flag is inconsistent.")

        pending_created = values.get("pending_created_at_monotonic")
        created = values.get("created_at_monotonic")
        expires = values.get("expires_at_monotonic")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (pending_created, created, expires)
        ) or created < pending_created or expires <= created:
            raise ApprovalGrantError("grant.time_invalid", "Grant lifetime is invalid.")

        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "validated_arguments", validated)
        object.__setattr__(self, "execution_arguments", execution)
        self.verify_integrity()

    @staticmethod
    def _verify_target(binding: TargetBinding) -> None:
        payload = {
            "capability_id": binding.capability_id,
            "contract_version": binding.contract_version,
            "manifest_digest": binding.manifest_digest,
            "resolver_id": binding.resolver_id,
            "target_kind": binding.target_kind,
            "identity": binding.identity,
        }
        expected = hashlib.sha256(_TARGET_DIGEST_DOMAIN + typed_value_bytes(payload)).hexdigest()
        if expected != binding.digest:
            raise ApprovalGrantError("grant.target_digest_mismatch", "Resolved target digest mismatch.")

    def verify_integrity(self) -> None:
        """Recompute all bindings; fail closed if a same-process mutation occurred."""
        if argument_digest(self.validated_arguments) != self.validated_argument_digest:
            raise ApprovalGrantError("grant.argument_digest_mismatch", "Validated arguments changed.")
        if argument_digest(self.execution_arguments) != self.execution_argument_digest:
            raise ApprovalGrantError("grant.execution_digest_mismatch", "Execution arguments changed.")
        if self.target_binding is not None:
            self._verify_target(self.target_binding)
        payload = {
            "invocation_id": self.invocation_id,
            "capability_id": self.capability_id,
            "contract_version": self.contract_version,
            "manifest_digest": self.manifest_digest,
            "validated_arguments": self.validated_arguments,
            "validated_argument_digest": self.validated_argument_digest,
            "execution_arguments": self.execution_arguments,
            "execution_argument_digest": self.execution_argument_digest,
            "target_required": self.target_required,
            "target_binding": _target_payload(self.target_binding),
            "policy_reason_code": self.policy_reason_code,
            "policy_risk": self.policy_risk,
            "policy_revision": self.policy_revision,
            "policy_revalidation_required": self.policy_revalidation_required,
            "pending_created_at_monotonic": self.pending_created_at_monotonic,
            "created_at_monotonic": self.created_at_monotonic,
            "expires_at_monotonic": self.expires_at_monotonic,
        }
        expected = _operation_digest(payload)
        if expected != self.operation_digest:
            raise ApprovalGrantError("grant.operation_digest_mismatch", "Grant operation binding changed.")

    def is_bound_to(self, pending: PendingInvocation, now_monotonic: float) -> bool:
        """Check exact identity and current pending lifecycle without consuming."""
        self.verify_integrity()
        if not isinstance(pending, PendingInvocation):
            return False
        if (
            pending.state is not PendingInvocationState.PENDING
            or isinstance(now_monotonic, bool)
            or not isinstance(now_monotonic, (int, float))
            or not math.isfinite(now_monotonic)
            or now_monotonic < self.created_at_monotonic
            or now_monotonic >= self.expires_at_monotonic
            or pending.created_at_monotonic != self.pending_created_at_monotonic
            or pending.expires_at_monotonic != self.expires_at_monotonic
            or pending.policy_result.policy_revision != self.policy_revision
            or pending.policy_result.reason_code is not self.policy_reason_code
            or pending.policy_result.risk != self.policy_risk
        ):
            return False
        return (
            pending.invocation_id == self.invocation_id
            and pending.capability_id == self.capability_id
            and pending.contract_version == self.contract_version
            and pending.manifest_digest == self.manifest_digest
            and pending.validated_arguments == self.validated_arguments
            and pending.validated_argument_digest == self.validated_argument_digest
            and pending.execution_arguments == self.execution_arguments
            and pending.execution_argument_digest == self.execution_argument_digest
            and pending.target_required == self.target_required
            and _target_payload(pending.target_binding) == _target_payload(self.target_binding)
        )

    def validate_current_policy(self, result: PolicyEvaluationResult) -> bool:
        """Validate fresh policy evidence before a later authorization boundary.

        When no trustworthy revision was available at grant issuance, callers
        must obtain this result by re-evaluating current trusted policy for the
        exact bound arguments. This method checks the evidence binding, but it
        cannot itself establish when the caller evaluated that policy.
        """
        self.verify_integrity()
        if type(result) is not PolicyEvaluationResult:
            raise ApprovalGrantError("grant.policy_result_untrusted", "A typed policy result is required.")
        if (
            result.disposition is not PolicyDisposition.REQUIRE_APPROVAL
            or result.capability_id != self.capability_id
            or result.contract_version != self.contract_version
            or result.manifest_digest != self.manifest_digest
            or result.policy_revision != self.policy_revision
            or result.argument_digest != self.validated_argument_digest
            or result.execution_argument_digest != self.execution_argument_digest
            or result.arguments != self.validated_arguments
            or result.execution_arguments != self.execution_arguments
            or result.reason_code is not self.policy_reason_code
            or result.risk != self.policy_risk
        ):
            return False
        return True

    def __repr__(self) -> str:
        return (
            "ApprovalGrant(invocation_id={!r}, capability_id={!r}, "
            "contract_version={!r}, operation_digest={!r}, policy_revision={!r})"
        ).format(
            self.invocation_id,
            self.capability_id,
            self.contract_version,
            self.operation_digest,
            self.policy_revision,
        )
