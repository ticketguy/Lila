"""Typed, read-only presentation and proof-carrying adapter for Harness approvals.

The daemon API bearer authorizes access to these routes only. Decisions still
require the separate Step 4 owner proof and are verified/consumed by Harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping

from src.harness.approval_authentication import ApprovalDecisionResponse, OwnerDecision
from src.harness.approval_continuation import ApprovalContinuationService
from src.harness.pending_invocations import PendingInvocationError


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CHALLENGE_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$")
_SENSITIVE_KEY = re.compile(r"(?i)(secret|token|password|credential|authorization|api[_-]?key|private[_-]?key|proof|content|payload|transcript|prompt|message|email|phone|ssn|credit[_-]?card)")
_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_SECRET_ASSIGNMENT = re.compile(r"(?i)\b(password|passwd|token|api[_-]?key|secret)\s*[:=]\s*([^\s,;]+)")
_URL_CREDENTIALS = re.compile(r"(://)[^/@\s]+:[^/@\s]+@")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")


class ApprovalApiErrorCode(str, Enum):
    INVALID_IDENTIFIER = "approval.invalid_identifier"
    INVALID_REQUEST = "approval.invalid_request"
    NOT_FOUND = "approval.not_found"
    NOT_PENDING = "approval.not_pending"
    AUTHENTICATION_REQUIRED = "approval.owner_authentication_required"
    AUTHENTICATION_FAILED = "approval.owner_authentication_failed"
    SERVICE_UNAVAILABLE = "approval.service_unavailable"


@dataclass(frozen=True, repr=False)
class ApprovalDecisionRequest:
    invocation_id: str
    challenge_id: str
    key_id: str
    operation_digest: str
    decision: OwnerDecision
    proof_hex: str

    def __repr__(self) -> str:
        return ("ApprovalDecisionRequest(invocation_id={!r}, decision={!r}, "
                "proof=<redacted>)").format(self.invocation_id, self.decision.value)

    @classmethod
    def parse(cls, invocation_id: object, body: object) -> "ApprovalDecisionRequest":
        if not isinstance(invocation_id, str) or not _IDENTIFIER.fullmatch(invocation_id):
            raise ValueError(ApprovalApiErrorCode.INVALID_IDENTIFIER.value)
        if isinstance(body, dict) and "proof" not in body:
            raise ValueError(ApprovalApiErrorCode.AUTHENTICATION_REQUIRED.value)
        if not isinstance(body, dict) or set(body) != {
            "challenge_id", "key_id", "operation_digest", "decision", "proof"
        }:
            raise ValueError(ApprovalApiErrorCode.INVALID_REQUEST.value)
        challenge_id, key_id = body["challenge_id"], body["key_id"]
        digest, decision, proof = body["operation_digest"], body["decision"], body["proof"]
        if (not isinstance(challenge_id, str) or not _CHALLENGE_ID.fullmatch(challenge_id)
                or not isinstance(key_id, str) or not _IDENTIFIER.fullmatch(key_id)
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or decision not in ("approve", "reject")
                or not isinstance(proof, str) or not re.fullmatch(r"[0-9a-f]{64}", proof)):
            raise ValueError(ApprovalApiErrorCode.INVALID_REQUEST.value)
        return cls(invocation_id, challenge_id, key_id, digest,
                   OwnerDecision(decision), proof)

    def to_harness_response(self) -> ApprovalDecisionResponse:
        return ApprovalDecisionResponse(
            challenge_id=self.challenge_id, key_id=self.key_id,
            invocation_id=self.invocation_id, operation_digest=self.operation_digest,
            decision=self.decision, proof=bytes.fromhex(self.proof_hex),
        )


@dataclass(frozen=True)
class ApprovalRequestView:
    invocation_id: str
    status: str
    capability_id: str
    contract_version: str
    operation_digest: str
    challenge_id: str
    voice_challenge_code: str
    key_id: str
    target_kind: str | None
    target_resolver_id: str
    target: Any
    arguments: Mapping[str, Any]
    policy_reason: str
    policy_detail: str
    risk: str | None
    expires_in_seconds: float
    approval_ready: bool
    display_action: str | None
    display_values: Mapping[str, Any]
    display_block_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class ApprovalStatusView:
    invocation_id: str | None
    status: str
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class ApprovalListView:
    requests: tuple[ApprovalRequestView, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"requests": [request.to_dict() for request in self.requests]}


@dataclass(frozen=True)
class ApprovalDetailView:
    kind: str
    request: ApprovalRequestView | None
    status: ApprovalStatusView | None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind,
                "request": self.request.to_dict() if self.request else None,
                "status": self.status.to_dict() if self.status else None}


@dataclass(frozen=True)
class ApprovalDecisionView:
    invocation_id: str | None
    status: str
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


class ApprovalApi:
    """Narrow projection of an already-injected Harness continuation service."""

    def __init__(self, service: ApprovalContinuationService | None):
        self._service = service

    def list_pending(self) -> ApprovalListView:
        service = self._require_service()
        return ApprovalListView(tuple(self._request_view(service, request)
                                      for request in service.list_pending_requests()))

    def get_pending(self, invocation_id: str) -> ApprovalDetailView:
        self._validate_id(invocation_id)
        service = self._require_service()
        try:
            request = service.get_request(invocation_id)
            return ApprovalDetailView("pending", self._request_view(service, request), None)
        except PendingInvocationError:
            return ApprovalDetailView("terminal", None, self.get_status(invocation_id))

    def get_status(self, invocation_id: str) -> ApprovalStatusView:
        self._validate_id(invocation_id)
        service = self._require_service()
        try:
            result = service.get_result(invocation_id)
        except Exception:
            raise LookupError(ApprovalApiErrorCode.NOT_FOUND.value) from None
        return ApprovalStatusView(result.invocation_id, result.status.value, result.reason_code)

    def require_reviewable(self, invocation_id: str) -> None:
        """Reject owner decisions without a trusted owner-facing display contract."""
        detail = self.get_pending(invocation_id)
        if detail.kind != "pending" or detail.request is None or not detail.request.approval_ready:
            raise PendingInvocationError(
                "approval.display_unreviewed",
                "This operation has no reviewed owner-facing display contract.",
            )

    def decide(self, request: ApprovalDecisionRequest) -> ApprovalDecisionView:
        service = self._require_service()
        try:
            outcome = service.respond(request.to_harness_response())
        except Exception:
            raise RuntimeError(ApprovalApiErrorCode.SERVICE_UNAVAILABLE.value) from None
        return ApprovalDecisionView(outcome.invocation_id, outcome.status.value, outcome.reason_code)

    def _require_service(self) -> ApprovalContinuationService:
        if not isinstance(self._service, ApprovalContinuationService):
            raise RuntimeError(ApprovalApiErrorCode.SERVICE_UNAVAILABLE.value)
        return self._service

    @staticmethod
    def _validate_id(invocation_id: str) -> None:
        if not isinstance(invocation_id, str) or not _IDENTIFIER.fullmatch(invocation_id):
            raise ValueError(ApprovalApiErrorCode.INVALID_IDENTIFIER.value)

    @classmethod
    def _request_view(cls, service, request) -> ApprovalRequestView:
        invocation = request.invocation
        policy = invocation.policy_result
        target = invocation.target_binding
        display = target.identity.get("display") if target else None
        reviewed = (
            isinstance(display, Mapping)
            and set(display) == {"reviewed", "action", "target", "values"}
            and display.get("reviewed") is True
            and target is not None
            and display.get("target") == target.identity.get("canonical_location")
            and cls._display_metadata_is_safe(display)
        )
        action = display.get("action") if reviewed else None
        values = display.get("values") if reviewed else None
        ready = bool(reviewed and isinstance(action, str) and isinstance(values, Mapping))
        try:
            remaining = service.remaining_lifetime(invocation.invocation_id)
        except Exception:
            remaining = 0.0
        return ApprovalRequestView(
            invocation.invocation_id, "pending", invocation.capability_id,
            invocation.contract_version, request.grant.operation_digest,
            request.challenge.challenge_id,
            f"{int.from_bytes(request.challenge.challenge_material[:4], 'big') % 1_000_000:06d}",
            request.challenge.key_id,
            target.target_kind if target else None,
            target.resolver_id if target else invocation.target_resolver_id,
            ({
                "type": target.identity.get("target_type"),
                "canonical_location": cls._safe_text(target.identity.get("canonical_location", "")),
                "parent_location": cls._safe_text(target.identity.get("parent_location", "")),
                "device": target.identity.get("device"),
                "inode": target.identity.get("inode"),
                "mode": target.identity.get("mode"),
                "size": target.identity.get("size"),
                "mtime_ns": target.identity.get("mtime_ns"),
            } if target and target.identity.get("target_type") else None),
            {},
            policy.reason_code.value, cls._safe_text(policy.detail),
            cls._safe_text(policy.risk) if policy.risk else request.grant.policy_risk,
            round(max(0.0, remaining), 3),
            ready,
            cls._safe_text(action) if ready else None,
            cls._safe_display_value(values) if ready else {},
            None if ready else "approval.display_metadata_unreviewed",
        )

    @classmethod
    def _safe_value(cls, value: Any, key: str = "") -> Any:
        if key and _SENSITIVE_KEY.search(key):
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {str(k)[:100]: cls._safe_value(v, str(k)) for k, v in list(value.items())[:100]}
        if isinstance(value, (tuple, list)):
            return [cls._safe_value(item) for item in value[:100]]
        if isinstance(value, str):
            # No approved argument-display classification exists yet. Keep
            # parameter names and primitive flags/numbers, redact free text.
            return "[REDACTED]"
        if value is None or type(value) in (bool, int, float):
            return value
        return "[REDACTED]"

    @classmethod
    def _safe_display_value(cls, value: Any, key: str = "") -> Any:
        """Render only already-reviewed metadata; sensitive labels remain redacted."""
        if key and _SENSITIVE_KEY.search(key):
            return "[REDACTED]"
        if isinstance(value, Mapping):
            return {str(k)[:100]: cls._safe_display_value(v, str(k))
                    for k, v in list(value.items())[:50]}
        if isinstance(value, (tuple, list)):
            return [cls._safe_display_value(item) for item in value[:50]]
        if isinstance(value, str):
            return cls._safe_text(value)
        if value is None or type(value) in (bool, int, float):
            return value
        return "[UNREVIEWABLE]"

    @classmethod
    def _display_metadata_is_safe(cls, value: Any, key: str = "") -> bool:
        """Require the screen projection to preserve every reviewed value exactly."""
        if key and _SENSITIVE_KEY.search(key):
            return False
        if isinstance(value, Mapping):
            return len(value) <= 50 and all(
                isinstance(name, str) and cls._display_metadata_is_safe(item, name)
                for name, item in value.items()
            )
        if isinstance(value, (tuple, list)):
            return len(value) <= 50 and all(cls._display_metadata_is_safe(item) for item in value)
        if isinstance(value, str):
            return len(value) <= 1000 and cls._safe_text(value) == value
        return value is None or type(value) in (bool, int, float)

    @classmethod
    def _safe_target_value(cls, value: Any, key: str = "") -> Any:
        if isinstance(value, Mapping):
            return {str(k)[:100]: cls._safe_target_value(v, str(k)) for k, v in list(value.items())[:100]}
        if isinstance(value, (tuple, list)):
            return [cls._safe_target_value(item) for item in value[:100]]
        if isinstance(value, str):
            if key.casefold() in {"path", "file_path", "directory", "device_id", "process_id", "resource_id", "uri"}:
                return cls._safe_text(value)
            return "[REDACTED]"
        if value is None or type(value) in (bool, int, float):
            return value
        return "[REDACTED]"

    @staticmethod
    def _safe_text(value: str) -> str:
        text = _BEARER.sub("Bearer [REDACTED]", str(value))
        text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
        text = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", text)
        text = _JWT.sub("[REDACTED TOKEN]", text)
        return "".join(ch for ch in text if ch in "\t\n" or ord(ch) >= 32)[:1000]
