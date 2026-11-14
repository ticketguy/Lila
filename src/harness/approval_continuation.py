"""In-memory one-use approval lifecycle and Harness continuation (M2B 5–6).

This service authenticates and consumes one exact approval, revalidates trusted
policy/manifest/target state, then dispatches the already-bound Tool once. It
is an application boundary, not an OS boundary or durable task queue. The
owner-facing adapter supplies credentials and transports challenges; it is not
implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Callable, Protocol

from .approval_authentication import (
    ApprovalAuthenticator,
    ApprovalChallenge,
    ApprovalDecisionResponse,
    AuthenticatedOwnerDecision,
    OwnerDecision,
)
from .approval_grants import ApprovalGrant
from .pending_invocations import (
    PendingInvocation,
    PendingInvocationError,
    PendingInvocationRegistry,
    PendingInvocationState,
)
from .permissions import PolicyDisposition, PolicyEvaluationResult
from .tools import Tool, ToolResult, get_registry


@dataclass(frozen=True, repr=False)
class OwnerAuthenticationSecret:
    key_id: str
    secret: bytes

    def __repr__(self) -> str:
        return f"OwnerAuthenticationSecret(key_id={self.key_id!r}, secret=<redacted>)"


class OwnerAuthenticationSecretProvider(Protocol):
    """Owner-controlled provider; no daemon token or generated fallback."""

    def get_owner_authentication_secret(self) -> OwnerAuthenticationSecret: ...


class ExecutionTargetAdapter(Protocol):
    """Trusted adapter that leases the exact target object for one dispatch."""

    def open_target(self, invocation): ...


class ContinuationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ContinuationResult:
    invocation_id: str | None
    status: ContinuationStatus
    reason_code: str
    tool_result: ToolResult | None = None


@dataclass(frozen=True)
class ContinuationAuditEvent:
    """Privacy-safe lifecycle transition emitted for continuation audit logs."""

    invocation_id: str | None
    tool_name: str
    argument_names: tuple[str, ...]
    lifecycle_stage: str
    status: ContinuationStatus
    reason_code: str
    success: bool = False
    has_output: bool = False
    has_error: bool = False
    duration_ms: float = 0.0


@dataclass(frozen=True, repr=False)
class ApprovalRequest:
    """Trusted adapter handoff; challenge material must not enter model output."""

    invocation: PendingInvocation
    grant: ApprovalGrant
    challenge: ApprovalChallenge

    def __repr__(self) -> str:
        return (
            "ApprovalRequest(invocation_id={!r}, operation_digest={!r}, "
            "challenge_id={!r}, challenge_material=<redacted>)"
        ).format(
            self.invocation.invocation_id,
            self.grant.operation_digest,
            self.challenge.challenge_id,
        )


@dataclass
class _Continuation:
    invocation: PendingInvocation
    grant: ApprovalGrant
    tool: Tool
    execute_handler: Callable
    manifest_digest: str
    revalidate_policy: Callable[[], PolicyEvaluationResult]
    execution_target_adapter: ExecutionTargetAdapter | None = None
    status: ContinuationStatus = ContinuationStatus.PENDING


class ApprovalContinuationService:
    """Instance-owned bounded in-memory approval lifecycle.

    Calls to ``respond`` serialize authentication, validation, consumption and
    dispatch. A continuation is removed from the pending set before dispatch,
    so repeated responses cannot execute it twice. Process/service loss drops
    in-memory continuations; restart recovery is deliberately not provided.
    """

    def __init__(
        self,
        *,
        pending_registry: PendingInvocationRegistry,
        owner_secret_provider: OwnerAuthenticationSecretProvider,
        challenge_lifetime_seconds: float = 60.0,
        capacity: int = 128,
        monotonic_clock: Callable[[], float] = time.monotonic,
        execution_target_adapters: dict[str, ExecutionTargetAdapter] | None = None,
    ) -> None:
        if not isinstance(pending_registry, PendingInvocationRegistry):
            raise TypeError("An instance-owned PendingInvocationRegistry is required.")
        if not callable(getattr(owner_secret_provider, "get_owner_authentication_secret", None)):
            raise TypeError("An explicit owner authentication secret provider is required.")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Continuation capacity must be a positive integer.")
        self._pending = pending_registry
        self._clock = monotonic_clock
        self._capacity = capacity
        self._continuations: dict[str, _Continuation] = {}
        self._requests: dict[str, ApprovalRequest] = {}
        self._results: dict[str, ContinuationResult] = {}
        self._tool_names: dict[str, str] = {}
        self._arg_names: dict[str, tuple[str, ...]] = {}
        self._audit_observers: list[Callable[[ContinuationAuditEvent], None]] = []
        self._lock = threading.RLock()
        self._closed = False
        adapters = {} if execution_target_adapters is None else dict(execution_target_adapters)
        if any(
            not isinstance(capability_id, str)
            or not callable(getattr(adapter, "open_target", None))
            for capability_id, adapter in adapters.items()
        ):
            raise TypeError("Execution target adapters must be keyed by capability ID and provide open_target().")
        self._execution_target_adapters = adapters
        try:
            credentials = owner_secret_provider.get_owner_authentication_secret()
        except Exception:
            raise ValueError("Owner authentication credentials are unavailable.") from None
        if type(credentials) is not OwnerAuthenticationSecret:
            raise TypeError("Secret provider must return OwnerAuthenticationSecret.")
        self._authenticator = ApprovalAuthenticator(
            pending_registry=pending_registry,
            owner_secret=credentials.secret,
            key_id=credentials.key_id,
            challenge_lifetime_seconds=challenge_lifetime_seconds,
            capacity=capacity,
            monotonic_clock=monotonic_clock,
        )

    def register_audit_observer(
        self, observer: Callable[[ContinuationAuditEvent], None]
    ) -> None:
        """Register a privacy-safe audit observer for continuation transitions."""
        if not callable(observer):
            raise TypeError("Audit observer must be callable.")
        with self._lock:
            if observer not in self._audit_observers:
                self._audit_observers.append(observer)

    def _emit_audit(self, event: ContinuationAuditEvent) -> None:
        for observer in tuple(self._audit_observers):
            try:
                observer(event)
            except Exception:
                # Audit observer failures must never break fail-closed execution.
                pass

    def submit(
        self,
        *,
        policy_result: PolicyEvaluationResult,
        tool: Tool,
        revalidate_policy: Callable[[], PolicyEvaluationResult],
    ) -> ApprovalRequest:
        """Capture a typed approval-required operation without blocking/dispatch."""
        if type(policy_result) is not PolicyEvaluationResult or policy_result.disposition is not PolicyDisposition.REQUIRE_APPROVAL:
            raise ValueError("Only a typed REQUIRE_APPROVAL result can be suspended.")
        if not isinstance(tool, Tool) or not callable(revalidate_policy):
            raise TypeError("A trusted Tool and fresh-policy callback are required.")
        with self._lock:
            if self._closed:
                raise PendingInvocationError("continuation.service_closed", "Approval continuation service is unavailable.")
            if len(self._continuations) >= self._capacity:
                raise PendingInvocationError("continuation.capacity_reached", "Pending continuation capacity reached.")
            invocation = self._pending.create_pending(policy_result)
            target_adapter = self._execution_target_adapters.get(invocation.capability_id)
            if (
                invocation.target_binding is not None
                and invocation.target_binding.target_kind == "filesystem.file"
                and target_adapter is None
            ):
                self._pending.invalidate(invocation.invocation_id, "execution.target_adapter_unavailable")
                raise PendingInvocationError(
                    "execution.target_adapter_unavailable",
                    "The approved file target cannot be safely passed to its handler.",
                )
            try:
                grant = self._pending._issue_approval_grant(invocation.invocation_id)
                challenge = self._authenticator.issue_challenge(invocation.invocation_id, grant)
            except Exception:
                try:
                    self._pending.invalidate(invocation.invocation_id, "continuation.setup_failed")
                except Exception:
                    pass
                raise
            request = ApprovalRequest(invocation, grant, challenge)
            self._requests[invocation.invocation_id] = request
            self._tool_names[invocation.invocation_id] = tool.name
            self._arg_names[invocation.invocation_id] = tuple(
                invocation.execution_arguments.keys()
            )
            self._continuations[invocation.invocation_id] = _Continuation(
                invocation, grant, tool, tool.execute,
                invocation.manifest_digest, revalidate_policy, target_adapter
            )
            result = ContinuationResult(
                invocation.invocation_id, ContinuationStatus.PENDING, "approval.pending"
            )
            self._results[invocation.invocation_id] = result
            return request

    def get_request(self, invocation_id: str) -> ApprovalRequest:
        """Return challenge data only to a trusted future owner adapter."""
        with self._lock:
            self._require_active(invocation_id)
            if not self._pending.validate_current(invocation_id):
                record = self._pending.get(invocation_id)
                status = ContinuationStatus.EXPIRED if record.state is PendingInvocationState.EXPIRED else ContinuationStatus.INVALIDATED
                self._finish(invocation_id, status, record.state_reason_code)
                raise PendingInvocationError("continuation.not_pending", "Approval assumptions are no longer current.")
            try:
                return self._requests[invocation_id]
            except KeyError:
                raise PendingInvocationError("continuation.unknown", "Pending approval was not found.") from None

    def list_pending_requests(self) -> tuple[ApprovalRequest, ...]:
        """Return current Harness requests to a trusted adapter (never serialize challenges)."""
        with self._lock:
            identifiers = tuple(self._continuations)
        requests = []
        for invocation_id in identifiers:
            try:
                requests.append(self.get_request(invocation_id))
            except PendingInvocationError:
                continue
        return tuple(requests)

    def remaining_lifetime(self, invocation_id: str) -> float:
        """Report bounded remaining time without exposing the monotonic deadline."""
        with self._lock:
            self._require_active(invocation_id)
            request = self._requests[invocation_id]
            return max(0.0, request.grant.expires_at_monotonic - self._clock())

    def get_result(self, invocation_id: str) -> ContinuationResult:
        with self._lock:
            try:
                return self._results[invocation_id]
            except KeyError:
                raise PendingInvocationError("continuation.unknown", "Continuation was not found.") from None

    def cancel(self, invocation_id: str) -> ContinuationResult:
        with self._lock:
            self._require_active(invocation_id)
            try:
                self._pending.cancel(invocation_id)
            except PendingInvocationError as exc:
                return self._finish(
                    invocation_id,
                    self._status_for_pending(invocation_id, ContinuationStatus.FAILED),
                    exc.reason_code,
                )
            return self._finish(invocation_id, ContinuationStatus.CANCELLED, "invocation.cancelled")

    def invalidate(self, invocation_id: str, reason_code: str = "invocation.assumptions_invalid") -> ContinuationResult:
        """Invalidate invocation and revoke its outstanding challenge atomically."""
        with self._lock:
            self._require_active(invocation_id)
            try:
                record = self._pending.invalidate(invocation_id, reason_code)
            except PendingInvocationError as exc:
                status = self._status_for_pending(invocation_id, ContinuationStatus.FAILED)
                return self._finish(invocation_id, status, exc.reason_code)
            status = self._status_for_state(
                record.state, ContinuationStatus.INVALIDATED
            )
            return self._finish(invocation_id, status, reason_code)

    def expire_due(self) -> tuple[ContinuationResult, ...]:
        """Expire elapsed invocations and revoke their outstanding challenges."""
        with self._lock:
            expired = self._pending.expire_due()
            results = []
            for invocation in expired:
                if invocation.invocation_id in self._continuations:
                    results.append(self._finish(
                        invocation.invocation_id,
                        ContinuationStatus.EXPIRED,
                        "invocation.expired",
                    ))
            return tuple(results)

    def respond(
        self, response: ApprovalDecisionResponse, *,
        _owner_secret: bytes | bytearray | None = None,
    ) -> ContinuationResult:
        """Authenticate, revalidate and atomically consume; dispatch at most once."""
        if not isinstance(response, ApprovalDecisionResponse) or not isinstance(response.invocation_id, str):
            return ContinuationResult(None, ContinuationStatus.FAILED, "approval.response_malformed")
        with self._lock:
            if self._closed:
                return ContinuationResult(None, ContinuationStatus.UNAVAILABLE, "continuation.service_closed")
            invocation_id = response.invocation_id
            continuation = self._continuations.get(invocation_id)
            if continuation is None or continuation.status is not ContinuationStatus.PENDING:
                tool_name = self._tool_names.get(invocation_id, "unknown")
                arg_names = self._arg_names.get(invocation_id, ())
                self._emit_audit(
                    ContinuationAuditEvent(
                        invocation_id=invocation_id,
                        tool_name=tool_name,
                        argument_names=arg_names,
                        lifecycle_stage="denied",
                        status=ContinuationStatus.FAILED,
                        reason_code="continuation.already_terminal",
                        success=False,
                        has_output=False,
                        has_error=True,
                        duration_ms=0.0,
                    )
                )
                return ContinuationResult(
                    invocation_id, ContinuationStatus.FAILED, "continuation.already_terminal"
                )
            try:
                assertion = self._authenticator.authenticate(
                    response, owner_secret=_owner_secret
                )
            except Exception as exc:
                code = getattr(exc, "reason_code", "approval.authentication_failed")
                try:
                    self._pending.invalidate(invocation_id, code)
                except Exception:
                    pass
                status = (
                    ContinuationStatus.EXPIRED
                    if code == "approval.challenge_expired"
                    else self._status_for_pending(
                        invocation_id, ContinuationStatus.FAILED
                    )
                )
                return self._finish(invocation_id, status, code)

            request = self._requests[invocation_id]
            if not self._assertion_matches(assertion, request):
                try:
                    self._pending.invalidate(
                        invocation_id, "approval.binding_mismatch"
                    )
                except PendingInvocationError:
                    pass
                return self._finish(
                    invocation_id,
                    self._status_for_pending(
                        invocation_id, ContinuationStatus.FAILED
                    ),
                    "approval.binding_mismatch",
                )

            if assertion.decision is OwnerDecision.REJECT:
                try:
                    self._authenticator.consume_authenticated(assertion)
                    self._pending.cancel(invocation_id)
                except Exception:
                    return self._finish(
                        invocation_id,
                        self._status_for_pending(
                            invocation_id, ContinuationStatus.FAILED
                        ),
                        "approval.rejection_state_invalid",
                    )
                return self._finish(invocation_id, ContinuationStatus.REJECTED, "approval.rejected")

            try:
                if not self._tool_binding_is_current(continuation):
                    self._pending.invalidate(invocation_id, "capability.handler_changed")
                    self._authenticator.consume_authenticated(assertion)
                    return self._finish(invocation_id, ContinuationStatus.INVALIDATED, "capability.handler_changed")
                fresh_policy = continuation.revalidate_policy()
                if not continuation.grant.validate_current_policy(fresh_policy):
                    self._pending.invalidate(invocation_id, "approval.policy_changed")
                    self._authenticator.consume_authenticated(assertion)
                    return self._finish(invocation_id, ContinuationStatus.INVALIDATED, "approval.policy_changed")
                if not self._pending.validate_current(invocation_id):
                    self._authenticator.consume_authenticated(assertion)
                    status = self._status_for_pending(
                        invocation_id, ContinuationStatus.INVALIDATED
                    )
                    return self._finish(invocation_id, status, "approval.invocation_stale")
                # Consume the challenge first. If invocation consumption then
                # fails, this response is still burned and can never be replayed.
                self._authenticator.consume_authenticated(assertion)
                consumed = self._pending._consume_exact_grant(
                    invocation_id, continuation.grant, fresh_policy, assertion
                )
                if consumed.state is not PendingInvocationState.CONSUMED:
                    return self._finish(
                        invocation_id,
                        self._status_for_state(
                            consumed.state, ContinuationStatus.FAILED
                        ),
                        consumed.state_reason_code,
                    )
            except Exception as exc:
                code = getattr(exc, "reason_code", "approval.revalidation_failed")
                try:
                    current = self._pending.get(invocation_id)
                    if current.state is PendingInvocationState.PENDING:
                        self._pending.invalidate(invocation_id, "approval.revalidation_failed")
                except Exception:
                    pass
                status = (
                    ContinuationStatus.EXPIRED
                    if code == "approval.challenge_expired"
                    else self._status_for_pending(
                        invocation_id, ContinuationStatus.FAILED
                    )
                )
                return self._finish(invocation_id, status, code)

            dispatch_started = time.perf_counter()
            arg_names = tuple(continuation.invocation.execution_arguments.keys())
            try:
                # Refuse replacement of the registered handler while waiting.
                if get_registry().get(continuation.tool.name) is not continuation.tool:
                    return self._finish(invocation_id, ContinuationStatus.FAILED, "capability.handler_changed")
                if not self._tool_binding_is_current(continuation):
                    return self._finish(invocation_id, ContinuationStatus.INVALIDATED, "capability.handler_changed")
                self._emit_audit(
                    ContinuationAuditEvent(
                        invocation_id=invocation_id,
                        tool_name=continuation.tool.name,
                        argument_names=arg_names,
                        lifecycle_stage="approved",
                        status=ContinuationStatus.APPROVED,
                        reason_code="approval.authenticated",
                        success=True,
                        has_output=False,
                        has_error=False,
                        duration_ms=0.0,
                    )
                )
                self._emit_audit(
                    ContinuationAuditEvent(
                        invocation_id=invocation_id,
                        tool_name=continuation.tool.name,
                        argument_names=arg_names,
                        lifecycle_stage="dispatched",
                        status=ContinuationStatus.APPROVED,
                        reason_code="execution.dispatched",
                        success=True,
                        has_output=False,
                        has_error=False,
                        duration_ms=0.0,
                    )
                )
                execution_args = dict(continuation.invocation.execution_arguments)
                if continuation.execution_target_adapter is not None:
                    with continuation.execution_target_adapter.open_target(continuation.invocation) as target:
                        outcome = continuation.tool.execute(
                            **execution_args,
                            _verified_file=target,
                        )
                else:
                    outcome = continuation.tool.execute(**execution_args)
                if not isinstance(outcome, ToolResult):
                    outcome = ToolResult(False, "", "Tool returned an invalid result.")
            except Exception as exc:
                outcome = ToolResult(False, "", f"Approved operation failed ({type(exc).__name__}).")
            dispatch_duration_ms = (time.perf_counter() - dispatch_started) * 1000.0
            if outcome.success:
                return self._finish(
                    invocation_id, ContinuationStatus.APPROVED,
                    "approval.consumed", outcome, dispatch_duration_ms,
                )
            return self._finish(
                invocation_id, ContinuationStatus.FAILED,
                "approved_operation.failed", outcome, dispatch_duration_ms,
            )

    def close(self) -> tuple[ContinuationResult, ...]:
        """Fail closed on owner/service shutdown; in-memory work is never resumed."""
        with self._lock:
            self._closed = True
            results = []
            for invocation_id in tuple(self._continuations):
                try:
                    self._pending.cancel(invocation_id)
                    status = ContinuationStatus.CANCELLED
                    reason = "continuation.service_closed"
                except PendingInvocationError as exc:
                    status = self._status_for_pending(
                        invocation_id, ContinuationStatus.FAILED
                    )
                    reason = exc.reason_code
                results.append(self._finish(invocation_id, status, reason))
            return tuple(results)

    def _require_active(self, invocation_id: str) -> _Continuation:
        continuation = self._continuations.get(invocation_id)
        if continuation is None or continuation.status is not ContinuationStatus.PENDING:
            raise PendingInvocationError("continuation.not_pending", "Continuation is not pending.")
        return continuation

    def _finish(self, invocation_id, status, reason_code, tool_result=None, duration_ms=0.0):
        self._authenticator.invalidate_invocation(invocation_id)
        self._requests.pop(invocation_id, None)
        continuation = self._continuations.pop(invocation_id, None)
        if continuation is not None:
            continuation.status = status
        result = ContinuationResult(invocation_id, status, reason_code, tool_result)
        self._results[invocation_id] = result
        if continuation is not None:
            if status is ContinuationStatus.APPROVED:
                stage = "completed"
            elif tool_result is not None:
                stage = "failed"
            elif status is ContinuationStatus.REJECTED:
                stage = "denied"
            else:
                stage = status.value
            self._emit_audit(
                ContinuationAuditEvent(
                    invocation_id=invocation_id,
                    tool_name=continuation.tool.name,
                    argument_names=tuple(continuation.invocation.execution_arguments.keys()),
                    lifecycle_stage=stage,
                    status=status,
                    reason_code=reason_code,
                    success=bool(tool_result is not None and tool_result.success),
                    has_output=bool(tool_result is not None and tool_result.output),
                    has_error=bool(
                        (tool_result is not None and tool_result.error)
                        or status is not ContinuationStatus.APPROVED
                    ),
                    duration_ms=float(duration_ms),
                )
            )
        return result

    def _status_for_pending(
        self, invocation_id: str, fallback: ContinuationStatus
    ) -> ContinuationStatus:
        try:
            record = self._pending.get(invocation_id)
        except PendingInvocationError:
            return fallback
        return self._status_for_state(record.state, fallback)

    @staticmethod
    def _status_for_state(
        state: PendingInvocationState, fallback: ContinuationStatus
    ) -> ContinuationStatus:
        return {
            PendingInvocationState.CANCELLED: ContinuationStatus.CANCELLED,
            PendingInvocationState.EXPIRED: ContinuationStatus.EXPIRED,
            PendingInvocationState.INVALIDATED: ContinuationStatus.INVALIDATED,
            PendingInvocationState.CONSUMED: ContinuationStatus.FAILED,
        }.get(state, fallback)

    @staticmethod
    def _assertion_matches(assertion: AuthenticatedOwnerDecision, request: ApprovalRequest) -> bool:
        return (
            type(assertion) is AuthenticatedOwnerDecision
            and assertion.invocation_id == request.invocation.invocation_id
            and assertion.challenge_id == request.challenge.challenge_id
            and assertion.operation_digest == request.grant.operation_digest
        )

    @staticmethod
    def _tool_binding_is_current(continuation: _Continuation) -> bool:
        if (
            get_registry().get(continuation.tool.name) is not continuation.tool
            or continuation.tool.execute is not continuation.execute_handler
        ):
            return False
        try:
            from .capabilities import legacy_tool_to_manifest

            current = legacy_tool_to_manifest(continuation.tool)
        except Exception:
            return False
        return (
            current.capability_id == continuation.invocation.capability_id
            and current.contract_version == continuation.invocation.contract_version
            and current.manifest_digest == continuation.manifest_digest
        )
