import hashlib
import hmac
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from src.harness.approval_authentication import (
    ApprovalDecisionResponse,
    OwnerDecision,
    approval_proof_message,
)
from src.harness.approval_continuation import (
    ApprovalContinuationService,
    ContinuationStatus,
    OwnerAuthenticationSecret,
)
from src.harness.capabilities import CapabilityRegistry, legacy_tool_to_manifest
from src.harness.executor import HarnessExecutor
from src.harness.pending_invocations import (
    PendingInvocationRegistry,
    TargetBindingResolverRegistry,
)
from src.harness.tools import Tool, ToolArg, ToolResult, get_registry, register_tool


class _SecretProvider:
    def __init__(self, secret):
        self.secret = secret

    def get_owner_authentication_secret(self):
        return OwnerAuthenticationSecret("test-owner-key", self.secret)


class ApprovalContinuationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.registry = get_registry()
        self.previous = dict(self.registry._tools)
        self.registry._tools.clear()
        self.calls = []
        self.tool = Tool(
            "bash", "test approval operation",
            [ToolArg("command", "string", "command")],
            lambda command: self._execute(command),
        )
        register_tool(self.tool)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._restore)
        self.secret = b"test-only-owner-secret-material-32-bytes!"
        manifest = legacy_tool_to_manifest(self.tool)
        self.catalog = CapabilityRegistry()
        self.catalog.register(manifest)
        self.resolvers = TargetBindingResolverRegistry()
        self.resolvers.register(
            manifest,
            resolver_id="test.bash.command-target",
            target_required=True,
            target_kind="shell-command",
            resolver=lambda _manifest, arguments, _execution: {
                "command": arguments["command"]
            },
        )
        self.pending = PendingInvocationRegistry(
            capabilities=self.catalog,
            target_resolvers=self.resolvers,
            lifetime_seconds=120,
        )
        self.service = ApprovalContinuationService(
            pending_registry=self.pending,
            owner_secret_provider=_SecretProvider(self.secret),
        )

    def _execute(self, command):
        self.calls.append(command)
        return ToolResult(True, "executed")

    def _restore(self):
        self.registry._tools.clear()
        self.registry._tools.update(self.previous)

    def _executor(self, service=None, callback=None):
        return HarnessExecutor(
            log_dir=self.tmp.name,
            approval_callback=callback,
            workspace_root=self.tmp.name,
            approval_continuation_service=service,
        )

    def _response(self, request, decision=OwnerDecision.APPROVE):
        challenge = request.challenge
        proof = hmac.new(
            self.secret, approval_proof_message(challenge, decision), hashlib.sha256
        ).digest()
        return ApprovalDecisionResponse(
            challenge_id=challenge.challenge_id,
            key_id=challenge.key_id,
            invocation_id=request.invocation.invocation_id,
            operation_digest=request.grant.operation_digest,
            decision=decision,
            proof=proof,
        )

    def test_production_default_is_unavailable_and_legacy_true_is_ignored(self):
        callback_calls = []
        result = self._executor(callback=lambda *args: callback_calls.append(args) or True).process(
            '<tool_call>{"name":"bash","arguments":{"command":"run"}}</tool_call>'
        )[0]
        self.assertEqual(result["approval"].status, ContinuationStatus.UNAVAILABLE)
        self.assertEqual(result["record"].permission_decision, "approval_unavailable")
        self.assertEqual(callback_calls, [])
        self.assertEqual(self.calls, [])

    def test_executor_suspends_then_exact_authenticated_approval_dispatches_once(self):
        executor = self._executor(self.service)
        from unittest.mock import Mock
        policy_evaluation = Mock(wraps=executor.permissions._evaluate_policy_once)
        executor.permissions._evaluate_policy_once = policy_evaluation
        result = executor.process(
            '<tool_call>{"name":"bash","arguments":{"command":"run"}}</tool_call>'
        )[0]
        self.assertEqual(result["approval"].status, ContinuationStatus.PENDING)
        self.assertEqual(result["record"].permission_decision, "approval_pending")
        self.assertEqual(policy_evaluation.call_count, 1)
        self.assertEqual(self.calls, [])

        invocation_id = result["approval"].invocation_id
        request = self.service.get_request(invocation_id)
        with self.assertRaises(TypeError):
            request.invocation.execution_arguments["command"] = "substituted"
        completed = self.service.respond(self._response(request))
        self.assertEqual(completed.status, ContinuationStatus.APPROVED, completed.reason_code)
        self.assertEqual(policy_evaluation.call_count, 2)  # one fresh recheck at resume
        self.assertTrue(completed.tool_result.success)
        self.assertEqual(self.calls, ["run"])
        self.assertEqual(self.pending.get(invocation_id).state.value, "consumed")

        replay = self.service.respond(self._response(request))
        self.assertEqual(replay.status, ContinuationStatus.FAILED)
        self.assertEqual(self.calls, ["run"])

    def test_rejection_terminates_without_dispatch(self):
        result = self._executor(self.service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"no"}}</tool_call>'
        )[0]
        request = self.service.get_request(result["approval"].invocation_id)
        rejected = self.service.respond(self._response(request, OwnerDecision.REJECT))
        self.assertEqual(rejected.status, ContinuationStatus.REJECTED)
        self.assertEqual(self.pending.get(request.invocation.invocation_id).state.value, "cancelled")
        self.assertEqual(self.calls, [])

    def test_forged_response_invalidates_without_dispatch(self):
        result = self._executor(self.service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"no"}}</tool_call>'
        )[0]
        request = self.service.get_request(result["approval"].invocation_id)
        response = self._response(request)
        forged = ApprovalDecisionResponse(
            response.challenge_id, response.key_id, response.invocation_id,
            response.operation_digest, response.decision, b"x" * 32,
        )
        outcome = self.service.respond(forged)
        self.assertEqual(outcome.status, ContinuationStatus.INVALIDATED)
        self.assertEqual(self.pending.get(response.invocation_id).state.value, "invalidated")
        self.assertEqual(self.calls, [])

    def test_missing_explicit_target_rule_fails_closed(self):
        no_resolver_pending = PendingInvocationRegistry(
            capabilities=self.catalog,
            target_resolvers=TargetBindingResolverRegistry(),
            lifetime_seconds=120,
        )
        service = ApprovalContinuationService(
            pending_registry=no_resolver_pending,
            owner_secret_provider=_SecretProvider(self.secret),
        )
        result = self._executor(service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"no"}}</tool_call>'
        )[0]
        self.assertEqual(result["approval"].status, ContinuationStatus.UNAVAILABLE)
        self.assertEqual(result["approval"].reason_code, "target.binding_policy_unavailable")
        self.assertEqual(self.calls, [])

    def test_current_policy_change_invalidates_before_dispatch(self):
        executor = self._executor(self.service)
        result = executor.process(
            '<tool_call>{"name":"bash","arguments":{"command":"stale"}}</tool_call>'
        )[0]
        request = self.service.get_request(result["approval"].invocation_id)
        # A typed result for a different current policy identity must not
        # validate the captured approval, even with a valid owner proof.
        from src.harness.tools import Tool
        disabled = Tool("memory_store", "disabled", [], lambda: ToolResult(True, ""))
        fresh_denial = executor.permissions.evaluate_policy(disabled, {})
        executor.permissions.evaluate_policy = lambda *_args: fresh_denial
        outcome = self.service.respond(self._response(request))
        self.assertEqual(outcome.status, ContinuationStatus.INVALIDATED)
        self.assertEqual(self.pending.get(request.invocation.invocation_id).state.value, "invalidated")
        self.assertEqual(self.calls, [])

    def test_cancellation_revokes_challenge_and_never_dispatches(self):
        result = self._executor(self.service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"cancel"}}</tool_call>'
        )[0]
        invocation_id = result["approval"].invocation_id
        cancelled = self.service.cancel(invocation_id)
        self.assertEqual(cancelled.status, ContinuationStatus.CANCELLED)
        self.assertEqual(self.pending.get(invocation_id).state.value, "cancelled")
        with self.assertRaises(Exception):
            self.service.get_request(invocation_id)
        self.assertEqual(self.calls, [])

    def test_explicit_invalidation_terminates_continuation_without_dispatch(self):
        result = self._executor(self.service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"invalidate"}}</tool_call>'
        )[0]
        invocation_id = result["approval"].invocation_id
        request = self.service.get_request(invocation_id)
        outcome = self.service.invalidate(invocation_id, "target.changed")
        self.assertEqual(outcome.status, ContinuationStatus.INVALIDATED)
        self.assertEqual(self.pending.get(invocation_id).state.value, "invalidated")
        self.assertEqual(self.calls, [])
        self.assertEqual(
            self.service.respond(self._response(request)).status,
            ContinuationStatus.FAILED,
        )

    def test_service_shutdown_cancels_pending_continuations(self):
        result = self._executor(self.service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"shutdown"}}</tool_call>'
        )[0]
        invocation_id = result["approval"].invocation_id
        request = self.service.get_request(invocation_id)

        closed = self.service.close()

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].status, ContinuationStatus.CANCELLED)
        self.assertEqual(self.pending.get(invocation_id).state.value, "cancelled")
        self.assertEqual(
            self.service.respond(self._response(request)).status,
            ContinuationStatus.UNAVAILABLE,
        )
        self.assertEqual(self.calls, [])

    def test_expired_challenge_invalidates_and_never_dispatches(self):
        now = [100.0]
        pending = PendingInvocationRegistry(
            capabilities=self.catalog,
            target_resolvers=self.resolvers,
            lifetime_seconds=120,
            monotonic_clock=lambda: now[0],
        )
        service = ApprovalContinuationService(
            pending_registry=pending,
            owner_secret_provider=_SecretProvider(self.secret),
            challenge_lifetime_seconds=10,
            monotonic_clock=lambda: now[0],
        )
        result = self._executor(service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"late"}}</tool_call>'
        )[0]
        invocation_id = result["approval"].invocation_id
        request = service.get_request(invocation_id)
        now[0] = 110.0
        outcome = service.respond(self._response(request))
        self.assertEqual(outcome.status, ContinuationStatus.EXPIRED)
        self.assertEqual(pending.get(invocation_id).state.value, "invalidated")
        self.assertEqual(self.calls, [])

    def test_expiry_during_atomic_consumption_is_reported_as_expired(self):
        now = [300.0]
        pending = PendingInvocationRegistry(
            capabilities=self.catalog,
            target_resolvers=self.resolvers,
            lifetime_seconds=5,
            monotonic_clock=lambda: now[0],
        )
        service = ApprovalContinuationService(
            pending_registry=pending,
            owner_secret_provider=_SecretProvider(self.secret),
            challenge_lifetime_seconds=60,
            monotonic_clock=lambda: now[0],
        )
        result = self._executor(service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"race-expiry"}}</tool_call>'
        )[0]
        invocation_id = result["approval"].invocation_id
        request = service.get_request(invocation_id)
        consume = pending._consume_exact_grant

        def expire_before_consume(*args):
            now[0] = 305.0
            return consume(*args)

        pending._consume_exact_grant = expire_before_consume
        outcome = service.respond(self._response(request))

        self.assertEqual(outcome.status, ContinuationStatus.EXPIRED)
        self.assertEqual(pending.get(invocation_id).state.value, "expired")
        self.assertEqual(self.calls, [])

    def test_cancel_or_invalidation_during_atomic_consumption_is_reported(self):
        terminal_cases = (
            ("cancel", ContinuationStatus.CANCELLED, "cancelled"),
            ("invalidate", ContinuationStatus.INVALIDATED, "invalidated"),
        )
        for action, expected_status, expected_state in terminal_cases:
            with self.subTest(action=action):
                pending = PendingInvocationRegistry(
                    capabilities=self.catalog,
                    target_resolvers=self.resolvers,
                    lifetime_seconds=120,
                )
                service = ApprovalContinuationService(
                    pending_registry=pending,
                    owner_secret_provider=_SecretProvider(self.secret),
                )
                result = self._executor(service).process(
                    '<tool_call>{"name":"bash","arguments":{"command":"race-' +
                    action +
                    '"}}</tool_call>'
                )[0]
                invocation_id = result["approval"].invocation_id
                request = service.get_request(invocation_id)
                consume = pending._consume_exact_grant

                def terminate_before_consume(*args):
                    if action == "cancel":
                        pending.cancel(invocation_id)
                    else:
                        pending.invalidate(invocation_id, "race.invalidated")
                    return consume(*args)

                pending._consume_exact_grant = terminate_before_consume
                outcome = service.respond(self._response(request))

                self.assertEqual(outcome.status, expected_status)
                self.assertEqual(
                    pending.get(invocation_id).state.value,
                    expected_state,
                )
                self.assertEqual(self.calls, [])

    def test_substituted_operation_digest_invalidates_without_dispatch(self):
        result = self._executor(self.service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"substitute"}}</tool_call>'
        )[0]
        invocation_id = result["approval"].invocation_id
        request = self.service.get_request(invocation_id)
        response = self._response(request)
        substituted = replace(response, operation_digest="0" * 64)

        outcome = self.service.respond(substituted)

        self.assertEqual(outcome.status, ContinuationStatus.INVALIDATED)
        self.assertEqual(self.pending.get(invocation_id).state.value, "invalidated")
        self.assertEqual(self.calls, [])

    def test_target_drift_invalidates_before_dispatch(self):
        target_state = ["original"]
        resolvers = TargetBindingResolverRegistry()
        manifest = legacy_tool_to_manifest(self.tool)
        resolvers.register(
            manifest,
            resolver_id="test.bash.target",
            target_required=True,
            target_kind="test-resource",
            resolver=lambda _manifest, args, _execution: {
                "command": args["command"], "target": target_state[0]
            },
        )
        pending = PendingInvocationRegistry(
            capabilities=self.catalog,
            target_resolvers=resolvers,
            lifetime_seconds=120,
        )
        service = ApprovalContinuationService(
            pending_registry=pending,
            owner_secret_provider=_SecretProvider(self.secret),
        )
        result = self._executor(service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"drift"}}</tool_call>'
        )[0]
        invocation_id = result["approval"].invocation_id
        request = service.get_request(invocation_id)
        target_state[0] = "changed"
        outcome = service.respond(self._response(request))
        self.assertEqual(outcome.status, ContinuationStatus.INVALIDATED)
        self.assertEqual(pending.get(invocation_id).state.value, "invalidated")
        self.assertEqual(self.calls, [])

    def test_manifest_or_handler_drift_invalidates_before_dispatch(self):
        result = self._executor(self.service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"manifest"}}</tool_call>'
        )[0]
        request = self.service.get_request(result["approval"].invocation_id)
        self.tool.description = "changed while approval was pending"
        outcome = self.service.respond(self._response(request))
        self.assertEqual(outcome.status, ContinuationStatus.INVALIDATED)
        self.assertEqual(self.pending.get(request.invocation.invocation_id).state.value, "invalidated")
        self.assertEqual(self.calls, [])

    def test_pending_invocation_expiry_prevents_valid_challenge_dispatch(self):
        now = [200.0]
        pending = PendingInvocationRegistry(
            capabilities=self.catalog,
            target_resolvers=self.resolvers,
            lifetime_seconds=5,
            monotonic_clock=lambda: now[0],
        )
        service = ApprovalContinuationService(
            pending_registry=pending,
            owner_secret_provider=_SecretProvider(self.secret),
            challenge_lifetime_seconds=60,
            monotonic_clock=lambda: now[0],
        )
        result = self._executor(service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"expired"}}</tool_call>'
        )[0]
        invocation_id = result["approval"].invocation_id
        request = service.get_request(invocation_id)
        now[0] = 205.0
        outcome = service.respond(self._response(request))
        self.assertEqual(outcome.status, ContinuationStatus.EXPIRED)
        self.assertEqual(pending.get(invocation_id).state.value, "expired")
        self.assertEqual(self.calls, [])

    def test_concurrent_identical_responses_dispatch_at_most_once(self):
        result = self._executor(self.service).process(
            '<tool_call>{"name":"bash","arguments":{"command":"once"}}</tool_call>'
        )[0]
        request = self.service.get_request(result["approval"].invocation_id)
        response = self._response(request)
        barrier = threading.Barrier(3)

        def consume():
            barrier.wait()
            return self.service.respond(response)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(consume) for _ in range(2)]
            barrier.wait()
            outcomes = [future.result() for future in futures]
        self.assertEqual(self.calls, ["once"])
        self.assertEqual(sum(item.status is ContinuationStatus.APPROVED for item in outcomes), 1)
        self.assertEqual(sum(item.status is ContinuationStatus.FAILED for item in outcomes), 1)


if __name__ == "__main__":
    unittest.main()
