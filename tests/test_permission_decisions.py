import tempfile
import unittest
from dataclasses import FrozenInstanceError

from src.harness.capabilities import CapabilityRegistry, legacy_tool_to_manifest
from src.harness.execution_context import ExecutionContextIssuer
from src.harness.permission_decisions import (
    ApprovalRequest,
    PermissionDecision,
    PermissionDecisionError,
    PermissionDecisionEvaluator,
    PermissionOutcome,
    PermissionReasonCode,
    approval_request_for,
)
from src.harness.permissions import PermissionManager
from src.harness.tools import Tool, ToolArg, ToolResult


class PermissionDecisionTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.issuer = ExecutionContextIssuer(
            lifetime_seconds=10.0, monotonic_clock=lambda: self.now[0]
        )
        self.handler_calls = []
        self.tool = Tool(
            "self_status",
            "Report status",
            [],
            lambda: self.handler_calls.append("called") or ToolResult(True, "unused"),
        )
        self.manifest = legacy_tool_to_manifest(self.tool)
        self.registry = CapabilityRegistry()
        self.registry.register(self.manifest)
        self.evaluator = PermissionDecisionEvaluator(
            registry=self.registry, issuer=self.issuer
        )

    def context(self, allowed=True, reason="from existing check", arguments=None, manifest=None):
        arguments = {} if arguments is None else arguments
        manifest = self.manifest if manifest is None else manifest
        result = (allowed, reason, dict(arguments))
        return self.issuer.issue(manifest, arguments, result)

    def evaluate(self, context, arguments=None, **overrides):
        identity = context
        values = {
            "capability_id": identity.capability_id,
            "contract_version": identity.contract_version,
            "manifest_digest": identity.manifest_digest,
            "arguments": {} if arguments is None else arguments,
            "context": identity,
        }
        values.update(overrides)
        return self.evaluator.evaluate(**values)

    def test_allow_and_ambiguous_false_evidence_are_typed_and_bound(self):
        allowed = self.context(True, "Automatically allowed by policy.")
        decision = self.evaluate(allowed)
        self.assertEqual(decision.outcome, PermissionOutcome.ALLOW)
        self.assertEqual(decision.reason_code, PermissionReasonCode.AUTHORIZATION_ALLOWED)
        self.assertEqual(decision.capability_id, self.manifest.capability_id)
        self.assertEqual(decision.contract_version, self.manifest.contract_version)
        self.assertEqual(decision.manifest_digest, self.manifest.manifest_digest)
        self.assertEqual(decision.argument_digest, allowed.argument_digest)
        self.assertEqual(decision.invocation_id, allowed.invocation_id)
        self.assertEqual(decision.detail, "Automatically allowed by policy.")
        self.assertIsNone(decision.policy_revision)
        self.assertEqual(decision.execution_status, "not_executed")

        # False may mean an unavailable/declined approval, policy denial, or
        # another failure. Human-readable reason text is retained, not parsed.
        denied_or_pending = self.context(False, "requires approval")
        ambiguous = self.evaluate(denied_or_pending)
        self.assertEqual(ambiguous.outcome, PermissionOutcome.INDETERMINATE)
        self.assertEqual(
            ambiguous.reason_code,
            PermissionReasonCode.AUTHORIZATION_OUTCOME_AMBIGUOUS,
        )
        self.assertEqual(ambiguous.detail, "requires approval")
        self.assertEqual(ambiguous.execution_status, "not_executed")

    def test_decisions_are_frozen_and_supports_all_typed_outcomes(self):
        context = self.context()
        for outcome, code in (
            (PermissionOutcome.ALLOW, PermissionReasonCode.AUTHORIZATION_ALLOWED),
            (PermissionOutcome.REQUIRE_APPROVAL, PermissionReasonCode.APPROVAL_REQUIRED),
            (PermissionOutcome.DENY, PermissionReasonCode.AUTHORIZATION_DENIED),
            (PermissionOutcome.INDETERMINATE, PermissionReasonCode.POLICY_EVALUATION_UNAVAILABLE),
        ):
            with self.subTest(outcome=outcome):
                decision = PermissionDecision(
                    outcome=outcome,
                    reason_code=code,
                    capability_id=context.capability_id,
                    contract_version=context.contract_version,
                    manifest_digest=context.manifest_digest,
                    argument_digest=context.argument_digest,
                    invocation_id=context.invocation_id,
                    detail="typed state fixture",
                )
                with self.assertRaises(FrozenInstanceError):
                    decision.outcome = PermissionOutcome.ALLOW
                self.assertEqual(decision.execution_status, "not_executed")

    def test_approval_request_is_bound_and_never_an_authorization_grant(self):
        context = self.context()
        request = approval_request_for(PermissionDecision(
            outcome=PermissionOutcome.REQUIRE_APPROVAL,
            reason_code=PermissionReasonCode.APPROVAL_REQUIRED,
            capability_id=context.capability_id,
            contract_version=context.contract_version,
            manifest_digest=context.manifest_digest,
            argument_digest=context.argument_digest,
            invocation_id=context.invocation_id,
            detail="explicit approval state from a future typed policy source",
        ))
        self.assertIsInstance(request, ApprovalRequest)
        self.assertEqual(request.capability_id, context.capability_id)
        self.assertEqual(request.contract_version, context.contract_version)
        self.assertEqual(request.manifest_digest, context.manifest_digest)
        self.assertEqual(request.argument_digest, context.argument_digest)
        self.assertEqual(request.invocation_id, context.invocation_id)
        self.assertEqual(request.status, "request_only_not_authorized")
        self.assertFalse(hasattr(request, "approved"))
        self.assertFalse(hasattr(request, "authorization_granted"))
        with self.assertRaises(FrozenInstanceError):
            request.status = "approved"
        with self.assertRaises(ValueError):
            approval_request_for(self.evaluate(context))

    def test_unknown_capability_and_stale_manifest_never_allow(self):
        context = self.context()
        empty = PermissionDecisionEvaluator(
            registry=CapabilityRegistry(), issuer=self.issuer
        )
        unknown = empty.evaluate(
            capability_id=context.capability_id,
            contract_version=context.contract_version,
            manifest_digest=context.manifest_digest,
            arguments={},
            context=context,
        )
        self.assertEqual(unknown.outcome, PermissionOutcome.INDETERMINATE)
        self.assertEqual(unknown.reason_code, PermissionReasonCode.CAPABILITY_UNKNOWN)

        changed_manifest = legacy_tool_to_manifest(Tool(
            "self_status", "Changed contract summary", [], self.tool.execute
        ))
        current = CapabilityRegistry()
        current.register(changed_manifest)
        stale_evaluator = PermissionDecisionEvaluator(
            registry=current, issuer=self.issuer
        )
        stale = stale_evaluator.evaluate(
            capability_id=context.capability_id,
            contract_version=context.contract_version,
            manifest_digest=context.manifest_digest,
            arguments={},
            context=context,
        )
        self.assertEqual(stale.outcome, PermissionOutcome.INDETERMINATE)
        self.assertEqual(stale.reason_code, PermissionReasonCode.MANIFEST_MISMATCH)

    def test_requested_binding_argument_mismatch_and_expiry_never_allow(self):
        context = self.context(arguments={"path": "a"})
        wrong_request = self.evaluate(
            context,
            arguments={"path": "a"},
            capability_id="org.lila.capability.other",
        )
        self.assertEqual(wrong_request.outcome, PermissionOutcome.INDETERMINATE)
        self.assertEqual(
            wrong_request.reason_code,
            PermissionReasonCode.REQUEST_BINDING_MISMATCH,
        )

        wrong_args = self.evaluate(context, arguments={"path": "b"})
        self.assertEqual(wrong_args.outcome, PermissionOutcome.INDETERMINATE)
        self.assertEqual(
            wrong_args.reason_code,
            PermissionReasonCode.CONTEXT_ARGUMENT_MISMATCH,
        )

        expired_context = self.context()
        self.now[0] = 111.0
        expired = self.evaluate(expired_context)
        self.assertEqual(expired.outcome, PermissionOutcome.INDETERMINATE)
        self.assertEqual(expired.reason_code, PermissionReasonCode.CONTEXT_EXPIRED)

    def test_missing_and_forged_context_fail_closed(self):
        with self.assertRaises(PermissionDecisionError) as missing:
            self.evaluator.evaluate(
                capability_id=self.manifest.capability_id,
                contract_version=self.manifest.contract_version,
                manifest_digest=self.manifest.manifest_digest,
                arguments={},
                context=None,
            )
        self.assertEqual(
            missing.exception.outcome, PermissionOutcome.INDETERMINATE
        )

        context = self.context()
        other_issuer = ExecutionContextIssuer(
            lifetime_seconds=10.0, monotonic_clock=lambda: self.now[0]
        )
        evaluator = PermissionDecisionEvaluator(
            registry=self.registry, issuer=other_issuer
        )
        with self.assertRaises(PermissionDecisionError):
            evaluator.evaluate(
                capability_id=context.capability_id,
                contract_version=context.contract_version,
                manifest_digest=context.manifest_digest,
                arguments={},
                context=context,
            )

    def test_evaluation_does_not_reauthorize_or_invoke_handlers(self):
        callback_calls = []
        manager = PermissionManager(
            approval_callback=lambda tool, args, risk: callback_calls.append(tool.name) or True
        )
        approved_tool = Tool(
            "file_write",
            "write",
            [ToolArg("path", "path", "")],
            lambda **_: self.handler_calls.append("write") or ToolResult(True, "unused"),
        )
        manifest = legacy_tool_to_manifest(approved_tool)
        registry = CapabilityRegistry()
        registry.register(manifest)
        with tempfile.TemporaryDirectory() as directory:
            arguments = {"path": directory + "/file.txt"}
            evidence = manager.authorize_with_context(approved_tool, arguments)
        context = self.issuer.issue(manifest, arguments, evidence)
        self.assertEqual(callback_calls, ["file_write"])

        evaluator = PermissionDecisionEvaluator(registry=registry, issuer=self.issuer)
        decision = evaluator.evaluate(
            capability_id=manifest.capability_id,
            contract_version=manifest.contract_version,
            manifest_digest=manifest.manifest_digest,
            arguments=arguments,
            context=context,
        )
        self.assertEqual(decision.outcome, PermissionOutcome.ALLOW)
        self.assertEqual(callback_calls, ["file_write"])
        self.assertEqual(self.handler_calls, [])
        self.assertFalse(hasattr(evaluator, "authorize"))
        self.assertFalse(hasattr(evaluator, "execute"))

    def test_permission_revision_is_not_fabricated(self):
        decision = self.evaluate(self.context())
        self.assertIsNone(decision.policy_revision)


if __name__ == "__main__":
    unittest.main()
