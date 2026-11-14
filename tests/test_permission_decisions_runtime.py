import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.harness.capabilities import CapabilityRegistry, legacy_tool_to_manifest
from src.harness.execution_context import ExecutionContextIssuer, argument_digest
from src.harness.executor import HarnessExecutor
from src.harness.permission_decisions import (
    PermissionDecision,
    PermissionDecisionObserver,
    PermissionOutcome,
    PermissionReasonCode,
)
from src.harness.tools import Tool, ToolArg, ToolResult, get_registry, register_tool
from src.core.lilacore import LilaCore


def tool_call(name, arguments=None):
    payload = {"name": name, "arguments": arguments or {}}
    return "<tool_call>" + json.dumps(payload) + "</tool_call>"


class PermissionDecisionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.global_registry = get_registry()
        self.previous_tools = dict(self.global_registry._tools)
        self.global_registry._tools.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.restore_global_registry)
        self.handler_calls = []

        def handler(**arguments):
            self.handler_calls.append(dict(arguments))
            return ToolResult(True, arguments.get("content", "handled"))

        definitions = (
            Tool("self_status", "status", [], handler),
            Tool(
                "bash", "sensitive operation",
                [ToolArg("command", "string", "command")], handler,
            ),
            Tool(
                "file_write", "write file",
                [ToolArg("path", "path", "path"), ToolArg("content", "string", "content")],
                handler,
            ),
        )
        for definition in definitions:
            register_tool(definition)
        self.tools = {tool.name: tool for tool in self.global_registry.all()}

        self.catalog = CapabilityRegistry()
        for tool in self.tools.values():
            self.catalog.register(legacy_tool_to_manifest(tool))
        self.now = [10.0]
        self.issuer = ExecutionContextIssuer(
            lifetime_seconds=5.0, monotonic_clock=lambda: self.now[0]
        )

    def restore_global_registry(self):
        self.global_registry._tools.clear()
        self.global_registry._tools.update(self.previous_tools)

    def make_executor(self, *, approval_callback=None, catalog=None):
        return HarnessExecutor(
            log_dir=self.temp_dir.name,
            approval_callback=approval_callback,
            workspace_root=self.temp_dir.name,
            shadow_capability_registry=(self.catalog if catalog is None else catalog),
            shadow_context_issuer=self.issuer,
            shadow_observation_enabled=True,
        )

    def manifest_for(self, name):
        matches = [
            manifest for manifest in self.catalog.all()
            if manifest.legacy_provenance.get("legacy_tool_name") == name
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_real_process_path_uses_one_permission_result_and_records_bound_decision(self):
        executor = self.make_executor()
        authorize = executor.permissions.evaluate_for_executor
        produced = []

        def authorize_once(tool, arguments):
            result = authorize(tool, arguments)
            produced.append(result)
            return result

        executor.permissions.evaluate_for_executor = Mock(side_effect=authorize_once)
        issue = Mock(wraps=self.issuer.issue)
        executor._shadow_context_issuer.issue = issue
        evaluate = Mock(wraps=executor._shadow_evaluator.evaluate)
        executor._shadow_evaluator.evaluate = evaluate

        result = executor.process(tool_call("self_status"))[0]

        self.assertTrue(result["result"].success)
        self.assertEqual(len(self.handler_calls), 1)
        executor.permissions.evaluate_for_executor.assert_called_once()
        issue.assert_called_once()
        self.assertEqual(
            issue.call_args.args[2],
            executor.permissions.legacy_result_for_policy(produced[0][0], produced[0][1]),
        )
        evaluate.assert_called_once()

        observation, = executor.permission_decision_observations
        self.assertEqual(observation.status, "evaluated")
        self.assertEqual(observation.outcome, PermissionOutcome.ALLOW)
        self.assertEqual(observation.reason_code, "authorization.allowed")
        self.assertEqual(observation.capability_id, self.manifest_for("self_status").capability_id)
        self.assertEqual(observation.argument_digest, argument_digest({}))
        self.assertTrue(observation.invocation_id)

    def test_live_denial_stays_denied_and_approval_ambiguity_never_runs_handler(self):
        executor = self.make_executor()
        result = executor.process(tool_call("bash", {"command": "do it"}))[0]

        self.assertFalse(result["result"].success)
        self.assertEqual(result["record"].permission_decision, "approval_unavailable")
        self.assertEqual(self.handler_calls, [])
        observation, = executor.permission_decision_observations
        self.assertEqual(observation.outcome, PermissionOutcome.INDETERMINATE)
        self.assertEqual(
            observation.reason_code,
            PermissionReasonCode.AUTHORIZATION_OUTCOME_AMBIGUOUS.value,
        )

    def test_shadow_allow_cannot_override_live_denial(self):
        executor = self.make_executor()
        manifest = self.manifest_for("bash")
        fake_allow = PermissionDecision(
            outcome=PermissionOutcome.ALLOW,
            reason_code=PermissionReasonCode.AUTHORIZATION_ALLOWED,
            capability_id=manifest.capability_id,
            contract_version=manifest.contract_version,
            manifest_digest=manifest.manifest_digest,
            argument_digest=argument_digest({"command": "do it"}),
            invocation_id="test-invocation",
            detail="shadow fixture",
        )
        executor._shadow_evaluator.evaluate = Mock(return_value=fake_allow)

        result = executor.process(tool_call("bash", {"command": "do it"}))[0]

        self.assertFalse(result["result"].success)
        self.assertEqual(result["record"].permission_decision, "approval_unavailable")
        self.assertEqual(self.handler_calls, [])

    def test_shadow_deny_cannot_override_live_allow(self):
        executor = self.make_executor()
        manifest = self.manifest_for("self_status")
        fake_deny = PermissionDecision(
            outcome=PermissionOutcome.DENY,
            reason_code=PermissionReasonCode.AUTHORIZATION_DENIED,
            capability_id=manifest.capability_id,
            contract_version=manifest.contract_version,
            manifest_digest=manifest.manifest_digest,
            argument_digest=argument_digest({}),
            invocation_id="test-invocation",
            detail="shadow fixture",
        )
        executor._shadow_evaluator.evaluate = Mock(return_value=fake_deny)

        result = executor.process(tool_call("self_status"))[0]

        self.assertTrue(result["result"].success)
        self.assertEqual(result["record"].permission_decision, "automatic")
        self.assertEqual(len(self.handler_calls), 1)
        observation, = executor.permission_decision_observations
        self.assertEqual(observation.outcome, PermissionOutcome.DENY)

    def test_context_issuer_exception_cannot_change_live_denial(self):
        executor = self.make_executor()
        executor._shadow_context_issuer.issue = Mock(
            side_effect=RuntimeError("private issuer failure detail")
        )

        with self.assertLogs("src.harness.executor", level="WARNING") as captured:
            result = executor.process(tool_call("bash", {"command": "do it"}))[0]

        self.assertFalse(result["result"].success)
        self.assertEqual(result["record"].permission_decision, "approval_unavailable")
        self.assertEqual(self.handler_calls, [])
        observation, = executor.permission_decision_observations
        self.assertEqual(observation.status, "unavailable")
        self.assertEqual(observation.outcome, PermissionOutcome.INDETERMINATE)
        self.assertNotIn("private issuer failure detail", "\n".join(captured.output))

    def test_indeterminate_or_approval_required_shadow_state_never_grants(self):
        executor = self.make_executor()
        manifest = self.manifest_for("bash")
        for outcome, reason_code in (
            (PermissionOutcome.INDETERMINATE, PermissionReasonCode.POLICY_EVALUATION_UNAVAILABLE),
            (PermissionOutcome.REQUIRE_APPROVAL, PermissionReasonCode.APPROVAL_REQUIRED),
        ):
            with self.subTest(outcome=outcome):
                executor._shadow_evaluator.evaluate = Mock(return_value=PermissionDecision(
                    outcome=outcome,
                    reason_code=reason_code,
                    capability_id=manifest.capability_id,
                    contract_version=manifest.contract_version,
                    manifest_digest=manifest.manifest_digest,
                    argument_digest=argument_digest({"command": "do it"}),
                    invocation_id="test-invocation",
                    detail="typed shadow fixture",
                ))
                result = executor.process(
                    tool_call("bash", {"command": "do it"})
                )[0]
                self.assertFalse(result["result"].success)
                self.assertEqual(self.handler_calls, [])

    def test_shadow_failures_are_sanitized_and_do_not_change_live_allow(self):
        executor = self.make_executor()
        executor._shadow_evaluator.evaluate = Mock(
            side_effect=RuntimeError("private payload must not enter diagnostics")
        )
        with self.assertLogs("src.harness.executor", level="WARNING") as captured:
            result = executor.process(tool_call("self_status"))[0]

        self.assertTrue(result["result"].success)
        self.assertEqual(len(self.handler_calls), 1)
        observation, = executor.permission_decision_observations
        self.assertEqual(observation.status, "unavailable")
        self.assertEqual(observation.outcome, PermissionOutcome.INDETERMINATE)
        self.assertNotIn("private payload", "\n".join(captured.output))
        self.assertNotIn("private payload", repr(observation))

    def test_observer_failure_does_not_change_live_allow(self):
        executor = self.make_executor()
        executor._permission_decision_observer.record_decision = Mock(
            side_effect=RuntimeError("sensitive observation failure detail")
        )
        with self.assertLogs("src.harness.executor", level="WARNING") as captured:
            result = executor.process(tool_call("self_status"))[0]
        self.assertTrue(result["result"].success)
        self.assertEqual(len(self.handler_calls), 1)
        self.assertNotIn("sensitive observation failure detail", "\n".join(captured.output))

    def test_missing_catalog_binding_records_indeterminate_without_shadow_allow(self):
        empty = CapabilityRegistry()
        executor = self.make_executor(catalog=empty)
        with self.assertLogs("src.harness.executor", level="WARNING"):
            result = executor.process(tool_call("self_status"))[0]

        self.assertTrue(result["result"].success)
        self.assertEqual(len(self.handler_calls), 1)
        observation, = executor.permission_decision_observations
        self.assertEqual(observation.outcome, PermissionOutcome.INDETERMINATE)
        self.assertEqual(observation.reason_code, "capability.binding_missing")
        self.assertIsNone(observation.capability_id)

    def test_invalid_context_never_allows_but_legacy_allow_is_preserved(self):
        executor = self.make_executor()
        executor._shadow_context_issuer.issue = Mock(return_value=None)
        with self.assertLogs("src.harness.executor", level="WARNING"):
            result = executor.process(tool_call("self_status"))[0]

        self.assertTrue(result["result"].success)
        self.assertEqual(len(self.handler_calls), 1)
        observation, = executor.permission_decision_observations
        self.assertEqual(observation.outcome, PermissionOutcome.INDETERMINATE)
        self.assertEqual(observation.reason_code, "authorization.evidence_unavailable")

    def test_observations_are_bounded_and_do_not_store_arguments_or_tool_results(self):
        callback = Mock(return_value=True)
        executor = self.make_executor(approval_callback=callback)
        secret = "secret-payload-do-not-record"
        result = executor.process(tool_call("file_write", {
            "path": self.temp_dir.name + "/safe.txt",
            "content": secret,
        }))[0]
        self.assertFalse(result["result"].success)
        self.assertEqual(result["approval"].status.value, "unavailable")
        callback.assert_not_called()
        snapshot = executor.permission_decision_observations
        self.assertEqual(len(snapshot), 1)
        self.assertNotIn(secret, repr(snapshot))
        self.assertNotIn(secret, repr(snapshot))

        small = PermissionDecisionObserver(capacity=1)
        manifest = self.manifest_for("file_write")
        decision = PermissionDecision(
            outcome=PermissionOutcome.ALLOW,
            reason_code=PermissionReasonCode.AUTHORIZATION_ALLOWED,
            capability_id=manifest.capability_id,
            contract_version=manifest.contract_version,
            manifest_digest=manifest.manifest_digest,
            argument_digest=snapshot[0].argument_digest,
            invocation_id=snapshot[0].invocation_id,
            detail=secret,
        )
        small.record_decision(decision)
        small.record_decision(decision)
        self.assertEqual(len(small.snapshot()), 1)
        self.assertNotIn(secret, repr(small.snapshot()))

    def test_model_cannot_supply_the_trusted_invocation_id(self):
        executor = self.make_executor()
        result = executor.process(tool_call("self_status", {"invocation_id": "model-choice"}))[0]
        self.assertFalse(result["result"].success)
        self.assertEqual(self.handler_calls, [])
        self.assertEqual(executor.permission_decision_observations, ())

    def test_lilacore_associates_its_catalog_with_the_live_executor_shadow(self):
        core = object.__new__(LilaCore)
        core.workstation_enabled = False
        core._capability_catalog = self.catalog
        core._capability_catalog_error = None
        core.log_dir = Path(self.temp_dir.name)
        core.workspace_root = Path(self.temp_dir.name)
        core.approval_callback = None
        core.approval_continuation_service = None
        core._harness = None
        with (
            patch.object(LilaCore, "_build_capability_catalog", return_value=self.catalog),
            patch("src.harness.system_tools.register_all_system_tools"),
            patch("src.harness.extended_tools.register_extended_tools"),
        ):
            core._boot_harness()
        self.assertIs(core._harness._shadow_capability_registry, self.catalog)
        self.assertIs(core._harness._shadow_evaluator._registry, self.catalog)
        self.assertIsInstance(core._harness._shadow_context_issuer, ExecutionContextIssuer)
        self.assertTrue(core._harness._shadow_observation_enabled)
        self.assertIsNone(core._harness._approval_continuation_service)


if __name__ == "__main__":
    unittest.main()
