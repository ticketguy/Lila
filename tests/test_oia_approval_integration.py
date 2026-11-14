import tempfile
import unittest

from src.harness.executor import HarnessExecutor
from src.harness.tools import Tool, ToolArg, ToolResult, get_registry, register_tool
from src.daemon.oia_harness import (
    _authentication_request,
    authenticate_spoken_decision,
    authenticate_typed_decision,
    build_approval_service,
)
from src.oia import OIAError, OwnerAuthenticationService, OwnerProfile


PHRASE = "violet orchard window copper lantern river"
OWNER_KEY = bytearray(b"integration-owner-secret-for-tests-32b")


class FakeOwnerStore:
    enrolled = True

    def profile(self):
        return OwnerProfile("test-owner", "owner-test-key", 0)

    def credential(self):
        raise AssertionError("The test authentication service overrides credential unlock.")


class OIAApprovalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.registry = get_registry()
        self.previous = dict(self.registry._tools)
        self.registry._tools.clear()
        self.addCleanup(self._restore_registry)
        self.calls = []
        self.tool = Tool(
            "bash", "run a command", [ToolArg("command", "string", "command")],
            lambda command: self._execute(command),
        )
        register_tool(self.tool)
        self.owner = OwnerAuthenticationService(FakeOwnerStore())
        self.owner._unlock = self._unlock
        self.service = build_approval_service(self.owner, self.registry)
        self.executor = HarnessExecutor(
            log_dir=self.temp.name,
            workspace_root=self.temp.name,
            approval_continuation_service=self.service,
        )

    @staticmethod
    def _unlock(phrase, *, source):
        if phrase != PHRASE:
            raise OIAError("owner_authentication.failed")
        return bytearray(OWNER_KEY)

    def _execute(self, command):
        self.calls.append(command)
        return ToolResult(True, "executed")

    def _restore_registry(self):
        self.registry._tools.clear()
        self.registry._tools.update(self.previous)

    def _pending(self):
        results = self.executor.process(
            '<tool_call>{"name":"bash","arguments":{"command":"exact target"}}</tool_call>'
        )
        request = self.service.get_request(results[0]["approval"].invocation_id)
        return request

    def test_typed_phrase_routes_bound_proof_through_harness_once(self):
        request = self._pending()
        oia_request, encoder = _authentication_request(request)
        evidence = self.owner.authenticate(
            oia_request,
            PHRASE,
            response="approve",
            proof_encoder=encoder,
        )
        self.assertEqual(evidence.context_id, request.invocation.invocation_id)
        self.assertEqual(evidence.context_digest, request.grant.operation_digest)
        self.assertEqual(self.calls, [])  # Evidence by itself does not dispatch.
        with self.assertRaises(OIAError):
            authenticate_typed_decision(
                self.owner,
                self.service,
                request.invocation.invocation_id,
                "violet orchard window brass lantern river wrong",
                "approve",
            )
        self.assertEqual(self.calls, [])
        result = authenticate_typed_decision(
            self.owner,
            self.service,
            request.invocation.invocation_id,
            PHRASE,
            "approve",
        )
        self.assertEqual(result.status.value, "approved")
        self.assertEqual(self.calls, ["exact target"])

    def test_spoken_path_requires_fresh_challenge_code_and_explicit_decision(self):
        request = self._pending()
        code = f"{int.from_bytes(request.challenge.challenge_material[:4], 'big') % 1_000_000:06d}"
        with self.assertRaises(OIAError):
            authenticate_spoken_decision(
                self.owner,
                self.service,
                request.invocation.invocation_id,
                f"{PHRASE} approve 000000",
            )
        self.assertEqual(self.calls, [])
        result = authenticate_spoken_decision(
            self.owner,
            self.service,
            request.invocation.invocation_id,
            f"{PHRASE} approve {code}",
        )
        self.assertEqual(result.status.value, "approved")
        self.assertEqual(self.calls, ["exact target"])


if __name__ == "__main__":
    unittest.main()
