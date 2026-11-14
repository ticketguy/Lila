import hashlib
import hmac
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.daemon.service import LilaDaemon, _make_api_handler
from src.harness.approval_authentication import OwnerDecision, approval_proof_message
from src.harness.approval_authentication import ApprovalDecisionResponse
from src.harness.approval_continuation import ApprovalContinuationService, OwnerAuthenticationSecret
from src.harness.capabilities import CapabilityRegistry, legacy_tool_to_manifest
from src.harness.executor import HarnessExecutor
from src.harness.pending_invocations import PendingInvocationRegistry, TargetBindingResolverRegistry
from src.harness.tools import Tool, ToolArg, ToolResult, get_registry, register_tool
from src.oia import OIAError, OwnerAuthenticationService, OwnerProfile


class SecretProvider:
    def __init__(self, secret):
        self.secret = secret

    def get_owner_authentication_secret(self):
        return OwnerAuthenticationSecret("test-owner-key", self.secret)


class DaemonApprovalApiTests(unittest.TestCase):
    def setUp(self):
        self.token = "daemon-api-token-that-is-not-an-owner-secret"
        self.secret = b"owner-secret-used-only-by-test-provider-32-bytes"
        self.registry = get_registry()
        self.previous = dict(self.registry._tools)
        self.registry._tools.clear()
        self.calls = []
        self.tool = Tool("bash", "execute a command", [ToolArg("command", "string", "command")], self.execute)
        register_tool(self.tool)
        manifest = legacy_tool_to_manifest(self.tool)
        catalog = CapabilityRegistry()
        catalog.register(manifest)
        resolvers = TargetBindingResolverRegistry()
        resolvers.register(manifest, resolver_id="test.command", target_required=True,
                           target_kind="command", resolver=lambda _m, args, _e: {"command": args["command"]})
        self.pending = PendingInvocationRegistry(capabilities=catalog, target_resolvers=resolvers,
                                                 lifetime_seconds=60)
        self.service = ApprovalContinuationService(pending_registry=self.pending,
                                                   owner_secret_provider=SecretProvider(self.secret))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.restore_registry)
        self.daemon = LilaDaemon(port=8765, api_token=self.token, voice=False,
                                 approval_continuation_service=self.service)
        self.daemon.lila = SimpleNamespace(approval_continuation_service=self.service)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _make_api_handler(self.daemon, self.token))
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.addCleanup(self.stop_server)

    def execute(self, command):
        self.calls.append(command)
        return ToolResult(True, "done")

    def restore_registry(self):
        self.registry._tools.clear()
        self.registry._tools.update(self.previous)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)

    def request(self, route, method="GET", body=None, token=None):
        data = None if body is None else json.dumps(body).encode()
        req = Request(f"http://127.0.0.1:{self.server.server_port}/{route}", data=data, method=method)
        if token is not None:
            req.add_header("Authorization", "Bearer " + token)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def create_pending(self, command="echo token=hide-this"):
        executor = HarnessExecutor(log_dir=self.temp.name, workspace_root=self.temp.name,
                                   approval_continuation_service=self.service)
        result = executor.process("<tool_call>" + json.dumps({"name": "bash", "arguments": {"command": command}}) + "</tool_call>")[0]
        self.assertEqual(result["approval"].status.value, "pending")
        return self.service.get_request(result["approval"].invocation_id)

    def proof_body(self, approval, decision=OwnerDecision.APPROVE):
        challenge = approval.challenge
        proof = hmac.new(self.secret, approval_proof_message(challenge, decision), hashlib.sha256).hexdigest()
        return {"challenge_id": challenge.challenge_id, "key_id": challenge.key_id,
                "operation_digest": approval.grant.operation_digest,
                "decision": decision.value, "proof": proof}

    def test_authorized_list_and_detail_redact_secrets_and_hide_challenge_material(self):
        approval = self.create_pending()
        status, result = self.request("approvals", token=self.token)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["requests"][0]["invocation_id"], approval.invocation.invocation_id)
        _, detail = self.request("approvals/" + approval.invocation.invocation_id, token=self.token)
        self.assertEqual(detail["kind"], "pending")
        encoded = json.dumps(detail["request"])
        self.assertFalse(detail["request"]["approval_ready"])
        self.assertNotIn("hide-this", encoded)
        self.assertNotIn("echo token", encoded)
        self.assertNotIn("challenge_material", encoded)
        self.assertNotIn(self.secret.decode(errors="ignore"), encoded)

    def test_unauthorized_api_requests_rejected(self):
        status, _ = self.request("approvals")
        self.assertEqual(status, 401)

    def test_bearer_alone_is_not_owner_authentication(self):
        approval = self.create_pending("safe command")
        body = self.proof_body(approval)
        body.pop("proof")
        status, result = self.request("approvals/" + approval.invocation.invocation_id + "/decision",
                                      "POST", body, self.token)
        self.assertEqual(status, 403)
        self.assertEqual(result["code"], "approval.owner_authentication_required")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.service.get_result(approval.invocation.invocation_id).status.value, "pending")

    def test_status_is_read_only_and_substitution_fields_are_rejected(self):
        approval = self.create_pending("read only")
        invocation_id = approval.invocation.invocation_id
        status, result = self.request("approvals/" + invocation_id + "/status", token=self.token)
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(self.service.get_result(invocation_id).status.value, "pending")
        body = self.proof_body(approval)
        body["arguments"] = {"command": "replacement"}
        status, error = self.request("approvals/" + invocation_id + "/decision", "POST", body, self.token)
        self.assertEqual(status, 400)
        self.assertEqual(error["code"], "approval.invalid_request")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.service.get_result(invocation_id).status.value, "pending")

    def test_authenticated_exact_decision_reaches_harness_and_dispatches_once(self):
        approval = self.create_pending("run exact")
        status, result = self.request("approvals/" + approval.invocation.invocation_id + "/decision",
                                      "POST", self.proof_body(approval), self.token)
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "approved")
        self.assertEqual(self.calls, ["run exact"])
        # A reloaded Island reads the terminal decision from the daemon.
        status, reloaded = self.request(
            "approvals/" + approval.invocation.invocation_id + "/status", token=self.token
        )
        self.assertEqual(status, 200)
        self.assertEqual(reloaded["status"], "approved")
        status, again = self.request("approvals/" + approval.invocation.invocation_id + "/decision",
                                     "POST", self.proof_body(approval), self.token)
        self.assertEqual(status, 200)
        self.assertNotEqual(again["status"], "approved")
        self.assertEqual(self.calls, ["run exact"])

    def test_rejection_and_mismatch_never_dispatch(self):
        rejected = self.create_pending("reject me")
        status, result = self.request("approvals/" + rejected.invocation.invocation_id + "/decision",
                                      "POST", self.proof_body(rejected, OwnerDecision.REJECT), self.token)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(self.calls, [])
        mismatched = self.create_pending("mismatch")
        body = self.proof_body(mismatched)
        body["operation_digest"] = "0" * 64
        _, result = self.request("approvals/" + mismatched.invocation.invocation_id + "/decision",
                                 "POST", body, self.token)
        self.assertIn(result["status"], {"failed", "invalidated"})
        self.assertEqual(self.calls, [])

    def test_service_loss_is_typed_unavailable(self):
        approval = self.create_pending("service loss")
        self.daemon.approval_continuation_service = None
        self.daemon.lila = SimpleNamespace(approval_continuation_service=None)
        status, result = self.request("approvals", token=self.token)
        self.assertEqual(status, 503)
        self.assertEqual(result["code"], "approval.service_unavailable")
        status, decision = self.request(
            f"approvals/{approval.invocation.invocation_id}/decision", "POST",
            self.proof_body(approval), self.token,
        )
        self.assertEqual(status, 503)
        self.assertEqual(decision["code"], "approval.service_unavailable")
        self.assertEqual(self.calls, [])

    def test_delayed_and_duplicate_http_decisions_after_terminal_states_never_dispatch(self):
        for action, expected in [(self.service.cancel, "cancelled"),
                                 (self.service.invalidate, "invalidated")]:
            with self.subTest(state=expected):
                approval = self.create_pending("terminated")
                invocation_id = approval.invocation.invocation_id
                body = self.proof_body(approval)
                action(invocation_id)
                for _ in range(2):
                    status, result = self.request(
                        f"approvals/{invocation_id}/decision", "POST", body, self.token,
                    )
                    self.assertEqual(status, 200)
                    self.assertNotEqual(result["status"], "approved")
                status, terminal = self.request(f"approvals/{invocation_id}/status", token=self.token)
                self.assertEqual(status, 200)
                self.assertEqual(terminal["status"], expected)
        self.assertEqual(self.calls, [])

    def test_expired_http_decision_and_replay_cannot_revive_pending_invocation(self):
        now = [100.0]
        self.pending._clock = lambda: now[0]
        self.service._clock = lambda: now[0]
        self.service._authenticator._clock = lambda: now[0]
        approval = self.create_pending("expires")
        invocation_id = approval.invocation.invocation_id
        body = self.proof_body(approval)
        now[0] = 1000.0
        for _ in range(2):
            status, result = self.request(f"approvals/{invocation_id}/decision", "POST", body, self.token)
            self.assertEqual(status, 200)
            self.assertNotEqual(result["status"], "approved")
        status, terminal = self.request(f"approvals/{invocation_id}/status", token=self.token)
        self.assertEqual(status, 200)
        self.assertEqual(terminal["status"], "expired")
        self.assertEqual(self.calls, [])

    def test_daemon_cleanup_cancels_pending_and_late_decision_cannot_dispatch(self):
        approval = self.create_pending("must be cancelled at shutdown")
        invocation_id = approval.invocation.invocation_id

        self.daemon._cleanup()

        status, terminal = self.request(f"approvals/{invocation_id}/status", token=self.token)
        self.assertEqual(status, 200, terminal)
        self.assertEqual(terminal["status"], "cancelled")
        body = self.proof_body(approval)
        late = self.service.respond(ApprovalDecisionResponse(
            challenge_id=body["challenge_id"],
            key_id=body["key_id"],
            invocation_id=invocation_id,
            operation_digest=body["operation_digest"],
            decision=OwnerDecision.APPROVE,
            proof=bytes.fromhex(body["proof"]),
        ))
        self.assertEqual(late.status.value, "unavailable")
        self.assertEqual(self.calls, [])

    def test_oia_service_loss_does_not_dispatch_or_terminally_approve(self):
        approval = self.create_pending("owner service unavailable")
        invocation_id = approval.invocation.invocation_id
        self.daemon.owner_authentication_service = None

        status, result = self.request(
            f"owner/approvals/{invocation_id}/decision", "POST",
            {"phrase": "controlled test input", "decision": "approve"}, self.token,
        )

        self.assertEqual(status, 503, result)
        self.assertEqual(self.service.get_result(invocation_id).status.value, "pending")
        self.assertEqual(self.calls, [])

    def test_typed_oia_route_rejects_operation_without_reviewed_display(self):
        phrase = "violet orchard window copper lantern river"
        profile = OwnerProfile("test-owner", "test-owner-key", 0)

        class TestOwnerStore:
            enrolled = True

            def profile(self):
                return profile

            def credential(self):
                raise AssertionError("Test authentication uses a controlled unlock.")

        owner = OwnerAuthenticationService(TestOwnerStore())

        def unlock(candidate, *, source):
            if candidate != phrase:
                raise OIAError("owner_authentication.failed")
            return bytearray(self.secret)

        owner._unlock = unlock

        class PlaceholderProvider:
            def get_owner_authentication_secret(self):
                return OwnerAuthenticationSecret(profile.key_id, b"x" * 32)

        self.service = ApprovalContinuationService(
            pending_registry=self.pending,
            owner_secret_provider=PlaceholderProvider(),
        )
        self.daemon.owner_authentication_service = owner
        self.daemon.approval_continuation_service = self.service
        self.daemon.lila = SimpleNamespace(
            approval_continuation_service=self.service,
            owner_authentication_service=owner,
        )
        approval = self.create_pending("OIA only").invocation.invocation_id
        status, result = self.request(
            f"owner/approvals/{approval}/decision", "POST",
            {"phrase": phrase, "decision": "approve"}, self.token,
        )
        self.assertEqual(status, 403, result)
        self.assertEqual(result["code"], "approval.display_unreviewed")
        self.assertNotIn(phrase, json.dumps(result))
        self.assertEqual(self.calls, [])

    def test_remote_owner_bootstrap_route_does_not_exist(self):
        status, _ = self.request(
            "owner/enroll", "POST",
            {"phrase": "violet orchard window copper lantern river"}, self.token,
        )
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
