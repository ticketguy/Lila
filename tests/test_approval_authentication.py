import hashlib
import hmac
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from src.harness.approval_authentication import (
    ApprovalAuthenticationError,
    ApprovalAuthenticator,
    ApprovalChallenge,
    ApprovalDecisionResponse,
    AuthenticatedOwnerDecision,
    OwnerDecision,
    approval_proof_message,
)
from src.harness.approval_grants import ApprovalGrant
from src.harness.capabilities import CapabilityRegistry, legacy_tool_to_manifest
from src.harness.pending_invocations import (
    PendingInvocationRegistry,
    TargetBindingResolverRegistry,
)
from src.harness.permissions import PermissionManager, PolicyDisposition
from src.harness.tools import Tool, ToolArg, ToolResult


class ApprovalAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.now = [100.0]
        self.handler_calls = []

        def handler(**arguments):
            self.handler_calls.append(arguments)
            return ToolResult(True, "must not dispatch")

        self.tool = Tool(
            "file_write", "write a file",
            [ToolArg("path", "path", "target"), ToolArg("content", "string", "body")],
            handler,
        )
        self.manifest = legacy_tool_to_manifest(self.tool)
        self.capabilities = CapabilityRegistry()
        self.capabilities.register(self.manifest)
        self.target_path = str(Path(self.temp_dir.name) / "target.txt")
        self.current_target = [self.target_path]
        self.resolvers = TargetBindingResolverRegistry()
        self.resolvers.register(
            self.manifest,
            resolver_id="files.write.path.v1",
            target_required=True,
            target_kind="filesystem.path",
            resolver=lambda _manifest, args, _execution: {
                "canonical_path": str(Path(self.current_target[0]).resolve(strict=False)),
            },
        )
        self.arguments = {"path": self.target_path, "content": "exact content"}
        policy = PermissionManager(workspace_root=self.temp_dir.name).evaluate_policy(
            self.tool, self.arguments
        )
        self.assertIs(policy.disposition, PolicyDisposition.REQUIRE_APPROVAL)
        self._id = [0]

        def next_id():
            self._id[0] += 1
            return f"pending-{self._id[0]}"

        self.registry = PendingInvocationRegistry(
            capabilities=self.capabilities,
            target_resolvers=self.resolvers,
            lifetime_seconds=120,
            capacity=16,
            monotonic_clock=lambda: self.now[0],
            invocation_id_factory=next_id,
        )
        self.pending = self.registry.create_pending(policy)
        self.grant = self.registry._issue_approval_grant(self.pending.invocation_id)
        self.secret = b"owner-controlled-test-key-material-32-bytes!"
        self.authenticator = self.make_authenticator()

    def make_authenticator(self, *, secret=None, lifetime=30, capacity=8):
        return ApprovalAuthenticator(
            pending_registry=self.registry,
            owner_secret=self.secret if secret is None else secret,
            key_id="owner-primary-v1",
            challenge_lifetime_seconds=lifetime,
            capacity=capacity,
            monotonic_clock=lambda: self.now[0],
        )

    def make_pending_and_grant(self):
        policy = PermissionManager(workspace_root=self.temp_dir.name).evaluate_policy(
            self.tool, self.arguments
        )
        pending = self.registry.create_pending(policy)
        grant = self.registry._issue_approval_grant(pending.invocation_id)
        return pending, grant

    def response_for(self, challenge, decision, *, secret=None, **overrides):
        proof = hmac.new(
            self.secret if secret is None else secret,
            approval_proof_message(challenge, decision),
            hashlib.sha256,
        ).digest()
        values = {
            "challenge_id": challenge.challenge_id,
            "key_id": challenge.key_id,
            "invocation_id": challenge.invocation_id,
            "operation_digest": challenge.operation_digest,
            "decision": decision,
            "proof": proof,
        }
        values.update(overrides)
        return ApprovalDecisionResponse(**values)

    def test_valid_approval_authenticates_for_exact_pending_and_grant(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        response = self.response_for(challenge, OwnerDecision.APPROVE)

        result = self.authenticator.authenticate(response)

        self.assertIsInstance(result, AuthenticatedOwnerDecision)
        self.assertIs(result.decision, OwnerDecision.APPROVE)
        self.assertEqual(result.invocation_id, self.pending.invocation_id)
        self.assertEqual(result.operation_digest, self.grant.operation_digest)
        self.assertEqual(result.challenge_id, challenge.challenge_id)
        self.assertEqual(result.authenticated_at_monotonic, self.now[0])
        self.assertFalse(isinstance(result, ApprovalGrant))
        self.assertEqual(self.handler_calls, [])

    def test_valid_rejection_is_distinct_and_authenticated(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        result = self.authenticator.authenticate(
            self.response_for(challenge, OwnerDecision.REJECT)
        )
        self.assertIs(result.decision, OwnerDecision.REJECT)
        self.assertNotEqual(result.decision, OwnerDecision.APPROVE)
        self.assertEqual(self.handler_calls, [])

    def test_missing_or_malformed_proof_fails_closed(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        response = self.response_for(challenge, OwnerDecision.APPROVE, proof=b"")
        with self.assertRaises(ApprovalAuthenticationError) as caught:
            self.authenticator.authenticate(response)
        self.assertEqual(caught.exception.reason_code, "approval.response_malformed")

    def test_wrong_secret_and_forged_proof_are_rejected(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        forged = self.response_for(challenge, OwnerDecision.APPROVE, secret=b"x" * 40)
        with self.assertRaises(ApprovalAuthenticationError) as caught:
            self.authenticator.authenticate(forged)
        self.assertEqual(caught.exception.reason_code, "approval.proof_invalid")
        self.assertEqual(self.handler_calls, [])

    def test_cross_invocation_challenge_or_response_substitution_is_rejected(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        another_challenge = self.authenticator.issue_challenge(
            self.pending.invocation_id, self.grant
        )
        substituted_response = self.response_for(
            challenge,
            OwnerDecision.APPROVE,
            challenge_id=another_challenge.challenge_id,
        )
        with self.assertRaises(ApprovalAuthenticationError) as caught:
            self.authenticator.authenticate(substituted_response)
        self.assertEqual(caught.exception.reason_code, "approval.proof_invalid")

        other_pending, _other_grant = self.make_pending_and_grant()
        response = self.response_for(
            challenge, OwnerDecision.APPROVE, invocation_id=other_pending.invocation_id
        )
        with self.assertRaises(ApprovalAuthenticationError) as caught:
            self.authenticator.authenticate(response)
        self.assertEqual(caught.exception.reason_code, "approval.binding_mismatch")
        with self.assertRaises(ApprovalAuthenticationError):
            self.authenticator.issue_challenge(other_pending.invocation_id, self.grant)

    def test_operation_digest_substitution_is_rejected(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        response = self.response_for(
            challenge, OwnerDecision.APPROVE, operation_digest="0" * 64
        )
        with self.assertRaises(ApprovalAuthenticationError) as caught:
            self.authenticator.authenticate(response)
        self.assertEqual(caught.exception.reason_code, "approval.binding_mismatch")

    def test_decision_tampering_invalidates_the_proof(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        approve_proof = self.response_for(challenge, OwnerDecision.APPROVE).proof
        tampered = ApprovalDecisionResponse(
            challenge_id=challenge.challenge_id,
            key_id=challenge.key_id,
            invocation_id=challenge.invocation_id,
            operation_digest=challenge.operation_digest,
            decision=OwnerDecision.REJECT,
            proof=approve_proof,
        )
        with self.assertRaises(ApprovalAuthenticationError) as caught:
            self.authenticator.authenticate(tampered)
        self.assertEqual(caught.exception.reason_code, "approval.proof_invalid")

    def test_expired_challenge_is_stale_and_cannot_authenticate(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        response = self.response_for(challenge, OwnerDecision.APPROVE)
        self.now[0] = challenge.expires_at_monotonic
        with self.assertRaises(ApprovalAuthenticationError) as caught:
            self.authenticator.authenticate(response)
        self.assertEqual(caught.exception.reason_code, "approval.challenge_expired")

    def test_cancelled_or_invalidated_pending_invocation_rejects_existing_challenge(self):
        for terminal in ("cancel", "invalidate"):
            with self.subTest(terminal=terminal):
                pending, grant = self.make_pending_and_grant()
                authenticator = self.make_authenticator()
                challenge = authenticator.issue_challenge(pending.invocation_id, grant)
                response = self.response_for(challenge, OwnerDecision.APPROVE)
                getattr(self.registry, terminal)(pending.invocation_id)
                with self.assertRaises(ApprovalAuthenticationError) as caught:
                    authenticator.authenticate(response)
                self.assertEqual(caught.exception.reason_code, "approval.binding_invalid")

    def test_target_drift_after_challenge_rejects_owner_response(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        response = self.response_for(challenge, OwnerDecision.APPROVE)
        self.current_target[0] = str(Path(self.temp_dir.name) / "different-target.txt")
        with self.assertRaises(ApprovalAuthenticationError) as caught:
            self.authenticator.authenticate(response)
        self.assertEqual(caught.exception.reason_code, "approval.binding_invalid")

    def test_secure_random_source_generates_challenge_material(self):
        real = __import__("secrets").token_bytes
        with patch("src.harness.approval_authentication.secrets.token_bytes", wraps=real) as random_bytes:
            challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        random_bytes.assert_called_once_with(48)
        self.assertEqual(len(challenge.challenge_material), 32)
        self.assertNotIn(challenge.challenge_material.hex(), repr(challenge))
        with patch(
            "src.harness.approval_authentication.secrets.token_bytes",
            side_effect=OSError("private entropy failure"),
        ):
            with self.assertRaises(ApprovalAuthenticationError) as caught:
                self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        self.assertEqual(caught.exception.reason_code, "approval.random_unavailable")

    def test_challenge_and_authenticated_decision_are_trusted_immutable_contracts(self):
        with self.assertRaises(TypeError):
            ApprovalChallenge(
                challenge_id="forged", key_id="owner-primary-v1",
                invocation_id=self.pending.invocation_id,
                operation_digest=self.grant.operation_digest,
                challenge_material=b"x" * 32,
                issued_at_monotonic=100.0,
                expires_at_monotonic=110.0,
            )
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        result = self.authenticator.authenticate(
            self.response_for(challenge, OwnerDecision.APPROVE)
        )
        with self.assertRaises(FrozenInstanceError):
            result.decision = OwnerDecision.REJECT
        self.assertNotIn(self.secret.hex(), repr(self.authenticator))
        self.assertNotIn(self.secret.hex(), repr(result))

    def test_authentication_failure_never_creates_grant_or_dispatches(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        with self.assertRaises(ApprovalAuthenticationError):
            self.authenticator.authenticate(
                self.response_for(challenge, OwnerDecision.APPROVE, secret=b"wrong" * 8)
            )
        self.assertEqual(self.handler_calls, [])
        self.assertEqual(self.registry.get(self.pending.invocation_id).state.value, "pending")

    def test_challenge_constructor_rejects_untrusted_callers_and_secret_must_be_strong(self):
        with self.assertRaises(ValueError):
            ApprovalAuthenticator(
                pending_registry=self.registry,
                owner_secret=b"short",
                key_id="owner-primary-v1",
                monotonic_clock=lambda: self.now[0],
            )

    def test_grant_mismatch_expiry_and_challenge_capacity_fail_closed(self):
        other_pending, other_grant = self.make_pending_and_grant()
        with self.assertRaises(ApprovalAuthenticationError):
            self.authenticator.issue_challenge(self.pending.invocation_id, other_grant)

        small = self.make_authenticator(capacity=1)
        small.issue_challenge(self.pending.invocation_id, self.grant)
        with self.assertRaises(ApprovalAuthenticationError) as caught:
            small.issue_challenge(self.pending.invocation_id, self.grant)
        self.assertEqual(caught.exception.reason_code, "approval.challenge_capacity")
        self.assertEqual(self.handler_calls, [])

    def test_response_repr_redacts_hmac_proof(self):
        challenge = self.authenticator.issue_challenge(self.pending.invocation_id, self.grant)
        response = self.response_for(challenge, OwnerDecision.APPROVE)
        self.assertNotIn(response.proof.hex(), repr(response))


if __name__ == "__main__":
    unittest.main()
