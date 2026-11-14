"""Lila's explicit OIA-to-Harness composition and proof adapter.

This module is application integration, not part of reusable OIA. Harness
continues to issue challenges and independently verifies evidence before its
existing continuation, grant-consumption and dispatch lifecycle.
"""
from __future__ import annotations

import secrets
import hashlib
from pathlib import Path
import shutil

from src.harness.approval_authentication import (
    ApprovalDecisionResponse,
    OwnerDecision,
    approval_proof_message,
)
from src.harness.approval_continuation import (
    ApprovalContinuationService,
    OwnerAuthenticationSecret,
)
from src.harness.capabilities import (
    LEGACY_CAPABILITY_IDS,
    CapabilityRegistry,
    legacy_tool_to_manifest,
)
from src.harness.file_targets import (
    FileReadExecutionTargetAdapter,
    describe_file_target,
)
from src.harness.pending_invocations import (
    PendingInvocationRegistry,
    TargetBindingResolverRegistry,
)
from src.harness.permissions import resolve_file_list_scope
from src.oia import (
    AuthenticationChallenge,
    AuthenticationRequest,
    OIAError,
    OwnerAuthenticationService,
)


class _ProcessPlaceholderSecretProvider:
    """Fail-closed boot key; the owner key is supplied only per OIA response."""

    def __init__(self, key_id: str):
        self._credential = OwnerAuthenticationSecret(key_id, secrets.token_bytes(32))

    def get_owner_authentication_secret(self) -> OwnerAuthenticationSecret:
        return self._credential


class _HarnessProofMessageEncoder:
    """Retain Harness's established Step 4 proof bytes and challenge binding."""

    def __init__(self, harness_challenge):
        self._challenge = harness_challenge

    def encode(self, challenge: AuthenticationChallenge, response: str) -> bytes:
        original = self._challenge
        if (
            challenge.challenge_id != original.challenge_id
            or challenge.key_id != original.key_id
            or challenge.context_id != original.invocation_id
            or challenge.context_digest != original.operation_digest
            or challenge.challenge_material != original.challenge_material
            or challenge.issued_at_monotonic != original.issued_at_monotonic
            or challenge.expires_at_monotonic != original.expires_at_monotonic
        ):
            raise OIAError("owner_authentication.challenge_binding_mismatch")
        try:
            decision = OwnerDecision(response)
        except ValueError:
            raise OIAError("owner_authentication.response_invalid") from None
        return approval_proof_message(original, decision)


def _authentication_request(approval_request):
    challenge = approval_request.challenge
    return AuthenticationRequest(
        challenge=AuthenticationChallenge(
            challenge_id=challenge.challenge_id,
            key_id=challenge.key_id,
            context_id=challenge.invocation_id,
            context_digest=challenge.operation_digest,
            challenge_material=challenge.challenge_material,
            issued_at_monotonic=challenge.issued_at_monotonic,
            expires_at_monotonic=challenge.expires_at_monotonic,
            purpose="harness.approval_decision",
        ),
        allowed_responses=(OwnerDecision.APPROVE.value, OwnerDecision.REJECT.value),
    ), _HarnessProofMessageEncoder(challenge)


def _respond_with_evidence(service, evidence, verifier_key):
    response = ApprovalDecisionResponse(
        challenge_id=evidence.challenge_id,
        key_id=evidence.key_id,
        invocation_id=evidence.context_id,
        operation_digest=evidence.context_digest,
        decision=OwnerDecision(evidence.response),
        proof=evidence.proof,
    )
    # Harness independently verifies this proof and the exact live request.
    # Only then does its pre-existing continuation consume and dispatch.
    return service.respond(response, _owner_secret=verifier_key)


def authenticate_typed_decision(
    owner: OwnerAuthenticationService,
    service: ApprovalContinuationService,
    invocation_id: str,
    phrase: str,
    decision: str,
):
    request = service.get_request(invocation_id)
    oia_request, encoder = _authentication_request(request)
    return owner.authenticate(
        oia_request,
        phrase,
        response=decision,
        proof_encoder=encoder,
        source="typed",
        consumer=lambda evidence, key: _respond_with_evidence(service, evidence, key),
    )


def authenticate_spoken_decision(
    owner: OwnerAuthenticationService,
    service: ApprovalContinuationService,
    invocation_id: str,
    transcript: str,
):
    request = service.get_request(invocation_id)
    oia_request, encoder = _authentication_request(request)
    return owner.authenticate_spoken(
        oia_request,
        transcript,
        proof_encoder=encoder,
        consumer=lambda evidence, key: _respond_with_evidence(service, evidence, key),
    )


def build_approval_service(owner_authentication, tool_registry):
    """Build the boot's Harness continuation from its registered tool set."""
    profile = owner_authentication.owner_profile()
    if profile is None:
        return None

    capabilities = CapabilityRegistry()
    target_resolvers = TargetBindingResolverRegistry()
    for tool in tool_registry.all():
        manifest = legacy_tool_to_manifest(tool)
        capabilities.register(manifest)

        # Only these legacy capabilities currently have reviewed target and
        # display semantics. Every other approval fails closed until its own
        # resolver is reviewed; a digest of arbitrary arguments is not a target.
        if tool.name in {"file_read", "file_list"}:
            def resolve(_manifest, validated, execution, *, name=tool.name):
                raw_path = execution.get("path", validated.get("path", "."))
                if not isinstance(raw_path, str) or not raw_path.strip():
                    return None
                if name == "file_read":
                    try:
                        descriptor = describe_file_target(raw_path)
                    except Exception:
                        return None
                    descriptor["display"] = {
                        "reviewed": True,
                        "action": "Read file contents",
                        "target": descriptor["canonical_location"],
                        "values": {},
                    }
                    return descriptor
                try:
                    path = Path(raw_path).expanduser().resolve(strict=True)
                    metadata = path.stat()
                except (OSError, RuntimeError, ValueError):
                    return None
                expected_kind = "directory"
                if not path.is_dir():
                    return None
                display = {
                    "reviewed": True,
                    "action": "List directory entries",
                    "target": str(path),
                    "values": {},
                }
                if name == "file_list":
                    pattern = validated.get("pattern", "*")
                    if not isinstance(pattern, str) or len(pattern) > 128:
                        return None
                    try:
                        _root, matches = resolve_file_list_scope(path, pattern)
                    except Exception:
                        return None
                    objects = []
                    for item in matches[:200]:
                        info = item.stat()
                        objects.append({
                            "canonical_location": str(item),
                            "device": int(info.st_dev),
                            "inode": int(info.st_ino),
                            "mode": int(info.st_mode),
                            "size": int(info.st_size),
                            "mtime_ns": int(info.st_mtime_ns),
                        })
                    display["values"] = {
                        "pattern": pattern,
                        "matched_locations": [item["canonical_location"] for item in objects],
                    }
                else:
                    objects = []
                return {
                    "target_type": expected_kind,
                    "canonical_location": str(path),
                    "parent_location": str(path.parent),
                    "device": int(metadata.st_dev),
                    "inode": int(metadata.st_ino),
                    "mode": int(metadata.st_mode),
                    "size": int(metadata.st_size),
                    "mtime_ns": int(metadata.st_mtime_ns),
                    "ctime_ns": int(metadata.st_ctime_ns),
                    "objects": objects,
                    "resolver_version": "1",
                    "display": display,
                }

            target_resolvers.register(
                manifest,
                resolver_id=f"lila.target.{tool.name}.v1",
                target_required=True,
                target_kind=("filesystem.file" if tool.name == "file_read"
                             else "filesystem.directory"),
                resolver=resolve,
            )
        elif tool.name == "bash":
            def resolve_process(_manifest, validated, execution):
                command = execution.get("command", validated.get("command"))
                executable = shutil.which("bash")
                if not isinstance(command, str) or not command or executable is None:
                    return None
                try:
                    shell = Path(executable).resolve(strict=True)
                    metadata = shell.stat()
                except OSError:
                    return None
                return {
                    "target_type": "process.command",
                    "executable": str(shell),
                    "executable_device": int(metadata.st_dev),
                    "executable_inode": int(metadata.st_ino),
                    "executable_mtime_ns": int(metadata.st_mtime_ns),
                    "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                    "resolver_version": "1",
                    "display": {
                        "reviewed": False,
                        "action": "Execute a shell command",
                        "values": {},
                    },
                }

            target_resolvers.register(
                manifest,
                resolver_id="lila.target.process.bash.v1",
                target_required=True,
                target_kind="process.command",
                resolver=resolve_process,
            )

    pending = PendingInvocationRegistry(
        capabilities=capabilities,
        target_resolvers=target_resolvers,
        lifetime_seconds=120.0,
    )
    return ApprovalContinuationService(
        pending_registry=pending,
        owner_secret_provider=_ProcessPlaceholderSecretProvider(profile.key_id),
        challenge_lifetime_seconds=60.0,
        execution_target_adapters={
            LEGACY_CAPABILITY_IDS["file_read"]: FileReadExecutionTargetAdapter(),
        },
    )
