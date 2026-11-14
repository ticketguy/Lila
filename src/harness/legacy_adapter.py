"""Non-executing shadow adapter for exact legacy capability bindings.

Bindings contain descriptive manifests and trusted legacy names only. This
module never stores handlers, consults the global ToolRegistry, or executes a
tool. A shadow validation result explicitly reports that no execution occurred.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping

from .capabilities import CapabilityManifest
from .execution_context import (
    ExecutionContext,
    ExecutionContextError,
    ExecutionContextIssuer,
    argument_digest,
)


class LegacyBindingError(ValueError):
    """A legacy capability binding is unsafe, ambiguous, or unavailable."""


class ShadowValidationError(ValueError):
    """A shadow invocation did not match its explicit trusted binding."""


_NON_BINDABLE_LEGACY_NAMES = frozenset({"bash", "ssh_exec"})
_DISABLED_FINDING = "disabled_placeholder"


@dataclass(frozen=True)
class LegacyCapabilityBinding:
    capability_id: str
    contract_version: str
    manifest_digest: str
    legacy_tool_name: str


@dataclass(frozen=True)
class ShadowValidationResult:
    """Validation evidence; this result never represents action completion."""

    validation_status: str
    execution_status: str
    capability_id: str
    contract_version: str
    manifest_digest: str
    argument_digest: str
    invocation_id: str


class LegacyCapabilityAdapter:
    """Instance-owned identity bindings with no handler-execution capability."""

    def __init__(self, issuer: ExecutionContextIssuer) -> None:
        if not isinstance(issuer, ExecutionContextIssuer):
            raise TypeError("An explicit ExecutionContextIssuer is required.")
        self._issuer = issuer
        self._bindings: dict[tuple[str, str], LegacyCapabilityBinding] = {}
        self._manifests: dict[tuple[str, str], CapabilityManifest] = {}
        self._used_invocations: set[str] = set()
        self._lock = threading.Lock()

    def bind(self, manifest: CapabilityManifest, legacy_tool_name: str) -> None:
        """Install one trusted-configuration binding; never accepts a handler.

        Application code must call this while building trusted configuration.
        It must never pass model-selected names or requests into this method.
        Binding permits shadow validation only, not consumer exposure or use.
        """
        if not isinstance(manifest, CapabilityManifest):
            raise TypeError("A CapabilityManifest is required.")
        if not isinstance(legacy_tool_name, str) or not legacy_tool_name:
            raise LegacyBindingError("A legacy tool name from trusted configuration is required.")
        provenance_name = manifest.legacy_provenance.get("legacy_tool_name")
        if provenance_name != legacy_tool_name:
            raise LegacyBindingError("Binding name does not match manifest provenance.")
        if legacy_tool_name in _NON_BINDABLE_LEGACY_NAMES:
            raise LegacyBindingError("Arbitrary local or remote shell tools cannot be bound.")
        if manifest.consumer_exposure == "prohibited_from_consumer_exposure":
            raise LegacyBindingError("Manifest prohibits consumer exposure.")
        if not manifest.availability_observations.get("registered_observation", False):
            raise LegacyBindingError("Capability is not present in the descriptive catalog.")
        if (
            manifest.availability_observations.get("legacy_permission_label") == "disabled"
            or any(finding.get("code") == _DISABLED_FINDING for finding in manifest.findings)
        ):
            raise LegacyBindingError("Disabled placeholder capabilities cannot be bound.")

        key = manifest.identity
        with self._lock:
            if key in self._bindings:
                raise LegacyBindingError(
                    f"Binding already exists for {key[0]} version {key[1]}."
                )
            self._bindings[key] = LegacyCapabilityBinding(
                capability_id=manifest.capability_id,
                contract_version=manifest.contract_version,
                manifest_digest=manifest.manifest_digest,
                legacy_tool_name=legacy_tool_name,
            )
            self._manifests[key] = manifest

    def bindings(self) -> tuple[LegacyCapabilityBinding, ...]:
        """Return an immutable snapshot of bindings, never executable objects."""
        with self._lock:
            return tuple(self._bindings[key] for key in sorted(self._bindings))

    def shadow_validate(
        self,
        *,
        capability_id: str,
        contract_version: str,
        manifest_digest: str,
        arguments: Mapping[str, Any],
        context: ExecutionContext | None,
    ) -> ShadowValidationResult:
        """Validate one exact invocation binding without calling a handler."""
        if context is None:
            raise ShadowValidationError("An execution context is required.")
        try:
            self._issuer.verify_authenticity(context)
        except ExecutionContextError as exc:
            raise ShadowValidationError(str(exc)) from exc

        # Consume authentic context IDs before checking the request binding, so
        # even a mismatched attempt cannot be retried as a second invocation.
        with self._lock:
            if context.invocation_id in self._used_invocations:
                raise ShadowValidationError("Execution context has already been used.")
            self._used_invocations.add(context.invocation_id)

        key = (capability_id, contract_version)
        with self._lock:
            binding = self._bindings.get(key)
            manifest = self._manifests.get(key)
        if binding is None:
            raise ShadowValidationError("No explicit binding exists for this capability version.")
        if manifest_digest != binding.manifest_digest:
            raise ShadowValidationError("Requested manifest digest does not match the binding.")
        if (
            context.capability_id != binding.capability_id
            or context.contract_version != binding.contract_version
            or context.manifest_digest != binding.manifest_digest
        ):
            raise ShadowValidationError("Context identity or manifest digest mismatched.")

        # The adapter retains only the immutable manifest and descriptive
        # binding, never a Tool/provider/handler. The issuer validates the
        # signed identity, arguments, decision and deadline from the context.
        try:
            if not isinstance(arguments, Mapping):
                raise ExecutionContextError("Invocation arguments must be an object.")
            if argument_digest(arguments) != context.argument_digest:
                raise ExecutionContextError("Invocation arguments do not match the context.")
            self._issuer.validate_for_invocation(context, manifest, arguments)
        except ExecutionContextError as exc:
            raise ShadowValidationError(str(exc)) from exc

        return ShadowValidationResult(
            validation_status="shadow_validated",
            execution_status="not_executed",
            capability_id=binding.capability_id,
            contract_version=binding.contract_version,
            manifest_digest=binding.manifest_digest,
            argument_digest=context.argument_digest,
            invocation_id=context.invocation_id,
        )

