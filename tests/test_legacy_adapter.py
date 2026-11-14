import unittest
from dataclasses import replace

from src.harness.capabilities import legacy_tool_to_manifest
from src.harness.execution_context import ExecutionContextIssuer
from src.harness.legacy_adapter import (
    LegacyBindingError,
    LegacyCapabilityAdapter,
    ShadowValidationError,
)
from src.harness.tools import Tool, ToolResult

HANDLER_CALLS = []


def make_manifest(name="self_status", version=None):
    def handler():
        HANDLER_CALLS.append(name)
        return ToolResult(success=True, output="must not run")

    tool = Tool(
        name=name,
        description="adapter test capability",
        args=[],
        execute=handler,
        risk="restricted" if name == "memory_store" else "low",
        permission="disabled" if name == "memory_store" else "auto",
    )
    manifest = legacy_tool_to_manifest(tool)
    if version is not None:
        manifest = replace(manifest, contract_version=version)
    return manifest


class LegacyCapabilityAdapterTests(unittest.TestCase):
    def setUp(self):
        self.now = [50.0]
        self.issuer = ExecutionContextIssuer(
            lifetime_seconds=10.0, monotonic_clock=lambda: self.now[0]
        )
        self.manifest = make_manifest()
        self.adapter = LegacyCapabilityAdapter(self.issuer)
        self.adapter.bind(self.manifest, "self_status")
        HANDLER_CALLS.clear()

    def context(self, manifest=None, arguments=None, allowed=True):
        manifest = self.manifest if manifest is None else manifest
        arguments = {} if arguments is None else arguments
        permission_result = (
            allowed,
            "allowed" if allowed else "denied",
            dict(arguments),
        )
        return self.issuer.issue(manifest, arguments, permission_result)

    def validate(self, context=None, **overrides):
        values = {
            "capability_id": self.manifest.capability_id,
            "contract_version": self.manifest.contract_version,
            "manifest_digest": self.manifest.manifest_digest,
            "arguments": {},
            "context": self.context() if context is None else context,
        }
        values.update(overrides)
        return self.adapter.shadow_validate(**values)

    def test_explicit_binding_succeeds_only_as_shadow_validation(self):
        result = self.validate()
        self.assertEqual(result.validation_status, "shadow_validated")
        self.assertEqual(result.execution_status, "not_executed")
        self.assertFalse(hasattr(result, "success"))
        self.assertFalse(hasattr(result, "executed"))
        self.assertFalse(any(callable(value) for value in vars(result).values()))
        self.assertEqual(HANDLER_CALLS, [])

    def test_unknown_capability_and_version_are_rejected(self):
        with self.assertRaisesRegex(ShadowValidationError, "No explicit binding"):
            self.validate(capability_id="org.lila.capability.unknown")
        with self.assertRaisesRegex(ShadowValidationError, "No explicit binding"):
            self.validate(contract_version="1.0.0")

    def test_manifest_digest_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ShadowValidationError, "digest"):
            self.validate(manifest_digest="0" * 64)

    def test_context_identity_version_and_digest_must_match(self):
        other_version = make_manifest(version="1.0.0")
        context = self.context(other_version)
        with self.assertRaisesRegex(ShadowValidationError, "identity or manifest digest"):
            self.validate(context=context)

    def test_argument_mismatch_is_rejected(self):
        context = self.context(arguments={"x": 1})
        with self.assertRaisesRegex(ShadowValidationError, "arguments"):
            self.validate(context=context)

    def test_missing_denied_expired_and_replayed_contexts_are_rejected(self):
        with self.assertRaisesRegex(ShadowValidationError, "required"):
            self.adapter.shadow_validate(
                capability_id=self.manifest.capability_id,
                contract_version=self.manifest.contract_version,
                manifest_digest=self.manifest.manifest_digest,
                arguments={},
                context=None,
            )

        denied = self.context(allowed=False)
        with self.assertRaisesRegex(ShadowValidationError, "denied"):
            self.validate(context=denied)

        expired = self.context()
        self.now[0] = 100.0
        with self.assertRaisesRegex(ShadowValidationError, "expired"):
            self.validate(context=expired)

        self.now[0] = 50.0
        fresh = self.context()
        self.validate(context=fresh)
        with self.assertRaisesRegex(ShadowValidationError, "already been used"):
            self.validate(context=fresh)

    def test_altered_context_signature_is_rejected(self):
        from dataclasses import replace

        context = self.context()
        altered = replace(context, manifest_digest="f" * 64)
        with self.assertRaisesRegex(ShadowValidationError, "integrity"):
            self.validate(context=altered)

    def test_bindings_are_instance_owned_and_immutable_snapshots(self):
        second = LegacyCapabilityAdapter(self.issuer)
        self.assertEqual(len(self.adapter.bindings()), 1)
        self.assertEqual(second.bindings(), ())
        with self.assertRaises(AttributeError):
            self.adapter.bindings()[0].legacy_tool_name = "bash"

    def test_shell_screen_capture_and_disabled_memory_cannot_be_bound(self):
        for name in ("bash", "ssh_exec", "screen_capture", "memory_store"):
            manifest = make_manifest(name)
            adapter = LegacyCapabilityAdapter(self.issuer)
            with self.subTest(name=name), self.assertRaises(LegacyBindingError):
                adapter.bind(manifest, name)

    def test_registration_does_not_make_unreviewed_capability_executable_or_approved(self):
        self.assertEqual(self.manifest.consumer_exposure, "unreviewed")
        result = self.validate()
        self.assertEqual(result.execution_status, "not_executed")
        self.assertEqual(result.validation_status, "shadow_validated")
        self.assertFalse(any(callable(value) for value in vars(self.adapter).values()))
        self.assertFalse(HANDLER_CALLS)

    def test_no_tool_handler_is_retained_or_invoked(self):
        self.assertFalse(any(callable(value) for value in vars(self.adapter).values()))
        self.validate()
        self.assertFalse(HANDLER_CALLS)

    def test_duplicate_binding_is_rejected(self):
        with self.assertRaisesRegex(LegacyBindingError, "already exists"):
            self.adapter.bind(self.manifest, "self_status")


if __name__ == "__main__":
    unittest.main()
