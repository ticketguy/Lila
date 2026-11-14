import hashlib
import json
import unittest

from src.harness.capabilities import (
    CANONICALIZATION_PROFILE,
    CapabilityManifest,
    CONSUMER_EXPOSURE_STATES,
    LEGACY_CONTRACT_VERSION,
    MANIFEST_DIGEST_DOMAIN,
    ManifestValidationError,
    PROVIDER_DIGEST_DOMAIN,
    canonicalize_restricted_jcs,
    legacy_tool_to_manifest,
    manifest_from_json,
    manifest_from_mapping,
    parse_json_strict,
    provider_artifact_digest,
    verify_provider_artifact_digest,
    _decimal_string,
)
from src.harness.system_tools import register_all_system_tools
from src.harness.tools import Tool, ToolArg, ToolRegistry


def sample_payload():
    return {
        "capability_id": "org.lila.capability.test.echo",
        "contract_version": "1.0.0",
        "display_name": "echo",
        "summary": "Echo input",
        "input_arguments": [{
            "name": "message", "type": "string", "description": "Text",
            "required": True, "default": None, "choices": None,
            "minimum": None, "maximum": None,
        }],
        "output_contract": {
            "kind": "observed",
            "fields": {"success": "boolean", "output": "text", "error": "optional text", "data": "optional untyped value"},
            "guarantees": [],
        },
        "declared_effects": [],
        "resource_bounds": {},
        "findings": [{"code": "observed_legacy", "evidence": "test"}],
        "availability_observations": {
            "registered_observation": True, "legacy_prompt_visible": True,
            "legacy_risk_label": "low", "legacy_permission_label": "auto",
            "daemon_without_approval_callback": "not_approval_gated",
        },
        "legacy_provenance": {
            "provider_module": None, "provider_digest_status": "unverified",
            "provider_integrity_digest": None, "legacy_tool_name": "echo",
        },
        "consumer_exposure": "unreviewed",
    }


class RestrictedJcsTests(unittest.TestCase):
    def test_jcs_utf16_property_order_differs_from_python_codepoint_order(self):
        value = {"\ue000": 1, "\U00010000": 2}
        self.assertEqual(
            canonicalize_restricted_jcs(value),
            '{"\U00010000":2,"\ue000":1}'.encode("utf-8"),
        )

    def test_unicode_is_preserved_without_normalization(self):
        value = {"a": "é", "b": "e\u0301"}
        encoded = canonicalize_restricted_jcs(value).decode("utf-8")
        self.assertIn("é", encoded)
        self.assertIn("e\u0301", encoded)
        self.assertNotEqual(value["a"], value["b"])

    def test_string_escaping_and_supported_jcs_fixture(self):
        self.assertEqual(
            canonicalize_restricted_jcs({"s": 'quote" slash/ line\n'}),
            b'{"s":"quote\\\" slash/ line\\n"}',
        )
        # RFC 8785-compatible subset fixture (no floating-point values).
        self.assertEqual(canonicalize_restricted_jcs({"a": 1, "b": True, "c": None}), b'{"a":1,"b":true,"c":null}')
        self.assertEqual(canonicalize_restricted_jcs({"€": "€"}), '{"€":"€"}'.encode("utf-8"))

    def test_independent_rfc8785_utf16_order_fixture(self):
        # Property order and expected sequence are from RFC 8785 §3.2.3's
        # published sorting test data (https://www.rfc-editor.org/rfc/rfc8785);
        # floats are intentionally omitted here.
        value = {
            "€": "Euro Sign",
            "\r": "Carriage Return",
            "דּ": "Hebrew Letter Dalet With Dagesh",
            "1": "One",
            "😀": "Emoji: Grinning Face",
            "\u0080": "Control",
            "ö": "Latin Small Letter O With Diaeresis",
        }
        expected = (
            '{"\\r":"Carriage Return","1":"One","\u0080":"Control",'
            '"ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign",'
            '"😀":"Emoji: Grinning Face","דּ":"Hebrew Letter Dalet With Dagesh"}'
        ).encode("utf-8")
        self.assertEqual(canonicalize_restricted_jcs(value), expected)

    def test_independent_rfc8785_string_and_unicode_fixture(self):
        # Escaping follows RFC 8785 §3.2.2.2: lowercase control escapes,
        # short escapes for the five named controls, raw Unicode, and no slash escaping.
        value = {"s": "€\x0f\n\t\r\b\f\"\\/e\u0301"}
        expected = b'{"s":"\xe2\x82\xac\\u000f\\n\\t\\r\\b\\f\\\"\\\\/e\xcc\x81"}'
        self.assertEqual(canonicalize_restricted_jcs(value), expected)

    def test_independent_integer_boundary_and_utf8_fixture(self):
        # The restricted profile permits only interoperable safe integers;
        # decimal spellings below are literal expected UTF-8 bytes.
        self.assertEqual(
            canonicalize_restricted_jcs([-9007199254740991, 0, 9007199254740991]),
            b"[-9007199254740991,0,9007199254740991]",
        )
        self.assertEqual(canonicalize_restricted_jcs("é😀"), b'"\xc3\xa9\xf0\x9f\x98\x80"')

    def test_independent_fixed_manifest_and_provider_digest_vectors(self):
        # Derived independently as SHA-256(domain bytes || canonical payload bytes).
        # RFC 8785 §§3.2.3-3.2.4 define sorting and UTF-8; the manifest profile
        # additionally excludes floats and limits integers to the safe range.
        manifest = manifest_from_mapping(sample_payload())
        expected_manifest_domain = b"lila-capability-manifest\x00rfc8785-jcs-v1-i53\x00"
        expected_manifest_payload = (
            b'{"availability_observations":{"daemon_without_approval_callback":"not_approval_gated",'
            b'"legacy_permission_label":"auto","legacy_prompt_visible":true,"legacy_risk_label":"low",'
            b'"registered_observation":true},"capability_id":"org.lila.capability.test.echo",'
            b'"consumer_exposure":"unreviewed","contract_version":"1.0.0","declared_effects":[],'
            b'"display_name":"echo","findings":[{"code":"observed_legacy","evidence":"test"}],'
            b'"input_arguments":[{"choices":null,"default":null,"description":"Text","maximum":null,'
            b'"minimum":null,"name":"message","required":true,"type":"string"}],'
            b'"legacy_provenance":{"legacy_tool_name":"echo","provider_digest_status":"unverified",'
            b'"provider_integrity_digest":null,"provider_module":null},"output_contract":{"fields":'
            b'{"data":"optional untyped value","error":"optional text","output":"text",'
            b'"success":"boolean"},"guarantees":[],"kind":"observed"},'
            b'"profile":"rfc8785-jcs-v1-i53","resource_bounds":{},"summary":"Echo input"}'
        )
        expected_manifest_digest = "c4c88a4a7616426ac019dff77ef0a38beb521390286538e960c522f89936594d"
        self.assertEqual(MANIFEST_DIGEST_DOMAIN, expected_manifest_domain)
        self.assertEqual(manifest.canonical_payload(), expected_manifest_payload)
        self.assertEqual(
            hashlib.sha256(expected_manifest_domain + expected_manifest_payload).hexdigest(),
            expected_manifest_digest,
        )
        self.assertEqual(manifest.manifest_digest, expected_manifest_digest)
        artifact = b"fixed synthetic provider artifact"
        expected_provider_domain = b"lila-provider-artifact\x00sha256-v1\x00"
        expected_provider_digest = "895febb3c1bd9d3b0c0c01f6d5cc3d64f7fdccb006e1740c4e37af3b7627075b"
        self.assertEqual(PROVIDER_DIGEST_DOMAIN, expected_provider_domain)
        self.assertEqual(
            hashlib.sha256(expected_provider_domain + artifact).hexdigest(),
            expected_provider_digest,
        )
        self.assertEqual(provider_artifact_digest(artifact), expected_provider_digest)

    def test_rejects_floats_non_finite_unsafe_integer_and_surrogates(self):
        for value in (1.25, float("nan"), 2**53, "\ud800"):
            with self.subTest(value=repr(value)), self.assertRaises(ManifestValidationError):
                canonicalize_restricted_jcs(value)
        self.assertEqual(canonicalize_restricted_jcs(2**53 - 1), b"9007199254740991")
        self.assertEqual(canonicalize_restricted_jcs(-(2**53 - 1)), b"-9007199254740991")

    def test_strict_json_rejects_duplicate_keys_and_non_finite_numbers(self):
        with self.assertRaises(ManifestValidationError):
            parse_json_strict('{"x":1,"x":2}')
        with self.assertRaises(ManifestValidationError):
            parse_json_strict('{"x":NaN}')

    def test_manifest_is_deeply_immutable_and_defensively_copied(self):
        payload = sample_payload()
        manifest = manifest_from_mapping(payload)
        payload["input_arguments"][0]["description"] = "mutated"
        payload["output_contract"]["guarantees"].append("made up")
        self.assertEqual(manifest.input_arguments[0]["description"], "Text")
        with self.assertRaises(TypeError):
            manifest.output_contract["kind"] = "changed"
        with self.assertRaises(AttributeError):
            manifest.output_contract["guarantees"].append("changed")

    def test_strict_manifest_rejects_missing_unknown_and_invalid_fields(self):
        payload = sample_payload()
        del payload["summary"]
        with self.assertRaises(ManifestValidationError):
            manifest_from_mapping(payload)
        payload = sample_payload()
        payload["unexpected"] = True
        with self.assertRaises(ManifestValidationError):
            manifest_from_mapping(payload)
        payload = sample_payload()
        payload["input_arguments"][0]["unknown"] = 1
        with self.assertRaises(ManifestValidationError):
            manifest_from_mapping(payload)
        payload = sample_payload()
        payload["input_arguments"][0]["minimum"] = 1.5
        with self.assertRaises(ManifestValidationError):
            manifest_from_mapping(payload)

    def test_digest_is_deterministic_sensitive_to_payload_and_domain_separated(self):
        first = manifest_from_mapping(sample_payload())
        reordered = sample_payload()
        reordered["summary"] = "Echo input"
        second = manifest_from_mapping(reordered)
        self.assertEqual(first.canonical_payload(), second.canonical_payload())
        self.assertEqual(first.manifest_digest, second.manifest_digest)
        changed = sample_payload()
        changed["summary"] = "Different"
        self.assertNotEqual(first.manifest_digest, manifest_from_mapping(changed).manifest_digest)
        expected = hashlib.sha256(MANIFEST_DIGEST_DOMAIN + first.canonical_payload()).hexdigest()
        self.assertEqual(first.manifest_digest, expected)
        self.assertEqual(first.canonicalization_profile, CANONICALIZATION_PROFILE)

    def test_manifest_json_parser_constructs_and_validates(self):
        manifest = manifest_from_json(json.dumps(sample_payload()))
        self.assertEqual(manifest.contract_version, "1.0.0")
        with self.assertRaises(ManifestValidationError):
            manifest_from_json('{"capability_id":"a","capability_id":"b"}')

    def test_provider_artifact_digest_is_only_a_byte_artifact_primitive(self):
        artifact = b"fixed synthetic provider artifact"
        digest = provider_artifact_digest(artifact)
        self.assertTrue(verify_provider_artifact_digest(digest, artifact))
        self.assertFalse(verify_provider_artifact_digest(digest, artifact + b" changed"))

    def test_legacy_provider_provenance_is_unverified(self):
        registry = ToolRegistry()
        register_all_system_tools(registry=registry)
        manifest = legacy_tool_to_manifest(registry.get("file_read"))
        self.assertEqual(manifest.contract_version, LEGACY_CONTRACT_VERSION)
        self.assertEqual(manifest.legacy_provenance["provider_digest_status"], "unverified")
        self.assertIsNone(manifest.legacy_provenance["provider_integrity_digest"])
        self.assertNotIn("execute", manifest.legacy_provenance)

    def test_screen_capture_is_historical_unsafe_not_future_screen_contract(self):
        registry = ToolRegistry()
        from src.harness.extended_tools import register_extended_tools
        register_extended_tools(registry=registry)
        manifest = legacy_tool_to_manifest(registry.get("screen_capture"))
        codes = {finding["code"] for finding in manifest.findings}
        self.assertIn("unsafe_legacy", codes)
        self.assertIn("incomplete_output", codes)
        self.assertTrue(manifest.capability_id.endswith("legacy-screen-capture"))
        self.assertEqual(manifest.consumer_exposure, "prohibited_from_consumer_exposure")
        self.assertNotIn("future_user_initiated", manifest.availability_observations)

    def test_legacy_exposure_is_unreviewed_or_prohibited_without_becoming_authority(self):
        standard = ToolRegistry()
        register_all_system_tools(registry=standard)
        ordinary = legacy_tool_to_manifest(standard.get("file_read"))
        self.assertEqual(ordinary.consumer_exposure, "unreviewed")

        self.assertEqual(
            CONSUMER_EXPOSURE_STATES,
            frozenset({"unreviewed", "approved_for_consumer_exposure", "prohibited_from_consumer_exposure"}),
        )
        reviewed_payload = sample_payload()
        reviewed_payload["consumer_exposure"] = "approved_for_consumer_exposure"
        reviewed = manifest_from_mapping(reviewed_payload)
        self.assertEqual(reviewed.consumer_exposure, "approved_for_consumer_exposure")
        baseline = manifest_from_mapping(sample_payload())
        self.assertNotEqual(baseline.consumer_exposure, reviewed.consumer_exposure)
        self.assertNotEqual(baseline.manifest_digest, reviewed.manifest_digest)

        for name in ("memory_store", "memory_recall", "memory_list"):
            placeholder = legacy_tool_to_manifest(standard.get(name))
            self.assertEqual(placeholder.consumer_exposure, "prohibited_from_consumer_exposure")
            self.assertEqual(placeholder.availability_observations["legacy_permission_label"], "disabled")

        screen_payload = sample_payload()
        screen_payload["capability_id"] = "org.lila.capability.perception.legacy-screen-capture"
        screen_payload["consumer_exposure"] = "approved_for_consumer_exposure"
        with self.assertRaises(ManifestValidationError):
            manifest_from_mapping(screen_payload)
        screen_payload = sample_payload()
        screen_payload["legacy_provenance"]["legacy_tool_name"] = "screen_capture"
        screen_payload["consumer_exposure"] = "approved_for_consumer_exposure"
        with self.assertRaises(ManifestValidationError):
            manifest_from_mapping(screen_payload)

    def _construct_both(self, payload):
        manifest_from_mapping(payload)
        CapabilityManifest(**payload)

    def _argument_payload(self, arg_type="int", **overrides):
        argument = {
            "name": "value", "type": arg_type, "description": "Value",
            "required": False, "default": None, "choices": None,
            "minimum": None, "maximum": None,
        }
        argument.update(overrides)
        payload = sample_payload()
        payload["input_arguments"] = [argument]
        return payload

    def test_argument_contract_rejects_invalid_values_through_both_paths(self):
        invalid_payloads = [
            self._argument_payload(default="7"),
            self._argument_payload(choices=["7"]),
            self._argument_payload(choices=[1, 1]),
            self._argument_payload(minimum="5", maximum="4"),
            self._argument_payload(default=4, minimum="1", maximum="3"),
            self._argument_payload(default=True),
            self._argument_payload(default={"not": "a scalar argument type"}),
            self._argument_payload(required=True, default=1),
            self._argument_payload(arg_type="string", minimum="1"),
            self._argument_payload(choices=[1, 3], minimum="1", maximum="2"),
        ]
        for payload in invalid_payloads:
            with self.subTest(argument=payload["input_arguments"][0]):
                with self.assertRaises(ManifestValidationError):
                    manifest_from_mapping(payload)
                with self.assertRaises(ManifestValidationError):
                    CapabilityManifest(**payload)

    def test_argument_contract_accepts_valid_boundaries_without_coercion(self):
        integer = self._argument_payload(
            default=2, choices=[-2, 0, 2], minimum="-2", maximum="2"
        )
        self._construct_both(integer)
        exact_float = self._argument_payload(
            arg_type="float", default="0.10000000000000000001",
            choices=["-0.1", "0.10000000000000000001"],
            minimum="-0.1", maximum="0.10000000000000000002",
        )
        self._construct_both(exact_float)

    def test_canonical_decimal_grammar_rejects_noncanonical_equivalent_values(self):
        for spelling in ("-0", "-0.0", "1.00", "01", "00.1", ".1", "1.", "1e2", "NaN", "Infinity"):
            payload = self._argument_payload(minimum=spelling)
            with self.subTest(spelling=spelling):
                with self.assertRaises(ManifestValidationError):
                    manifest_from_mapping(payload)
                with self.assertRaises(ManifestValidationError):
                    CapabilityManifest(**payload)
        for spelling in ("0", "-1", "0.01", "-0.01", "123.456"):
            payload = self._argument_payload(arg_type="float", minimum=spelling)
            with self.subTest(spelling=spelling):
                self._construct_both(payload)

    def test_decimal_projection_is_exact_context_independent_and_canonical(self):
        large_integer = 12345678901234567890123456789012345678901234567890
        tool = Tool(
            name="tcp_connect",
            description="Synthetic large-bound projection fixture",
            args=[ToolArg(
                "limit", "float", "Exact bound", required=False,
                minimum=large_integer, maximum=large_integer + 2,
            )],
            execute=lambda **kwargs: None,
        )
        manifest = legacy_tool_to_manifest(tool)
        projected = manifest.input_arguments[0]
        self.assertEqual(projected["minimum"], str(large_integer))
        self.assertEqual(projected["maximum"], str(large_integer + 2))
        self.assertIn(
            b'"minimum":"' + str(large_integer).encode("ascii") + b'"',
            manifest.canonical_payload(),
        )

        self.assertEqual(_decimal_string(123.4500), "123.45")
        self.assertEqual(_decimal_string(1e-7), "0.0000001")
        self.assertEqual(_decimal_string(-1.25), "-1.25")
        self.assertEqual(_decimal_string(-0.0), "0")
        self.assertEqual(_decimal_string(-10), "-10")
        self.assertEqual(_decimal_string(0), "0")

        for invalid in (True, "1.25", float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ManifestValidationError):
                _decimal_string(invalid)

    def test_semver_2_validation(self):
        valid = (
            "0.0.0-legacy", "1.2.3-alpha", "1.2.3-alpha.1",
            "1.2.3-0A-x+build.007", "1.2.3+build-metadata.01",
        )
        invalid = (
            "1.0.0-", "1.0.0-alpha..1", "1.0.0-01", "1.0.0-00",
            "01.0.0", "1.01.0", "1.0.01", "1.0", "1.0.0+",
            "1.0.0-alpha+build..x", "1.0.0-ä", "v1.0.0",
        )
        for version in valid:
            payload = sample_payload()
            payload["contract_version"] = version
            with self.subTest(version=version):
                self._construct_both(payload)
        for version in invalid:
            payload = sample_payload()
            payload["contract_version"] = version
            with self.subTest(version=version), self.assertRaises(ManifestValidationError):
                manifest_from_mapping(payload)
            with self.subTest(direct_version=version), self.assertRaises(ManifestValidationError):
                CapabilityManifest(**payload)


if __name__ == "__main__":
    unittest.main()
