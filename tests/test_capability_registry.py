import unittest

from src.harness.capabilities import (
    CapabilityRegistry,
    DuplicateCapabilityError,
    ManifestValidationError,
    manifest_from_mapping,
)
from tests.test_capability_manifests import sample_payload


class CapabilityRegistryTests(unittest.TestCase):
    def manifest(self, version="1.0.0"):
        payload = sample_payload()
        payload["contract_version"] = version
        return manifest_from_mapping(payload)

    def test_registries_are_instance_owned_and_snapshots_are_immutable(self):
        left, right = CapabilityRegistry(), CapabilityRegistry()
        item = self.manifest()
        left.register(item)
        self.assertEqual(right.all(), ())
        self.assertIsNot(left.snapshot(), right.snapshot())
        with self.assertRaises(TypeError):
            left.snapshot()[("x", "1.0.0")] = item
        self.assertIs(left.get(item.capability_id, "1.0.0"), item)

    def test_duplicate_identity_and_version_is_rejected(self):
        registry = CapabilityRegistry()
        item = self.manifest()
        registry.register(item)
        with self.assertRaises(DuplicateCapabilityError):
            registry.register(item)

    def test_versions_coexist_and_lookup_requires_exact_version(self):
        registry = CapabilityRegistry()
        v1, v2 = self.manifest("1.0.0"), self.manifest("2.0.0")
        registry.register(v1)
        registry.register(v2)
        self.assertIs(registry.get(v1.capability_id, "1.0.0"), v1)
        self.assertIs(registry.get(v2.capability_id, "2.0.0"), v2)
        with self.assertRaises(TypeError):
            registry.get(v1.capability_id)
        with self.assertRaises(KeyError):
            registry.get(v1.capability_id, "3.0.0")

    def test_registry_rejects_non_manifest_and_retains_no_provider(self):
        registry = CapabilityRegistry()
        with self.assertRaises(TypeError):
            registry.register(object())
        item = self.manifest()
        registry.register(item)
        self.assertFalse(hasattr(registry, "dispatch"))
        self.assertFalse(any(callable(value) for value in vars(item).values()))

    def test_invalid_manifest_digest_domain_is_not_a_registry_provider_binding(self):
        payload = sample_payload()
        payload["findings"] = [{"code": "arbitrary", "evidence": "bad"}]
        with self.assertRaises(ManifestValidationError):
            manifest_from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
