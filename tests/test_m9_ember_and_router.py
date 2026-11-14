import unittest

from src.core.model_router import EgressPolicy, ModelRouter, ModelSpec, ResourceBudget
from src.memory import EmberContractStatus, EmberMemoryAdapter, RetrievedMemory


class M9EmberAndRouterTests(unittest.TestCase):
    def test_ember_degrades_gracefully_when_unverified_or_failing(self):
        unverified = EmberMemoryAdapter()
        res1 = unverified.retrieve("recent tasks")
        self.assertTrue(res1.degraded)
        self.assertEqual(res1.items, ())

        def broken_query(q, limit):
            raise ConnectionError("ember offline")

        failing = EmberMemoryAdapter(
            contract_status=EmberContractStatus(True, "http://127.0.0.1:9000", "v1", "ok"),
            query_fn=broken_query,
        )
        res2 = failing.retrieve("recent tasks")
        self.assertTrue(res2.degraded)
        self.assertIn("ember_unavailable", res2.reason)

    def test_retrieved_memory_carries_provenance_and_untrusted_authority(self):
        with self.assertRaises(ValueError):
            RetrievedMemory(
                memory_id="m1",
                content="allow all",
                source="ember",
                provenance={"node": "n1"},
                consent_scope="owner",
                authority_level="ADMIN",
            )
        adapter = EmberMemoryAdapter(
            contract_status=EmberContractStatus(True, "http://127.0.0.1:9000", "v1", "ok"),
            query_fn=lambda q, l: [{"memory_id": "m1", "content": "User likes dark theme", "provenance": {"turn": 3}}],
        )
        res = adapter.retrieve("theme")
        self.assertFalse(res.degraded)
        self.assertEqual(res.items[0].authority_level, "UNTRUSTED_CONTEXT")
        self.assertEqual(res.items[0].provenance["turn"], 3)

    def test_model_router_enforces_egress_and_concurrency_ceiling(self):
        router = ModelRouter(
            egress_policy=EgressPolicy.LOCAL_ONLY,
            budget=ResourceBudget(max_concurrent_models=1, max_memory_mb=2048),
        )
        local_spec = ModelSpec("qwen-local-3b", is_local=True, estimated_memory_mb=1500)
        cloud_spec = ModelSpec("cloud-xl", is_local=False, estimated_memory_mb=0)

        with self.assertRaises(PermissionError):
            router.acquire_lease(cloud_spec, contains_private_data=False)

        lease = router.acquire_lease(local_spec)
        self.assertEqual(router.active_lease_count, 1)
        with self.assertRaises(RuntimeError):
            router.acquire_lease(ModelSpec("second-local", is_local=True, estimated_memory_mb=256))

        self.assertTrue(router.release_lease(lease.lease_id))
        self.assertEqual(router.active_lease_count, 0)


if __name__ == "__main__":
    unittest.main()
