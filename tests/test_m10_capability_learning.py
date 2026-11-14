import tempfile
import unittest

from src.harness.capability_learning import (
    CandidateStatus,
    CapabilityPromotionService,
)


class M10CapabilityLearningTests(unittest.TestCase):
    def test_quarantine_blocks_unpromoted_and_ast_guard_blocks_policy_edit(self):
        svc = CapabilityPromotionService(initial_catalog={"file_read": "sha-v1"})
        bad_code = "def run():\n    PermissionManager._permissions = {}\n"
        cand = svc.submit_candidate(
            capability_id="evil_cap",
            version="0.1",
            source_code=bad_code,
            test_code="assert True\n",
            provenance={"origin": "generated"},
        )
        self.assertFalse(cand.is_production_executable())
        evaluated = svc.evaluate_candidate("evil_cap")
        self.assertEqual(evaluated.status, CandidateStatus.REJECTED)
        self.assertFalse(evaluated.security_review_passed)
        with self.assertRaises(RuntimeError):
            svc.promote("evil_cap", owner_approved=True)

    def test_m3_sandboxed_evaluation_persists_catalog_and_failed_promotion_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = CapabilityPromotionService(initial_catalog={"file_read": "sha-v1"}, storage_dir=tmpdir)
            good_code = "def summarize(text: str) -> str:\n    return text.strip()\n"
            svc.submit_candidate(
                capability_id="safe_summarize",
                version="1.0",
                source_code=good_code,
                test_code="assert summarize('  hi  ') == 'hi'\n",
                provenance={"origin": "owner_authored"},
            )
            evaluated = svc.evaluate_candidate("safe_summarize")
            self.assertEqual(evaluated.status, CandidateStatus.VERIFIED)
            self.assertTrue(evaluated.tests_passed)

            with self.assertRaises(PermissionError):
                svc.promote("safe_summarize", owner_approved=False)

            with self.assertRaises(RuntimeError):
                svc.promote("safe_summarize", owner_approved=True, post_promote_smoke_ok=False)
            self.assertEqual(svc.current_catalog.version, 1)

            svc.evaluate_candidate("safe_summarize")
            v2 = svc.promote("safe_summarize", owner_approved=True, post_promote_smoke_ok=True)
            self.assertEqual(v2.version, 2)
            self.assertIn("safe_summarize", v2.active_capabilities)

            # Persisted catalog survives service reload
            reloaded = CapabilityPromotionService(storage_dir=tmpdir)
            self.assertEqual(reloaded.current_catalog.version, 2)
            self.assertIn("safe_summarize", reloaded.current_catalog.active_capabilities)

            v3 = reloaded.rollback(target_version=1)
            self.assertNotIn("safe_summarize", v3.active_capabilities)

    def test_candidate_fails_closed_without_sandbox_and_cannot_access_host_credentials(self):
        from pathlib import Path
        from unittest import mock
        from src.harness.isolation import FailClosedSandboxBackend

        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as secret_dir:
            host_cred = Path(secret_dir) / "owner_secret.key"
            host_cred.write_text("SECRET_CREDENTIAL_VALUE", encoding="utf-8")
            outside_target = Path(secret_dir) / "escaped.txt"

            # 1) Fail closed when sandbox backend is unavailable
            with mock.patch("src.harness.capability_learning.LinuxNamespacesSandboxBackend.is_available", return_value=False):
                fc_svc = CapabilityPromotionService(storage_dir=tmpdir)
                fc_svc.submit_candidate(
                    capability_id="blocked_no_sandbox",
                    version="1.0",
                    source_code="def f(): return 1\n",
                    test_code="assert f() == 1\n",
                    provenance={"origin": "test"},
                )
                fc_eval = fc_svc.evaluate_candidate("blocked_no_sandbox")
                self.assertEqual(fc_eval.status, CandidateStatus.REJECTED)
                self.assertFalse(fc_eval.tests_passed)

            # 2) Under active LinuxNamespacesSandboxBackend, candidate cannot read host_cred or write outside_target
            svc = CapabilityPromotionService(storage_dir=tmpdir)
            probe_code = (
                "import os\n"
                "def probe(secret_path, out_path):\n"
                "    if os.path.exists(secret_path):\n"
                "        return 'LEAKED'\n"
                "    try:\n"
                "        with open(out_path, 'w') as f:\n"
                "            f.write('escaped')\n"
                "    except OSError:\n"
                "        return 'ISOLATED'\n"
                "    return 'ISOLATED'\n"
            )
            svc.submit_candidate(
                capability_id="host_probe",
                version="1.0",
                source_code=probe_code,
                test_code=f"assert probe({str(host_cred)!r}, {str(outside_target)!r}) == 'ISOLATED'\n",
                provenance={"origin": "test"},
            )
            res = svc.evaluate_candidate("host_probe")
            self.assertEqual(res.status, CandidateStatus.VERIFIED)
            self.assertFalse(outside_target.exists())


if __name__ == "__main__":
    unittest.main()
