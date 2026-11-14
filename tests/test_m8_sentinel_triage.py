import unittest

from src.sentinel import (
    AlertTier,
    CoverageGap,
    SentinelTriageEngine,
    TelemetryFact,
)


class M8SentinelTriageTests(unittest.TestCase):
    def test_synthetic_triage_separates_facts_indicators_gaps_and_tiers(self):
        def fake_collector(sources):
            facts = [
                TelemetryFact(source="system_resources", metric="disk_free_pct", value=4.2),
                TelemetryFact(source="defender", metric="defender_realtime_enabled", value=False),
            ]
            gaps = [
                CoverageGap(
                    source="firewall",
                    reason="Access denied to host firewall profile.",
                    permission_required="read_only_firewall_status",
                )
            ]
            return facts, ["system_resources", "defender"], gaps

        engine = SentinelTriageEngine()
        report = engine.run_triage(
            opted_in_sources=["system_resources", "defender", "firewall", "unapproved_camera"],
            collector_fn=fake_collector,
        )
        self.assertFalse(report.autonomous_remediation_attempted)
        self.assertFalse(report.screen_capture_used)
        self.assertEqual(report.highest_tier, AlertTier.CRITICAL)
        self.assertEqual(len(report.facts), 2)
        self.assertEqual(len(report.indicators), 2)
        self.assertEqual(report.indicators[0].severity, "high")
        self.assertGreater(report.indicators[0].confidence, 0.9)
        gap_sources = {g.source for g in report.coverage.gaps}
        self.assertIn("firewall", gap_sources)
        self.assertIn("unapproved_camera", gap_sources)

    def test_readonly_fact_rejects_mutating_flag(self):
        with self.assertRaises(ValueError):
            TelemetryFact(source="system_resources", metric="x", value=1, read_only=False)


if __name__ == "__main__":
    unittest.main()
