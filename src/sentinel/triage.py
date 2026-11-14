"""M8: Sentinel read-only collectors and triage engine."""

from __future__ import annotations

import os
import platform
import shutil
from typing import Callable, Iterable, Sequence

from src.sentinel.contracts import (
    AlertTier,
    CoverageGap,
    SecurityIndicator,
    SourceCoverageReport,
    TelemetryFact,
    TriageReport,
)


_TIER_RANK = {
    AlertTier.INFO: 0,
    AlertTier.ADVISORY: 1,
    AlertTier.WARNING: 2,
    AlertTier.CRITICAL: 3,
}


def collect_local_readonly_facts(
    enabled_sources: Sequence[str] = ("system_resources", "defender", "firewall"),
) -> tuple[list[TelemetryFact], list[str], list[CoverageGap]]:
    """Collect read-only local facts; unavailable permissions become CoverageGaps."""
    facts: list[TelemetryFact] = []
    available: list[str] = []
    gaps: list[CoverageGap] = []

    for src in enabled_sources:
        if src == "system_resources":
            usage = shutil.disk_usage("/")
            free_pct = round((usage.free / max(1, usage.total)) * 100.0, 2)
            facts.append(TelemetryFact(source="system_resources", metric="disk_free_pct", value=free_pct))
            try:
                load1 = round(os.getloadavg()[0], 2)
                facts.append(TelemetryFact(source="system_resources", metric="load_avg_1m", value=load1))
            except (AttributeError, OSError):
                pass
            available.append("system_resources")
        elif src in ("defender", "firewall", "windows_security_events"):
            # On WSL/non-elevated context, report explicit coverage gap instead of guessing.
            gaps.append(
                CoverageGap(
                    source=src,
                    reason="Windows host collector not invoked in WSL read-only pass; opt-in host bridge required.",
                    permission_required="read_only_windows_security_telemetry",
                )
            )
        else:
            gaps.append(
                CoverageGap(
                    source=src,
                    reason="Source is not in the Sentinel allowlist.",
                    permission_required="allowlist_registration",
                )
            )
    return facts, available, gaps


class SentinelTriageEngine:
    """Evaluates read-only facts from opt-in collectors without any actuation."""

    def __init__(self, allowlisted_sources: Sequence[str] = ("system_resources", "defender", "firewall")) -> None:
        self._allowlisted = tuple(allowlisted_sources)

    def run_triage(
        self,
        *,
        opted_in_sources: Sequence[str],
        collector_fn: Callable[[Sequence[str]], tuple[list[TelemetryFact], list[str], list[CoverageGap]]] = collect_local_readonly_facts,
    ) -> TriageReport:
        allowed = [s for s in opted_in_sources if s in self._allowlisted]
        blocked = [s for s in opted_in_sources if s not in self._allowlisted]

        facts, available, gaps = collector_fn(allowed)
        for b in blocked:
            gaps.append(CoverageGap(source=b, reason="Source not opted-in/allowlisted.", permission_required="allowlist"))

        indicators: list[SecurityIndicator] = []
        for fact in facts:
            if fact.metric == "disk_free_pct" and isinstance(fact.value, (int, float)) and fact.value < 10.0:
                indicators.append(
                    SecurityIndicator(
                        indicator_id="sentinel.disk_low",
                        source=fact.source,
                        summary=f"Low disk space ({fact.value}% free).",
                        severity="high" if fact.value < 5.0 else "medium",
                        confidence=0.98,
                        alert_tier=AlertTier.WARNING,
                        evidence_metrics=("disk_free_pct",),
                    )
                )
            elif fact.metric == "defender_realtime_enabled" and fact.value is False:
                indicators.append(
                    SecurityIndicator(
                        indicator_id="sentinel.defender_disabled",
                        source=fact.source,
                        summary="Windows Defender real-time protection is reported disabled.",
                        severity="high",
                        confidence=0.95,
                        alert_tier=AlertTier.CRITICAL,
                        evidence_metrics=("defender_realtime_enabled",),
                    )
                )
            elif fact.metric == "firewall_enabled" and fact.value is False:
                indicators.append(
                    SecurityIndicator(
                        indicator_id="sentinel.firewall_disabled",
                        source=fact.source,
                        summary="Host firewall profile is reported disabled.",
                        severity="high",
                        confidence=0.92,
                        alert_tier=AlertTier.WARNING,
                        evidence_metrics=("firewall_enabled",),
                    )
                )

        highest = AlertTier.INFO
        for ind in indicators:
            if _TIER_RANK[ind.alert_tier] > _TIER_RANK[highest]:
                highest = ind.alert_tier

        coverage = SourceCoverageReport(
            available_sources=tuple(available),
            gaps=tuple(gaps),
            os_platform=platform.system(),
            os_version=platform.release(),
        )
        return TriageReport(
            facts=tuple(facts),
            indicators=tuple(indicators),
            coverage=coverage,
            highest_tier=highest,
            autonomous_remediation_attempted=False,
            screen_capture_used=False,
        )
