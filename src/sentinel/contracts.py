"""M8: Sentinel read-only telemetry contracts (facts, indicators, gaps, alert tiers)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AlertTier(str, Enum):
    INFO = "info"
    ADVISORY = "advisory"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class TelemetryFact:
    """Raw, un-interpreted observation from an allowlisted read-only collector."""
    source: str
    metric: str
    value: Any
    observed_at: float = field(default_factory=time.time)
    read_only: bool = True

    def __post_init__(self) -> None:
        if not self.read_only:
            raise ValueError("Sentinel telemetry facts must strictly be read_only=True.")


@dataclass(frozen=True)
class SecurityIndicator:
    """Interpreted indicator separating severity from confidence."""
    indicator_id: str
    source: str
    summary: str
    severity: str       # low | medium | high | critical
    confidence: float   # 0.0 .. 1.0
    alert_tier: AlertTier
    evidence_metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageGap:
    source: str
    reason: str
    permission_required: Optional[str] = None


@dataclass(frozen=True)
class SourceCoverageReport:
    available_sources: tuple[str, ...]
    gaps: tuple[CoverageGap, ...]
    os_platform: str
    os_version: str


@dataclass(frozen=True)
class TriageReport:
    facts: tuple[TelemetryFact, ...]
    indicators: tuple[SecurityIndicator, ...]
    coverage: SourceCoverageReport
    highest_tier: AlertTier
    autonomous_remediation_attempted: bool = False
    screen_capture_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "highest_tier": self.highest_tier.value,
            "autonomous_remediation_attempted": self.autonomous_remediation_attempted,
            "screen_capture_used": self.screen_capture_used,
            "facts": [
                {
                    "source": f.source,
                    "metric": f.metric,
                    "value": f.value,
                    "observed_at": f.observed_at,
                }
                for f in self.facts
            ],
            "indicators": [
                {
                    "indicator_id": i.indicator_id,
                    "source": i.source,
                    "summary": i.summary,
                    "severity": i.severity,
                    "confidence": i.confidence,
                    "alert_tier": i.alert_tier.value,
                    "evidence_metrics": list(i.evidence_metrics),
                }
                for i in self.indicators
            ],
            "coverage": {
                "available_sources": list(self.coverage.available_sources),
                "gaps": [
                    {
                        "source": g.source,
                        "reason": g.reason,
                        "permission_required": g.permission_required,
                    }
                    for g in self.coverage.gaps
                ],
                "os_platform": self.coverage.os_platform,
                "os_version": self.coverage.os_version,
            },
        }
