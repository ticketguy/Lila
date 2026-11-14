"""M9: Provenance-bearing Ember memory adapter with graceful degradation.

- Verifies Ember API contract readiness before querying.
- Gracefully degrades when Ember is unconfigured, unreachable, or errors.
- Retrieved memory records always carry provenance + `authority_level="UNTRUSTED_CONTEXT"`
  so retrieved memory can never alter policy or grant tool authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class EmberContractStatus:
    verified: bool
    endpoint: Optional[str]
    api_version: Optional[str]
    reason: str


@dataclass(frozen=True)
class RetrievedMemory:
    memory_id: str
    content: str
    source: str
    provenance: dict[str, Any]
    consent_scope: str
    authority_level: str = "UNTRUSTED_CONTEXT"

    def __post_init__(self) -> None:
        if self.authority_level != "UNTRUSTED_CONTEXT":
            raise ValueError("Retrieved memory cannot hold elevated authority.")


@dataclass(frozen=True)
class MemoryQueryResult:
    items: tuple[RetrievedMemory, ...]
    degraded: bool
    reason: Optional[str] = None


class EmberMemoryAdapter:
    """Adapter for external Ember memory service with strict contract & authority bounds."""

    def __init__(
        self,
        *,
        contract_status: Optional[EmberContractStatus] = None,
        query_fn: Optional[Callable[[str, int], list[dict[str, Any]]]] = None,
    ) -> None:
        self._contract = contract_status or EmberContractStatus(
            verified=False,
            endpoint=None,
            api_version=None,
            reason="Ember API contract not yet verified for this environment.",
        )
        self._query_fn = query_fn

    @property
    def contract_status(self) -> EmberContractStatus:
        return self._contract

    def retrieve(self, query: str, *, limit: int = 5) -> MemoryQueryResult:
        if not self._contract.verified or self._query_fn is None:
            return MemoryQueryResult(items=(), degraded=True, reason=self._contract.reason)
        try:
            raw_items = self._query_fn(query, limit)
            items: list[RetrievedMemory] = []
            for idx, entry in enumerate(raw_items[:limit]):
                items.append(
                    RetrievedMemory(
                        memory_id=str(entry.get("memory_id") or f"mem-{idx}"),
                        content=str(entry.get("content") or ""),
                        source=str(entry.get("source") or "ember"),
                        provenance=dict(entry.get("provenance") or {"endpoint": self._contract.endpoint}),
                        consent_scope=str(entry.get("consent_scope") or "owner_session"),
                        authority_level="UNTRUSTED_CONTEXT",
                    )
                )
            return MemoryQueryResult(items=tuple(items), degraded=False)
        except Exception as exc:
            return MemoryQueryResult(items=(), degraded=True, reason=f"ember_unavailable:{type(exc).__name__}")
