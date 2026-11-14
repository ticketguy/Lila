"""M9: Local-first model router with egress policy and concurrent load ceiling."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EgressPolicy(str, Enum):
    LOCAL_ONLY = "local_only"
    ALLOWLISTED_CLOUD = "allowlisted_cloud"


@dataclass(frozen=True)
class ResourceBudget:
    max_concurrent_models: int = 1
    max_memory_mb: int = 2048


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    is_local: bool
    estimated_memory_mb: int
    role: str = "general"


@dataclass(frozen=True)
class ModelLease:
    lease_id: str
    model_id: str
    is_local: bool
    estimated_memory_mb: int


class ModelRouter:
    """Routes tasks local-first while enforcing egress policy and load ceilings."""

    def __init__(
        self,
        *,
        egress_policy: EgressPolicy = EgressPolicy.LOCAL_ONLY,
        budget: Optional[ResourceBudget] = None,
    ) -> None:
        self._egress_policy = egress_policy
        self._budget = budget or ResourceBudget()
        self._lock = threading.RLock()
        self._active_leases: dict[str, ModelLease] = {}

    @property
    def active_lease_count(self) -> int:
        with self._lock:
            return len(self._active_leases)

    def acquire_lease(
        self,
        spec: ModelSpec,
        *,
        contains_private_data: bool = True,
    ) -> ModelLease:
        if not spec.is_local:
            if self._egress_policy == EgressPolicy.LOCAL_ONLY or contains_private_data:
                raise PermissionError(
                    f"Egress policy ({self._egress_policy.value}) blocks remote model {spec.model_id!r}."
                )
        with self._lock:
            if len(self._active_leases) >= self._budget.max_concurrent_models:
                raise RuntimeError(
                    f"Concurrent model load ceiling ({self._budget.max_concurrent_models}) reached."
                )
            used_mb = sum(l.estimated_memory_mb for l in self._active_leases.values())
            if used_mb + spec.estimated_memory_mb > self._budget.max_memory_mb:
                raise RuntimeError(
                    f"Memory budget ({self._budget.max_memory_mb} MB) exceeded by {spec.model_id!r}."
                )
            lease = ModelLease(
                lease_id=f"lease-{uuid.uuid4().hex[:10]}",
                model_id=spec.model_id,
                is_local=spec.is_local,
                estimated_memory_mb=spec.estimated_memory_mb,
            )
            self._active_leases[lease.lease_id] = lease
            return lease

    def release_lease(self, lease_id: str) -> bool:
        with self._lock:
            return self._active_leases.pop(lease_id, None) is not None
