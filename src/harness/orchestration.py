"""Milestone M5 (WS-ORCH): Durable Task Orchestration and Bounded Parallelism.

Provides:
- Durable task state machine (PENDING, RUNNING, WAITING_APPROVAL, COMPLETED, FAILED, UNCERTAIN, CANCELLED)
- Atomic JSON checkpoint persistence (`DurableCheckpointStore`)
- Crash/restart recovery that marks interrupted non-idempotent side effects as `uncertain`
- Deadlock-free canonical resource lock manager (`CanonicalResourceLockManager`)
- Dependency-respecting bounded worker pool (`BoundedTaskOrchestrator`)
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .verification import format_failure_for_lila


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({
    TaskState.COMPLETED,
    TaskState.FAILED,
    TaskState.UNCERTAIN,
    TaskState.CANCELLED,
})


@dataclass
class TaskNode:
    """One unit of work in a durable task graph."""

    task_id: str
    name: str
    dependencies: tuple[str, ...] = ()
    resource_keys: tuple[str, ...] = ()
    idempotent: bool = True
    state: TaskState = TaskState.PENDING
    attempts: int = 0
    error: str | None = None
    reason_code: str | None = None
    result_summary: str | None = None
    lila_failure_report: dict[str, Any] | None = None
    updated_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "dependencies": list(self.dependencies),
            "resource_keys": list(self.resource_keys),
            "idempotent": self.idempotent,
            "state": self.state.value,
            "attempts": self.attempts,
            "error": self.error,
            "reason_code": self.reason_code,
            "result_summary": self.result_summary,
            "lila_failure_report": self.lila_failure_report,
            "updated_utc": self.updated_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TaskNode:
        return cls(
            task_id=str(raw["task_id"]),
            name=str(raw.get("name", raw["task_id"])),
            dependencies=tuple(raw.get("dependencies") or ()),
            resource_keys=tuple(raw.get("resource_keys") or ()),
            idempotent=bool(raw.get("idempotent", True)),
            state=TaskState(raw.get("state", TaskState.PENDING.value)),
            attempts=int(raw.get("attempts", 0)),
            error=raw.get("error"),
            reason_code=raw.get("reason_code"),
            result_summary=raw.get("result_summary"),
            lila_failure_report=raw.get("lila_failure_report"),
            updated_utc=str(raw.get("updated_utc") or datetime.now(timezone.utc).isoformat()),
        )


def canonicalize_resource_key(key: str) -> str:
    raw = str(key or "").strip()
    if not raw:
        return ""
    if raw.startswith(("file:", "path:")):
        prefix, _, rest = raw.partition(":")
        return f"{prefix}:{Path(rest).expanduser().resolve(strict=False)}"
    if os.sep in raw or raw.startswith(".") or raw.startswith("/"):
        return f"path:{Path(raw).expanduser().resolve(strict=False)}"
    return raw.lower()


class CanonicalResourceLockManager:
    """Deadlock-free canonical resource lock manager (always acquires in sorted order)."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._held: set[str] = set()

    def try_acquire_all(self, keys: Sequence[str]) -> tuple[bool, tuple[str, ...]]:
        canonical = tuple(sorted({canonicalize_resource_key(k) for k in keys if canonicalize_resource_key(k)}))
        with self._guard:
            if any(k in self._held for k in canonical):
                return False, ()
            self._held.update(canonical)
            return True, canonical

    def release_all(self, canonical_keys: Sequence[str]) -> None:
        with self._guard:
            for k in canonical_keys:
                self._held.discard(k)


class DurableCheckpointStore:
    """Atomic JSON checkpoint store with crash/restart recovery."""

    def __init__(self, checkpoint_path: Path | str) -> None:
        self._path = Path(checkpoint_path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def save(self, run_id: str, nodes: Mapping[str, TaskNode]) -> None:
        payload = {
            "run_id": run_id,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "tasks": {tid: node.to_dict() for tid, node in nodes.items()},
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(encoded, encoding="utf-8")
            os.replace(tmp, self._path)

    def load(self) -> tuple[str | None, dict[str, TaskNode]]:
        if not self._path.is_file():
            return None, {}
        with self._lock:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        tasks = {
            tid: TaskNode.from_dict(node_dict)
            for tid, node_dict in (raw.get("tasks") or {}).items()
        }
        return raw.get("run_id"), tasks

    def recover_on_restart(self) -> tuple[str | None, dict[str, TaskNode]]:
        """Recover interrupted tasks after a crash/restart.

        Non-idempotent tasks that were RUNNING are marked UNCERTAIN (never COMPLETED).
        Idempotent tasks that were RUNNING are reset to PENDING for safe re-execution.
        """
        run_id, tasks = self.load()
        changed = False
        for node in tasks.values():
            if node.state is TaskState.RUNNING:
                changed = True
                node.updated_utc = datetime.now(timezone.utc).isoformat()
                if node.idempotent:
                    node.state = TaskState.PENDING
                    node.reason_code = "orchestration.recovered_idempotent_retry"
                else:
                    node.state = TaskState.UNCERTAIN
                    node.reason_code = "orchestration.interrupted_non_idempotent"
                    node.error = "Task was interrupted mid-execution and is non-idempotent; outcome is uncertain."
                    node.lila_failure_report = format_failure_for_lila(
                        "harness.orchestration",
                        node.reason_code,
                        node.error,
                        tool_name=node.name,
                        recoverable=False,
                    )
        if changed and run_id is not None:
            self.save(run_id, tasks)
        return run_id, tasks


@dataclass(frozen=True)
class OrchestrationRunSummary:
    run_id: str
    tasks: Mapping[str, TaskNode]
    all_succeeded: bool
    failure_reports: tuple[dict[str, Any], ...]


class BoundedTaskOrchestrator:
    """Executes a durable task DAG with bounded parallelism and canonical resource locks."""

    def __init__(
        self,
        checkpoint_store: DurableCheckpointStore | None = None,
        *,
        max_workers: int = 3,
    ) -> None:
        if max_workers < 1 or max_workers > 16:
            raise ValueError("max_workers must be between 1 and 16.")
        self._max_workers = max_workers
        self._checkpoints = checkpoint_store
        self._locks = CanonicalResourceLockManager()
        self._cancelled = False

    def cancel_all(self) -> None:
        self._cancelled = True

    def run(
        self,
        run_id: str,
        nodes: Sequence[TaskNode],
        handlers: Mapping[str, Callable[[TaskNode], str | None]],
    ) -> OrchestrationRunSummary:
        node_map: dict[str, TaskNode] = {n.task_id: n for n in nodes}
        if len(node_map) != len(nodes):
            raise ValueError("Duplicate task_id in orchestration graph.")

        if self._checkpoints is not None:
            self._checkpoints.save(run_id, node_map)

        active: dict[Future[tuple[str, bool, str | None, str | None]], tuple[str, tuple[str, ...]]] = {}

        def save_checkpoint() -> None:
            if self._checkpoints is not None:
                self._checkpoints.save(run_id, node_map)

        with ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="lila-orch") as pool:
            while True:
                if self._cancelled:
                    for node in node_map.values():
                        if node.state is TaskState.PENDING:
                            node.state = TaskState.CANCELLED
                            node.reason_code = "orchestration.cancelled"
                    save_checkpoint()

                # Cancel tasks whose dependency failed/uncertain/cancelled
                for node in node_map.values():
                    if node.state is not TaskState.PENDING:
                        continue
                    dep_states = [node_map[d].state for d in node.dependencies if d in node_map]
                    if any(s in {TaskState.FAILED, TaskState.UNCERTAIN, TaskState.CANCELLED} for s in dep_states):
                        node.state = TaskState.CANCELLED
                        node.reason_code = "orchestration.dependency_failed"
                        node.error = "Skipped because an upstream dependency did not complete."
                        node.lila_failure_report = format_failure_for_lila(
                            "harness.orchestration",
                            node.reason_code,
                            node.error,
                            tool_name=node.name,
                        )
                        save_checkpoint()

                # Dispatch ready tasks up to max_workers
                for node in node_map.values():
                    if len(active) >= self._max_workers:
                        break
                    if node.state is not TaskState.PENDING or self._cancelled:
                        continue
                    if not all(node_map[d].state is TaskState.COMPLETED for d in node.dependencies):
                        continue
                    handler = handlers.get(node.task_id)
                    if handler is None:
                        node.state = TaskState.FAILED
                        node.reason_code = "orchestration.missing_handler"
                        node.error = f"No handler registered for task '{node.task_id}'."
                        node.lila_failure_report = format_failure_for_lila(
                            "harness.orchestration", node.reason_code, node.error, tool_name=node.name
                        )
                        save_checkpoint()
                        continue

                    acquired, locked_keys = self._locks.try_acquire_all(node.resource_keys)
                    if not acquired:
                        continue

                    node.state = TaskState.RUNNING
                    node.attempts += 1
                    node.updated_utc = datetime.now(timezone.utc).isoformat()
                    save_checkpoint()

                    def _exec(target_node: TaskNode = node, fn: Callable[[TaskNode], str | None] = handler):
                        try:
                            out = fn(target_node)
                            return target_node.task_id, True, out, None
                        except Exception as exc:
                            return target_node.task_id, False, None, f"{type(exc).__name__}: {exc}"

                    fut = pool.submit(_exec)
                    active[fut] = (node.task_id, locked_keys)

                if not active:
                    # Check if any pending tasks remain blocked on missing deps
                    if not any(n.state is TaskState.PENDING for n in node_map.values()):
                        break
                    time.sleep(0.01)
                    continue

                done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    tid, locked_keys = active.pop(fut)
                    self._locks.release_all(locked_keys)
                    node = node_map[tid]
                    _, ok, summary, err = fut.result()
                    node.updated_utc = datetime.now(timezone.utc).isoformat()
                    if ok:
                        node.state = TaskState.COMPLETED
                        node.result_summary = summary
                        node.reason_code = "orchestration.completed"
                    else:
                        node.state = TaskState.FAILED
                        node.error = err or "Task execution failed."
                        node.reason_code = "orchestration.task_failed"
                        node.lila_failure_report = format_failure_for_lila(
                            "harness.orchestration",
                            node.reason_code,
                            node.error,
                            tool_name=node.name,
                        )
                    save_checkpoint()

        failures = tuple(
            n.lila_failure_report
            for n in node_map.values()
            if n.lila_failure_report is not None
        )
        return OrchestrationRunSummary(
            run_id=run_id,
            tasks=node_map,
            all_succeeded=all(n.state is TaskState.COMPLETED for n in node_map.values()),
            failure_reports=failures,
        )
