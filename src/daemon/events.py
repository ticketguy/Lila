"""M6: Versioned asynchronous event stream and task center for Lila Daemon.

Provides:
- VersionedEventStream: thread-safe ring buffer with monotonic integer cursors,
  replay with gap detection, and SSE formatting.
- DaemonTaskCenter: asynchronous task lifecycle tracking (created, running,
  completed, failed, cancelled) emitting versioned events.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class StreamEvent:
    event_id: int
    event_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "cursor": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }


def format_sse_event(event: StreamEvent) -> bytes:
    """Format a StreamEvent as a UTF-8 Server-Sent Events frame."""
    data = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
    frame = f"id: {event.event_id}\nevent: {event.event_type}\ndata: {data}\n\n"
    return frame.encode("utf-8")


class VersionedEventStream:
    """Bounded, monotonic-cursor event stream with replay and gap detection."""

    def __init__(self, max_events: int = 512) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._events: deque[StreamEvent] = deque(maxlen=max_events)
        self._next_id = 1
        self._subscribers: list[Callable[[StreamEvent], None]] = []

    @property
    def current_cursor(self) -> int:
        with self._lock:
            return self._next_id - 1

    def publish(self, event_type: str, payload: Optional[dict[str, Any]] = None) -> StreamEvent:
        if not event_type or not isinstance(event_type, str):
            raise ValueError("event_type must be a non-empty string")
        with self._cond:
            event = StreamEvent(
                event_id=self._next_id,
                event_type=event_type,
                payload=dict(payload or {}),
            )
            self._next_id += 1
            self._events.append(event)
            self._cond.notify_all()
            subs = list(self._subscribers)
        for sub in subs:
            try:
                sub(event)
            except Exception:
                pass
        return event

    def poll_since(self, since_cursor: int = 0, limit: int = 100) -> dict[str, Any]:
        if since_cursor < 0:
            since_cursor = 0
        if limit < 1:
            limit = 1
        with self._lock:
            oldest_id = self._events[0].event_id if self._events else self._next_id
            # Gap occurs if client asks for a cursor older than (oldest_id - 1)
            gap_detected = bool(self._events and since_cursor > 0 and since_cursor < (oldest_id - 1))
            matched = [e for e in self._events if e.event_id > since_cursor][:limit]
            next_cursor = matched[-1].event_id if matched else max(since_cursor, self._next_id - 1)
            return {
                "schema_version": SCHEMA_VERSION,
                "since_cursor": since_cursor,
                "next_cursor": next_cursor,
                "gap_detected": gap_detected,
                "events": [e.to_dict() for e in matched],
            }

    def wait_for_events(self, since_cursor: int, timeout: float = 15.0, limit: int = 50) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.05, timeout)
        with self._cond:
            while True:
                batch = self.poll_since(since_cursor=since_cursor, limit=limit)
                if batch["events"]:
                    return batch
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return batch
                self._cond.wait(timeout=remaining)


@dataclass
class DaemonTaskRecord:
    task_id: str
    kind: str
    description: str
    status: str = "queued"  # queued | running | completed | failed | cancelled
    progress: float = 0.0
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "description": self.description,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class DaemonTaskCenter:
    """Tracks asynchronous daemon tasks and publishes lifecycle events."""

    def __init__(self, event_stream: VersionedEventStream, max_tasks: int = 256) -> None:
        self._stream = event_stream
        self._lock = threading.RLock()
        self._tasks: dict[str, DaemonTaskRecord] = {}
        self._order: deque[str] = deque(maxlen=max_tasks)
        self._cancel_flags: dict[str, threading.Event] = {}

    def create_task(self, kind: str, description: str, *, task_id: Optional[str] = None) -> DaemonTaskRecord:
        tid = task_id or f"task-{uuid.uuid4().hex[:12]}"
        with self._lock:
            if tid in self._tasks:
                return self._tasks[tid]
            rec = DaemonTaskRecord(task_id=tid, kind=kind, description=description)
            if len(self._order) == self._order.maxlen:
                evict = self._order.popleft()
                self._tasks.pop(evict, None)
                self._cancel_flags.pop(evict, None)
            self._order.append(tid)
            self._tasks[tid] = rec
            self._cancel_flags[tid] = threading.Event()
        self._stream.publish("task.created", rec.to_dict())
        return rec

    def update_task(self, task_id: str, *, status: Optional[str] = None,
                    progress: Optional[float] = None,
                    result: Optional[dict[str, Any]] = None,
                    error: Optional[str] = None) -> DaemonTaskRecord:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                raise KeyError(f"Unknown task_id: {task_id}")
            if rec.status in ("completed", "failed", "cancelled"):
                return rec
            if status is not None:
                rec.status = status
            if progress is not None:
                rec.progress = max(0.0, min(1.0, float(progress)))
            if result is not None:
                rec.result = result
            if error is not None:
                rec.error = error
            rec.updated_at = time.time()
            snapshot = rec.to_dict()
        ev_type = f"task.{rec.status}" if rec.status in ("completed", "failed", "cancelled") else "task.updated"
        self._stream.publish(ev_type, snapshot)
        return rec

    def cancel_task(self, task_id: str, reason: str = "owner_cancelled") -> DaemonTaskRecord:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                raise KeyError(f"Unknown task_id: {task_id}")
            flag = self._cancel_flags.get(task_id)
            if flag is not None:
                flag.set()
            if rec.status in ("completed", "failed", "cancelled"):
                return rec
            rec.status = "cancelled"
            rec.error = reason
            rec.updated_at = time.time()
            snapshot = rec.to_dict()
        self._stream.publish("task.cancelled", snapshot)
        return rec

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            flag = self._cancel_flags.get(task_id)
            return bool(flag and flag.is_set())

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            rec = self._tasks.get(task_id)
            return rec.to_dict() if rec else None

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._tasks[tid].to_dict() for tid in reversed(self._order) if tid in self._tasks]
