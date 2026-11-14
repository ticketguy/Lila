import json
import unittest

from src.daemon.events import DaemonTaskCenter, VersionedEventStream, format_sse_event


class M6EventsAndTasksTests(unittest.TestCase):
    def test_event_stream_replay_dedup_and_gap_detection(self):
        stream = VersionedEventStream(max_events=3)
        e1 = stream.publish("task.created", {"task_id": "t1"})
        e2 = stream.publish("task.updated", {"task_id": "t1", "progress": 0.5})
        self.assertEqual((e1.event_id, e2.event_id), (1, 2))

        batch1 = stream.poll_since(since_cursor=0)
        self.assertFalse(batch1["gap_detected"])
        self.assertEqual([e["event_id"] for e in batch1["events"]], [1, 2])
        self.assertEqual(batch1["next_cursor"], 2)

        # Replay from cursor 2 returns no duplicates
        batch2 = stream.poll_since(since_cursor=2)
        self.assertEqual(batch2["events"], [])
        self.assertEqual(batch2["next_cursor"], 2)

        # Evict event 1 by pushing 3 more events
        stream.publish("task.updated", {"task_id": "t1", "progress": 0.7})
        stream.publish("task.completed", {"task_id": "t1"})
        stream.publish("notification.posted", {"title": "Done"})

        # Asking from cursor 1 now detects gap (oldest retained is 3)
        gap_batch = stream.poll_since(since_cursor=1)
        self.assertTrue(gap_batch["gap_detected"])
        self.assertEqual([e["event_id"] for e in gap_batch["events"]], [3, 4, 5])

    def test_sse_formatting(self):
        stream = VersionedEventStream()
        ev = stream.publish("voice.state", {"state": "recording"})
        sse = format_sse_event(ev).decode("utf-8")
        self.assertIn("id: 1\n", sse)
        self.assertIn("event: voice.state\n", sse)
        data_line = [ln for ln in sse.splitlines() if ln.startswith("data: ")][0]
        parsed = json.loads(data_line[6:])
        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(parsed["payload"]["state"], "recording")

    def test_task_center_lifecycle_and_cancellation(self):
        stream = VersionedEventStream()
        center = DaemonTaskCenter(stream)
        rec = center.create_task("research", "Inspect docs", task_id="task-abc")
        self.assertEqual(rec.status, "queued")
        center.update_task("task-abc", status="running", progress=0.4)
        cancelled = center.cancel_task("task-abc", reason="user_stopped")
        self.assertEqual(cancelled.status, "cancelled")
        self.assertTrue(center.is_cancelled("task-abc"))
        events = stream.poll_since(0)["events"]
        self.assertEqual([e["event_type"] for e in events], ["task.created", "task.updated", "task.cancelled"])


if __name__ == "__main__":
    unittest.main()
