import base64
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.daemon.service import LilaDaemon, _make_api_handler
from src.harness.orchestration import DurableCheckpointStore, TaskNode, TaskState


class DaemonM6M8HttpTests(unittest.TestCase):
    def setUp(self):
        self.token = "t" * 40
        self.daemon = LilaDaemon(voice=False, port=0, api_token=self.token)
        handler = _make_api_handler(self.daemon, self.token)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _get(self, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
        req = Request(f"http://127.0.0.1:{self.port}{path}")
        req.add_header("Authorization", f"Bearer {self.token}")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8")
        req = Request(f"http://127.0.0.1:{self.port}{path}", data=data, method="POST")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Length", str(len(data)))
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_events_replay_sse_tasks_cancel_and_sentinel_triage(self):
        rec = self.daemon.task_center.create_task("analysis", "Check logs", task_id="task-http-1")
        self.assertEqual(rec.status, "queued")

        status, raw = self._get("/events?since=0")
        self.assertEqual(status, 200)
        batch = json.loads(raw.decode("utf-8"))
        self.assertEqual(batch["schema_version"], "1.0")
        created_events = [e for e in batch["events"] if e["event_type"] == "task.created" and e["payload"].get("task_id") == "task-http-1"]
        self.assertEqual(len(created_events), 1)
        created_cursor = created_events[0]["event_id"]

        # SSE stream from since=0 returns SSE frame
        status_sse, sse_raw = self._get("/events/stream?since=0&once=1")
        self.assertEqual(status_sse, 200)
        sse_text = sse_raw.decode("utf-8")
        self.assertIn("event: task.created\n", sse_text)

        # List and cancel task via HTTP
        status_tasks, tasks_raw = self._get("/tasks")
        self.assertEqual(status_tasks, 200)
        self.assertEqual(json.loads(tasks_raw.decode("utf-8"))["tasks"][0]["task_id"], "task-http-1")

        status_cancel, cancel_resp = self._post("/tasks/task-http-1/cancel", {"reason": "owner_stopped"})
        self.assertEqual(status_cancel, 200)
        self.assertEqual(cancel_resp["status"], "cancelled")

        # Replay from created_cursor gets only task.cancelled
        _, batch2_raw = self._get(f"/events?since={created_cursor}")
        batch2 = json.loads(batch2_raw.decode("utf-8"))
        self.assertEqual([e["event_type"] for e in batch2["events"]], ["task.cancelled"])

        # M8 read-only Sentinel triage endpoint (GET & POST with Windows host facts)
        status_sent, sent_raw = self._get("/sentinel/triage?sources=system_resources,defender")
        self.assertEqual(status_sent, 200)
        triage = json.loads(sent_raw.decode("utf-8"))
        self.assertFalse(triage["autonomous_remediation_attempted"])
        self.assertFalse(triage["screen_capture_used"])
        self.assertIn("system_resources", triage["coverage"]["available_sources"])
        self.assertEqual(triage["coverage"]["gaps"][0]["source"], "defender")

        status_post_sent, post_triage = self._post("/sentinel/triage", {
            "sources": ["system_resources", "defender", "firewall"],
            "host_facts": [
                {"source": "defender", "metric": "defender_realtime_enabled", "value": False, "read_only": True},
                {"source": "firewall", "metric": "firewall_enabled", "value": True, "read_only": True},
            ],
        })
        self.assertEqual(status_post_sent, 200)
        self.assertEqual(post_triage["highest_tier"], "critical")
        self.assertIn("defender", post_triage["coverage"]["available_sources"])

        # M7 gesture token + explicit capture + discard
        _, gest = self._post("/perception/gesture", {"scope": "region"})
        _, cap = self._post("/perception/capture", {
            "initiator": "owner_gesture",
            "owner_gesture_token": gest["owner_gesture_token"],
            "scope": "region",
            "target_label": "Editor",
            "image_base64": "AQID",
            "width": 100,
            "height": 80,
        })
        self.assertFalse(cap["persisted"])
        _, disc = self._post("/perception/discard", {"capture_id": cap["capture_id"]})
        self.assertTrue(disc["discarded"])

        # M4 traces & M10 candidates endpoints
        st_tr, _ = self._get("/traces?limit=10")
        self.assertEqual(st_tr, 200)
        st_ca, _ = self._get("/capabilities/candidates")
        self.assertEqual(st_ca, 200)

    def test_m4_trace_write_and_http_get_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HOME": tmp}):
                trace_path = Path(tmp) / ".local" / "share" / "lila" / "harness_traces.jsonl"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(
                    json.dumps({
                        "invocation_id": "inv-m4-e2e",
                        "correlation_id": "corr-m4-e2e",
                        "capability_id": "file_write",
                        "status": "succeeded",
                        "exit_code": 0,
                        "verification_status": "passed",
                        "verification_summary": "Postcondition hash verified",
                    }) + "\n",
                    encoding="utf-8",
                )
                status, raw = self._get("/traces?limit=5")
                self.assertEqual(status, 200)
                traces = json.loads(raw.decode("utf-8"))["traces"]
                self.assertTrue(any(
                    t.get("invocation_id") == "inv-m4-e2e"
                    and t.get("verification_status") == "passed"
                    and t.get("exit_code") == 0
                    for t in traces
                ))

    def test_m5_checkpoint_restart_marks_non_idempotent_uncertain_and_emits_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"HOME": tmp}):
                ckpt_path = Path(tmp) / ".local" / "share" / "lila" / "harness_checkpoints.json"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                store = DurableCheckpointStore(ckpt_path)
                store.save(
                    "run-restart-1",
                    {
                        "task-restart-1": TaskNode(
                            task_id="task-restart-1",
                            name="Send external webhook",
                            state=TaskState.RUNNING,
                            idempotent=False,
                        )
                    },
                )
                daemon2 = LilaDaemon(voice=False, port=0, api_token=self.token)
                self.assertTrue(any(
                    r.get("task_id") == "task-restart-1" and r.get("state") == "uncertain"
                    for r in daemon2.recovered_uncertain_tasks
                ))
                events = daemon2.event_stream.poll_since(since_cursor=0)["events"]
                self.assertTrue(any(
                    e.get("event_type") == "task.uncertain"
                    and (e.get("payload") or {}).get("task_id") == "task-restart-1"
                    for e in events
                ))

    def test_m7_perception_capture_observe_ocr_discard_and_stale_403(self):
        png_1x1 = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        png_b64 = base64.b64encode(png_1x1).decode("ascii")
        _, gest = self._post("/perception/gesture", {"scope": "region"})
        st_cap, cap = self._post("/perception/capture", {
            "initiator": "owner_gesture",
            "owner_gesture_token": gest["owner_gesture_token"],
            "scope": "region",
            "target_label": "Synthetic PNG",
            "image_base64": png_b64,
            "width": 1,
            "height": 1,
        })
        self.assertEqual(st_cap, 200)
        st_obs, obs = self._post("/perception/observe", {"capture_id": cap["capture_id"]})
        self.assertEqual(st_obs, 200)
        self.assertEqual(obs["capture"]["capture_id"], cap["capture_id"])
        st_ocr, ocr = self._post("/perception/ocr", {"capture_id": cap["capture_id"], "uia_text": "Hello Lila"})
        self.assertEqual(st_ocr, 200)
        self.assertIn("Hello Lila", json.dumps(ocr))
        st_disc, disc = self._post("/perception/discard", {"capture_id": cap["capture_id"]})
        self.assertEqual(st_disc, 200)
        self.assertTrue(disc["discarded"])
        with self.assertRaises(HTTPError) as ctx:
            self._post("/perception/observe", {"capture_id": cap["capture_id"]})
        self.assertIn(ctx.exception.code, (403, 404, 410))

    def test_m8_sentinel_rejects_non_readonly_mutation_fact(self):
        with self.assertRaises(HTTPError) as ctx:
            self._post("/sentinel/triage", {
                "sources": ["defender"],
                "host_facts": [
                    {"source": "defender", "metric": "remediate", "value": "enable", "read_only": False},
                ],
            })
        self.assertEqual(ctx.exception.code, 400)

    def test_m10_http_promote_persists_catalog_and_rollback(self):
        from src.harness.capability_learning import CapabilityPromotionService

        with tempfile.TemporaryDirectory() as tmp:
            svc = CapabilityPromotionService(initial_catalog={"file_read": "sha-v1"}, storage_dir=tmp)
            cand = svc.submit_candidate(
                capability_id="cap.http.e2e",
                version="0.1.0",
                source_code="def run_cap(x: str) -> str:\n    return x.upper()\n",
                test_code="assert run_cap('ok') == 'OK'\n",
                provenance={"origin": "owner_authored"},
            )
            evaluated = svc.evaluate_candidate(cand.capability_id)
            self.assertTrue(evaluated.tests_passed)
            orig_svc = self.daemon.capability_learning
            self.daemon.capability_learning = svc
            try:
                st_get, raw_get = self._get("/capabilities/candidates")
                self.assertEqual(st_get, 200)
                listed = json.loads(raw_get.decode("utf-8"))
                self.assertEqual(listed["catalog_version"]["version"], 1)
                self.assertEqual(listed["candidates"][0]["capability_id"], cand.capability_id)

                st_prom, prom = self._post(
                    f"/capabilities/candidates/{cand.capability_id}/promote",
                    {"owner_approved": True},
                )
                self.assertEqual(st_prom, 200)
                prom_ver = prom["catalog"]["version"] if "catalog" in prom else prom["version"]
                self.assertGreaterEqual(prom_ver, 2)

                # Reload from disk to prove persistence
                reloaded = CapabilityPromotionService(storage_dir=tmp)
                self.assertEqual(reloaded.current_catalog.version, prom_ver)
                self.assertIn(cand.capability_id, reloaded.current_catalog.active_capabilities)

                st_rb, rb = self._post("/capabilities/rollback", {"target_version": 1})
                self.assertEqual(st_rb, 200)
                rb_ver = rb["catalog"]["version"] if "catalog" in rb else rb["version"]
                self.assertGreater(rb_ver, prom_ver)

                reloaded_after = CapabilityPromotionService(storage_dir=tmp)
                self.assertEqual(reloaded_after.current_catalog.version, rb_ver)
                self.assertNotIn(cand.capability_id, reloaded_after.current_catalog.active_capabilities)
            finally:
                self.daemon.capability_learning = orig_svc


if __name__ == "__main__":
    unittest.main()
