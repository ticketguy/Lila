"""Milestone M5 (WS-ORCH) tests: durable checkpoints, restart recovery
(non-idempotent -> UNCERTAIN), canonical resource locks, and bounded parallelism.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from src.harness.orchestration import (
    BoundedTaskOrchestrator,
    CanonicalResourceLockManager,
    DurableCheckpointStore,
    TaskNode,
    TaskState,
)


class M5DurableOrchestrationTests(unittest.TestCase):
    def test_restart_recovery_marks_non_idempotent_running_task_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "run_checkpoint.json"
            store = DurableCheckpointStore(cp)
            nodes = {
                "t1": TaskNode("t1", "write_config", idempotent=False, state=TaskState.RUNNING),
                "t2": TaskNode("t2", "read_status", idempotent=True, state=TaskState.RUNNING),
                "t3": TaskNode("t3", "done_step", idempotent=True, state=TaskState.COMPLETED),
            }
            store.save("run-crash", nodes)

            run_id, recovered = store.recover_on_restart()
            self.assertEqual(run_id, "run-crash")
            self.assertEqual(recovered["t1"].state, TaskState.UNCERTAIN)
            self.assertIsNotNone(recovered["t1"].lila_failure_report)
            self.assertEqual(recovered["t2"].state, TaskState.PENDING)
            self.assertEqual(recovered["t3"].state, TaskState.COMPLETED)

    def test_canonical_resource_locks_prevent_concurrent_write_races(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "shared.txt"
            locks = CanonicalResourceLockManager()
            ok1, held1 = locks.try_acquire_all([str(target)])
            ok2, held2 = locks.try_acquire_all([f"{tmp}/./shared.txt"])
            self.assertTrue(ok1)
            self.assertFalse(ok2)
            self.assertEqual(held2, ())
            locks.release_all(held1)
            ok3, held3 = locks.try_acquire_all([f"{tmp}/./shared.txt"])
            self.assertTrue(ok3)
            locks.release_all(held3)

    def test_bounded_orchestrator_respects_worker_cap_and_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = DurableCheckpointStore(Path(tmp) / "orch.json")
            orch = BoundedTaskOrchestrator(store, max_workers=2)

            active_count = 0
            peak_active = 0
            guard = threading.Lock()
            order: list[str] = []

            def make_worker(tid: str):
                def _run(_node: TaskNode) -> str:
                    nonlocal active_count, peak_active
                    with guard:
                        active_count += 1
                        peak_active = max(peak_active, active_count)
                    time.sleep(0.05)
                    with guard:
                        active_count -= 1
                        order.append(tid)
                    return f"ok:{tid}"
                return _run

            nodes = [
                TaskNode("a", "step_a", resource_keys=("res-a",)),
                TaskNode("b", "step_b", resource_keys=("res-b",)),
                TaskNode("c", "step_c", resource_keys=("res-c",)),
                TaskNode("d", "step_d", dependencies=("a", "b", "c")),
            ]
            handlers = {n.task_id: make_worker(n.task_id) for n in nodes}
            summary = orch.run("run-1", nodes, handlers)

            self.assertTrue(summary.all_succeeded)
            self.assertLessEqual(peak_active, 2)
            self.assertEqual(order[-1], "d")

    def test_failed_upstream_cancels_downstream_and_emits_failure_reports(self) -> None:
        orch = BoundedTaskOrchestrator(max_workers=2)

        def fail_fn(_node: TaskNode) -> str:
            raise RuntimeError("disk full")

        nodes = [
            TaskNode("fetch", "web_fetch"),
            TaskNode("save", "file_write", dependencies=("fetch",)),
        ]
        summary = orch.run("run-fail", nodes, {"fetch": fail_fn, "save": lambda _: "ok"})
        self.assertFalse(summary.all_succeeded)
        self.assertEqual(summary.tasks["fetch"].state, TaskState.FAILED)
        self.assertEqual(summary.tasks["save"].state, TaskState.CANCELLED)
        self.assertGreaterEqual(len(summary.failure_reports), 1)

    def test_lilacore_multi_step_task_enters_orchestrator_checkpoints_and_recovers(self) -> None:
        import threading
        import time
        from src.core.lilacore import LilaCore
        from src.harness.tools import get_registry

        get_registry()._tools.clear()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "core_checkpoints.json"
            store = DurableCheckpointStore(ckpt_path)
            orch = BoundedTaskOrchestrator(checkpoint_store=store, max_workers=2)
            core = LilaCore(api_mode=True, orchestrator=orch)

            active = 0
            peak = 0
            order: list[str] = []
            lock = threading.Lock()

            def make_step(tid: str, deps: tuple[str, ...] = ()):
                def _run(_node: TaskNode) -> str:
                    nonlocal active, peak
                    with lock:
                        active += 1
                        peak = max(peak, active)
                    time.sleep(0.02)
                    with lock:
                        active -= 1
                        order.append(tid)
                    return f"done:{tid}"

                return {
                    "task_id": tid,
                    "name": f"step_{tid}",
                    "dependencies": deps,
                    "resource_keys": (f"res-{tid}",),
                    "handler": _run,
                }

            summary, results = core.execute_multi_step_task(
                [make_step("a"), make_step("b"), make_step("c"), make_step("d", ("a", "b", "c"))],
                run_id="core-dag-1",
            )
            self.assertTrue(summary.all_succeeded)
            self.assertLessEqual(peak, 2)
            self.assertEqual(order[-1], "d")
            self.assertEqual(len(results), 4)
            self.assertTrue(ckpt_path.exists())

            # Simulate crash mid-flight on a non-idempotent task in the same store
            run_id, nodes = store.load()
            self.assertEqual(run_id, "core-dag-1")
            nodes["d"].state = TaskState.RUNNING
            nodes["d"].idempotent = False
            store.save(run_id, nodes)

            _, recovered = DurableCheckpointStore(ckpt_path).recover_on_restart()
            self.assertEqual(recovered["d"].state, TaskState.UNCERTAIN)

    def test_lilacore_think_multi_tool_turn_executes_via_orchestrator_and_checkpoints(self):
        from src.core.lilacore import LilaCore
        from src.harness.tools import get_registry

        get_registry()._tools.clear()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "think_checkpoints.json"
            store = DurableCheckpointStore(ckpt_path)
            orch = BoundedTaskOrchestrator(checkpoint_store=store, max_workers=2)
            core = LilaCore(api_mode=True, orchestrator=orch)
            core.boot()
            core._llm = object()

            turns = iter([
                (
                    '<tool_call>{"name": "self_status", "arguments": {}}</tool_call>\n'
                    '<tool_call>{"name": "schedule_list", "arguments": {}}</tool_call>'
                ),
                "Here are your status and schedule details.",
            ])
            core._generate_gguf = lambda _messages: next(turns)

            response = core.think("Check my status and schedule")

            self.assertIn("Here are your status and schedule details.", response.text)
            self.assertEqual(len(response.actions), 2)
            self.assertEqual([a["tool"] for a in response.actions], ["self_status", "schedule_list"])
            self.assertTrue(all(a["success"] for a in response.actions))
            self.assertTrue(ckpt_path.exists())

            run_id, nodes = store.load()
            self.assertTrue(run_id.startswith("core-"))
            self.assertEqual(nodes["step-1"].state, TaskState.COMPLETED)
            self.assertEqual(nodes["step-2"].state, TaskState.COMPLETED)

    def test_lilacore_dependent_tasks_intermediate_failure_cancels_downstream_and_checkpoints(self):
        from src.core.lilacore import LilaCore
        from src.harness.tools import get_registry

        get_registry()._tools.clear()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "fail_checkpoints.json"
            store = DurableCheckpointStore(ckpt_path)
            orch = BoundedTaskOrchestrator(checkpoint_store=store, max_workers=2)
            core = LilaCore(api_mode=True, orchestrator=orch)
            core.boot()

            executed: list[str] = []

            def step1(_node: TaskNode) -> str:
                executed.append("s1")
                return "s1-ok"

            def step2(_node: TaskNode) -> str:
                executed.append("s2")
                raise RuntimeError("intermediate failure")

            def step3(_node: TaskNode) -> str:
                executed.append("s3")
                return "should-not-run"

            summary, _ = core.execute_multi_step_task(
                [
                    {"task_id": "s1", "name": "prepare", "dependencies": (), "handler": step1},
                    {"task_id": "s2", "name": "validate", "dependencies": ("s1",), "handler": step2},
                    {"task_id": "s3", "name": "finalize", "dependencies": ("s2",), "handler": step3},
                ],
                run_id="core-dep-fail",
            )

            self.assertFalse(summary.all_succeeded)
            self.assertEqual(executed, ["s1", "s2"])
            self.assertGreaterEqual(len(summary.failure_reports), 1)
            run_id, nodes = store.load()
            self.assertEqual(run_id, "core-dep-fail")
            self.assertEqual(nodes["s1"].state, TaskState.COMPLETED)
            self.assertEqual(nodes["s2"].state, TaskState.FAILED)
            self.assertEqual(nodes["s3"].state, TaskState.CANCELLED)


if __name__ == "__main__":
    unittest.main()
