import hashlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from src.harness.extended_tools import _screen_capture
from src.harness.isolation import (
    BoundedProcessBroker,
    BubblewrapSandboxBackend,
    CancellationToken,
    FailClosedSandboxBackend,
    LinuxNamespacesSandboxBackend,
    ProcessExecutionSpec,
    ResourceClass,
    ResourceLeaseError,
    ResourceLeaseManager,
    StagedPatchWorkspace,
    assess_host_isolation,
    build_scrubbed_environment,
    validate_fixed_argv,
)


class M3ExecutionIsolationTests(unittest.TestCase):
    def test_host_assessment_is_read_only_and_never_treats_venv_as_sandbox(self):
        assessment = assess_host_isolation()
        self.assertFalse(assessment.venv_is_security_boundary)
        self.assertIsNone(assessment.owner_approved_backend)
        self.assertIn(
            "Python virtualenv is dependency isolation only and is never treated as a security boundary.",
            assessment.notes,
        )

    def test_fixed_argv_rejects_shell_strings_traversal_and_secret_env(self):
        with self.assertRaises(ValueError):
            validate_fixed_argv("ls -la /tmp")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            validate_fixed_argv([])
        with self.assertRaises(ValueError):
            validate_fixed_argv(["../bin/sh", "-c", "echo hi"])

        with tempfile.TemporaryDirectory() as tmp:
            env = build_scrubbed_environment(
                tmp,
                explicit_env={"SAFE_FLAG": "1"},
                host_env={
                    "PATH": "/usr/bin:/bin",
                    "LILA_API_TOKEN": "top-secret-token",
                    "AWS_SECRET_ACCESS_KEY": " leaked ",
                    "HOME": "/home/realuser",
                },
            )
            self.assertNotIn("LILA_API_TOKEN", env)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
            self.assertEqual(env["HOME"], tmp)
            self.assertEqual(env["TMPDIR"], tmp)
            self.assertEqual(env["SAFE_FLAG"], "1")

            with self.assertRaises(ValueError):
                build_scrubbed_environment(tmp, explicit_env={"LILA_SECRET_KEY": "nope"})

    def test_sandbox_required_fails_closed_without_approved_strong_backend(self):
        broker = BoundedProcessBroker(sandbox_backend=FailClosedSandboxBackend())
        with tempfile.TemporaryDirectory() as tmp:
            spec = ProcessExecutionSpec(
                argv=(sys.executable, "-c", "print('should never run')"),
                cwd=tmp,
                execution_backend="sandbox-required",
            )
            outcome = broker.execute(spec)
            self.assertEqual(outcome.state, "failed")
            self.assertEqual(outcome.reason_code, "sandbox.unavailable_or_unapproved")
            self.assertIsNone(outcome.exit_code)
            self.assertEqual(outcome.stdout, "")

    def test_sandbox_network_egress_is_denied_by_default(self):
        broker = BoundedProcessBroker(sandbox_backend=BubblewrapSandboxBackend("/bin/true"))
        with tempfile.TemporaryDirectory() as tmp:
            spec = ProcessExecutionSpec(
                argv=(sys.executable, "-c", "print('net')"),
                cwd=tmp,
                execution_backend="sandbox-required",
                allow_network_egress=True,
            )
            outcome = broker.execute(spec)
            self.assertEqual(outcome.state, "failed")
            self.assertEqual(outcome.reason_code, "sandbox.network_egress_denied")

    def test_timeout_and_cancellation_terminate_descendant_processes_and_mark_ambiguous(self):
        broker = BoundedProcessBroker()
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            script = (
                "import subprocess, sys, time, pathlib\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
                "sys.stdout.flush()\n"
                "time.sleep(60)\n"
            )
            spec = ProcessExecutionSpec(
                argv=(sys.executable, "-c", script),
                cwd=tmp,
                timeout_seconds=0.35,
                grace_period_seconds=0.1,
                idempotency="non-idempotent",
            )
            outcome = broker.execute(spec)
            self.assertEqual(outcome.state, "unknown_outcome")
            self.assertEqual(outcome.reason_code, "process.timed_out")
            self.assertTrue(outcome.ambiguous_side_effects)

            if pid_file.exists():
                child_pid = int(pid_file.read_text(encoding="utf-8").strip())
                time.sleep(0.05)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)

            # Now test explicit CancellationToken on an idempotent job
            token = CancellationToken()
            threading.Timer(0.15, lambda: token.cancel("owner_cancelled")).start()
            idem_spec = ProcessExecutionSpec(
                argv=(sys.executable, "-c", "import time; time.sleep(60)"),
                cwd=tmp,
                timeout_seconds=10.0,
                grace_period_seconds=0.1,
                idempotency="idempotent",
            )
            cancelled_outcome = broker.execute(idem_spec, cancellation_token=token)
            self.assertEqual(cancelled_outcome.state, "cancelled")
            self.assertEqual(cancelled_outcome.reason_code, "owner_cancelled")
            self.assertFalse(cancelled_outcome.ambiguous_side_effects)

    def test_output_caps_and_secret_redaction_in_broker(self):
        broker = BoundedProcessBroker()
        with tempfile.TemporaryDirectory() as tmp:
            script = (
                "import sys\n"
                "sys.stdout.write('Authorization: Bearer super_secret_token_value_12345\\n')\n"
                "sys.stdout.write('A' * 2048)\n"
            )
            spec = ProcessExecutionSpec(
                argv=(sys.executable, "-c", script),
                cwd=tmp,
                max_output_bytes=512,
                idempotency="idempotent",
                redaction_secrets=("custom_inline_secret",),
            )
            outcome = broker.execute(spec)
            self.assertEqual(outcome.state, "succeeded")
            self.assertTrue(outcome.stdout_truncated)
            self.assertTrue(outcome.credentials_redacted)
            self.assertNotIn("super_secret_token_value_12345", outcome.stdout)
            self.assertIn("[REDACTED]", outcome.stdout)

    def test_resource_lease_manager_enforces_ceilings_and_releases(self):
        leases = ResourceLeaseManager({ResourceClass.CPU_HEAVY: 1, ResourceClass.IO_BOUND: 2})
        with leases.acquire(ResourceClass.CPU_HEAVY):
            self.assertEqual(leases.active_count(ResourceClass.CPU_HEAVY), 1)
            with self.assertRaises(ResourceLeaseError):
                leases.acquire(ResourceClass.CPU_HEAVY)
        self.assertEqual(leases.active_count(ResourceClass.CPU_HEAVY), 0)

    def test_staged_patch_workspace_keeps_primary_untouched_until_owner_promotion(self):
        with tempfile.TemporaryDirectory() as primary:
            primary_path = Path(primary)
            target_file = primary_path / "module.py"
            target_file.write_text("VALUE = 1\n", encoding="utf-8")

            with StagedPatchWorkspace(primary_path, redaction_secrets=("my-secret-phrase-999",)) as stage:
                staged_file = stage.staging_root / "module.py"
                staged_file.write_text("VALUE = 2\n# token: my-secret-phrase-999\n", encoding="utf-8")

                # Primary checkout is completely untouched during staging
                self.assertTrue(stage.verify_primary_unchanged())
                self.assertEqual(target_file.read_text(encoding="utf-8"), "VALUE = 1\n")

                # Artifacts are scanned and redacted before promotion
                artifacts = stage.collect_sanitized_artifacts()
                self.assertEqual(len(artifacts), 1)
                self.assertTrue(artifacts[0].credentials_redacted)

                diff_text = stage.collect_unified_diff()
                self.assertIn("+VALUE = 2", diff_text)
                self.assertNotIn("my-secret-phrase-999", diff_text)

                diff_sha = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
                with self.assertRaises(PermissionError):
                    stage.promote_reviewed_patch(owner_approved=False, expected_diff_sha256=diff_sha)

                promoted = stage.promote_reviewed_patch(owner_approved=True, expected_diff_sha256=diff_sha)
                self.assertEqual(promoted, ("module.py",))
                self.assertIn("VALUE = 2", target_file.read_text(encoding="utf-8"))
                self.assertNotIn("my-secret-phrase-999", target_file.read_text(encoding="utf-8"))

    def test_legacy_screen_capture_tool_is_disabled_without_shell_execution(self):
        res = _screen_capture("/tmp/should_not_exist_lila_test.png")
        self.assertFalse(res.success)
        self.assertIn("Lila Island Screen Highlight", res.error or "")

    def test_linux_namespaces_sandbox_executes_and_blocks_network_egress(self):
        backend = LinuxNamespacesSandboxBackend()
        if not backend.is_available():
            self.skipTest("unshare user/net namespaces not available on this host")
        broker = BoundedProcessBroker(sandbox_backend=backend)
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as host_secret_dir:
            host_secret = Path(host_secret_dir) / "host_token.txt"
            host_secret.write_text("TOP_SECRET_HOST_CREDENTIAL", encoding="utf-8")
            host_outside_target = Path(host_secret_dir) / "should_not_exist.txt"

            script = (
                "import os, resource, socket\n"
                "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "s.settimeout(0.5)\n"
                "try:\n"
                "    s.connect(('1.1.1.1', 80))\n"
                "    print('NETWORK_ALLOWED')\n"
                "except OSError as exc:\n"
                "    print(f'NETWORK_BLOCKED:{exc.errno}')\n"
                f"print('HOST_SECRET_EXISTS:', os.path.exists({str(host_secret)!r}))\n"
                "try:\n"
                f"    with open({str(host_outside_target)!r}, 'w') as f:\n"
                "        f.write('escaped')\n"
                "    print('HOST_WRITE_IN_TMPFS')\n"
                "except OSError:\n"
                "    print('HOST_WRITE_DENIED')\n"
                "try:\n"
                "    with open('/etc/lila_sandbox_write_test', 'w') as f:\n"
                "        f.write('escaped')\n"
                "    print('ETC_WRITE_ALLOWED')\n"
                "except OSError:\n"
                "    print('ETC_WRITE_DENIED')\n"
                "pids = sorted(int(p) for p in os.listdir('/proc') if p.isdigit())\n"
                "print('SANDBOX_PIDS:', pids)\n"
                "print('NOFILE_LIMIT:', resource.getrlimit(resource.RLIMIT_NOFILE)[0])\n"
                "print('FSIZE_LIMIT:', resource.getrlimit(resource.RLIMIT_FSIZE)[0])\n"
            )
            outcome = broker.execute(
                ProcessExecutionSpec(
                    argv=[sys.executable, "-c", script],
                    execution_backend="sandbox-required",
                    cwd=ws,
                    timeout_seconds=10.0,
                )
            )
            self.assertEqual(outcome.exit_code, 0, msg=outcome.stderr)
            self.assertEqual(outcome.state, "succeeded")
            self.assertIn("NETWORK_BLOCKED:", outcome.stdout)
            self.assertNotIn("NETWORK_ALLOWED", outcome.stdout)
            self.assertIn("HOST_SECRET_EXISTS: False", outcome.stdout)
            self.assertIn("ETC_WRITE_DENIED", outcome.stdout)
            self.assertFalse(host_outside_target.exists())
            self.assertIn("SANDBOX_PIDS: [1]", outcome.stdout)
            self.assertIn("NOFILE_LIMIT: 256", outcome.stdout)
            self.assertIn("FSIZE_LIMIT: 16777216", outcome.stdout)

    def test_linux_namespaces_sandbox_blocks_symlinks_alt_paths_inherited_fds_and_resource_abuse(self):
        backend = LinuxNamespacesSandboxBackend()
        if not backend.is_available():
            self.skipTest("unshare user/net namespaces not available on this host")
        broker = BoundedProcessBroker(sandbox_backend=backend)
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as secret_dir:
            ws_path = Path(ws)
            outside_secret = Path(secret_dir) / "outside_cred.txt"
            outside_secret.write_text("SECRET_CREDENTIAL_XYZ", encoding="utf-8")

            # Symlink inside workspace pointing outside workspace
            symlink_escape = ws_path / "escape_link.txt"
            os.symlink(outside_secret, symlink_escape)

            # Open an inheritable FD in parent at a fixed high slot pointing to outside_secret
            raw_fd = os.open(str(outside_secret), os.O_RDONLY)
            leaked_fd = 42
            os.dup2(raw_fd, leaked_fd)
            os.close(raw_fd)
            os.set_inheritable(leaked_fd, True)
            try:
                script = (
                    "import os, resource\n"
                    "print('SYMLINK_READABLE:', os.path.exists('escape_link.txt'))\n"
                    "print('VAR_TMP_MISSING:', not os.path.exists('/var/tmp'))\n"
                    "print('RUN_MISSING:', not os.path.exists('/run'))\n"
                    "print('ETC_MISSING:', not os.path.exists('/etc'))\n"
                    "open_fds = sorted(int(f) for f in os.listdir('/proc/self/fd') if f.isdigit() and int(f) > 2)\n"
                    f"print('INHERITED_FD_ABSENT:', {leaked_fd} not in open_fds)\n"
                    "print('NPROC_LIMITS:', resource.getrlimit(resource.RLIMIT_NPROC))\n"
                    "print('CPU_LIMITS:', resource.getrlimit(resource.RLIMIT_CPU))\n"
                    "print('AS_LIMITS:', resource.getrlimit(resource.RLIMIT_AS))\n"
                    "try:\n"
                    "    _ = bytearray(2 * 1024 * 1024 * 1024)\n"
                    "    print('MEM_ALLOC_ALLOWED')\n"
                    "except MemoryError:\n"
                    "    print('MEM_ALLOC_DENIED')\n"
                    "# Workspace itself remains writable\n"
                    "with open('ws_ok.txt', 'w') as wf:\n"
                    "    wf.write('ok')\n"
                    "print('WS_WRITABLE: True')\n"
                )
                outcome = broker.execute(
                    ProcessExecutionSpec(
                        argv=[sys.executable, "-c", script],
                        execution_backend="sandbox-required",
                        cwd=ws,
                        timeout_seconds=10.0,
                    )
                )
            finally:
                os.close(leaked_fd)

            self.assertEqual(outcome.exit_code, 0, msg=outcome.stderr)
            self.assertIn("SYMLINK_READABLE: False", outcome.stdout)
            self.assertIn("VAR_TMP_MISSING: True", outcome.stdout)
            self.assertIn("RUN_MISSING: True", outcome.stdout)
            self.assertIn("ETC_MISSING: True", outcome.stdout)
            self.assertIn("INHERITED_FD_ABSENT: True", outcome.stdout)
            self.assertIn("NPROC_LIMITS: (32, 32)", outcome.stdout)
            self.assertIn("CPU_LIMITS: (15, 15)", outcome.stdout)
            self.assertIn("AS_LIMITS: (1610612736, 1610612736)", outcome.stdout)
            self.assertIn("MEM_ALLOC_DENIED", outcome.stdout)
            self.assertIn("WS_WRITABLE: True", outcome.stdout)
            self.assertTrue((ws_path / "ws_ok.txt").exists())

    def test_linux_namespaces_pivot_root_and_hard_rlimits_resist_breakout_attempts(self):
        backend = LinuxNamespacesSandboxBackend()
        if not backend.is_available():
            self.skipTest("unshare user/net namespaces not available on this host")
        broker = BoundedProcessBroker(sandbox_backend=backend)
        with tempfile.TemporaryDirectory() as ws:
            script = (
                "import os, resource\n"
                "root_entries = set(os.listdir('/'))\n"
                "print('HOST_DIRS_ABSENT:', root_entries.isdisjoint({'home', 'root', 'etc', 'var', 'run', 'mnt', 'opt', 'srv'}))\n"
                "raised = []\n"
                "for lim in (resource.RLIMIT_AS, resource.RLIMIT_CPU, resource.RLIMIT_NOFILE, resource.RLIMIT_FSIZE, resource.RLIMIT_NPROC):\n"
                "    soft, hard = resource.getrlimit(lim)\n"
                "    try:\n"
                "        resource.setrlimit(lim, (hard + 1, hard + 1))\n"
                "        raised.append(lim)\n"
                "    except (ValueError, OSError):\n"
                "        pass\n"
                "print('HARD_CEILINGS_LOCKED:', len(raised) == 0)\n"
                "try:\n"
                "    with open('/usr/breakout_write_test', 'w') as f:\n"
                "        f.write('x')\n"
                "    print('USR_RO: False')\n"
                "except OSError:\n"
                "    print('USR_RO: True')\n"
                "pids = []\n"
                "fork_blocked = False\n"
                "for _ in range(40):\n"
                "    try:\n"
                "        pid = os.fork()\n"
                "        if pid == 0:\n"
                "            os._exit(0)\n"
                "        pids.append(pid)\n"
                "    except OSError:\n"
                "        fork_blocked = True\n"
                "        break\n"
                "for pid in pids:\n"
                "    os.waitpid(pid, 0)\n"
                "print('FORK_BOMB_BOUNDED:', fork_blocked or len(pids) <= 32)\n"
            )
            outcome = broker.execute(
                ProcessExecutionSpec(
                    argv=[sys.executable, "-c", script],
                    execution_backend="sandbox-required",
                    cwd=ws,
                    timeout_seconds=10.0,
                )
            )
            self.assertEqual(outcome.exit_code, 0, msg=outcome.stderr)
            self.assertIn("HOST_DIRS_ABSENT: True", outcome.stdout)
            self.assertIn("HARD_CEILINGS_LOCKED: True", outcome.stdout)
            self.assertIn("USR_RO: True", outcome.stdout)
            self.assertIn("FORK_BOMB_BOUNDED: True", outcome.stdout)


if __name__ == "__main__":
    unittest.main()
