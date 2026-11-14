"""Explicitly authorized Windows/WSL test fixture; never a deployment command.

Run with the existing isolated crypto environment. It imports only test-owned
short-lived client certificates, uses production TLS/approval paths, removes
exact imported thumbprints, and deletes its temporary credentials after testing.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    import ipaddress
    from src.daemon.service import LilaDaemon
    from src.daemon.oia_harness import build_approval_service
    from src.harness.executor import HarnessExecutor
    from src.harness.system_tools import _file_read
    from src.harness.tools import Tool, ToolArg, get_registry, register_tool
    from src.oia import OwnerAuthenticationService, OwnerProfileStore

    fixture = Path(tempfile.mkdtemp(prefix="lila-m2-live-resume-"))
    fixture.chmod(0o700)
    private_values = []
    output_capture = io.StringIO()
    error_capture = io.StringIO()
    cleanup_confirmed = False
    daemon = None
    worker = None
    stop_worker = threading.Event()
    dispatched = []
    restart_statuses = []
    fixture_failures = []
    token = secrets.token_urlsafe(40)
    phrase = " ".join(secrets.token_hex(4) for _ in range(6))
    sentinel = secrets.token_urlsafe(40)
    password = secrets.token_urlsafe(32).encode()
    private_values.extend([token, phrase, sentinel, password.decode()])

    def write_private(name, data):
        path = fixture / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        return path

    def issue(name, issuer=None, usage=None, expired=False):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Lila M2 Resume " + name)])
        now = datetime.now(timezone.utc)
        builder = (x509.CertificateBuilder().subject_name(subject)
                   .issuer_name(issuer[1].subject if issuer else subject)
                   .public_key(key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before(now - timedelta(days=2) if expired else now - timedelta(minutes=5))
                   .not_valid_after(now - timedelta(days=1) if expired else now + timedelta(hours=2))
                   .add_extension(x509.BasicConstraints(ca=issuer is None, path_length=None), critical=True))
        if usage is not None:
            builder = builder.add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
            builder = builder.add_extension(x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=True,
                data_encipherment=False, key_agreement=False, key_cert_sign=False,
                crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
        else:
            builder = builder.add_extension(x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True,
                crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
        if usage == ExtendedKeyUsageOID.SERVER_AUTH:
            builder = builder.add_extension(x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        cert = builder.sign(issuer[0] if issuer else key, hashes.SHA256())
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
        private_values.append(pem.decode())
        write_private(name + ".key", pem)
        write_private(name + ".crt", cert.public_bytes(serialization.Encoding.PEM))
        if usage == ExtendedKeyUsageOID.CLIENT_AUTH:
            write_private(name + ".pfx", pkcs12.serialize_key_and_certificates(
                name.encode(), key, cert, [issuer[1]], serialization.BestAvailableEncryption(password)))
        return key, cert

    registry = get_registry()
    previous = dict(registry._tools)
    windows_output = ""
    result = None
    try:
        ca = issue("ca")
        outsider_ca = issue("outsider-ca")
        server = issue("server", ca, ExtendedKeyUsageOID.SERVER_AUTH)
        clients = {name: issue(name, issuer, ExtendedKeyUsageOID.CLIENT_AUTH, expired)
                   for name, issuer, expired in [
                       ("allowed", ca, False), ("unlisted", ca, False),
                       ("untrusted", outsider_ca, False), ("expired", ca, True)]}
        write_private("pfx.password", password)
        env = {
            "LILA_TLS_SERVER_CERT": str(fixture / "server.crt"),
            "LILA_TLS_SERVER_KEY": str(fixture / "server.key"),
            "LILA_TLS_CLIENT_CA": str(fixture / "ca.crt"),
            "LILA_TLS_ALLOWED_CLIENT_SHA256": clients["allowed"][1].fingerprint(hashes.SHA256()).hex(),
        }
        os.environ.update(env)
        registry._tools.clear()

        def read(path, *, _verified_file=None):
            result = _file_read(path, _verified_file=_verified_file)
            if result.success:
                dispatched.append(path)
            return result

        register_tool(Tool("file_read", "Read test file", [ToolArg("path", "path", "Target")], read))
        target = write_private("reviewed-target.txt", b"M2 controlled fixture contents")
        owner_store = OwnerProfileStore(fixture / "owner.json")
        owner_store.create(phrase, phrase)
        private_values.append((fixture / "owner.json").read_text())

        def boot(port):
            owner = OwnerAuthenticationService(owner_store)
            service = build_approval_service(owner, registry)
            active = LilaDaemon(port=port, api_token=token, voice=False,
                                approval_continuation_service=service)
            active.owner_authentication_service = owner
            active.lila = SimpleNamespace(approval_continuation_service=service,
                                         owner_authentication_service=owner, gguf_path="test")
            active._service_state = "running"
            active._start_api()
            return active

        def pending():
            executor = HarnessExecutor(log_dir=str(fixture / "harness-logs"),
                                       workspace_root=str(fixture / "separate-workspace"),
                                       approval_continuation_service=daemon.approval_continuation_service)
            value = executor.process("<tool_call>" + json.dumps({
                "name": "file_read", "arguments": {"path": str(target)}}) + "</tool_call>")[0]
            if value["approval"].status.value != "pending":
                raise RuntimeError("The controlled operation did not suspend for approval.")

        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(error_capture):
            daemon = boot(0)
            port = daemon.port
            pending()
            windows_fixture = subprocess.check_output(["wslpath", "-w", str(fixture)], text=True).strip()
            settings = {
                "LILA_RUN_LIVE_MTLS_TEST": "1", "LILA_LIVE_MTLS_API_TOKEN": token,
                "LILA_LIVE_MTLS_OWNER_PHRASE": phrase, "LILA_LIVE_MTLS_SECRET_SENTINEL": sentinel,
                "LILA_LIVE_MTLS_PORT": str(port), "LILA_LIVE_MTLS_FIXTURE_DIRECTORY": windows_fixture,
                "LILA_LIVE_MTLS_SERVER_PIN": server[1].fingerprint(hashes.SHA256()).hex(),
                "LILA_LIVE_MTLS_WRONG_SERVER_PIN": "f" * 64,
            }
            for name, (_, cert) in clients.items():
                settings["LILA_LIVE_MTLS_" + name.upper() + "_CLIENT_PIN"] = cert.fingerprint(hashes.SHA256()).hex()
            write_private("settings.json", json.dumps(settings).encode())
            script = r'''param([string]$FixtureDirectory)
$ErrorActionPreference = 'Stop'
$imported = @()
$exitCode = 1
try {
    Set-Content -LiteralPath (Join-Path $FixtureDirectory 'windows-script-started') -Value 'ok'
    $settings = Get-Content -LiteralPath (Join-Path $FixtureDirectory 'settings.json') -Raw | ConvertFrom-Json
    foreach ($item in $settings.PSObject.Properties) { [Environment]::SetEnvironmentVariable($item.Name, [string]$item.Value, 'Process') }
    $password = ConvertTo-SecureString (Get-Content -LiteralPath (Join-Path $FixtureDirectory 'pfx.password') -Raw) -AsPlainText -Force
    foreach ($name in @('allowed', 'unlisted', 'untrusted', 'expired')) {
        $cert = Import-PfxCertificate -FilePath (Join-Path $FixtureDirectory ($name + '.pfx')) -CertStoreLocation 'Cert:\CurrentUser\My' -Password $password
        $imported += $cert.Thumbprint
        if (-not $cert.HasPrivateKey) { throw 'Test client private key unavailable.' }
    }
    Set-Location 'REPOSITORY_PATH'
    dotnet build .\windows\Lila.Desktop\Lila.Desktop.csproj -c Release --no-restore
    if ($LASTEXITCODE -ne 0) { throw 'Desktop build failed.' }
    dotnet test .\windows\Lila.Desktop.Tests\Lila.Desktop.Tests.csproj -c Release --no-restore --logger 'console;verbosity=normal'
    $exitCode = $LASTEXITCODE
} finally {
    foreach ($thumbprint in $imported) {
        $path = 'Cert:\CurrentUser\My\' + $thumbprint
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -DeleteKey }
    }
    $remaining = @(Get-ChildItem Cert:\CurrentUser\My | Where-Object { $imported -contains $_.Thumbprint })
    if ($remaining.Count -ne 0) { throw 'Temporary client certificate cleanup failed.' }
    Set-Content -LiteralPath (Join-Path $FixtureDirectory 'windows-cleanup-confirmed') -Value 'ok'
    Write-Output 'TemporaryTestCertificatesRemaining=0'
}
exit $exitCode
'''
            windows_root = subprocess.check_output(["wslpath", "-w", str(ROOT)], text=True).strip()
            write_private("run.ps1", script.replace("REPOSITORY_PATH", windows_root).encode())

            def commands():
                nonlocal daemon
                while not stop_worker.wait(0.05):
                    for command in fixture.glob("*.command"):
                        action = command.read_text()
                        answer = "ok"
                        try:
                            if action == "create_pending":
                                pending()
                            elif action == "stop_daemon":
                                service = daemon.approval_continuation_service
                                requests = service.list_pending_requests()
                                daemon._cleanup()
                                restart_statuses.extend(service.get_result(
                                    item.invocation.invocation_id).status.value for item in requests)
                            elif action == "restart_daemon":
                                daemon = boot(port)
                            elif action == "assert_evidence":
                                if len(dispatched) != 1 or restart_statuses != ["cancelled"]:
                                    raise RuntimeError("Dispatch or shutdown evidence did not match.")
                            else:
                                raise RuntimeError("Unknown fixture command.")
                        except Exception as exc:
                            fixture_failures.append(type(exc).__name__)
                            answer = "failed"
                        response = command.with_suffix(".result")
                        staging = response.with_suffix(".staging")
                        staging.write_text(answer)
                        staging.replace(response)
                        command.unlink()

            worker = threading.Thread(target=commands, daemon=True)
            worker.start()
            process = subprocess.run([
                "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", windows_fixture + "\\run.ps1",
                "-FixtureDirectory", windows_fixture], capture_output=True, text=True, timeout=240)
            windows_output = process.stdout + process.stderr
            cleanup_confirmed = (fixture / "windows-cleanup-confirmed").exists()
            stop_worker.set()
            worker.join(timeout=3)
            daemon._cleanup()
            daemon = None
        captured = output_capture.getvalue() + error_capture.getvalue() + windows_output
        harness_logs = "".join(p.read_text(errors="replace") for p in (fixture / "harness-logs").rglob("*") if p.is_file())
        captured += harness_logs
        leaked = any(value in captured for value in private_values) or "PRIVATE KEY" in captured
        result = {"windows_exit_code": process.returncode, "fixture_failures": fixture_failures,
                  "dispatch_count": len(dispatched), "restart_terminal_statuses": restart_statuses,
                  "windows_certificate_cleanup": cleanup_confirmed, "captured_secret_leak": leaked,
                  "channels": ["daemon stdout", "daemon stderr", "Harness log files", "Windows build/test stdout/stderr",
                               "Windows API RequestCompleted", "Windows exception strings", "Island InteractionDiagnostics file"]}
        if leaked:
            print("Captured output contained a sensitive fixture value; output suppressed.")
        else:
            print(windows_output)
        print(json.dumps(result))
        Path('/tmp/lila-m2-resume-live-result.json').write_text(json.dumps(result))
        return 0 if process.returncode == 0 and cleanup_confirmed and not leaked and not fixture_failures else 1
    finally:
        stop_worker.set()
        if worker:
            worker.join(timeout=3)
        if daemon:
            with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(error_capture):
                daemon._cleanup()
        registry._tools.clear()
        registry._tools.update(previous)
        if cleanup_confirmed:
            shutil.rmtree(fixture)
            print("Temporary fixture keys and credentials removed.")
        else:
            print("Temporary fixture retained pending certificate cleanup verification: " + str(fixture))


if __name__ == "__main__":
    raise SystemExit(main())
