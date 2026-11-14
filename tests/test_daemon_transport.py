import os
import http.client
import hashlib
import io
import json
import socket
import ssl
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stderr

from src.daemon.transport import (TransportConfigurationError,
                                  allowed_client_fingerprints,
                                  build_mutual_tls_context)
from src.daemon.approvals import ApprovalApi
from src.daemon.oia_harness import build_approval_service
from src.daemon.service import LilaDaemon
from src.harness.executor import HarnessExecutor
from src.harness.tools import Tool, ToolArg, ToolResult, get_registry, register_tool
from src.harness.system_tools import _file_read
from src.harness.file_targets import FileReadExecutionTargetAdapter
from src.daemon.oia_harness import authenticate_typed_decision
from src.oia import OIAError, OwnerAuthenticationService, OwnerProfile


@unittest.skipUnless(subprocess.run(["openssl", "version"], capture_output=True).returncode == 0,
                     "OpenSSL command line is required for certificate fixtures")
class MutualTlsTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)

        def run(*args):
            subprocess.run(["openssl", *map(str, args)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
            "-keyout", cls.root / "ca.key", "-out", cls.root / "ca.pem",
            "-subj", "/CN=Lila test CA", "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign,cRLSign")
        cls._issue(run, "server", "serverAuth", "subjectAltName=IP:127.0.0.1")
        cls._issue(run, "client", "clientAuth")
        cls._issue(run, "expired", "clientAuth", days="-1")
        cls._issue(run, "outsider", "clientAuth", ca_name="outsider-ca")

    @classmethod
    def _issue(cls, run, name, usage, extra=None, ca_name="ca", days="2"):
        key, csr, cert = (cls.root / f"{name}.key", cls.root / f"{name}.csr", cls.root / f"{name}.pem")
        run("req", "-new", "-newkey", "rsa:2048", "-nodes", "-keyout", key,
            "-out", csr, "-subj", f"/CN={name}")
        if ca_name == "outsider-ca":
            ca_key, ca_cert = cls.root / "outsider-ca.key", cls.root / "outsider-ca.pem"
            run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
                "-keyout", ca_key, "-out", ca_cert, "-subj", "/CN=outsider CA",
                "-addext", "basicConstraints=critical,CA:TRUE")
        else:
            ca_key, ca_cert = cls.root / "ca.key", cls.root / "ca.pem"
        ext = cls.root / f"{name}.ext"
        ext.write_text(f"basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage={usage}\n" + (f"{extra}\n" if extra else ""))
        run("x509", "-req", "-in", csr, "-CA", ca_cert, "-CAkey", ca_key,
            "-CAcreateserial", "-days", days, "-out", cert, "-extfile", ext)
        key.chmod(0o600)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def env(self):
        client_der = subprocess.run(
            ["openssl", "x509", "-in", str(self.root / "client.pem"), "-outform", "DER"],
            check=True, capture_output=True,
        ).stdout
        return {
            "LILA_TLS_SERVER_CERT": str(self.root / "server.pem"),
            "LILA_TLS_SERVER_KEY": str(self.root / "server.key"),
            "LILA_TLS_CLIENT_CA": str(self.root / "ca.pem"),
            "LILA_TLS_ALLOWED_CLIENT_SHA256": hashlib.sha256(client_der).hexdigest(),
        }

    def handshake(self, client_cert=True, outsider=False, expired=False):
        server_context = build_mutual_tls_context(self.env())
        client_context = ssl.create_default_context(cafile=str(self.root / "ca.pem"))
        if client_cert:
            name = "outsider" if outsider else "expired" if expired else "client"
            client_context.load_cert_chain(str(self.root / f"{name}.pem"), str(self.root / f"{name}.key"))
        left, right = socket.socketpair()
        outcome = []

        def accept():
            try:
                with server_context.wrap_socket(left, server_side=True) as connection:
                    outcome.append(connection.recv(4))
            except (ssl.SSLError, OSError) as exc:
                outcome.append(exc)

        worker = threading.Thread(target=accept)
        worker.start()
        try:
            with client_context.wrap_socket(right, server_hostname="127.0.0.1") as connection:
                connection.sendall(b"ping")
        except (ssl.SSLError, OSError):
            pass
        finally:
            left.close()
            right.close()
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        return outcome[0]

    def test_trusted_client_completes_mutual_tls_exchange(self):
        self.assertEqual(self.handshake(), b"ping")

    def test_missing_client_certificate_is_rejected(self):
        self.assertIsInstance(self.handshake(client_cert=False), (ssl.SSLError, OSError))

    def test_client_from_untrusted_ca_is_rejected(self):
        self.assertIsInstance(self.handshake(outsider=True), (ssl.SSLError, OSError))

    def test_expired_client_certificate_is_rejected(self):
        self.assertIsInstance(self.handshake(expired=True), (ssl.SSLError, OSError))

    def test_missing_configuration_and_permissive_key_fail_closed(self):
        with self.assertRaises(TransportConfigurationError):
            build_mutual_tls_context({})
        with self.assertRaises(TransportConfigurationError):
            allowed_client_fingerprints({})
        key = self.root / "server.key"
        key.chmod(0o644)
        try:
            with self.assertRaises(TransportConfigurationError):
                build_mutual_tls_context(self.env())
        finally:
            key.chmod(0o600)

    def test_api_listener_has_no_plain_http_fallback(self):
        daemon = LilaDaemon(port=0, api_token="test-daemon-token-that-is-not-owner-proof", voice=False)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(TransportConfigurationError):
                daemon._start_api()
        self.assertIsNone(daemon._api_server)

    def test_client_certificate_allowlist_supports_immediate_rotation(self):
        old_pin = "a" * 64
        new_pin = "b" * 64
        self.assertEqual(allowed_client_fingerprints({
            "LILA_TLS_ALLOWED_CLIENT_SHA256": old_pin + "," + new_pin
        }), frozenset({old_pin, new_pin}))
        self.assertEqual(allowed_client_fingerprints({
            "LILA_TLS_ALLOWED_CLIENT_SHA256": new_pin
        }), frozenset({new_pin}))

    def test_production_api_listener_serves_only_over_mutual_tls(self):
        env = self.env()
        env["LILA_API_TOKEN"] = "test-daemon-token-that-is-not-owner-proof"
        daemon = LilaDaemon(port=0, api_token=env["LILA_API_TOKEN"], voice=False)
        with patch.dict(os.environ, env, clear=False):
            daemon._start_api()
        thread = threading.Thread(target=daemon._api_server.serve_forever, daemon=True)
        thread.start()
        try:
            client_context = ssl.create_default_context(cafile=str(self.root / "ca.pem"))
            client_context.load_cert_chain(str(self.root / "client.pem"), str(self.root / "client.key"))
            connection = http.client.HTTPSConnection("127.0.0.1", daemon.port,
                                                      context=client_context, timeout=3)
            connection.request("GET", "/owner/status", headers={
                "Authorization": "Bearer " + env["LILA_API_TOKEN"]
            })
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn(b'"enrolled": false', response.read())
            connection.close()

            no_client = ssl.create_default_context(cafile=str(self.root / "ca.pem"))
            rejected = http.client.HTTPSConnection("127.0.0.1", daemon.port,
                                                    context=no_client, timeout=3)
            with self.assertRaises((ssl.SSLError, OSError, http.client.HTTPException)):
                rejected.request("GET", "/owner/status")
                rejected.getresponse()
            rejected.close()
        finally:
            daemon._stop_api()
            thread.join(timeout=3)

    def test_revoked_client_certificate_is_denied_after_valid_tls_handshake(self):
        env = self.env()
        env["LILA_TLS_ALLOWED_CLIENT_SHA256"] = "0" * 64
        daemon = LilaDaemon(port=0, api_token="test-daemon-token-that-is-not-owner-proof", voice=False)
        with patch.dict(os.environ, env, clear=False):
            daemon._start_api()
        thread = threading.Thread(target=daemon._api_server.serve_forever, daemon=True)
        thread.start()
        try:
            context = ssl.create_default_context(cafile=str(self.root / "ca.pem"))
            context.load_cert_chain(str(self.root / "client.pem"), str(self.root / "client.key"))
            connection = http.client.HTTPSConnection("127.0.0.1", daemon.port,
                                                      context=context, timeout=3)
            connection.request("GET", "/owner/status", headers={
                "Authorization": "Bearer test-daemon-token-that-is-not-owner-proof"
            })
            response = connection.getresponse()
            self.assertEqual(response.status, 401)
            connection.close()
        finally:
            daemon._stop_api()
            thread.join(timeout=3)

    def test_mtls_oia_harness_path_approves_only_reviewed_exact_target(self):
        token = "integration-daemon-token-that-is-not-owner-proof"
        phrase = "test-only violet orchard owner phrase"
        owner_secret = b"test-only-owner-secret-material-32bytes"
        received_handles = []
        previous = dict(get_registry()._tools)
        get_registry()._tools.clear()
        target_file = self.root / "sensitive.txt"
        target_file.write_text("fixture data")
        workspace = self.root / "workspace"
        workspace.mkdir()
        def observed_file_read(path, *, _verified_file=None):
            received_handles.append(_verified_file)
            return _file_read(path, _verified_file=_verified_file)

        tool = Tool("file_read", "read", [ToolArg("path", "path", "target")], observed_file_read)
        register_tool(tool)

        class Store:
            enrolled = True

            def profile(self):
                return OwnerProfile("owner-test", "owner-test-key", 0)

            def credential(self):
                raise AssertionError("The controlled test unlock is used.")

        owner = OwnerAuthenticationService(Store())

        def unlock(candidate, *, source):
            if candidate != phrase:
                raise OIAError("owner_authentication.failed")
            return bytearray(owner_secret)

        owner._unlock = unlock
        service = build_approval_service(owner, get_registry())
        daemon = LilaDaemon(port=0, api_token=token, voice=False,
                            approval_continuation_service=service)
        daemon.owner_authentication_service = owner
        daemon.lila = SimpleNamespace(approval_continuation_service=service,
                                      owner_authentication_service=owner)
        executor = HarnessExecutor(log_dir=str(self.root / "logs"),
                                   workspace_root=str(workspace),
                                   approval_continuation_service=service)
        try:
            call = json.dumps({"name": "file_read", "arguments": {"path": str(target_file)}})
            results = executor.process("<tool_call>" + call + "</tool_call>")
            request = service.get_request(results[0]["approval"].invocation_id)
            self.assertEqual(ApprovalApi(service).get_pending(request.invocation.invocation_id)
                             .request.approval_ready, True)
            env = self.env()
            env["LILA_API_TOKEN"] = token
            with patch.dict(os.environ, env, clear=False):
                daemon._start_api()
            thread = threading.Thread(target=daemon._api_server.serve_forever, daemon=True)
            thread.start()
            try:
                diagnostics = io.StringIO()
                client_context = ssl.create_default_context(cafile=str(self.root / "ca.pem"))
                client_context.load_cert_chain(str(self.root / "client.pem"), str(self.root / "client.key"))
                connection = http.client.HTTPSConnection("127.0.0.1", daemon.port,
                                                          context=client_context, timeout=3)
                with redirect_stderr(diagnostics):
                    connection.request("POST", f"/owner/approvals/{request.invocation.invocation_id}/decision",
                                       body=json.dumps({"decision": "approve"}), headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "application/json",
                    })
                    unauthenticated = connection.getresponse()
                    self.assertEqual(unauthenticated.status, 403)
                    unauthenticated.read()
                    self.assertEqual(received_handles, [])

                    body = json.dumps({"phrase": phrase, "decision": "approve"})
                    connection.request("POST", f"/owner/approvals/{request.invocation.invocation_id}/decision",
                                       body=body, headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "application/json",
                    })
                    response = connection.getresponse()
                    response_body = response.read()
                self.assertEqual(response.status, 200, response_body)
                self.assertEqual(json.loads(response_body)["status"], "approved")
                self.assertFalse(phrase.encode() in response_body,
                                 "ordinary response leaked the controlled owner phrase")
                diagnostic_text = diagnostics.getvalue()
                self.assertFalse(any(secret in diagnostic_text for secret in (
                    phrase, token, owner_secret.decode(),
                )), "diagnostics leaked a controlled test secret")
                self.assertEqual(len(received_handles), 1)
                self.assertTrue(received_handles[0]._closed)
                self.assertEqual(service.get_result(request.invocation.invocation_id)
                                 .tool_result.output, "fixture data")
                connection.close()
            finally:
                daemon._stop_api()
                thread.join(timeout=3)

            # Replace the pathname after Harness target revalidation and grant
            # consumption but immediately before the production target adapter
            # opens it. The pinned target verifier must reject the replacement.
            original_file = self.root / "race-target.txt"
            replacement = self.root / "replacement.txt"
            original_file.write_text("approved original")
            replacement.write_text("unapproved replacement")
            race_call = json.dumps({"name": "file_read", "arguments": {"path": str(original_file)}})
            race_request = service.get_request(executor.process("<tool_call>" + race_call + "</tool_call>")[0]
                                               ["approval"].invocation_id)
            first_handle_count = len(received_handles)
            original_open = FileReadExecutionTargetAdapter.open_target

            @__import__("contextlib").contextmanager
            def replace_then_open(adapter, invocation):
                os.replace(replacement, original_file)
                with original_open(adapter, invocation) as target:
                    yield target

            with patch.object(FileReadExecutionTargetAdapter, "open_target", replace_then_open):
                race_result = authenticate_typed_decision(
                    owner, service, race_request.invocation.invocation_id,
                    phrase, "approve",
                )
            self.assertEqual(race_result.status.value, "failed")
            self.assertEqual(len(received_handles), first_handle_count)
            self.assertNotEqual(service.get_result(race_request.invocation.invocation_id)
                                .tool_result.output if service.get_result(race_request.invocation.invocation_id).tool_result else "",
                                "unapproved replacement")
        finally:
            get_registry()._tools.clear()
            get_registry()._tools.update(previous)


if __name__ == "__main__":
    unittest.main()
