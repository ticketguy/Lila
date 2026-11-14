#!/usr/bin/env python3
"""
ASI Backend Qualification Suite (items 1-6):
1. Real-model inference correctness (eval_token vs eval_batch parity, greedy determinism)
2. Multi-turn conversations, long prompts, and tool-call outputs
3. Cancellation, session isolation, and simultaneous inference requests
4. Memory cleanup, model unloading, and prefix-cache invalidation
5. Actual Lila daemon (LilaDaemon HTTP /chat) with LILA_BACKEND=asi
6. Controlled N=5 benchmark medians against llama-cpp-python on the same machine
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.asi_backend import AsiLlamaAdapter, _AsiV2ModelConfig

GGUF_PATH = "/home/ticket/workspaces/lila/models/Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf"
ASI_PATH = "/tmp/qwen35_0.8b.asi"
N_THREADS = 2


def _rss_mb() -> float:
    with open("/proc/self/statm", "r", encoding="utf-8") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_1_correctness() -> None:
    print("\n[1] Real-model inference correctness (batch vs token-by-token parity)...")
    llm = AsiLlamaAdapter(ASI_PATH, n_ctx=512, n_threads=N_THREADS)
    try:
        ids = llm.tokenizer.encode("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n")
        assert len(ids) >= 8, f"unexpected token count: {len(ids)}"

        # Token-by-token eval
        llm._lib.lila_asi_session_reset(llm._session)
        logits_tok = (ctypes.c_float * llm._cfg.vocab_size)()
        for pos, tid in enumerate(ids):
            rc = llm._lib.lila_asi_eval_token(llm._session, tid, pos, logits_tok)
            assert rc == 0, f"eval_token failed at pos={pos}"
        rng1 = ctypes.c_uint64(1)
        tok_next_1 = int(llm._lib.lila_asi_sample(logits_tok, llm._cfg.vocab_size, 0.0, 1, 1.0, ctypes.byref(rng1)))

        # Batched eval
        llm._lib.lila_asi_session_reset(llm._session)
        arr = (ctypes.c_int32 * len(ids))(*ids)
        logits_bat = (ctypes.c_float * llm._cfg.vocab_size)()
        rc = llm._lib.lila_asi_eval_batch(llm._session, arr, len(ids), 0, logits_bat)
        assert rc == 0, "eval_batch failed"
        rng2 = ctypes.c_uint64(1)
        tok_next_2 = int(llm._lib.lila_asi_sample(logits_bat, llm._cfg.vocab_size, 0.0, 1, 1.0, ctypes.byref(rng2)))

        assert tok_next_1 == tok_next_2, f"token vs batch argmax mismatch: {tok_next_1} != {tok_next_2}"
        top5_t = set(sorted(range(llm._cfg.vocab_size), key=lambda i: logits_tok[i], reverse=True)[:5])
        top5_b = set(sorted(range(llm._cfg.vocab_size), key=lambda i: logits_bat[i], reverse=True)[:5])
        overlap = len(top5_t & top5_b)
        assert overlap >= 4, f"top-5 overlap too low: {overlap}/5 ({top5_t} vs {top5_b})"
        num = sum((logits_tok[i] - logits_bat[i]) ** 2 for i in range(0, llm._cfg.vocab_size, 17))
        den = sum(logits_tok[i] ** 2 for i in range(0, llm._cfg.vocab_size, 17)) + 1e-12
        rel_l2 = (num / den) ** 0.5
        assert rel_l2 < 0.15, f"relative L2 logit drift too large: {rel_l2}"
        print(f"  PASS: argmax={tok_next_1} matched, top5_overlap={overlap}/5, rel_l2={rel_l2:.6f}")
    finally:
        llm.close()


def test_2_multiturn_long_toolcall() -> None:
    print("\n[2] Multi-turn conversations, long prompts, and tool-call outputs...")
    llm = AsiLlamaAdapter(ASI_PATH, n_ctx=512, n_threads=N_THREADS)
    try:
        # Multi-turn
        msgs = [
            {"role": "system", "content": "You are Lila, a concise assistant."},
            {"role": "user", "content": "My favorite number is 42. Acknowledge in 5 words."},
        ]
        r1 = llm.create_chat_completion(msgs, max_tokens=20, temperature=0.0)["choices"][0]["message"]["content"]
        assert r1.strip(), "Turn 1 empty"
        msgs.append({"role": "assistant", "content": r1})
        msgs.append({"role": "user", "content": "What is my favorite number?"})
        r2 = llm.create_chat_completion(msgs, max_tokens=24, temperature=0.0)["choices"][0]["message"]["content"]
        assert "42" in r2, f"Multi-turn lost context: {r2!r}"
        print(f"  PASS multi-turn: Turn2={r2.strip()!r}")

        # Long prompt (~300+ tokens within n_ctx=512)
        filler = " ".join(f"item_{i}={i*3}" for i in range(50))
        long_msgs = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": f"Data: {filler}\nReply with OK."},
        ]
        n_toks = len(llm.tokenizer.encode(llm._format_chat(long_msgs)))
        assert 250 < n_toks < 500, f"expected 250..500 tokens, got {n_toks}"
        rl = llm.create_chat_completion(long_msgs, max_tokens=16, temperature=0.0)["choices"][0]["message"]["content"]
        assert rl.strip(), "Long prompt returned empty output"
        print(f"  PASS long prompt ({n_toks} tokens): {rl.strip()[:80]!r}")

        # Over-context prompt (> n_ctx=512) must fail cleanly without crashing
        over_msgs = [{"role": "user", "content": " ".join(f"item_{i}={i*3}" for i in range(100))}]
        try:
            llm.create_chat_completion(over_msgs, max_tokens=8, temperature=0.0)
            raise AssertionError("Expected RuntimeError when prompt exceeds n_ctx=512")
        except RuntimeError as exc:
            print(f"  PASS over-context guard: {exc}")

        # Tool-call output
        tool_msgs = [
            {
                "role": "system",
                "content": (
                    "You have a tool: get_time(). "
                    'When asked for the time, respond ONLY with: <tool_call>{"name": "get_time", "arguments": {}}</tool_call>'
                ),
            },
            {"role": "user", "content": "What time is it?"},
        ]
        rt = llm.create_chat_completion(tool_msgs, max_tokens=40, temperature=0.0)["choices"][0]["message"]["content"]
        assert "get_time" in rt or "tool_call" in rt, f"Tool call not emitted: {rt!r}"
        print(f"  PASS tool-call output: {rt.strip()[:100]!r}")
    finally:
        llm.close()


def test_3_cancel_isolation_simultaneous() -> None:
    print("\n[3] Cancellation, session isolation, and simultaneous inference...")
    # 3a. Cancellation via cancel_event & early stream close
    llm1 = AsiLlamaAdapter(ASI_PATH, n_ctx=512, n_threads=N_THREADS)
    llm2 = AsiLlamaAdapter(ASI_PATH, n_ctx=512, n_threads=N_THREADS)
    try:
        cancel = threading.Event()
        gen = llm1.create_chat_completion(
            [{"role": "user", "content": "Count from 1 to 50 slowly."}],
            max_tokens=40,
            temperature=0.0,
            stream=True,
            cancel_event=cancel,
        )
        seen = 0
        for _ev in gen:
            seen += 1
            if seen >= 3:
                cancel.set()
        gen.close()
        assert 3 <= seen <= 5, f"expected early cancel around 3 tokens, got {seen}"

        # Subsequent call on same adapter must still work cleanly
        after = llm1.create_chat_completion(
            [{"role": "user", "content": "Say hello."}], max_tokens=12, temperature=0.0
        )["choices"][0]["message"]["content"]
        assert after.strip(), "Post-cancel generation failed"
        print(f"  PASS cancellation: stopped at {seen} chunks, post-cancel={after.strip()[:60]!r}")

        # 3b. Session isolation + simultaneous requests on same adapter (lock-serialized)
        res_a, res_b = [], []

        def run_a():
            res_a.append(
                llm1.create_chat_completion(
                    [{"role": "user", "content": "Reply with ALPHA."}], max_tokens=15, temperature=0.0
                )["choices"][0]["message"]["content"]
            )

        def run_b():
            res_b.append(
                llm1.create_chat_completion(
                    [{"role": "user", "content": "Reply with BETA."}], max_tokens=15, temperature=0.0
                )["choices"][0]["message"]["content"]
            )

        ta = threading.Thread(target=run_a)
        tb = threading.Thread(target=run_b)
        ta.start()
        tb.start()
        ta.join(timeout=30)
        tb.join(timeout=30)
        assert res_a and res_b and res_a[0].strip() and res_b[0].strip(), "Simultaneous same-adapter failed"
        print(f"  PASS simultaneous same-adapter: A={res_a[0].strip()[:40]!r}, B={res_b[0].strip()[:40]!r}")

        # 3c. Simultaneous inference on TWO independent sessions sharing ONE LilaAsiModel
        lib = llm1._lib
        err = ctypes.create_string_buffer(256)
        s1 = lib.lila_asi_session_create(llm1._model, 256, 1, err, 256)
        s2 = lib.lila_asi_session_create(llm1._model, 256, 1, err, 256)
        assert s1 and s2, "Failed to create two sessions on shared model"
        try:
            ids1 = llm1.tokenizer.encode("<|im_start|>user\nSay ONE.<|im_end|>\n<|im_start|>assistant\n")
            ids2 = llm1.tokenizer.encode("<|im_start|>user\nSay TWO.<|im_end|>\n<|im_start|>assistant\n")
            out_par = {}

            def worker(key, sess, toks):
                arr = (ctypes.c_int32 * len(toks))(*toks)
                logits = (ctypes.c_float * llm1._cfg.vocab_size)()
                rc = lib.lila_asi_eval_batch(sess, arr, len(toks), 0, logits)
                rng = ctypes.c_uint64(1)
                tok = int(lib.lila_asi_sample(logits, llm1._cfg.vocab_size, 0.0, 1, 1.0, ctypes.byref(rng)))
                out_par[key] = (rc, tok)

            p1 = threading.Thread(target=worker, args=("s1", s1, ids1))
            p2 = threading.Thread(target=worker, args=("s2", s2, ids2))
            p1.start()
            p2.start()
            p1.join(timeout=30)
            p2.join(timeout=30)
            assert out_par["s1"][0] == 0 and out_par["s2"][0] == 0, f"Parallel shared-model eval failed: {out_par}"
            print(f"  PASS parallel shared-model sessions: s1_tok={out_par['s1'][1]}, s2_tok={out_par['s2'][1]}")
        finally:
            lib.lila_asi_session_free(s1)
            lib.lila_asi_session_free(s2)
    finally:
        llm1.close()
        llm2.close()


def test_4_memory_and_prefix_invalidation() -> None:
    print("\n[4] Memory cleanup, model unloading, and prefix-cache invalidation...")
    # 4a. Prefix-cache invalidation
    llm = AsiLlamaAdapter(ASI_PATH, n_ctx=512, n_threads=N_THREADS)
    try:
        sys_a = [{"role": "system", "content": "Always include the word MANGO."}, {"role": "user", "content": "Hi"}]
        sys_b = [{"role": "system", "content": "Always include the word ZEBRA."}, {"role": "user", "content": "Hi"}]
        no_sys = [{"role": "user", "content": "Reply with just: OK."}]

        ra = llm.create_chat_completion(sys_a, max_tokens=20, temperature=0.0)["choices"][0]["message"]["content"]
        cached_a = list(llm._cached_prefix_ids or [])
        assert cached_a, "System A was not cached"

        rb = llm.create_chat_completion(sys_b, max_tokens=20, temperature=0.0)["choices"][0]["message"]["content"]
        cached_b = list(llm._cached_prefix_ids or [])
        assert cached_b and cached_b != cached_a, "Prefix cache did not invalidate & replace A with B"

        rn = llm.create_chat_completion(no_sys, max_tokens=10, temperature=0.0)["choices"][0]["message"]["content"]
        assert rn.strip(), "No-system prompt failed after cached prefix"
        print(f"  PASS prefix invalidation: A_len={len(cached_a)}, B_len={len(cached_b)}, no_sys={rn.strip()[:40]!r}")
    finally:
        llm.close()

    # 4b. Memory cleanup across 5 open/eval/close cycles (after warmup, gc + malloc_trim)
    import gc
    libc = ctypes.CDLL("libc.so.6")
    gc.collect()
    libc.malloc_trim(0)
    rss_start = _rss_mb()
    for _ in range(5):
        a = AsiLlamaAdapter(ASI_PATH, n_ctx=256, n_threads=N_THREADS)
        a.create_chat_completion([{"role": "user", "content": "Hi"}], max_tokens=4, temperature=0.0)
        a.close()
        del a
        gc.collect()
        libc.malloc_trim(0)
    rss_end = _rss_mb()
    drift = rss_end - rss_start
    assert drift < 8.0, f"RSS grew too much across 5 open/close cycles: {drift:.2f} MB"
    print(f"  PASS memory cleanup: RSS start={rss_start:.1f} MB, end={rss_end:.1f} MB (drift={drift:+.1f} MB)")


def test_5_daemon_with_lila_backend_asi() -> None:
    print("\n[5] Actual Lila daemon (LilaDaemon mTLS HTTPS /chat) with LILA_BACKEND=asi...")
    import hashlib
    import shutil
    import ssl
    import subprocess
    import tempfile
    from src.daemon.service import LilaDaemon

    tls_keys = (
        "LILA_BACKEND", "LILA_ASI_PATH", "LILA_THREADS",
        "LILA_TLS_SERVER_CERT", "LILA_TLS_SERVER_KEY",
        "LILA_TLS_CLIENT_CA", "LILA_TLS_ALLOWED_CLIENT_SHA256",
    )
    prev_env = {k: os.environ.get(k) for k in tls_keys}
    tls_dir = Path(tempfile.mkdtemp(prefix="lila_asi_qual_tls_"))
    try:
        ca_key, ca_crt = tls_dir / "ca.key", tls_dir / "ca.crt"
        srv_key, srv_csr, srv_crt = tls_dir / "srv.key", tls_dir / "srv.csr", tls_dir / "srv.crt"
        cli_key, cli_csr, cli_crt = tls_dir / "cli.key", tls_dir / "cli.csr", tls_dir / "cli.crt"
        ext = tls_dir / "srv.ext"
        ext.write_text("subjectAltName=IP:127.0.0.1,DNS:localhost\n", encoding="utf-8")

        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", str(ca_key), "-out", str(ca_crt), "-days", "1", "-subj", "/CN=LilaTestCA"],
                       check=True, capture_output=True)
        subprocess.run(["openssl", "req", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", str(srv_key), "-out", str(srv_csr), "-subj", "/CN=127.0.0.1"],
                       check=True, capture_output=True)
        subprocess.run(["openssl", "x509", "-req", "-in", str(srv_csr), "-CA", str(ca_crt), "-CAkey", str(ca_key),
                        "-CAcreateserial", "-out", str(srv_crt), "-days", "1", "-extfile", str(ext)],
                       check=True, capture_output=True)
        subprocess.run(["openssl", "req", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", str(cli_key), "-out", str(cli_csr), "-subj", "/CN=LilaTestIsland"],
                       check=True, capture_output=True)
        subprocess.run(["openssl", "x509", "-req", "-in", str(cli_csr), "-CA", str(ca_crt), "-CAkey", str(ca_key),
                        "-CAcreateserial", "-out", str(cli_crt), "-days", "1"],
                       check=True, capture_output=True)
        for p in (ca_key, srv_key, cli_key):
            os.chmod(p, 0o600)

        der = ssl.PEM_cert_to_DER_cert(cli_crt.read_text(encoding="utf-8"))
        cli_sha256 = hashlib.sha256(der).hexdigest()

        os.environ["LILA_BACKEND"] = "asi"
        os.environ["LILA_ASI_PATH"] = ASI_PATH
        os.environ["LILA_THREADS"] = str(N_THREADS)
        os.environ["LILA_TLS_SERVER_CERT"] = str(srv_crt)
        os.environ["LILA_TLS_SERVER_KEY"] = str(srv_key)
        os.environ["LILA_TLS_CLIENT_CA"] = str(ca_crt)
        os.environ["LILA_TLS_ALLOWED_CLIENT_SHA256"] = cli_sha256

        token = "0123456789abcdef0123456789abcdef"
        port = _free_port()
        daemon = LilaDaemon(
            model_path=GGUF_PATH,
            voice=False,
            port=port,
            api_token=token,
            headless=True,
        )
        try:
            daemon.lila = daemon._create_lila_core()
            daemon.lila.boot()
            daemon.lila.max_tokens = 32
            daemon._running = True
            daemon._service_state = "running"
            daemon._start_api()

            ctx = ssl.create_default_context(cafile=str(ca_crt))
            ctx.load_cert_chain(certfile=str(cli_crt), keyfile=str(cli_key))
            req = urllib.request.Request(
                f"https://127.0.0.1:{port}/chat",
                data=json.dumps({"text": "Say hello in one short sentence."}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                method="POST",
            )
            t_req = time.monotonic()
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=420) as resp:
                    assert resp.status == 200, f"Unexpected HTTP status: {resp.status}"
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                raise RuntimeError(f"HTTP {err.code}: {err.read().decode('utf-8', 'replace')}") from err
            elapsed_req = time.monotonic() - t_req
            assert isinstance(daemon.lila._llm, AsiLlamaAdapter), f"Expected AsiLlamaAdapter, got {type(daemon.lila._llm)}"
            reply = body.get("reply") or body.get("response") or ""
            assert isinstance(reply, str) and reply.strip(), f"Empty daemon reply: {body}"
            print(f"  PASS daemon mTLS /chat (HTTPS 200 in {elapsed_req:.1f}s): {reply.strip()[:120]!r}")
        finally:
            daemon._stop_api()
            if daemon.lila and getattr(daemon.lila, "_llm", None) and hasattr(daemon.lila._llm, "close"):
                daemon.lila._llm.close()
    finally:
        shutil.rmtree(tls_dir, ignore_errors=True)
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_6_controlled_medians() -> None:
    print("\n[6] Controlled N=5 benchmark medians + [min..max] spread (Lila ASI v2 vs llama-cpp-python)...")
    from llama_cpp import Llama

    messages = [
        {"role": "system", "content": "You are Lila, a concise local AI assistant."},
        {"role": "user", "content": "Give me a 1-sentence greeting and explain what 2+2 equals."},
    ]

    def run_once(loader):
        t0 = time.monotonic()
        llm = loader()
        load_s = time.monotonic() - t0
        res = {"load": load_s}
        try:
            for turn in (1, 2):
                t1 = time.monotonic()
                ttft = None
                chunks = 0
                for ev in llm.create_chat_completion(
                    messages=messages, max_tokens=20, temperature=0.0, top_p=1.0, top_k=1, stream=True
                ):
                    delta = (ev.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.monotonic() - t1
                        chunks += 1
                total = time.monotonic() - t1
                gen_s = max(1e-6, total - (ttft or total))
                dec = max(0, chunks - 1) / gen_s
                res[f"t{turn}_ttft"] = ttft or total
                res[f"t{turn}_dec"] = dec
        finally:
            if hasattr(llm, "close"):
                llm.close()
        return res

    asi_runs = [run_once(lambda: AsiLlamaAdapter(ASI_PATH, n_ctx=512, n_threads=N_THREADS, verbose=False)) for _ in range(5)]
    llc_runs = [run_once(lambda: Llama(model_path=GGUF_PATH, n_ctx=512, n_threads=N_THREADS, verbose=False)) for _ in range(5)]

    for metric in ("load", "t1_ttft", "t1_dec", "t2_ttft", "t2_dec"):
        a_vals = [r[metric] for r in asi_runs]
        l_vals = [r[metric] for r in llc_runs]
        a_med, a_min, a_max = statistics.median(a_vals), min(a_vals), max(a_vals)
        l_med, l_min, l_max = statistics.median(l_vals), min(l_vals), max(l_vals)
        unit = "tok/s" if "dec" in metric else "s"
        print(
            f"  {metric:8s}: ASI = {a_med:6.2f} [{a_min:.2f}..{a_max:.2f}] {unit}"
            f"  |  llama.cpp = {l_med:6.2f} [{l_min:.2f}..{l_max:.2f}] {unit}"
        )


if __name__ == "__main__":
    test_1_correctness()
    test_2_multiturn_long_toolcall()
    test_3_cancel_isolation_simultaneous()
    test_4_memory_and_prefix_invalidation()
    test_5_daemon_with_lila_backend_asi()
    test_6_controlled_medians()
    print("\nALL 6 ASI QUALIFICATION CHECKS PASSED")
