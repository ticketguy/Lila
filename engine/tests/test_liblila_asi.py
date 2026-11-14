#!/usr/bin/env python3
"""
Phase 3 end-to-end test of asi/liblila_asi.so C ABI via ctypes.
"""

import ctypes
import json
import sys
import time
from pathlib import Path

LIB_PATH = Path(__file__).resolve().parent.parent / "asi" / "liblila_asi.so"
ASI_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/qwen35_0.8b.asi")

SEC_TOKENIZER = 0x04
SEC_CHAT_TEMPLATE = 0x09


class AsiV2ModelConfig(ctypes.Structure):
    _fields_ = [
        ("arch_id", ctypes.c_uint32),
        ("n_layers", ctypes.c_uint32),
        ("hidden_size", ctypes.c_uint32),
        ("intermediate_size", ctypes.c_uint32),
        ("n_heads", ctypes.c_uint32),
        ("n_kv_heads", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("rope_dim", ctypes.c_uint32),
        ("rope_theta", ctypes.c_float),
        ("rms_norm_eps", ctypes.c_float),
        ("embed_scale", ctypes.c_float),
        ("full_attention_interval", ctypes.c_uint32),
        ("ssm_conv_kernel", ctypes.c_uint32),
        ("ssm_state_size", ctypes.c_uint32),
        ("ssm_group_count", ctypes.c_uint32),
        ("ssm_inner_size", ctypes.c_uint32),
        ("ssm_time_step_rank", ctypes.c_uint32),
        ("default_quant_type", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def decode_tokens(tokens_vocab, ids):
    u2b = gpt2_bytes_to_unicode()
    out = bytearray()
    for tid in ids:
        s = tokens_vocab[tid]
        for ch in s:
            if ch in u2b:
                out.append(u2b[ch])
            else:
                out.extend(ch.encode("utf-8"))
    return out.decode("utf-8", errors="replace")


def main() -> int:
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.lila_asi_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.lila_asi_open.restype = ctypes.c_void_p
    lib.lila_asi_close.argtypes = [ctypes.c_void_p]
    lib.lila_asi_model_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(AsiV2ModelConfig)]
    lib.lila_asi_model_info.restype = ctypes.c_int
    lib.lila_asi_get_section.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.lila_asi_get_section.restype = ctypes.c_int
    lib.lila_asi_session_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.lila_asi_session_create.restype = ctypes.c_void_p
    lib.lila_asi_session_free.argtypes = [ctypes.c_void_p]
    lib.lila_asi_eval_token.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.lila_asi_eval_token.restype = ctypes.c_int
    lib.lila_asi_eval_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.lila_asi_eval_batch.restype = ctypes.c_int
    lib.lila_asi_sample.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.lila_asi_sample.restype = ctypes.c_int32

    err = ctypes.create_string_buffer(256)
    model = lib.lila_asi_open(str(ASI_PATH).encode("utf-8"), err, 256)
    if not model:
        print(f"FAIL: lila_asi_open: {err.value.decode()}")
        return 1

    cfg = AsiV2ModelConfig()
    lib.lila_asi_model_info(model, ctypes.byref(cfg))
    print(f"Opened {ASI_PATH}: arch={cfg.arch_id} layers={cfg.n_layers} hidden={cfg.hidden_size} vocab={cfg.vocab_size}")

    sec_ptr = ctypes.c_void_p()
    sec_sz = ctypes.c_size_t()
    assert lib.lila_asi_get_section(model, SEC_TOKENIZER, ctypes.byref(sec_ptr), ctypes.byref(sec_sz)) == 0
    tok_json = json.loads(ctypes.string_at(sec_ptr.value, sec_sz.value).decode("utf-8"))
    tokens_vocab = tok_json["tokens"]
    eos_id = tok_json.get("eos_token_id", 248046)
    print(f"Loaded SEC_TOKENIZER: {len(tokens_vocab)} tokens, eos={eos_id}")

    assert lib.lila_asi_get_section(model, SEC_CHAT_TEMPLATE, ctypes.byref(sec_ptr), ctypes.byref(sec_sz)) == 0
    print(f"Loaded SEC_CHAT_TEMPLATE: {sec_sz.value} bytes")

    sess = lib.lila_asi_session_create(model, 512, 0, err, 256)
    if not sess:
        print(f"FAIL: session_create: {err.value.decode()}")
        lib.lila_asi_close(model)
        return 1

    # Prompt: "<|im_start|>user\nSay hi in 3 words.<|im_end|>\n<|im_start|>assistant\n"
    vocab_map = {t: i for i, t in enumerate(tokens_vocab)}
    b2u = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    def simple_encode(text):
        # Greedy longest-match over byte-encoded vocab for quick test
        out = []
        parts = text.split("<|im_start|>")
        for idx, p in enumerate(parts):
            if idx > 0:
                out.append(vocab_map["<|im_start|>"])
            subparts = p.split("<|im_end|>")
            for j, sp in enumerate(subparts):
                if j > 0:
                    out.append(vocab_map["<|im_end|>"])
                bstr = "".join(b2u[b] for b in sp.encode("utf-8"))
                pos = 0
                while pos < len(bstr):
                    matched = None
                    for end in range(min(len(bstr), pos + 32), pos, -1):
                        cand = bstr[pos:end]
                        if cand in vocab_map:
                            matched = vocab_map[cand]
                            pos = end
                            break
                    if matched is None:
                        pos += 1
                    else:
                        out.append(matched)
        return out

    prompt_ids = simple_encode("<|im_start|>user\nSay hi in 3 words.<|im_end|>\n<|im_start|>assistant\n")
    arr = (ctypes.c_int32 * len(prompt_ids))(*prompt_ids)
    logits = (ctypes.c_float * cfg.vocab_size)()

    t0 = time.monotonic()
    assert lib.lila_asi_eval_batch(sess, arr, len(prompt_ids), 0, logits) == 0
    t_prefill = time.monotonic() - t0

    rng = ctypes.c_uint64(42)
    gen_ids = []
    pos = len(prompt_ids)
    t1 = time.monotonic()
    for _ in range(12):
        tok = lib.lila_asi_sample(logits, cfg.vocab_size, 0.0, 40, 0.95, ctypes.byref(rng))
        if tok == eos_id:
            break
        gen_ids.append(tok)
        assert lib.lila_asi_eval_token(sess, tok, pos, logits) == 0
        pos += 1
    t_gen = time.monotonic() - t1

    print(f"Prompt ({len(prompt_ids)} tokens) prefill: {t_prefill:.2f}s")
    print(f"Generated ({len(gen_ids)} tokens in {t_gen:.2f}s, {len(gen_ids)/max(t_gen,1e-6):.2f} tok/s): {decode_tokens(tokens_vocab, gen_ids)!r}")

    lib.lila_asi_session_free(sess)
    lib.lila_asi_close(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
