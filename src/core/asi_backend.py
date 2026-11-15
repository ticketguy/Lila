"""
Lila ASI v2 Runtime Backend — Python ctypes adapter for liblila_asi.so.
Drop-in compatible with llama_cpp.Llama.create_chat_completion().
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_SEC_TOKENIZER = 0x04
_SEC_CHAT_TEMPLATE = 0x09


class _AsiV2ModelConfig(ctypes.Structure):
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


def _gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class AsiBpeTokenizer:
    """Byte-level BPE tokenizer backed by SEC_TOKENIZER inside model.asi."""

    _PRE_RE = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
    )

    def __init__(self, tok_json: dict):
        self.tokens: List[str] = tok_json.get("tokens", [])
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.eos_token_id: int = int(tok_json.get("eos_token_id") or 248046)
        self.b2u = _gpt2_bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

        merges_raw = tok_json.get("merges", [])
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for idx, m in enumerate(merges_raw):
            if isinstance(m, str) and " " in m:
                a, b = m.split(" ", 1)
                self.bpe_ranks[(a, b)] = idx

        self._special_tokens = [t for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>") if t in self.vocab]
        self._bpe_cache: Dict[str, Tuple[str, ...]] = {}

    def _bpe(self, token: str) -> Tuple[str, ...]:
        if token in self._bpe_cache:
            return self._bpe_cache[token]
        word = tuple(token)
        if len(word) <= 1 or not self.bpe_ranks:
            return word
        while True:
            best_pair = None
            best_rank = 1 << 30
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        self._bpe_cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        if self._special_tokens:
            pattern = "(" + "|".join(re.escape(s) for s in self._special_tokens) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        ids: List[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.vocab and chunk in self._special_tokens:
                ids.append(self.vocab[chunk])
                continue
            for piece in self._PRE_RE.findall(chunk) or [chunk]:
                bstr = "".join(self.b2u[b] for b in piece.encode("utf-8"))
                for btok in self._bpe(bstr):
                    tid = self.vocab.get(btok)
                    if tid is not None:
                        ids.append(tid)
                    else:
                        for ch in btok:
                            if ch in self.vocab:
                                ids.append(self.vocab[ch])
        return ids

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for tid in ids:
            if 0 <= tid < len(self.tokens):
                s = self.tokens[tid]
                for ch in s:
                    if ch in self.u2b:
                        out.append(self.u2b[ch])
                    else:
                        out.extend(ch.encode("utf-8"))
        return out.decode("utf-8", errors="replace")


def _default_lib_path() -> Path:
    env_path = os.environ.get("LILA_ASI_LIB")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "engine" / "asi" / "liblila_asi.so"


class AsiLlamaAdapter:
    """
    Drop-in replacement for llama_cpp.Llama backed by liblila_asi.so + model.asi.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 0,
        lib_path: Optional[str] = None,
        **_: object,
    ):
        so = Path(lib_path) if lib_path else _default_lib_path()
        if not so.exists():
            raise FileNotFoundError(f"liblila_asi.so not found at {so}")

        self._lib = ctypes.CDLL(str(so))
        self._bind_abi()

        err = ctypes.create_string_buffer(256)
        self._model = self._lib.lila_asi_open(model_path.encode("utf-8"), err, 256)
        if not self._model:
            raise RuntimeError(f"lila_asi_open failed: {err.value.decode('utf-8', 'replace')}")

        self._cfg = _AsiV2ModelConfig()
        self._lib.lila_asi_model_info(self._model, ctypes.byref(self._cfg))

        sec_ptr = ctypes.c_void_p()
        sec_sz = ctypes.c_size_t()
        if self._lib.lila_asi_get_section(self._model, _SEC_TOKENIZER, ctypes.byref(sec_ptr), ctypes.byref(sec_sz)) != 0:
            self._lib.lila_asi_close(self._model)
            raise RuntimeError("SEC_TOKENIZER missing from ASI cartridge")

        tok_json = json.loads(ctypes.string_at(sec_ptr.value, sec_sz.value).decode("utf-8"))
        self.tokenizer = AsiBpeTokenizer(tok_json)

        self.chat_template = ""
        if self._lib.lila_asi_get_section(self._model, _SEC_CHAT_TEMPLATE, ctypes.byref(sec_ptr), ctypes.byref(sec_sz)) == 0:
            self.chat_template = ctypes.string_at(sec_ptr.value, sec_sz.value).decode("utf-8", "replace")

        self._session = self._lib.lila_asi_session_create(self._model, int(n_ctx), int(n_threads), err, 256)
        if not self._session:
            self._lib.lila_asi_close(self._model)
            raise RuntimeError(f"lila_asi_session_create failed: {err.value.decode('utf-8', 'replace')}")

        self._logits = (ctypes.c_float * self._cfg.vocab_size)()
        self._rng = ctypes.c_uint64(0x9E3779B97F4A7C15)
        self._cached_prefix_ids: Optional[List[int]] = None
        self._cached_prefix_text: Optional[str] = None
        self._lock = threading.Lock()

    def _bind_abi(self) -> None:
        lib = self._lib
        lib.lila_asi_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.lila_asi_open.restype = ctypes.c_void_p
        lib.lila_asi_close.argtypes = [ctypes.c_void_p]
        lib.lila_asi_model_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AsiV2ModelConfig)]
        lib.lila_asi_model_info.restype = ctypes.c_int
        lib.lila_asi_get_section.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.lila_asi_get_section.restype = ctypes.c_int
        lib.lila_asi_session_create.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t
        ]
        lib.lila_asi_session_create.restype = ctypes.c_void_p
        lib.lila_asi_session_free.argtypes = [ctypes.c_void_p]
        lib.lila_asi_session_reset.argtypes = [ctypes.c_void_p]
        lib.lila_asi_session_save_prefix.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.lila_asi_session_save_prefix.restype = ctypes.c_int
        lib.lila_asi_session_restore_prefix.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        lib.lila_asi_session_restore_prefix.restype = ctypes.c_int
        lib.lila_asi_eval_token.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.lila_asi_eval_token.restype = ctypes.c_int
        lib.lila_asi_eval_batch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)
        ]
        lib.lila_asi_eval_batch.restype = ctypes.c_int
        lib.lila_asi_sample.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32, ctypes.c_float, ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.lila_asi_sample.restype = ctypes.c_int32

    def close(self) -> None:
        self._cached_prefix_ids = None
        self._cached_prefix_text = None
        if getattr(self, "_session", None):
            self._lib.lila_asi_session_free(self._session)
            self._session = None
        if getattr(self, "_model", None):
            self._lib.lila_asi_close(self._model)
            self._model = None

    def __del__(self):
        self.close()

    def _format_message(self, role: str, content: object) -> str:
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
            )
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def _format_chat(self, messages: List[Dict]) -> str:
        parts = [self._format_message(m.get("role", "user"), m.get("content", "")) for m in messages]
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _save_prefix_locked(self, sys_text: str, sys_ids: List[int]) -> None:
        self._lib.lila_asi_session_reset(self._session)
        sys_arr = (ctypes.c_int32 * len(sys_ids))(*sys_ids)
        if self._lib.lila_asi_eval_batch(self._session, sys_arr, len(sys_ids), 0, self._logits) != 0:
            raise RuntimeError("lila_asi_eval_batch(prefix) failed")
        if self._lib.lila_asi_session_save_prefix(self._session, len(sys_ids)) == 0:
            self._cached_prefix_ids = sys_ids
            self._cached_prefix_text = sys_text

    def warm_prefix(self, system_prompt: str, role: str = "system") -> int:
        """Pre-evaluate and snapshot the system prompt prefix so Turn 1 skips system prefill."""
        if not system_prompt:
            return 0
        sys_text = self._format_message(role, system_prompt)
        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiLlamaAdapter is closed")
            if self._cached_prefix_text == sys_text and self._cached_prefix_ids:
                return len(self._cached_prefix_ids)
            sys_ids = self.tokenizer.encode(sys_text)
            if not sys_ids:
                return 0
            if self._cached_prefix_ids == sys_ids:
                self._cached_prefix_text = sys_text
                return len(sys_ids)
            self._save_prefix_locked(sys_text, sys_ids)
            return len(sys_ids)

    def _stream_tokens(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[Dict]:
        has_system_prefix = len(messages) > 1 and messages[0].get("role") == "system"
        sys_text = ""
        if has_system_prefix:
            sys_text = self._format_message(
                messages[0].get("role", "system"),
                messages[0].get("content", ""),
            )

        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiLlamaAdapter is closed")

            # Fast path: reuse cached system-prompt tokenization when boundary tokens are atomic.
            can_split_sys = (
                has_system_prefix
                and self._cached_prefix_text == sys_text
                and self._cached_prefix_ids
                and "<|im_start|>" in self.tokenizer._special_tokens
                and "<|im_end|>" in self.tokenizer._special_tokens
            )
            if can_split_sys:
                sys_ids = self._cached_prefix_ids
                rest_ids = self.tokenizer.encode(self._format_chat(messages[1:]))
                prompt_ids = sys_ids + rest_ids
            else:
                prompt = self._format_chat(messages)
                prompt_ids = self.tokenizer.encode(prompt)
                sys_ids = self.tokenizer.encode(sys_text) if has_system_prefix else []

            if not prompt_ids:
                return

            # Automatic system-prompt prefix state caching (DeltaNet + KV snapshot)
            prefix_len = 0
            if sys_ids and len(prompt_ids) > len(sys_ids) and prompt_ids[: len(sys_ids)] == sys_ids:
                n_restored = ctypes.c_uint32(0)
                if (
                    self._cached_prefix_ids == sys_ids
                    and self._lib.lila_asi_session_restore_prefix(self._session, ctypes.byref(n_restored)) == 0
                    and n_restored.value == len(sys_ids)
                ):
                    self._cached_prefix_text = sys_text
                    prefix_len = len(sys_ids)
                else:
                    self._save_prefix_locked(sys_text, sys_ids)
                    prefix_len = len(sys_ids)

            if prefix_len == 0:
                self._lib.lila_asi_session_reset(self._session)

            rem_ids = prompt_ids[prefix_len:]
            if rem_ids:
                arr = (ctypes.c_int32 * len(rem_ids))(*rem_ids)
                if self._lib.lila_asi_eval_batch(self._session, arr, len(rem_ids), prefix_len, self._logits) != 0:
                    raise RuntimeError("lila_asi_eval_batch failed")

            pos = len(prompt_ids)
            eos = self.tokenizer.eos_token_id
            for _ in range(max_tokens):
                if cancel_event is not None and cancel_event.is_set():
                    break
                tok = int(
                    self._lib.lila_asi_sample(
                        self._logits,
                        self._cfg.vocab_size,
                        float(temperature),
                        int(top_k),
                        float(top_p),
                        ctypes.byref(self._rng),
                    )
                )
                if tok == eos or tok < 0:
                    break
                piece = self.tokenizer.decode([tok])
                if piece:
                    yield {"choices": [{"delta": {"content": piece}}]}
                if cancel_event is not None and cancel_event.is_set():
                    break
                if self._lib.lila_asi_eval_token(self._session, tok, pos, self._logits) != 0:
                    break
                pos += 1

    def create_chat_completion(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        stream: bool = False,
        cancel_event: Optional[threading.Event] = None,
        **_: object,
    ):
        if stream:
            return self._stream_tokens(messages, max_tokens, temperature, top_p, top_k, cancel_event=cancel_event)
        text = "".join(
            ev["choices"][0]["delta"]["content"]
            for ev in self._stream_tokens(messages, max_tokens, temperature, top_p, top_k, cancel_event=cancel_event)
        )
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}
