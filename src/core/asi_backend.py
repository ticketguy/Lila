"""
Lila ASI v2 Runtime Backend — Python ctypes adapter for liblila_asi.so.
Drop-in compatible with llama_cpp.Llama.create_chat_completion().
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import struct
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

try:
    from .private_storage import restrictive_umask, secure_private_directory
except ImportError:
    from src.core.private_storage import restrictive_umask, secure_private_directory

_SEC_TOKENIZER = 0x04
_SEC_CHAT_TEMPLATE = 0x09
_DEFAULT_PREFIX_CACHE_MAX_FILES = 8


class _AsiV2Header(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("arch_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("n_sections", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("total_size", ctypes.c_uint64),
        ("section_table_offset", ctypes.c_uint64),
        ("identity_hash", ctypes.c_uint8 * 24),
    ]


assert ctypes.sizeof(_AsiV2Header) == 64


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
        prefix_cache_dir: Optional[Union[str, Path]] = None,
        **_: object,
    ):
        so = Path(lib_path) if lib_path else _default_lib_path()
        if not so.exists():
            raise FileNotFoundError(f"liblila_asi.so not found at {so}")

        self._lib = ctypes.CDLL(str(so))
        self._has_model_header = False
        self._supports_prefix_file = False
        self._bind_abi()

        err = ctypes.create_string_buffer(256)
        self._model = self._lib.lila_asi_open(model_path.encode("utf-8"), err, 256)
        if not self._model:
            raise RuntimeError(f"lila_asi_open failed: {err.value.decode('utf-8', 'replace')}")

        self._cfg = _AsiV2ModelConfig()
        self._lib.lila_asi_model_info(self._model, ctypes.byref(self._cfg))

        self._model_hdr = _AsiV2Header()
        if self._has_model_header:
            if self._lib.lila_asi_model_header(self._model, ctypes.byref(self._model_hdr)) != 0:
                self._model_hdr = _AsiV2Header()
                self._supports_prefix_file = False

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

        raw_cache_dir: Optional[Union[str, Path]] = prefix_cache_dir
        if raw_cache_dir is None and os.environ.get("LILA_ASI_DISABLE_DISK_PREFIX") != "1":
            env_cache_dir = os.environ.get("LILA_ASI_PREFIX_CACHE_DIR", "").strip()
            raw_cache_dir = env_cache_dir or None
        elif isinstance(raw_cache_dir, str):
            raw_cache_dir = raw_cache_dir.strip() or None

        self._prefix_cache_dir: Optional[Path] = (
            Path(os.path.abspath(Path(raw_cache_dir).expanduser()))
            if raw_cache_dir is not None
            else None
        )

        self._logits = (ctypes.c_float * self._cfg.vocab_size)()
        self._rng = ctypes.c_uint64(0x9E3779B97F4A7C15)
        self._cached_prefix_ids: Optional[List[int]] = None
        self._cached_prefix_text: Optional[str] = None
        self.last_prefix_cache_hit: Optional[str] = None
        self.prefix_cache_stats: Dict[str, int] = {"ram": 0, "disk": 0, "miss": 0}
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

        self._has_model_header = hasattr(lib, "lila_asi_model_header")
        if self._has_model_header:
            lib.lila_asi_model_header.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_AsiV2Header),
            ]
            lib.lila_asi_model_header.restype = ctypes.c_int

        has_save_prefix_file = hasattr(lib, "lila_asi_session_save_prefix_file")
        if has_save_prefix_file:
            lib.lila_asi_session_save_prefix_file.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_char_p,
            ]
            lib.lila_asi_session_save_prefix_file.restype = ctypes.c_int

        has_load_prefix_file = hasattr(lib, "lila_asi_session_load_prefix_file")
        if has_load_prefix_file:
            lib.lila_asi_session_load_prefix_file.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            lib.lila_asi_session_load_prefix_file.restype = ctypes.c_int

        self._supports_prefix_file = bool(
            self._has_model_header and has_save_prefix_file and has_load_prefix_file
        )

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

    @staticmethod
    def _fp32_ref_flag() -> int:
        raw = (os.environ.get("LILA_ASI_FP32_REF") or "").strip().lower()
        if raw in ("1", "true", "yes"):
            return 1
        try:
            return 1 if raw and int(raw) != 0 else 0
        except ValueError:
            return 0

    @staticmethod
    def _max_prefix_cache_files() -> int:
        raw = (os.environ.get("LILA_ASI_PREFIX_CACHE_MAX_FILES") or "").strip()
        if not raw:
            return _DEFAULT_PREFIX_CACHE_MAX_FILES
        try:
            return max(1, int(raw))
        except ValueError:
            return _DEFAULT_PREFIX_CACHE_MAX_FILES

    def _compute_prefix_key_hash(self, sys_ids: Sequence[int], sys_text: str = "") -> bytes:
        """Compute the 32-byte SHA-256 cache key for a model, fp32_ref mode, text, and token sequence."""
        ids = [int(t) for t in sys_ids]
        fp32_ref = self._fp32_ref_flag()
        packed_ids = struct.pack(f"<{len(ids)}i", *ids) if ids else b""
        payload = (
            b"ASIPFX02"
            + bytes(self._model_hdr.identity_hash)
            + struct.pack("<III", int(self._model_hdr.arch_id), fp32_ref, len(ids))
            + sys_text.encode("utf-8")
            + packed_ids
        )
        return hashlib.sha256(payload).digest()

    def _record_prefix_hit(self, kind: str) -> None:
        self.last_prefix_cache_hit = kind
        if kind in self.prefix_cache_stats:
            self.prefix_cache_stats[kind] += 1

    @staticmethod
    def _prune_prefix_cache(
        cache_dir: Path,
        max_files: int,
        keep_path: Optional[Path] = None,
    ) -> None:
        """Evict oldest regular .asip snapshot files beyond max_files."""
        if max_files < 1:
            return
        try:
            entries = []
            for p in cache_dir.iterdir():
                if p.suffix != ".asip" or p.is_symlink() or not p.is_file():
                    continue
                try:
                    st = p.stat(follow_symlinks=False)
                except OSError:
                    continue
                is_keep = 1 if (keep_path is not None and p == keep_path) else 0
                entries.append((is_keep, st.st_mtime_ns, p.name, p))
            if len(entries) <= max_files:
                return
            entries.sort(key=lambda item: (item[0], item[1], item[2]))
            to_remove = len(entries) - max_files
            for _, _, _, victim in entries[:to_remove]:
                try:
                    victim.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass

    def _eval_and_save_ram_prefix_locked(self, ids: List[int], sys_text: str) -> bool:
        """Reset session, prefill prefix token IDs, and snapshot into session RAM."""
        self._lib.lila_asi_session_reset(self._session)
        sys_arr = (ctypes.c_int32 * len(ids))(*ids)
        if self._lib.lila_asi_eval_batch(self._session, sys_arr, len(ids), 0, self._logits) != 0:
            raise RuntimeError("lila_asi_eval_batch(prefix) failed")
        if self._lib.lila_asi_session_save_prefix(self._session, len(ids)) == 0:
            self._cached_prefix_ids = list(ids)
            self._cached_prefix_text = sys_text
            return True
        return False

    def _save_prefix_file_locked(self, target: Path, key_arr: ctypes.Array) -> bool:
        """Write current RAM prefix snapshot to target with 0600 permissions, rejecting symlinks."""
        if target.is_symlink():
            return False
        with restrictive_umask():
            rc = self._lib.lila_asi_session_save_prefix_file(
                self._session,
                key_arr,
                str(target).encode("utf-8"),
            )
        if rc != 0:
            return False
        if not target.is_symlink() and target.is_file():
            os.chmod(target, 0o600)
        return True

    def _load_prefix_file_locked(
        self,
        target: Path,
        key_arr: ctypes.Array,
        ids: List[int],
        sys_text: str,
    ) -> Optional[int]:
        """Load and validate a disk prefix snapshot into session RAM and live state."""
        if target.is_symlink() or not target.is_file():
            return None
        out_n = ctypes.c_uint32(0)
        rc = self._lib.lila_asi_session_load_prefix_file(
            self._session,
            key_arr,
            str(target).encode("utf-8"),
            ctypes.byref(out_n),
        )
        if rc != 0 or int(out_n.value) != len(ids):
            return None
        restored_n = ctypes.c_uint32(0)
        self._lib.lila_asi_session_restore_prefix(self._session, ctypes.byref(restored_n))
        self._cached_prefix_ids = list(ids)
        self._cached_prefix_text = sys_text
        self._record_prefix_hit("disk")
        return int(out_n.value)

    def _try_load_prefix_cache_file(
        self,
        sys_text: str,
        ids: List[int],
    ) -> Tuple[bool, Optional[Path], Optional[ctypes.Array]]:
        """Try loading a cached disk prefix snapshot; return (loaded, cache_path, key_arr)."""
        if self._prefix_cache_dir is None or not self._supports_prefix_file:
            return False, None, None
        try:
            cache_dir = secure_private_directory(self._prefix_cache_dir)
            key_hash = self._compute_prefix_key_hash(ids, sys_text)
            key_arr = (ctypes.c_uint8 * 32).from_buffer_copy(key_hash)
            cache_path = cache_dir / f"prefix_{key_hash[:16].hex()}.asip"
            if self._load_prefix_file_locked(cache_path, key_arr, ids, sys_text) is not None:
                try:
                    os.utime(cache_path, None)
                except OSError:
                    pass
                return True, cache_path, key_arr
            return False, cache_path, key_arr
        except Exception:
            return False, None, None

    def _save_prefix_locked(self, sys_text: str, sys_ids: Sequence[int]) -> None:
        ids = [int(t) for t in sys_ids]
        if not ids:
            return

        loaded, cache_path, key_arr = self._try_load_prefix_cache_file(sys_text, ids)
        if loaded:
            return

        saved_ram = self._eval_and_save_ram_prefix_locked(ids, sys_text)
        self._record_prefix_hit("miss")

        if saved_ram and cache_path is not None and key_arr is not None:
            try:
                if self._save_prefix_file_locked(cache_path, key_arr):
                    self._prune_prefix_cache(
                        cache_path.parent,
                        self._max_prefix_cache_files(),
                        keep_path=cache_path,
                    )
            except Exception:
                pass

    def save_prefix_snapshot(
        self,
        filepath: Union[str, Path],
        token_ids: Sequence[int],
        sys_text: str = "",
    ) -> bool:
        """Evaluate (if needed) and save a prefix snapshot for token_ids + sys_text to filepath."""
        ids = [int(t) for t in token_ids]
        if not ids:
            return False
        target = Path(filepath).expanduser()
        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiLlamaAdapter is closed")
            if not self._supports_prefix_file or target.is_symlink():
                return False
            try:
                if self._cached_prefix_ids != ids:
                    if not self._eval_and_save_ram_prefix_locked(ids, sys_text):
                        return False
                key_hash = self._compute_prefix_key_hash(ids, sys_text)
                key_arr = (ctypes.c_uint8 * 32).from_buffer_copy(key_hash)
                with restrictive_umask():
                    target.parent.mkdir(parents=True, exist_ok=True)
                return self._save_prefix_file_locked(target, key_arr)
            except Exception:
                return False

    def load_prefix_snapshot(
        self,
        filepath: Union[str, Path],
        token_ids: Sequence[int],
        sys_text: str = "",
    ) -> Optional[int]:
        """Load and validate a prefix snapshot from filepath into the session."""
        ids = [int(t) for t in token_ids]
        if not ids:
            return None
        target = Path(filepath).expanduser()
        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiLlamaAdapter is closed")
            if not self._supports_prefix_file:
                return None
            try:
                key_hash = self._compute_prefix_key_hash(ids, sys_text)
                key_arr = (ctypes.c_uint8 * 32).from_buffer_copy(key_hash)
                return self._load_prefix_file_locked(target, key_arr, ids, sys_text)
            except Exception:
                return None

    def warm_prefix(self, system_prompt: str, role: str = "system") -> int:
        """Pre-evaluate and snapshot the system prompt prefix so Turn 1 skips system prefill."""
        if not system_prompt:
            return 0
        sys_text = self._format_message(role, system_prompt)
        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiLlamaAdapter is closed")
            if self._cached_prefix_text == sys_text and self._cached_prefix_ids:
                self._record_prefix_hit("ram")
                return len(self._cached_prefix_ids)
            sys_ids = self.tokenizer.encode(sys_text)
            if not sys_ids:
                return 0
            if self._cached_prefix_ids == sys_ids:
                self._cached_prefix_text = sys_text
                self._record_prefix_hit("ram")
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
                    self._record_prefix_hit("ram")
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
