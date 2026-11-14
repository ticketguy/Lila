"""M7: Explicit-request-only screen capture and visual perception broker.

Enforces M7 invariants:
- No screenshot occurs from Sentinel, model request alone, or background task.
- Requires a daemon-issued, HMAC-SHA256-signed short-lived owner gesture token
  originating from a native desktop gesture (never forgeable by model/tool text).
- Transient image objects in memory by default (no disk persistence unless
  explicitly saved by owner) with strict TTL expiry and pause/stop/discard.
- Stale-image action revalidation rejects actions targeting expired captures.
- Separates capabilities: `vision_observe`, `ocr_uia_extract`, and
  `computer_interaction` (never bundled into one unrestricted tool).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CaptureScope(str, Enum):
    WINDOW = "window"
    REGION = "region"


class PerceptionCapability(str, Enum):
    VISION_OBSERVE = "vision_observe"
    OCR_UIA_EXTRACT = "ocr_uia_extract"
    COMPUTER_INTERACTION = "computer_interaction"


class GestureTokenIssuer:
    """Issues and verifies short-lived HMAC-signed native owner gesture tokens."""

    def __init__(self, ttl_seconds: float = 15.0) -> None:
        self._secret = secrets.token_bytes(32)
        self._ttl_seconds = max(1.0, float(ttl_seconds))
        self._lock = threading.RLock()
        self._consumed: set[str] = set()

    def issue(self, scope: CaptureScope) -> str:
        nonce = secrets.token_urlsafe(16)
        exp = int(time.time() + self._ttl_seconds)
        msg = f"{scope.value}:{exp}:{nonce}"
        sig = hmac.new(self._secret, msg.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{msg}:{sig}"

    def verify_and_consume(self, token: str, expected_scope: CaptureScope) -> str:
        if not token or not isinstance(token, str):
            raise PermissionError("Missing owner_gesture_token.")
        parts = token.split(":")
        if len(parts) != 4:
            raise PermissionError("Malformed owner_gesture_token.")
        scope_str, exp_str, nonce, sig = parts
        if scope_str != expected_scope.value:
            raise PermissionError("Gesture token scope mismatch.")
        try:
            exp = int(exp_str)
        except ValueError as exc:
            raise PermissionError("Invalid gesture token expiry.") from exc
        if time.time() > exp:
            raise PermissionError("Expired owner_gesture_token.")
        msg = f"{scope_str}:{exp_str}:{nonce}"
        expected_sig = hmac.new(self._secret, msg.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            raise PermissionError("Forged or invalid owner_gesture_token.")
        with self._lock:
            if nonce in self._consumed:
                raise PermissionError("Replayed owner_gesture_token.")
            self._consumed.add(nonce)
            if len(self._consumed) > 1024:
                self._consumed.clear()
        return nonce


@dataclass
class TransientScreenObservation:
    capture_id: str
    scope: CaptureScope
    owner_gesture_id: str
    target_label: str
    width: int
    height: int
    image_digest: str
    created_at: float
    expires_at: float
    persisted: bool = False
    paused: bool = False
    discarded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_stale(self, now: Optional[float] = None) -> bool:
        t = time.monotonic() if now is None else now
        return self.discarded or t >= self.expires_at

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "scope": self.scope.value,
            "target_label": self.target_label,
            "width": self.width,
            "height": self.height,
            "image_digest": self.image_digest,
            "persisted": self.persisted,
            "paused": self.paused,
            "discarded": self.discarded,
            "stale": self.is_stale(),
            "ttl_remaining_s": max(0.0, round(self.expires_at - time.monotonic(), 2)),
        }


class VisualPerceptionBroker:
    """Manages transient screen observations and separates visual capabilities."""

    ALLOWED_INITIATORS = frozenset({"owner_gesture"})
    BLOCKED_INITIATORS = frozenset({"sentinel", "model_request", "background_task"})

    def __init__(self, default_ttl_s: float = 30.0) -> None:
        self._default_ttl_s = max(1.0, float(default_ttl_s))
        self._lock = threading.RLock()
        self._captures: dict[str, TransientScreenObservation] = {}
        self._raw_bytes: dict[str, bytes] = {}
        self._indicator_active = False
        self._paused = False
        self.gesture_issuer = GestureTokenIssuer()

    @property
    def indicator_active(self) -> bool:
        with self._lock:
            return self._indicator_active

    def issue_owner_gesture_token(self, scope: CaptureScope) -> str:
        return self.gesture_issuer.issue(scope)

    def capture_explicit(
        self,
        *,
        initiator: str,
        owner_gesture_id: Optional[str],
        scope: CaptureScope,
        target_label: str,
        image_bytes: bytes,
        width: int,
        height: int,
        ttl_s: Optional[float] = None,
        require_signed_token: bool = True,
    ) -> TransientScreenObservation:
        if initiator in self.BLOCKED_INITIATORS or initiator not in self.ALLOWED_INITIATORS:
            raise PermissionError(
                f"Screen capture denied for initiator={initiator!r}; explicit owner gesture required."
            )
        if not owner_gesture_id or not owner_gesture_id.strip():
            raise PermissionError("Missing owner_gesture_id; cannot capture without explicit gesture.")
        if require_signed_token:
            gesture_nonce = self.gesture_issuer.verify_and_consume(owner_gesture_id.strip(), scope)
        else:
            gesture_nonce = owner_gesture_id.strip()
        if self._paused:
            raise RuntimeError("Visual capture is currently paused by the owner.")
        if not image_bytes:
            raise ValueError("Empty image payload.")
        now = time.monotonic()
        ttl = self._default_ttl_s if ttl_s is None else max(1.0, float(ttl_s))
        cid = f"cap-{uuid.uuid4().hex[:12]}"
        digest = hashlib.sha256(image_bytes).hexdigest()
        obs = TransientScreenObservation(
            capture_id=cid,
            scope=scope,
            owner_gesture_id=gesture_nonce,
            target_label=target_label,
            width=int(width),
            height=int(height),
            image_digest=digest,
            created_at=now,
            expires_at=now + ttl,
            persisted=False,
        )
        with self._lock:
            self._captures[cid] = obs
            self._raw_bytes[cid] = bytes(image_bytes)
            self._indicator_active = True
        return obs

    def observe_vision_metadata(self, capture_id: str, expected_digest: Optional[str] = None) -> dict[str, Any]:
        obs = self.revalidate_for_action(capture_id, PerceptionCapability.VISION_OBSERVE, expected_digest)
        return {
            "capability": PerceptionCapability.VISION_OBSERVE.value,
            "capture": obs.to_public_dict(),
        }

    def extract_ocr_uia_text(self, capture_id: str, uia_text: str = "", expected_digest: Optional[str] = None) -> dict[str, Any]:
        obs = self.revalidate_for_action(capture_id, PerceptionCapability.OCR_UIA_EXTRACT, expected_digest)
        return {
            "capability": PerceptionCapability.OCR_UIA_EXTRACT.value,
            "capture_id": obs.capture_id,
            "image_digest": obs.image_digest,
            "extracted_text": uia_text.strip(),
        }

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            for cap in self._captures.values():
                cap.paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            for cap in self._captures.values():
                cap.paused = False

    def discard(self, capture_id: str) -> bool:
        with self._lock:
            obs = self._captures.get(capture_id)
            if obs is None:
                return False
            obs.discarded = True
            self._raw_bytes.pop(capture_id, None)
            self._refresh_indicator()
            return True

    def stop_all(self) -> int:
        with self._lock:
            count = len(self._captures)
            for obs in self._captures.values():
                obs.discarded = True
            self._raw_bytes.clear()
            self._indicator_active = False
            return count

    def revalidate_for_action(
        self,
        capture_id: str,
        capability: PerceptionCapability,
        expected_digest: Optional[str] = None,
    ) -> TransientScreenObservation:
        """Revalidate that a transient capture is fresh before vision/OCR/interaction."""
        with self._lock:
            obs = self._captures.get(capture_id)
            if obs is None:
                raise LookupError(f"Unknown capture_id: {capture_id}")
            if obs.is_stale():
                self._raw_bytes.pop(capture_id, None)
                self._refresh_indicator()
                raise PermissionError(
                    f"Stale screen capture {capture_id} cannot be used for {capability.value}."
                )
            if expected_digest and obs.image_digest != expected_digest:
                raise PermissionError("Screen capture digest mismatch during revalidation.")
            return obs

    def _refresh_indicator(self) -> None:
        self._indicator_active = any(not c.is_stale() for c in self._captures.values())
