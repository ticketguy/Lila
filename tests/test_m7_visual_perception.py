import unittest

from src.harness.visual_perception import (
    CaptureScope,
    PerceptionCapability,
    VisualPerceptionBroker,
)


class M7VisualPerceptionTests(unittest.TestCase):
    def test_blocks_sentinel_model_and_background_initiators_and_forged_tokens(self):
        broker = VisualPerceptionBroker(default_ttl_s=10.0)
        tok = broker.issue_owner_gesture_token(CaptureScope.REGION)
        for bad_initiator in ("sentinel", "model_request", "background_task"):
            with self.assertRaises(PermissionError):
                broker.capture_explicit(
                    initiator=bad_initiator,
                    owner_gesture_id=tok,
                    scope=CaptureScope.REGION,
                    target_label="editor",
                    image_bytes=b"png-bytes",
                    width=100,
                    height=100,
                )
        # Forged / model-supplied gesture string must be rejected
        for forged in ("", "model-forged-gesture-123", "region:9999999999:abc:deadbeef"):
            with self.assertRaises(PermissionError):
                broker.capture_explicit(
                    initiator="owner_gesture",
                    owner_gesture_id=forged,
                    scope=CaptureScope.REGION,
                    target_label="editor",
                    image_bytes=b"png-bytes",
                    width=100,
                    height=100,
                )

    def test_explicit_capture_is_transient_and_stale_revalidation_fails(self):
        broker = VisualPerceptionBroker(default_ttl_s=1.0)
        tok = broker.issue_owner_gesture_token(CaptureScope.REGION)
        obs = broker.capture_explicit(
            initiator="owner_gesture",
            owner_gesture_id=tok,
            scope=CaptureScope.REGION,
            target_label="Terminal",
            image_bytes=b"sample-frame",
            width=640,
            height=360,
            ttl_s=1.0,
        )
        self.assertFalse(obs.persisted)
        self.assertTrue(broker.indicator_active)

        # Replay of consumed token must fail
        with self.assertRaises(PermissionError):
            broker.capture_explicit(
                initiator="owner_gesture",
                owner_gesture_id=tok,
                scope=CaptureScope.REGION,
                target_label="Terminal",
                image_bytes=b"sample-frame",
                width=640,
                height=360,
            )

        # Fresh capture can be revalidated for separate capabilities
        meta = broker.observe_vision_metadata(obs.capture_id)
        self.assertEqual(meta["capability"], "vision_observe")
        ocr = broker.extract_ocr_uia_text(obs.capture_id, uia_text="Ready")
        self.assertEqual(ocr["extracted_text"], "Ready")

        # Discarding clears indicator and blocks stale action
        self.assertTrue(broker.discard(obs.capture_id))
        self.assertFalse(broker.indicator_active)
        with self.assertRaises(PermissionError):
            broker.revalidate_for_action(obs.capture_id, PerceptionCapability.COMPUTER_INTERACTION)

    def test_pause_and_stop_all(self):
        broker = VisualPerceptionBroker()
        broker.pause()
        tok1 = broker.issue_owner_gesture_token(CaptureScope.WINDOW)
        with self.assertRaises(RuntimeError):
            broker.capture_explicit(
                initiator="owner_gesture",
                owner_gesture_id=tok1,
                scope=CaptureScope.WINDOW,
                target_label="Browser",
                image_bytes=b"123",
                width=10,
                height=10,
            )
        broker.resume()
        tok2 = broker.issue_owner_gesture_token(CaptureScope.WINDOW)
        broker.capture_explicit(
            initiator="owner_gesture",
            owner_gesture_id=tok2,
            scope=CaptureScope.WINDOW,
            target_label="Browser",
            image_bytes=b"123",
            width=10,
            height=10,
        )
        self.assertEqual(broker.stop_all(), 1)
        self.assertFalse(broker.indicator_active)


if __name__ == "__main__":
    unittest.main()
