from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from main import app
except Exception:  # pragma: no cover - depends on local runtime
    cv2 = None
    np = None
    TestClient = None
    app = None


def _fake_pipeline(ctx) -> None:
    ctx.quality = {
        "passed_resolution": True,
        "passed_blur": True,
        "quality_score": 1.0,
        "blur_score": 100.0,
    }
    ctx.scene = {
        "type": "apparel_candidate",
        "is_apparel": True,
        "apparel_source": "ml",
        "apparel_confidence": 0.94,
    }
    ctx.ml = {
        "apparel": {
            "label": "apparel",
            "confidence": 0.94,
            "isReliable": True,
            "source": "ml",
        }
    }
    ctx.detections = {
        "ocr": [],
        "lines": [],
        "ip": {"exactHits": [], "suspiciousHits": [], "blocked": False, "needsReview": False},
        "logoLikeMarks": [],
        "visualLogoMarks": [],
        "qrMarks": [],
        "watermarkMarks": [],
    }
    ctx.score = 94
    ctx.set_verdict("PASS")
    ctx.explain = ["Автоматична перевірка пройдена, критичних порушень не виявлено."]
    ctx.mark_step_done("fake_pipeline")


@unittest.skipUnless(
    TestClient is not None and app is not None and cv2 is not None and np is not None,
    "API runtime is unavailable",
)
class AnalyzeApiTests(unittest.TestCase):
    @patch("api.routes.run_pipeline", side_effect=_fake_pipeline)
    def test_analyze_endpoint_returns_presentation_ready_summary(self, _mocked_pipeline):
        client = TestClient(app)
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)

        response = client.post(
            "/analyze",
            files={"file": ("sample.png", encoded.tobytes(), "image/png")},
            data={"profile_id": "demo"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["profileId"], "demo")
        self.assertEqual(body["verdict"], "PASS")
        self.assertIn("summary", body)
        self.assertEqual(body["summary"]["apparelSignal"]["label"], "apparel")
        self.assertEqual(body["summary"]["decision"], "PASS")
