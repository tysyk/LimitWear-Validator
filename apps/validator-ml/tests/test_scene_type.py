from __future__ import annotations

import unittest

try:
    import numpy as np

    from pipeline.context import PipelineContext
    from pipeline.steps.scene_type import run as scene_type_run
except Exception:  # pragma: no cover - depends on local cv2 runtime
    np = None
    PipelineContext = None
    scene_type_run = None


@unittest.skipUnless(np is not None and PipelineContext is not None and scene_type_run is not None, "OpenCV runtime is unavailable")
class SceneTypeTests(unittest.TestCase):
    def test_reliable_ml_apparel_sets_scene_type_to_apparel(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        ctx = PipelineContext(
            image_id="scene-type",
            profile_id="default",
            bgr=image,
            width=512,
            height=512,
        )
        ctx.bgr_used = image
        ctx.ml = {
            "apparel": {
                "label": "apparel",
                "confidence": 0.91,
                "isReliable": True,
            }
        }
        ctx.scene = {
            "is_apparel": True,
            "apparel_source": "ml",
            "apparel_confidence": 0.91,
        }
        ctx.detections = {"ocr": []}

        scene_type_run(ctx)

        self.assertEqual(ctx.scene["type"], "apparel")
        self.assertEqual(ctx.scene["type_source"], "ml_apparel")
        self.assertTrue(ctx.scene["is_apparel"])
