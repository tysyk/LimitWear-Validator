from __future__ import annotations

import unittest

try:
    import numpy as np

    from pipeline.context import PipelineContext
    from pipeline.steps.quality_gate import run as quality_gate_run
except Exception:  # pragma: no cover - depends on local cv2 runtime
    np = None
    PipelineContext = None
    quality_gate_run = None


@unittest.skipUnless(np is not None and PipelineContext is not None and quality_gate_run is not None, "OpenCV runtime is unavailable")
class QualityGateTests(unittest.TestCase):
    def test_blurry_image_keeps_pipeline_running(self):
        image = np.full((512, 512, 3), 127, dtype=np.uint8)
        ctx = PipelineContext(
            image_id="test-analysis",
            profile_id="default",
            bgr=image,
            width=512,
            height=512,
        )

        quality_gate_run(ctx)

        blur_rule = next(rule for rule in ctx.rule_results if rule["ruleId"] == "BLURRY_IMAGE")
        self.assertFalse(blur_rule["passed"])
        self.assertFalse(ctx.stop_pipeline)
        self.assertIn("quality_gate", ctx.steps_completed)
