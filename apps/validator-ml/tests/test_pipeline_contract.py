from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import numpy as np

    from pipeline.context import PipelineContext
    from pipeline.runner import run_pipeline
except Exception:  # pragma: no cover - depends on local runtime
    np = None
    PipelineContext = None
    run_pipeline = None


def _step(name):
    def inner(ctx):
        ctx.mark_step_done(name)

    return inner


@unittest.skipUnless(np is not None and PipelineContext is not None and run_pipeline is not None, "Pipeline runtime is unavailable")
class PipelineContractTests(unittest.TestCase):
    def test_pipeline_runs_all_steps_without_exception(self):
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        ctx = PipelineContext(
            image_id="pipeline-contract",
            profile_id="default",
            bgr=image,
            width=320,
            height=320,
        )

        with patch("pipeline.runner.quality_gate", _step("quality_gate")):
            with patch("pipeline.runner.ml_apparel", _step("ml_apparel")):
                with patch("pipeline.runner.roi_extract", _step("roi_extract")):
                    with patch("pipeline.runner.detectors", _step("detectors")):
                        with patch("pipeline.runner.scene_type", _step("scene_type")):
                            with patch("pipeline.runner.moderation", _step("moderation")):
                                with patch("pipeline.runner.rules", _step("rules")):
                                    with patch("pipeline.runner.aggregate", _step("aggregate")):
                                        with patch("pipeline.runner.explain", _step("explain")):
                                            run_pipeline(ctx)

        self.assertEqual(
            ctx.steps_completed,
            [
                "quality_gate",
                "ml_apparel",
                "roi_extract",
                "detectors",
                "scene_type",
                "moderation",
                "rules",
                "aggregate",
                "explain",
            ],
        )
        self.assertFalse(ctx.errors)
