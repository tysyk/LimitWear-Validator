from __future__ import annotations

import unittest

from conftest import base_ml, base_quality, base_scene, build_ctx, empty_detections
from pipeline.steps.aggregate import run as aggregate_run


class AggregateStepTests(unittest.TestCase):
    def test_scene_type_metadata_does_not_override_ml_apparel_signal(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene(is_apparel=True, scene_type="poster_like", confidence=0.93)
        ctx.ml = base_ml(label="apparel", confidence=0.93)
        ctx.detections = empty_detections()

        aggregate_run(ctx)

        self.assertEqual(ctx.verdict, "PASS")
        self.assertNotIn("need_review_reason", ctx.debug)

    def test_non_apparel_ml_signal_moves_result_to_manual_review(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene(is_apparel=False, scene_type="apparel_candidate", confidence=0.88)
        ctx.ml = base_ml(label="non_apparel", confidence=0.88)
        ctx.detections = empty_detections()

        aggregate_run(ctx)

        self.assertEqual(ctx.verdict, "FAIL")
        self.assertIn("non-apparel", ctx.debug.get("fail_reason", ""))
