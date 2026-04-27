from __future__ import annotations

import unittest

from conftest import base_ml, base_quality, base_scene, build_ctx, empty_detections
from pipeline.steps.aggregate import run as aggregate_run
from pipeline.steps.rules import run as rules_run


class PolicyDecisionTests(unittest.TestCase):
    def _run_rules_and_aggregate(self, ctx):
        rules_run(ctx)
        aggregate_run(ctx)
        return ctx

    def test_valid_apparel_without_violations_is_not_fail(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene(is_apparel=True, scene_type="apparel", confidence=0.95)
        ctx.ml = base_ml(label="apparel", confidence=0.95)
        ctx.detections = empty_detections()

        self._run_rules_and_aggregate(ctx)

        self.assertNotEqual(ctx.verdict, "FAIL")

    def test_visual_logo_like_without_ip_is_not_fail(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene(is_apparel=True, scene_type="apparel", confidence=0.95)
        ctx.ml = base_ml(label="apparel", confidence=0.95)
        ctx.detections = empty_detections()
        ctx.detections["visualLogoMarks"] = [
            {
                "bbox": [300, 300, 700, 500],
                "emblem_score": 0.86,
                "area_ratio": 0.05,
                "center_dist": 0.20,
            }
        ]

        self._run_rules_and_aggregate(ctx)

        self.assertIn(ctx.verdict, {"WARN", "NEED_REVIEW"})
        self.assertNotEqual(ctx.verdict, "FAIL")

    def test_confirmed_ip_risk_fails(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene(is_apparel=True, scene_type="apparel", confidence=0.95)
        ctx.ml = base_ml(label="apparel", confidence=0.95)
        ctx.detections = empty_detections()
        ctx.detections["ip"] = {
            "blocked": True,
            "needsReview": False,
            "brandTextHits": [{"keyword": "nike", "matchKind": "exact_substring"}],
            "exactHits": [{"type": "brand", "keyword": "nike", "matchKind": "exact_substring"}],
            "suspiciousHits": [],
        }

        self._run_rules_and_aggregate(ctx)

        self.assertEqual(ctx.verdict, "FAIL")

    def test_weak_watermark_on_apparel_is_not_fail(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene(is_apparel=True, scene_type="apparel", confidence=0.95)
        ctx.ml = base_ml(label="apparel", confidence=0.95)
        ctx.detections = empty_detections()
        ctx.detections["watermarkMarks"] = [
            {
                "bbox": [200, 200, 520, 270],
                "score": 0.80,
                "meta": {"areaRatio": 0.006, "centeredness": 0.35},
            }
        ]

        self._run_rules_and_aggregate(ctx)

        self.assertNotEqual(ctx.verdict, "FAIL")

    def test_low_support_skew_on_apparel_is_not_fail(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene(is_apparel=True, scene_type="apparel", confidence=0.95)
        ctx.ml = base_ml(label="apparel", confidence=0.95)
        ctx.detections = empty_detections()
        ctx.debug = {"detectors": {"skew": {"angleDeg": 16.0, "supportLines": 2, "confidence": 0.30}}}

        self._run_rules_and_aggregate(ctx)

        skew_rule = next(rule for rule in ctx.rule_results if rule["ruleId"] == "HIGH_SKEW")
        self.assertTrue(skew_rule["passed"])
        self.assertNotEqual(ctx.verdict, "FAIL")
