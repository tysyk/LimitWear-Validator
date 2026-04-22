from __future__ import annotations

import unittest

from conftest import base_ml, base_quality, base_scene, build_ctx, empty_detections
from pipeline.steps.rules import run as rules_run


class GeometryRuleTests(unittest.TestCase):
    def test_skew_rule_is_relaxed_for_normal_apparel_images(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene(is_apparel=True, confidence=0.92)
        ctx.ml = base_ml(label="apparel", confidence=0.92)
        ctx.detections = empty_detections()
        ctx.debug = {
            "detectors": {
                "skew": {
                    "angleDeg": 9.0,
                    "supportLines": 7,
                    "confidence": 0.78,
                }
            }
        }

        rules_run(ctx)

        skew_rule = next(rule for rule in ctx.rule_results if rule["ruleId"] == "HIGH_SKEW")
        self.assertTrue(skew_rule["passed"])

    def test_skew_rule_still_flags_non_apparel_geometry_issues(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene(is_apparel=False, scene_type="poster_like", confidence=0.90)
        ctx.ml = base_ml(label="non_apparel", confidence=0.90)
        ctx.detections = empty_detections()
        ctx.debug = {
            "detectors": {
                "skew": {
                    "angleDeg": 9.0,
                    "supportLines": 7,
                    "confidence": 0.78,
                }
            }
        }

        rules_run(ctx)

        skew_rule = next(rule for rule in ctx.rule_results if rule["ruleId"] == "HIGH_SKEW")
        self.assertFalse(skew_rule["passed"])
