from __future__ import annotations

import unittest

from conftest import base_ml, base_quality, base_scene, build_ctx, empty_detections
from pipeline.steps.rules import run as rules_run


class MarginRuleTests(unittest.TestCase):
    def test_text_near_edge_is_reported(self):
        ctx = build_ctx(width=1000, height=1000)
        ctx.quality = base_quality()
        ctx.scene = base_scene()
        ctx.ml = base_ml()
        ctx.detections = empty_detections()
        ctx.detections["ocr"] = [
            {"text": "LIMIT", "bbox": [5, 30, 140, 90]},
        ]

        rules_run(ctx)

        edge_rules = [rule for rule in ctx.rule_results if rule["ruleId"] == "TEXT_NEAR_EDGE" and not rule["passed"]]
        self.assertEqual(len(edge_rules), 1)
