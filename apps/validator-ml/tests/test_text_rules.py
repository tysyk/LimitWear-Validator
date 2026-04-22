from __future__ import annotations

import unittest

from conftest import base_ml, base_quality, base_scene, build_ctx, empty_detections
from pipeline.steps.rules import run as rules_run


class TextRuleTests(unittest.TestCase):
    def test_too_much_text_rule_flags_dense_layouts(self):
        ctx = build_ctx()
        ctx.quality = base_quality()
        ctx.scene = base_scene()
        ctx.ml = base_ml()
        ctx.detections = empty_detections()
        ctx.detections["ocr"] = [
            {"text": "one two three four five six"},
            {"text": "seven eight nine ten eleven twelve thirteen"},
        ]

        rules_run(ctx)

        text_rule = next(rule for rule in ctx.rule_results if rule["ruleId"] == "TOO_MUCH_TEXT")
        self.assertFalse(text_rule["passed"])
