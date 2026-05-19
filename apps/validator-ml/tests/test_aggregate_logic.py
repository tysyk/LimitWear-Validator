from __future__ import annotations

import pytest

from pipeline.steps.aggregate import run as aggregate
from pipeline.steps.rules import run as rules
from tests.helpers import (
    apparel_ml,
    apparel_scene,
    empty_detections,
    make_blank_context,
    minimal_quality,
    violation_ids,
)


def _prepared_context():
    ctx = make_blank_context()
    ctx.quality = minimal_quality()
    ctx.scene = apparel_scene()
    ctx.ml = apparel_ml()
    ctx.detections = empty_detections()
    ctx.moderation = {
        "ok": True,
        "blocked": False,
        "needsReview": False,
        "labels": [],
        "textSignals": {},
    }
    return ctx


@pytest.mark.unit
def test_clean_apparel_context_passes_after_rules_and_aggregate() -> None:
    ctx = _prepared_context()

    rules(ctx)
    aggregate(ctx)

    assert ctx.verdict == "PASS"
    assert ctx.score == 100
    assert not ctx.violations


@pytest.mark.unit
def test_visual_logo_without_confirmed_brand_moves_to_review_not_fail() -> None:
    ctx = _prepared_context()
    ctx.ml["logo_presence"] = {
        "label": "logo",
        "confidence": 0.98,
        "isLogo": True,
        "isReliable": True,
    }
    ctx.ml["brand_crop_classifier"] = {
        "label": "no_brand",
        "brand_label": "no_brand",
        "confidence": 0.99,
        "isKnownBrand": False,
        "suspectedKnownBrand": False,
    }
    ctx.detections["visualLogoMarks"] = [
        {"bbox": [180, 120, 240, 180], "emblem_score": 0.85},
        {"bbox": [245, 120, 305, 180], "emblem_score": 0.85},
        {"bbox": [310, 120, 370, 180], "emblem_score": 0.85},
    ]

    rules(ctx)
    aggregate(ctx)

    assert ctx.verdict == "NEED_REVIEW"
    assert "VISUAL_LOGO_REVIEW" in violation_ids(ctx)
    assert all(
        not violation["meta"].get("blocking", False)
        for violation in ctx.violations
    )


@pytest.mark.unit
def test_confirmed_ip_or_moderation_block_remains_fail() -> None:
    ctx = _prepared_context()
    ctx.moderation = {
        "ok": False,
        "blocked": True,
        "needsReview": False,
        "labels": [{"label": "sexual_text", "blocked": True}],
    }

    aggregate(ctx)

    assert ctx.verdict == "FAIL"
    assert ctx.score == 0


@pytest.mark.unit
def test_high_confidence_non_apparel_is_blocking_business_decision() -> None:
    ctx = _prepared_context()
    ctx.scene = apparel_scene(is_apparel=False, confidence=0.98)
    ctx.ml = apparel_ml(label="non_apparel", confidence=0.98)

    rules(ctx)
    aggregate(ctx)

    assert ctx.verdict == "FAIL"
    assert "NON_APPAREL" in violation_ids(ctx)


@pytest.mark.unit
def test_uncertain_non_apparel_goes_to_need_review() -> None:
    ctx = _prepared_context()
    ctx.scene = apparel_scene(is_apparel=False, confidence=0.72)
    ctx.ml = apparel_ml(label="non_apparel", confidence=0.72)

    rules(ctx)
    aggregate(ctx)

    assert ctx.verdict == "NEED_REVIEW"
    assert ctx.debug["aggregate"]["uncertainApparel"]
