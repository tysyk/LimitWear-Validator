from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import run_pipeline_on_image, violation_ids


@pytest.mark.integration
@pytest.mark.slow
def test_confirmed_brand_crop_classifier_signal_requires_review(
    need_review_image: Path,
) -> None:
    ctx = run_pipeline_on_image(need_review_image)
    brand = ctx.ml["brand_crop_classifier"]

    assert ctx.verdict == "NEED_REVIEW"
    assert brand["isKnownBrand"] or brand["suspectedKnownBrand"]
    assert brand["brand_label"] not in {"unknown", "unknown_logo", "no_brand"}
    assert "KNOWN_BRAND_REVIEW" in violation_ids(ctx)


@pytest.mark.integration
@pytest.mark.slow
def test_unsupported_brand_crop_suspicion_does_not_pollute_pass_bucket(
    evaluation_root: Path,
) -> None:
    ctx = run_pipeline_on_image(evaluation_root / "pass" / "3.jpeg")
    brand = ctx.ml["brand_crop_classifier"]

    assert ctx.verdict == "PASS"
    assert brand["isKnownBrand"] is False
    assert brand["suspectedKnownBrand"] is False
    assert "KNOWN_BRAND_REVIEW" not in violation_ids(ctx)


@pytest.mark.integration
@pytest.mark.slow
def test_logo_like_without_brand_evidence_goes_to_review_not_fail(
    evaluation_root: Path,
) -> None:
    ctx = run_pipeline_on_image(evaluation_root / "need_review" / "11.png")

    assert ctx.verdict == "NEED_REVIEW"
    assert "VISUAL_LOGO_REVIEW" in violation_ids(ctx)
    assert all(
        not violation["meta"].get("blocking", False)
        for violation in ctx.violations
    )
