from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import (
    EXPECTED_PIPELINE_STEPS,
    assert_pipeline_completed,
    assert_response_contract,
    run_pipeline_on_image,
)


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_execution_on_real_image(
    pass_image: Path,
    response_contract_keys: set[str],
) -> None:
    ctx = run_pipeline_on_image(pass_image)

    assert_pipeline_completed(ctx)
    assert ctx.steps_completed == EXPECTED_PIPELINE_STEPS
    assert not ctx.errors
    assert ctx.verdict == "PASS"
    assert "apparel" in ctx.ml
    assert "apparel_type" in ctx.ml
    assert "brand_crop_classifier" in ctx.ml
    assert "adult_safety" in ctx.ml
    assert ctx.scene["type"] == "apparel"

    body = ctx.to_response()
    assert_response_contract(body, response_contract_keys)


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_keeps_ml_as_signal_and_aggregate_as_decision(
    need_review_image: Path,
) -> None:
    ctx = run_pipeline_on_image(need_review_image)

    assert_pipeline_completed(ctx)
    assert not ctx.errors
    assert ctx.verdict == "NEED_REVIEW"
    assert ctx.ml["apparel"]["label"] == "apparel"
    assert any(not rule["passed"] for rule in ctx.rule_results)

    body = ctx.to_response()
    assert body["summary"]["decision"] == "NEED_REVIEW"
    assert body["summary"]["apparelSignal"]["label"] == "apparel"
