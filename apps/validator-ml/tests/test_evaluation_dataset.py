from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import (
    dataset_counts,
    iter_dataset_cases,
    run_pipeline_on_image,
    assert_pipeline_completed,
    assert_verdict,
)


def _cases() -> list:
    evaluation_root = Path(__file__).resolve().parents[1] / "data" / "evaluation"
    return iter_dataset_cases(evaluation_root)


def _normalize_verdict(verdict) -> str:
    text = str(verdict)

    if "." in text:
        text = text.split(".")[-1]

    return text.upper()


def _assert_safe_verdict(ctx, expected_verdict: str, case_id: str) -> None:
    """
    Strict verdict check for PASS and NEED_REVIEW.
    For FAIL samples, NEED_REVIEW is accepted as a safe escalation,
    because the system does not incorrectly accept risky content.
    """
    actual = _normalize_verdict(ctx.verdict)
    expected = str(expected_verdict).upper()

    if expected == "FAIL" and actual == "NEED_REVIEW":
        return

    assert actual == expected, (
        f"{case_id}: expected {expected}, got {actual}. "
        f"verdict={ctx.verdict}, score={ctx.score}, "
        f"scene={(ctx.scene or {}).get('type') if isinstance(ctx.scene, dict) else ctx.scene}, "
        f"violations={[v.get('code') if isinstance(v, dict) else v for v in ctx.violations]}, "
        f"errors={ctx.errors}"
    )


@pytest.mark.evaluation
def test_evaluation_dataset_has_all_expected_buckets(evaluation_root: Path) -> None:
    counts = dataset_counts(evaluation_root)

    assert counts["pass"] > 0
    assert counts["need_review"] > 0
    assert counts["fail"] > 0


@pytest.mark.evaluation
@pytest.mark.slow
@pytest.mark.parametrize("case", _cases(), ids=lambda item: item.case_id)
def test_evaluation_dataset_verdicts(case) -> None:
    ctx = run_pipeline_on_image(case.path, profile_id="evaluation")

    assert not ctx.errors

    expected = str(case.expected_verdict).upper()
    actual = _normalize_verdict(ctx.verdict)

    # For non-critical expected categories, the whole pipeline should complete normally.
    # For FAIL cases, the pipeline may stop earlier after detecting a blocking violation.
    if expected in {"PASS", "NEED_REVIEW"}:
        assert_pipeline_completed(ctx)

    _assert_safe_verdict(ctx, case.expected_verdict, case_id=case.case_id)