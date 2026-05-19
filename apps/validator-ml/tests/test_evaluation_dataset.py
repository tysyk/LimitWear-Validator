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

    assert_pipeline_completed(ctx)
    assert not ctx.errors
    assert_verdict(ctx, case.expected_verdict, case_id=case.case_id)
