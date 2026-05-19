from __future__ import annotations

import pytest

from moderation.moderation_service import moderate_image_and_text


def _moderate_text(text: str) -> dict:
    return moderate_image_and_text(
        scene={"type": "apparel"},
        detections={"ocr": [{"text": text}] if text else []},
        quality={"blur_score": 100.0},
    )


@pytest.mark.unit
def test_empty_ocr_text_does_not_block_moderation() -> None:
    result = _moderate_text("")

    assert result["blocked"] is False
    assert result["needsReview"] is False
    assert result["labels"] == []


@pytest.mark.unit
def test_weak_adult_text_signal_needs_review_not_block() -> None:
    result = _moderate_text("sexy summer print")

    assert result["blocked"] is False
    assert result["needsReview"] is True
    assert "sexual_text_review" in {
        label["label"]
        for label in result["labels"]
    }


@pytest.mark.unit
@pytest.mark.parametrize("text", ["nude print", "nazi slogan", "self harm"])
def test_strong_unsafe_text_signal_blocks(text: str) -> None:
    result = _moderate_text(text)

    assert result["blocked"] is True
    assert result["needsReview"] is False
    assert result["blockedReasons"]


@pytest.mark.unit
def test_brand_words_from_ocr_become_review_signal_not_content_block() -> None:
    result = _moderate_text("Nike limited edition")

    assert result["blocked"] is False
    assert result["needsReview"] is True
    assert result["textSignals"]["brandHits"] == ["nike"]
