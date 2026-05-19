from __future__ import annotations

from pathlib import Path

import pytest

from detectors.easyocr_detector import detect_ocr
from tests.helpers import load_bgr_image


@pytest.mark.integration
@pytest.mark.slow
def test_ocr_detector_extracts_real_text_from_evaluation_image(
    evaluation_root: Path,
) -> None:
    image = load_bgr_image(evaluation_root / "need_review" / "6.jpg")
    ocr_items = detect_ocr(image)
    normalized_text = " ".join(
        str(item.get("text", "")).lower()
        for item in ocr_items
    )

    assert ocr_items
    assert "sex" in normalized_text
