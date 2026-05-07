from __future__ import annotations

from typing import Any, Dict, List

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None


_EASY_OCR_READER = None


def _get_reader():
    global _EASY_OCR_READER

    if easyocr is None:
        return None

    if _EASY_OCR_READER is None:
        _EASY_OCR_READER = easyocr.Reader(["en"], gpu=False)

    return _EASY_OCR_READER


def _clamp_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> List[int]:
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))

    return [x1, y1, x2, y2]


def detect_ocr(image) -> List[Dict[str, Any]]:
    if image is None:
        return []

    reader = _get_reader()

    if reader is None:
        return []

    try:
        results = reader.readtext(image)
    except Exception:
        return []

    height, width = image.shape[:2]
    items: List[Dict[str, Any]] = []

    for item in results:
        try:
            box, text, confidence = item

            xs = [p[0] for p in box]
            ys = [p[1] for p in box]

            bbox = _clamp_bbox(
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                width,
                height,
            )

            value = str(text).strip()

            if not value:
                continue

            items.append(
                {
                    "text": value,
                    "confidence": float(confidence),
                    "bbox": bbox,
                }
            )

        except Exception:
            continue

    return items