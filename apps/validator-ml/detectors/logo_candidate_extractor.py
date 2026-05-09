from __future__ import annotations

from typing import Any

import numpy as np


def _crop_with_padding(
    image: np.ndarray,
    bbox: list[int],
    padding_ratio: float = 0.2,
) -> tuple[np.ndarray | None, list[int] | None]:
    h, w = image.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    if x2 <= x1 or y2 <= y1:
        return None, None

    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * padding_ratio)

    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(w, x2 + pad)
    py2 = min(h, y2 + pad)

    crop = image[py1:py2, px1:px2]

    if crop is None or crop.size == 0:
        return None, None

    return crop, [px1, py1, px2, py2]


def build_logo_candidates(
    image: np.ndarray,
    visual_logo_marks: list[dict[str, Any]],
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    candidates = []

    for index, mark in enumerate(visual_logo_marks[:max_candidates]):
        bbox = mark.get("bbox")

        if not bbox:
            continue

        crop, crop_bbox = _crop_with_padding(image, bbox)

        if crop is None or crop_bbox is None:
            continue

        candidates.append({
            "id": f"logo_candidate_{index + 1}",
            "bbox": crop_bbox,
            "original_bbox": bbox,
            "crop": crop,
            "source": mark.get("type", "visual_logo_mark"),
            "emblem_score": mark.get("emblem_score"),
            "area_ratio": mark.get("area_ratio"),
            "aspect_ratio": mark.get("aspect_ratio"),
            "center_dist": mark.get("center_dist"),
        })

    return candidates