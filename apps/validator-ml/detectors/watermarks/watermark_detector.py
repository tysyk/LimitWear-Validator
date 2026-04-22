from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


def _bbox_from_rect(x: int, y: int, w: int, h: int) -> list[int]:
    return [int(x), int(y), int(x + w), int(y + h)]


def _bbox_overlap(b1: list[int], b2: list[int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])

    return inter / max(area1, 1)


def detect_watermark_like_regions(image, ocr_items, is_apparel: bool = False) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    image_area = float(w * h)

    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    enhanced = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)

    _, th = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    merged = cv2.morphologyEx(th, cv2.MORPH_CLOSE, merge_kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hits: List[Dict[str, Any]] = []

    for cnt in contours[:500]:
        x, y, cw, ch = cv2.boundingRect(cnt)
        bbox = _bbox_from_rect(x, y, cw, ch)

        overlaps = 0
        for ocr in ocr_items or []:
            obox = ocr.get("bbox")
            if isinstance(obox, list) and len(obox) == 4 and _bbox_overlap(bbox, obox) > 0.5:
                overlaps += 1

        if overlaps >= 1:
            continue

        area = float(cw * ch)
        area_ratio = area / max(image_area, 1.0)

        if area_ratio < 0.003:
            continue
        if area_ratio > 0.12:
            continue

        aspect = cw / max(ch, 1)
        if aspect < 2.8:
            continue

        roi = gray[y:y + ch, x:x + cw]
        if roi.size == 0:
            continue

        std_val = float(np.std(roi))
        mean_val = float(np.mean(roi))

        if std_val < 18:
            continue

        # близькість до краю
        center_x = x + cw / 2
        center_y = y + ch / 2
        dist_x = abs(center_x - w / 2) / max(w / 2, 1)
        dist_y = abs(center_y - h / 2) / max(h / 2, 1)
        centeredness = 1.0 - (dist_x + dist_y) / 2.0

        score = 0.0

        if aspect >= 3.5:
            score += 0.20
        if area_ratio >= 0.008:
            score += 0.15
        if mean_val > 140:
            score += 0.15
        if std_val >= 24:
            score += 0.10
        if centeredness > 0.45:
            score += 0.10

        # apparel-safe режим: на одязі значно жорсткіше
        if is_apparel:
            if area_ratio < 0.006:
                continue
            if aspect < 3.5:
                continue
            score -= 0.15

        score = max(0.0, min(score, 0.95))

        # відсікаємо слабкі хіти
        min_score = 0.65 if not is_apparel else 0.78
        if score < min_score:
            continue

        hits.append(
            {
                "bbox": bbox,
                "score": round(float(score), 4),
                "kind": "watermark_like_region",
                "meta": {
                    "aspect": round(float(aspect), 2),
                    "areaRatio": round(float(area_ratio), 5),
                    "std": round(std_val, 2),
                    "mean": round(mean_val, 2),
                    "centeredness": round(float(centeredness), 3),
                    "isApparelMode": bool(is_apparel),
                },
            }
        )

    hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
    return hits[:10]