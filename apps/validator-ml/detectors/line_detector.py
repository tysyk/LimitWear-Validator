from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


def detect_lines(image: np.ndarray) -> List[Dict[str, Any]]:
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 180)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=40,
        maxLineGap=8,
    )

    output: List[Dict[str, Any]] = []

    if lines is None:
        return output

    for line in lines[:300]:
        x1, y1, x2, y2 = line[0].tolist()

        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        output.append(
            {
                "p1": [int(x1), int(y1)],
                "p2": [int(x2), int(y2)],
                "length": round(length, 2),
                "angle": round(angle, 2),
            }
        )

    return output


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)

    values_sorted = values[order]
    weights_sorted = weights[order]

    cumulative = np.cumsum(weights_sorted)
    midpoint = cumulative[-1] / 2.0

    return float(values_sorted[np.searchsorted(cumulative, midpoint)])


def estimate_skew(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not lines:
        return {
            "angleDeg": None,
            "supportLines": 0,
            "candidateLines": 0,
            "confidence": 0.0,
            "spread": None,
        }

    angles: List[float] = []
    weights: List[float] = []

    for line in lines:
        angle = float(line.get("angle", 0.0))
        length = float(line.get("length", 0.0))

        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180

        if length < 80 or abs(angle) > 35:
            continue

        angles.append(angle)
        weights.append(length)

    if len(angles) < 4:
        return {
            "angleDeg": None,
            "supportLines": len(angles),
            "candidateLines": len(angles),
            "confidence": 0.0,
            "spread": None,
        }

    angle_array = np.array(angles, dtype=np.float32)
    weight_array = np.array(weights, dtype=np.float32)

    median_angle = _weighted_median(angle_array, weight_array)

    support_mask = np.abs(angle_array - median_angle) <= 5.0
    support_lines = int(np.sum(support_mask))

    if support_lines == 0:
        return {
            "angleDeg": None,
            "supportLines": 0,
            "candidateLines": len(angles),
            "confidence": 0.0,
            "spread": None,
        }

    support_angles = angle_array[support_mask]
    support_weights = weight_array[support_mask]

    support_ratio = float(np.sum(support_weights) / max(np.sum(weight_array), 1.0))
    spread = float(np.median(np.abs(support_angles - median_angle)))

    confidence = (
        0.45 * support_ratio
        + 0.35 * min(1.0, support_lines / 10.0)
        + 0.20 * max(0.0, 1.0 - min(spread / 6.0, 1.0))
    )

    if support_lines < 5:
        confidence *= 0.65

    return {
        "angleDeg": round(float(median_angle), 2),
        "supportLines": support_lines,
        "candidateLines": len(angles),
        "confidence": round(float(min(confidence, 0.99)), 4),
        "spread": round(spread, 4),
    }