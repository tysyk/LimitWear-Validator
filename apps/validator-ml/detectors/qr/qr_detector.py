from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


def _normalize_points(points: np.ndarray) -> list[list[int]]:
    pts = []
    for p in points:
        x, y = p
        pts.append([int(x), int(y)])
    return pts


def _bbox_from_points(points: np.ndarray) -> list[int]:
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def detect_qr_codes(image) -> List[Dict[str, Any]]:
    detector = cv2.QRCodeDetector()

    hits: List[Dict[str, Any]] = []

    try:
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image)

        if retval and points is not None:
            for i, quad in enumerate(points):
                decoded_text = ""
                if decoded_info is not None and i < len(decoded_info):
                    decoded_text = str(decoded_info[i] or "").strip()

                quad_points = np.array(quad, dtype=np.float32)
                bbox = _bbox_from_points(quad_points)

                hits.append(
                    {
                        "bbox": bbox,
                        "points": _normalize_points(quad_points),
                        "decodedText": decoded_text,
                        "score": 0.95 if decoded_text else 0.75,
                        "kind": "qr_code",
                    }
                )
    except Exception:
        pass

    if hits:
        return hits

    try:
        decoded_text, points, _ = detector.detectAndDecode(image)
        if points is not None and len(points) > 0:
            quad_points = np.array(points[0], dtype=np.float32)
            hits.append(
                {
                    "bbox": _bbox_from_points(quad_points),
                    "points": _normalize_points(quad_points),
                    "decodedText": str(decoded_text or "").strip(),
                    "score": 0.95 if decoded_text else 0.75,
                    "kind": "qr_code",
                }
            )
    except Exception:
        pass

    return hits