from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


def _safe_bbox(bbox: Any) -> BBox | None:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
    except Exception:
        return None


def _bbox_iou(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = float((x2 - x1) * (y2 - y1))
    area_a = float(max(1, (a[2] - a[0]) * (a[3] - a[1])))
    area_b = float(max(1, (b[2] - b[0]) * (b[3] - b[1])))
    union = area_a + area_b - inter
    return inter / max(union, 1.0)


def _bbox_overlap_ratio(inner: BBox, outer: BBox) -> float:
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = float((x2 - x1) * (y2 - y1))
    inner_area = float(max(1, (inner[2] - inner[0]) * (inner[3] - inner[1])))
    return inter / inner_area


def _normalize_center_distance(cx: float, cy: float, w: int, h: int) -> float:
    img_cx = w / 2.0
    img_cy = h / 2.0
    dx = cx - img_cx
    dy = cy - img_cy
    dist = (dx * dx + dy * dy) ** 0.5
    max_dist = ((img_cx * img_cx) + (img_cy * img_cy)) ** 0.5
    return float(dist / max(max_dist, 1.0))


def _collect_text_boxes(detections: Dict[str, Any]) -> List[BBox]:
    result: List[BBox] = []
    ocr = detections.get("ocr", []) if isinstance(detections, dict) else []

    if isinstance(ocr, dict):
        items = ocr.get("items", [])
    elif isinstance(ocr, list):
        items = ocr
    else:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = _safe_bbox(item.get("bbox"))
        if bbox:
            result.append(bbox)

    return result


def _collect_boxes(detections: Dict[str, Any], key: str) -> List[BBox]:
    result: List[BBox] = []
    items = detections.get(key, []) if isinstance(detections, dict) else []

    for item in items:
        bbox = _safe_bbox(item.get("bbox"))
        if bbox:
            result.append(bbox)

    return result


def _deduplicate_marks(marks: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    if not marks:
        return []

    marks = sorted(marks, key=lambda x: x.get("emblem_score", 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []

    for mark in marks:
        bbox_a = _safe_bbox(mark.get("bbox"))
        if not bbox_a:
            continue

        should_keep = True
        for existing in kept:
            bbox_b = _safe_bbox(existing.get("bbox"))
            if not bbox_b:
                continue
            if _bbox_iou(bbox_a, bbox_b) >= iou_threshold:
                should_keep = False
                break

        if should_keep:
            kept.append(mark)

    return kept


def detect_visual_logo_marks(
    image: np.ndarray,
    detections: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Heuristic visual logo detector.
    Ловить logo / emblem-like форми без тексту.
    """
    if image is None or image.size == 0:
        return []

    detections = detections or {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    image_area = float(max(1, w * h))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(blur)

    edges = cv2.Canny(norm, 80, 180)
    th = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.bitwise_or(edges, th)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=2)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_boxes = _collect_text_boxes(detections)
    qr_boxes = _collect_boxes(detections, "qrMarks")
    watermark_boxes = _collect_boxes(detections, "watermarkMarks")

    marks: List[Dict[str, Any]] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 120:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 12 or bh < 12:
            continue

        bbox = (x, y, x + bw, y + bh)
        bbox_area = float(max(1, bw * bh))
        area_ratio = bbox_area / image_area

        if area_ratio < 0.0004 or area_ratio > 0.18:
            continue

        aspect_ratio = float(bw) / max(float(bh), 1.0)
        if aspect_ratio > 5.5 or aspect_ratio < 0.18:
            continue

        perimeter = float(cv2.arcLength(cnt, True))
        if perimeter <= 0:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        extent = area / max(bbox_area, 1.0)
        circularity = (4.0 * np.pi * area) / max(perimeter * perimeter, 1.0)

        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        vertices = len(approx)

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        center_dist = _normalize_center_distance(cx, cy, w, h)

        text_overlap = 0.0
        for tb in text_boxes:
            text_overlap = max(text_overlap, _bbox_overlap_ratio(bbox, tb))
        if text_overlap > 0.45:
            continue

        qr_overlap = 0.0
        for qb in qr_boxes:
            qr_overlap = max(qr_overlap, _bbox_overlap_ratio(bbox, qb))
        if qr_overlap > 0.35:
            continue

        wm_overlap = 0.0
        for wb in watermark_boxes:
            wm_overlap = max(wm_overlap, _bbox_overlap_ratio(bbox, wb))
        if wm_overlap > 0.35:
            continue

        shape_score = 0.0

        if 0.45 <= solidity <= 0.98:
            shape_score += 0.22
        if 0.18 <= extent <= 0.90:
            shape_score += 0.18
        if 0.08 <= circularity <= 1.35:
            shape_score += 0.18
        if 3 <= vertices <= 24:
            shape_score += 0.18
        if 0.33 <= aspect_ratio <= 3.0:
            shape_score += 0.14
        if 0.001 <= area_ratio <= 0.08:
            shape_score += 0.10

        center_score = max(0.0, 1.0 - center_dist)
        emblem_score = 0.7 * shape_score + 0.3 * center_score

        if emblem_score < 0.46:
            continue

        marks.append(
            {
                "bbox": [x, y, x + bw, y + bh],
                "area_ratio": round(area_ratio, 6),
                "center_dist": round(center_dist, 4),
                "emblem_score": round(float(emblem_score), 4),
                "solidity": round(float(solidity), 4),
                "extent": round(float(extent), 4),
                "aspect_ratio": round(float(aspect_ratio), 4),
                "vertices": int(vertices),
                "type": "visual_logo_like",
            }
        )

    marks = _deduplicate_marks(marks, iou_threshold=0.45)

    filtered: List[Dict[str, Any]] = []
    for mark in marks:
        area_ratio = float(mark.get("area_ratio", 0.0))
        emblem_score = float(mark.get("emblem_score", 0.0))
        center_dist = float(mark.get("center_dist", 1.0))

        keep = False
        if emblem_score >= 0.68:
            keep = True
        elif emblem_score >= 0.58 and area_ratio >= 0.002:
            keep = True
        elif emblem_score >= 0.54 and area_ratio >= 0.01 and center_dist <= 0.55:
            keep = True

        if keep:
            filtered.append(mark)

    return filtered