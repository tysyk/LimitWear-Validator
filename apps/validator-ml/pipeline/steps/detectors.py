from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

try:
    import easyocr
except Exception:  # pragma: no cover - depends on local OCR runtime
    easyocr = None

from detectors.logos.logo_visual_detector import detect_visual_logo_marks
from detectors.qr.qr_detector import detect_qr_codes
from detectors.watermarks.watermark_detector import detect_watermark_like_regions
from ip import analyze_ip_risk


_EASY_OCR_READER = None


def _get_reader():
    global _EASY_OCR_READER
    if _EASY_OCR_READER is None and easyocr is not None:
        _EASY_OCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASY_OCR_READER


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> List[int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return [x1, y1, x2, y2]


def _run_easyocr(image: np.ndarray) -> List[Dict[str, Any]]:
    reader = _get_reader()
    if reader is None:
        return []

    try:
        results = reader.readtext(image)
    except Exception:
        return []

    h, w = image.shape[:2]
    out: List[Dict[str, Any]] = []

    for item in results:
        try:
            box, text, confidence = item
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            bbox = _clamp_bbox(min(xs), min(ys), max(xs), max(ys), w, h)

            value = str(text).strip()
            if not value:
                continue

            out.append(
                {
                    "text": value,
                    "confidence": float(confidence),
                    "bbox": bbox,
                }
            )
        except Exception:
            continue

    return out


def _detect_lines(image: np.ndarray) -> List[Dict[str, Any]]:
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

    out: List[Dict[str, Any]] = []
    if lines is None:
        return out

    for line in lines[:300]:
        x1, y1, x2, y2 = line[0].tolist()
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        out.append(
            {
                "p1": [int(x1), int(y1)],
                "p2": [int(x2), int(y2)],
                "length": round(length, 2),
                "angle": round(angle, 2),
            }
        )

    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.cumsum(weights_sorted)
    midpoint = cumulative[-1] / 2.0
    return float(values_sorted[np.searchsorted(cumulative, midpoint)])


def _estimate_skew(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def _detect_simple_logo_like_marks(image: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    area_img = float(w * h)

    hits: List[Dict[str, Any]] = []

    for cnt in contours[:400]:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = float(cw * ch)

        if area < area_img * 0.002:
            continue
        if area > area_img * 0.25:
            continue

        aspect = cw / max(ch, 1)
        if aspect < 0.2 or aspect > 5.0:
            continue

        cx = x + cw / 2.0
        cy = y + ch / 2.0
        center_dist = float(
            np.hypot(cx - (w / 2.0), cy - (h / 2.0))
            / max(np.hypot(w / 2.0, h / 2.0), 1.0)
        )

        bbox = [int(x), int(y), int(x + cw), int(y + ch)]
        hits.append(
            {
                "bbox": bbox,
                "score": 0.35,
                "kind": "logo_like_shape",
                "areaRatio": round(area / max(area_img, 1.0), 6),
                "centerDist": round(center_dist, 4),
            }
        )

    return hits[:30]


def _filter_logo_like_marks(
    marks: List[Dict[str, Any]],
    *,
    is_apparel: bool,
) -> List[Dict[str, Any]]:
    if not is_apparel:
        return marks

    filtered = []
    for mark in marks:
        area_ratio = float(mark.get("areaRatio", 0.0))
        if area_ratio >= 0.006:
            filtered.append(mark)
    return filtered[:15]


def _filter_visual_logo_marks(
    marks: List[Dict[str, Any]],
    *,
    is_apparel: bool,
    apparel_confidence: float,
) -> List[Dict[str, Any]]:
    if not is_apparel:
        return marks

    filtered: List[Dict[str, Any]] = []
    for mark in marks:
        score = float(mark.get("emblem_score", 0.0))
        area_ratio = float(mark.get("area_ratio", 0.0))
        center_dist = float(mark.get("center_dist", 1.0))

        keep = score >= 0.68
        if not keep and apparel_confidence < 0.75:
            keep = score >= 0.62 and area_ratio >= 0.01 and center_dist <= 0.45

        if keep:
            filtered.append(mark)

    return filtered


def run(ctx) -> None:
    image = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr
    scene = ctx.scene or {}
    apparel_ml = (ctx.ml or {}).get("apparel", {})
    is_apparel = bool(scene.get("is_apparel", True))
    apparel_confidence = float(
        scene.get("apparel_confidence", apparel_ml.get("confidence", 0.0)) or 0.0
    )

    ocr_items = _run_easyocr(image)
    lines = _detect_lines(image)
    skew_meta = _estimate_skew(lines)
    logo_like = _filter_logo_like_marks(
        _detect_simple_logo_like_marks(image),
        is_apparel=is_apparel,
    )
    qr_hits = detect_qr_codes(image)
    watermark_hits = detect_watermark_like_regions(
        image,
        ocr_items,
        is_apparel=is_apparel,
    )
    ip_result = analyze_ip_risk(detections={"ocr": ocr_items})
    visual_logo_hits = _filter_visual_logo_marks(
        detect_visual_logo_marks(image=image, detections={"ocr": ocr_items, "qrMarks": qr_hits, "watermarkMarks": watermark_hits}),
        is_apparel=is_apparel,
        apparel_confidence=apparel_confidence,
    )

    ctx.detections = {
        **(ctx.detections or {}),
        "ocr": ocr_items,
        "lines": lines,
        "logoLikeMarks": logo_like,
        "qrMarks": qr_hits,
        "watermarkMarks": watermark_hits,
        "visualLogoMarks": visual_logo_hits,
        "ip": ip_result,
    }

    ctx.debug["ocrCount"] = len(ocr_items)
    ctx.debug["lineCount"] = len(lines)
    ctx.debug["skew_angle_deg"] = skew_meta.get("angleDeg")
    ctx.debug["logoLikeCount"] = len(logo_like)
    ctx.debug["qrCount"] = len(qr_hits)
    ctx.debug["watermarkCount"] = len(watermark_hits)
    ctx.debug["visualLogoCount"] = len(visual_logo_hits)

    ctx.set_debug_section(
        "detectors",
        {
            "ocrCount": len(ocr_items),
            "lineCount": len(lines),
            "logoLikeCount": len(logo_like),
            "qrCount": len(qr_hits),
            "watermarkCount": len(watermark_hits),
            "visualLogoCount": len(visual_logo_hits),
            "apparelMode": is_apparel,
            "apparelConfidence": round(apparel_confidence, 4),
            "skew": skew_meta,
        },
    )

    ctx.mark_step_done("detectors")
