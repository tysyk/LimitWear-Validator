from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

try:
    import easyocr
except Exception:
    easyocr = None

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

    results = reader.readtext(image)
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


def _estimate_skew_deg(lines: List[Dict[str, Any]]) -> float | None:
    if not lines:
        return None

    angles: List[float] = []
    for line in lines:
        angle = float(line.get("angle", 0.0))
        normalized = angle

        while normalized <= -90:
            normalized += 180
        while normalized > 90:
            normalized -= 180

        if -45 <= normalized <= 45:
            angles.append(normalized)

    if not angles:
        return None

    return round(float(np.median(angles)), 2)


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

        bbox = [int(x), int(y), int(x + cw), int(y + ch)]
        hits.append(
            {
                "bbox": bbox,
                "score": 0.35,
                "kind": "logo_like_shape",
            }
        )

    return hits[:30]


def run(ctx) -> None:
    image = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr

    ocr_items = _run_easyocr(image)
    lines = _detect_lines(image)
    skew_angle_deg = _estimate_skew_deg(lines)
    logo_like = _detect_simple_logo_like_marks(image)
    qr_hits = detect_qr_codes(image)
    watermark_hits = detect_watermark_like_regions(image, ocr_items)
    ip_result = analyze_ip_risk(detections={"ocr": ocr_items})

    ctx.detections = {
        **(ctx.detections or {}),
        "ocr": ocr_items,
        "lines": lines,
        "logoLikeMarks": logo_like,
        "qrMarks": qr_hits,
        "watermarkMarks": watermark_hits,
        "ip": ip_result,
    }

    ctx.debug["ocrCount"] = len(ocr_items)
    ctx.debug["lineCount"] = len(lines)
    ctx.debug["skew_angle_deg"] = skew_angle_deg
    ctx.debug["logoLikeCount"] = len(logo_like)
    ctx.debug["qrCount"] = len(qr_hits)
    ctx.debug["watermarkCount"] = len(watermark_hits)

    ctx.mark_step_done("detectors")