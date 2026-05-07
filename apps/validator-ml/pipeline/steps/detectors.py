from __future__ import annotations

from detectors.lines.line_detector import detect_lines, estimate_skew
from detectors.logos.logo_visual_detector import detect_visual_logo_marks
from detectors.ocr.easyocr_detector import detect_ocr
from detectors.qr.qr_detector import detect_qr_codes
from detectors.watermarks.watermark_detector import detect_watermark_like_regions
from ip import analyze_ip_risk


def _filter_visual_logo_marks(
    marks,
    *,
    is_apparel: bool,
    apparel_confidence: float,
):
    if not marks:
        return []

    if not is_apparel:
        return marks[:10]

    filtered = []

    for mark in marks:
        score = float(mark.get("emblem_score", 0.0))
        area_ratio = float(mark.get("area_ratio", 0.0))
        center_dist = float(mark.get("center_dist", 1.0))

        keep = (
            score >= 0.90
            and area_ratio >= 0.08
            and center_dist <= 0.22
        )

        if not keep and apparel_confidence < 0.75:
            keep = (
                score >= 0.86
                and area_ratio >= 0.05
                and center_dist <= 0.30
            )

        if keep:
            filtered.append(mark)

    return filtered[:5]


def run(ctx) -> None:
    image = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr

    scene = ctx.scene or {}
    apparel_ml = (ctx.ml or {}).get("apparel", {})

    is_apparel = bool(scene.get("is_apparel", True))
    apparel_confidence = float(
        scene.get("apparel_confidence", apparel_ml.get("confidence", 0.0)) or 0.0
    )

    ocr_items = detect_ocr(image)

    qr_hits = detect_qr_codes(image)

    watermark_hits = detect_watermark_like_regions(
        image,
        ocr_items,
        is_apparel=is_apparel,
    )

    visual_logo_hits = _filter_visual_logo_marks(
        detect_visual_logo_marks(
            image=image,
            detections={
                "ocr": ocr_items,
                "qrMarks": qr_hits,
                "watermarkMarks": watermark_hits,
            },
        ),
        is_apparel=is_apparel,
        apparel_confidence=apparel_confidence,
    )

    lines = detect_lines(image)
    skew_meta = estimate_skew(lines)

    ip_result = analyze_ip_risk(
        detections={
            "ocr": ocr_items,
        }
    )

    ctx.detections = {
        **(ctx.detections or {}),
        "ocr": ocr_items,
        "lines": lines,
        "logoLikeMarks": [],
        "qrMarks": qr_hits,
        "watermarkMarks": watermark_hits,
        "visualLogoMarks": visual_logo_hits,
        "adultSafety": (ctx.ml or {}).get("adult_safety", {}),
        "ip": ip_result,
    }   

    ctx.debug["ocrCount"] = len(ocr_items)
    ctx.debug["lineCount"] = len(lines)
    ctx.debug["skew_angle_deg"] = skew_meta.get("angleDeg")
    ctx.debug["logoLikeCount"] = 0
    ctx.debug["qrCount"] = len(qr_hits)
    ctx.debug["watermarkCount"] = len(watermark_hits)
    ctx.debug["visualLogoCount"] = len(visual_logo_hits)

    ctx.set_debug_section(
        "detectors",
        {
            "ocrCount": len(ocr_items),
            "lineCount": len(lines),
            "logoLikeCount": 0,
            "qrCount": len(qr_hits),
            "watermarkCount": len(watermark_hits),
            "visualLogoCount": len(visual_logo_hits),
            "apparelMode": is_apparel,
            "apparelConfidence": round(apparel_confidence, 4),
            "skew": skew_meta,
        },
    )

    ctx.mark_step_done("detectors")