from __future__ import annotations

from detectors.easyocr_detector import detect_ocr
from detectors.line_detector import detect_lines, estimate_skew
from detectors.logo_candidate_extractor import build_logo_candidates
from detectors.logo_visual_detector import detect_visual_logo_marks
from detectors.qr_detector import detect_qr_codes
from detectors.watermark_detector import detect_watermark_like_regions
from ip import analyze_ip_risk


def _get_working_image(ctx):
    return ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr


def _get_apparel_meta(ctx) -> tuple[bool, float]:
    scene = ctx.scene or {}
    apparel_ml = (ctx.ml or {}).get("apparel", {})

    is_apparel = bool(scene.get("is_apparel", True))

    apparel_confidence = float(
        scene.get(
            "apparel_confidence",
            apparel_ml.get("confidence", 0.0),
        )
        or 0.0
    )

    return is_apparel, apparel_confidence


def _filter_visual_logo_marks(
    marks,
    *,
    is_apparel: bool,
    apparel_confidence: float,
):
    if not marks:
        return []

    marks = sorted(
        marks,
        key=lambda item: float(item.get("emblem_score", 0.0)),
        reverse=True,
    )

    return marks[:8]


def _public_logo_candidates(logo_candidates):
    return [
        {
            "id": item["id"],
            "bbox": item["bbox"],
            "original_bbox": item["original_bbox"],
            "source": item["source"],
            "emblem_score": item["emblem_score"],
            "area_ratio": item["area_ratio"],
            "aspect_ratio": item["aspect_ratio"],
            "center_dist": item["center_dist"],
        }
        for item in logo_candidates
    ]


def _logo_candidate_artifacts(logo_candidates):
    return [
        {
            "id": item["id"],
            "bbox": item["bbox"],
            "original_bbox": item["original_bbox"],
            "source": item["source"],
            "emblem_score": item["emblem_score"],
        }
        for item in logo_candidates
    ]


def run(ctx) -> None:
    image = _get_working_image(ctx)

    is_apparel, apparel_confidence = _get_apparel_meta(ctx)

    ocr_items = detect_ocr(image)
    qr_hits = detect_qr_codes(image)

    watermark_hits = detect_watermark_like_regions(
        image,
        ocr_items,
        is_apparel=is_apparel,
    )

    raw_visual_logo_hits = detect_visual_logo_marks(
        image=image,
        detections={
            "ocr": ocr_items,
            "qrMarks": qr_hits,
            "watermarkMarks": watermark_hits,
        },
    )

    visual_logo_hits = _filter_visual_logo_marks(
        raw_visual_logo_hits,
        is_apparel=is_apparel,
        apparel_confidence=apparel_confidence,
    )

    logo_candidates = build_logo_candidates(
        image=image,
        visual_logo_marks=visual_logo_hits,
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
        "logoCandidates": _public_logo_candidates(logo_candidates),
        "adultSafety": (ctx.ml or {}).get("adult_safety", {}),
        "ip": ip_result,
    }

    ctx.logo_candidate_crops = logo_candidates

    ctx.artifacts["logoCandidateCrops"] = _logo_candidate_artifacts(
        logo_candidates
    )

    ctx.debug["ocrCount"] = len(ocr_items)
    ctx.debug["lineCount"] = len(lines)
    ctx.debug["skew_angle_deg"] = skew_meta.get("angleDeg")
    ctx.debug["logoLikeCount"] = 0
    ctx.debug["qrCount"] = len(qr_hits)
    ctx.debug["watermarkCount"] = len(watermark_hits)
    ctx.debug["visualLogoRawCount"] = len(raw_visual_logo_hits)
    ctx.debug["visualLogoCount"] = len(visual_logo_hits)
    ctx.debug["logoCandidateCount"] = len(logo_candidates)

    ctx.set_debug_section(
        "detectors",
        {
            "ocrCount": len(ocr_items),
            "lineCount": len(lines),
            "logoLikeCount": 0,
            "qrCount": len(qr_hits),
            "watermarkCount": len(watermark_hits),
            "visualLogoRawCount": len(raw_visual_logo_hits),
            "visualLogoCount": len(visual_logo_hits),
            "logoCandidateCount": len(logo_candidates),
            "apparelMode": is_apparel,
            "apparelConfidence": round(apparel_confidence, 4),
            "skew": skew_meta,
        },
    )

    ctx.mark_step_done("detectors")