from __future__ import annotations

from core.config import (
    CHEST_LOGO_FALLBACK_ENABLED,
    CHEST_LOGO_FALLBACK_SCORE,
    CHEST_LOGO_ZONES,
)
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


def _clip_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def _make_candidate_from_bbox(
    *,
    image,
    candidate_id: str,
    bbox,
    source: str,
    emblem_score: float,
):
    height, width = image.shape[:2]

    clipped = _clip_bbox(bbox, width, height)

    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    crop = image[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    box_width = x2 - x1
    box_height = y2 - y1

    image_area = max(1, width * height)
    area_ratio = round((box_width * box_height) / image_area, 6)
    aspect_ratio = round(box_width / max(1, box_height), 4)

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    image_center_x = width / 2
    image_center_y = height / 2

    center_dist = (
        ((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2) ** 0.5
        / max(width, height)
    )

    return {
        "id": candidate_id,
        "bbox": clipped,
        "original_bbox": clipped,
        "source": source,
        "emblem_score": emblem_score,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "center_dist": round(center_dist, 4),
        "crop": crop,
    }


def _bbox_from_ratio(width: int, height: int, bbox_ratio):
    x1, y1, x2, y2 = bbox_ratio

    return [
        width * float(x1),
        height * float(y1),
        width * float(x2),
        height * float(y2),
    ]


def _build_chest_logo_candidates(
    *,
    image,
    is_apparel: bool,
):
    if (
        image is None
        or not is_apparel
        or not CHEST_LOGO_FALLBACK_ENABLED
    ):
        return []

    height, width = image.shape[:2]

    candidates = []

    for zone in CHEST_LOGO_ZONES:
        bbox = _bbox_from_ratio(
            width=width,
            height=height,
            bbox_ratio=zone["bbox_ratio"],
        )

        candidate = _make_candidate_from_bbox(
            image=image,
            candidate_id=zone["id"],
            bbox=bbox,
            source=zone["source"],
            emblem_score=CHEST_LOGO_FALLBACK_SCORE,
        )

        if candidate is not None:
            candidates.append(candidate)

    return candidates


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

    chest_logo_candidates = _build_chest_logo_candidates(
        image=image,
        is_apparel=is_apparel,
    )

    logo_candidates.extend(chest_logo_candidates)

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
    ctx.debug["chestLogoCandidateCount"] = len(chest_logo_candidates)

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
            "chestLogoCandidateCount": len(chest_logo_candidates),
            "chestFallbackEnabled": CHEST_LOGO_FALLBACK_ENABLED,
            "apparelMode": is_apparel,
            "apparelConfidence": round(apparel_confidence, 4),
            "skew": skew_meta,
        },
    )

    ctx.mark_step_done("detectors")