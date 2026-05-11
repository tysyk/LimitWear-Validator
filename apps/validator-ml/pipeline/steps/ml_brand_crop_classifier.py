from __future__ import annotations

from pathlib import Path

import cv2

from core.config import (
    BRAND_CONFIDENCE_THRESHOLD,
    BRAND_SUSPECTED_THRESHOLD,
    DEBUG_BRAND_CROPS_DIR,
    SAVE_DEBUG_BRAND_CROPS,
)
from ml.brand_crop_classifier.inference_brand_crop_classifier import (
    predict_brand_crop_classifier,
)


def _get_logo_detections(ctx):
    logo_candidates = getattr(ctx, "logo_candidate_crops", None)

    if not logo_candidates:
        logo_candidates = (ctx.detections or {}).get("logoCandidates", [])

    detections = []

    for item in logo_candidates or []:
        crop = item.get("crop")

        if crop is None:
            continue

        detections.append(
            {
                "id": item.get("id"),
                "bbox": item.get("bbox"),
                "original_bbox": item.get("original_bbox"),
                "confidence": item.get("emblem_score", item.get("confidence", 0.5)),
                "emblem_score": item.get("emblem_score", item.get("confidence", 0.5)),
                "source": item.get("source", "logo_candidate_crop"),
                "crop": crop,
            }
        )

    return detections


def _save_debug_crops(ctx, logo_detections) -> None:
    if not SAVE_DEBUG_BRAND_CROPS:
        return

    output_dir = Path(DEBUG_BRAND_CROPS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_id = getattr(ctx, "image_id", "unknown")

    for index, item in enumerate(logo_detections, start=1):
        crop = item.get("crop")

        if crop is None:
            continue

        path = output_dir / f"{image_id}_brand_crop_{index}.png"
        cv2.imwrite(str(path), crop)


def _fallback_result(reason: str, error: str | None = None):
    result = {
        "label": "unknown",
        "brand_label": "unknown",
        "raw_label": "unknown",
        "confidence": 0.0,
        "isReliable": False,
        "isKnownBrand": False,
        "threshold": BRAND_CONFIDENCE_THRESHOLD,
        "suspectedKnownBrand": False,
        "suspectedBrandLabel": None,
        "suspectedBrandConfidence": 0.0,
        "suspectedBrandThreshold": BRAND_SUSPECTED_THRESHOLD,
        "source": "ml_brand_crop_classifier",
        "reason": reason,
        "crop_results": [],
    }

    if error:
        result["error"] = error

    return result


def run(ctx):
    if ctx.ml is None:
        ctx.ml = {}

    try:
        logo_detections = _get_logo_detections(ctx)

        _save_debug_crops(ctx, logo_detections)

        result = predict_brand_crop_classifier(
            logo_candidates=logo_detections,
        )

        ctx.ml["brand_crop_classifier"] = result

        ctx.debug["brandCropClassifier"] = {
            "candidateCount": len(logo_detections),
            "resultLabel": result.get("label"),
            "brandLabel": result.get("brand_label"),
            "confidence": result.get("confidence"),
            "isKnownBrand": result.get("isKnownBrand"),
            "suspectedKnownBrand": result.get("suspectedKnownBrand"),
            "suspectedBrandLabel": result.get("suspectedBrandLabel"),
            "suspectedBrandConfidence": result.get("suspectedBrandConfidence"),
            "debugCropsSaved": SAVE_DEBUG_BRAND_CROPS,
        }

    except Exception as error:
        ctx.ml["brand_crop_classifier"] = _fallback_result(
            reason="brand_crop_classifier_error",
            error=str(error),
        )

        ctx.add_error(
            "ml_brand_crop_classifier",
            str(error),
            critical=False,
        )

        ctx.set_debug_section(
            "brandCropClassifier",
            {
                "candidateCount": 0,
                "resultLabel": "unknown",
                "brandLabel": "unknown",
                "confidence": 0.0,
                "isKnownBrand": False,
                "suspectedKnownBrand": False,
                "error": str(error),
            },
        )

    return ctx