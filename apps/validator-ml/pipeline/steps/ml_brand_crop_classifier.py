from ml.brand_crop_classifier.inference_brand_crop_classifier import (
    predict_brand_crop_classifier,
)


def _get_logo_detections(ctx):
    logo_candidates = getattr(ctx, "logo_candidate_crops", [])

    if logo_candidates:
        return [
            {
                "id": item.get("id"),
                "bbox": item.get("bbox"),
                "original_bbox": item.get("original_bbox"),
                "confidence": item.get("emblem_score", 0.5),
                "source": item.get("source", "logo_candidate_crop"),
                "crop": item.get("crop"),
            }
            for item in logo_candidates
            if item.get("bbox") and item.get("crop") is not None
        ]

    return []


def run(ctx):
    if ctx.ml is None:
        ctx.ml = {}

    try:
        logo_detections = _get_logo_detections(ctx)

        result = predict_brand_crop_classifier(
            logo_detections=logo_detections,
        )

        ctx.ml["brand_crop_classifier"] = result

    except Exception as error:
        ctx.ml["brand_crop_classifier"] = {
            "label": "unknown",
            "brand_label": "unknown",
            "raw_label": "unknown",
            "confidence": 0.0,
            "isReliable": False,
            "isKnownBrand": False,
            "threshold": 0.75,
            "source": "ml_brand_crop_classifier",
            "error": str(error),
            "crop_results": [],
        }

    return ctx