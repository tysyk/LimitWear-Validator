from ml.brand_crop_classifier.inference_brand_crop_classifier import (
    predict_brand_crop_classifier,
)


def _get_logo_detections(ctx):
    detections = ctx.detections or {}

    candidates = []

    visual = detections.get("visual_logo_detector")

    if isinstance(visual, list):
        candidates.extend(visual)

    elif isinstance(visual, dict):
        if "detections" in visual:
            candidates.extend(visual["detections"] or [])

        elif "boxes" in visual:
            candidates.extend(visual["boxes"] or [])

        elif "bbox" in visual:
            candidates.append(visual)

    for item in detections.get("visualLogoMarks", []) or []:
        if item.get("bbox"):
            candidates.append({
                "bbox": item.get("bbox"),
                "confidence": item.get(
                    "score",
                    item.get("confidence", 0.5),
                ),
                "source": "visualLogoMarks",
            })

    # for item in detections.get("ocr", []) or []:
    #     text = str(item.get("text") or "").strip()

    #     if item.get("bbox") and text:
    #         candidates.append({
    #             "bbox": item.get("bbox"),
    #             "confidence": item.get("confidence", 0.8),
    #             "source": "ocr_text_bbox",
    #             "text": text,
    #         })

    return candidates


def run(ctx):
    if ctx.ml is None:
        ctx.ml = {}

    try:
        logo_detections = _get_logo_detections(ctx)

        logo_detections = [
            item
            for item in logo_detections
            if item.get("source") == "visual_logo_detector"
            and float(item.get("confidence", 0.0) or 0.0) >= 0.50
        ]

        print("[DEBUG] filtered logo detections:", logo_detections)

        result = predict_brand_crop_classifier(
            bgr=(
                ctx.bgr_used
                if ctx.bgr_used is not None
                else ctx.bgr
            ),
            logo_detections=logo_detections,
            use_fallback=False,
        )

        ctx.ml["brand_crop_classifier"] = result

    except Exception as error:
        ctx.ml["brand_crop_classifier"] = {
            "label": "unknown",
            "confidence": 0.0,
            "isReliable": False,
            "source": "ml_brand_crop_classifier",
            "error": str(error),
        }

    return ctx