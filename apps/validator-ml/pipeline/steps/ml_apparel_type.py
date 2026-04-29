from __future__ import annotations


APPAREL_TYPE_CONFIDENCE_THRESHOLD = 0.70


def run(ctx) -> None:
    if not isinstance(ctx.ml, dict):
        ctx.ml = {}

    if not isinstance(ctx.scene, dict):
        ctx.scene = {}

    is_apparel = bool(ctx.scene.get("is_apparel", False))

    if not is_apparel:
        ctx.ml["apparel_type"] = {
            "label": "unknown",
            "confidence": 0.0,
            "isReliable": False,
            "threshold": APPAREL_TYPE_CONFIDENCE_THRESHOLD,
            "source": "skipped",
            "reason": "Image is not reliable apparel",
        }

        ctx.scene["apparel_type"] = "unknown"
        ctx.scene["apparel_type_confidence"] = 0.0
        ctx.scene["apparel_type_source"] = "skipped"

        ctx.set_debug_section(
            "ml_apparel_type",
            {
                "label": "unknown",
                "confidence": 0.0,
                "isReliable": False,
                "skipped": True,
                "reason": "Image is not reliable apparel",
            },
        )

        ctx.mark_step_done("ml_apparel_type")
        return

    try:
        from ml.apparel_type.inference import predict_apparel_type

        result = predict_apparel_type(ctx.bgr)

        confidence = float(result.get("confidence", 0.0) or 0.0)
        label = str(result.get("label", "unknown"))
        is_reliable = confidence >= APPAREL_TYPE_CONFIDENCE_THRESHOLD

        enriched_result = {
            **result,
            "isReliable": is_reliable,
            "threshold": APPAREL_TYPE_CONFIDENCE_THRESHOLD,
            "source": "ml",
        }

        ctx.ml["apparel_type"] = enriched_result

        ctx.scene["apparel_type"] = label
        ctx.scene["apparel_type_confidence"] = round(confidence, 4)
        ctx.scene["apparel_type_source"] = "ml"

        ctx.set_debug_section(
            "ml_apparel_type",
            {
                "label": label,
                "confidence": round(confidence, 4),
                "isReliable": is_reliable,
            },
        )

    except Exception as exc:
        ctx.ml["apparel_type"] = {
            "label": "unknown",
            "confidence": None,
            "isReliable": False,
            "source": "unavailable",
            "error": str(exc),
        }

        ctx.scene["apparel_type"] = "unknown"
        ctx.scene["apparel_type_confidence"] = None
        ctx.scene["apparel_type_source"] = "unavailable"

        ctx.add_warning("ML apparel type classifier is unavailable.")
        ctx.add_error("ml_apparel_type", str(exc), critical=False)

        ctx.set_debug_section(
            "ml_apparel_type",
            {
                "label": "unknown",
                "confidence": None,
                "isReliable": False,
                "error": str(exc),
            },
        )

    ctx.mark_step_done("ml_apparel_type")