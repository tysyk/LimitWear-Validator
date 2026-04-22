from __future__ import annotations


RELIABLE_APPAREL_CONFIDENCE = 0.65


def run(ctx) -> None:
    if not isinstance(ctx.ml, dict):
        ctx.ml = {}

    if not isinstance(ctx.scene, dict):
        ctx.scene = {}

    try:
        from ml.apparel.inference import predict_apparel

        result = predict_apparel(ctx.bgr)
        confidence = float(result.get("confidence", 0.0) or 0.0)
        label = str(result.get("label", "unknown"))
        is_apparel = label == "apparel"

        enriched_result = {
            **result,
            "isReliable": confidence >= RELIABLE_APPAREL_CONFIDENCE,
            "threshold": RELIABLE_APPAREL_CONFIDENCE,
            "source": "ml",
        }

        ctx.ml["apparel"] = enriched_result
        ctx.scene["is_apparel"] = is_apparel
        ctx.scene["apparel_confidence"] = round(confidence, 4)
        ctx.scene["apparel_label"] = label
        ctx.scene["apparel_source"] = "ml"

        ctx.set_debug_section(
            "ml_apparel",
            {
                "label": label,
                "confidence": round(confidence, 4),
                "isReliable": enriched_result["isReliable"],
            },
        )
    except Exception as exc:
        ctx.ml["apparel"] = {
            "label": "unknown",
            "confidence": None,
            "isReliable": False,
            "source": "unavailable",
            "error": str(exc),
        }
        ctx.scene["apparel_source"] = "scene_fallback"
        ctx.add_warning("ML apparel classifier is unavailable, so scene heuristics will be used as fallback.")
        ctx.add_error("ml_apparel", str(exc), critical=False)
        ctx.set_debug_section(
            "ml_apparel",
            {
                "label": "unknown",
                "confidence": None,
                "isReliable": False,
                "error": str(exc),
            },
        )

    ctx.mark_step_done("ml_apparel")
