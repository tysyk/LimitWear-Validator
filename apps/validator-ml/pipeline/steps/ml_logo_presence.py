from __future__ import annotations


LOGO_CONFIDENCE_THRESHOLD = 0.75


def run(ctx) -> None:
    if not isinstance(ctx.ml, dict):
        ctx.ml = {}

    try:
        from ml.logo_presence.inference import predict_logo_presence

        result = predict_logo_presence(ctx.bgr)

        confidence = float(result.get("confidence", 0.0) or 0.0)
        label = str(result.get("label", "unknown"))

        is_logo = label == "logo"
        is_reliable = confidence >= LOGO_CONFIDENCE_THRESHOLD

        enriched = {
            **result,
            "isLogo": is_logo,
            "isReliable": is_reliable,
            "threshold": LOGO_CONFIDENCE_THRESHOLD,
            "source": "ml",
        }

        ctx.ml["logo_presence"] = enriched

        ctx.scene["has_logo"] = is_logo and is_reliable
        ctx.scene["logo_confidence"] = round(confidence, 4)
        ctx.scene["logo_source"] = "ml"

        ctx.set_debug_section(
            "ml_logo_presence",
            {
                "label": label,
                "confidence": round(confidence, 4),
                "isReliable": is_reliable,
            },
        )

    except Exception as exc:
        ctx.ml["logo_presence"] = {
            "label": "unknown",
            "confidence": None,
            "isReliable": False,
            "source": "unavailable",
            "error": str(exc),
        }

        ctx.add_warning("Logo ML unavailable")
        ctx.add_error("ml_logo_presence", str(exc), critical=False)

    ctx.mark_step_done("ml_logo_presence")