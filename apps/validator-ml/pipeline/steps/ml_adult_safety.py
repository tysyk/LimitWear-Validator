from __future__ import annotations

from ml.adult_safety.inference_adult_safety import predict_adult_safety


def _fallback_result(error: str | None = None):
    result = {
        "label": "unknown",
        "confidence": 0.0,
        "adultScore": 0.0,
        "safeScore": 0.0,
        "scores": {},
        "isReliable": False,
        "reliability": "unavailable",
        "riskLevel": "unknown",
        "source": "unavailable",
    }

    if error:
        result["error"] = error

    return result


def run(ctx):
    if not isinstance(ctx.ml, dict):
        ctx.ml = {}

    if not isinstance(ctx.detections, dict):
        ctx.detections = {}

    try:
        result = predict_adult_safety(ctx.bgr)

        ctx.ml["adult_safety"] = result
        ctx.detections["ml_adult_safety"] = result
        ctx.detections["adultSafety"] = result

        ctx.set_debug_section(
            "ml_adult_safety",
            {
                "label": result.get("label"),
                "confidence": result.get("confidence"),
                "adultScore": result.get("adultScore"),
                "safeScore": result.get("safeScore"),
                "riskLevel": result.get("riskLevel"),
                "isReliable": result.get("isReliable"),
            },
        )

    except Exception as exc:
        result = _fallback_result(str(exc))

        ctx.ml["adult_safety"] = result
        ctx.detections["ml_adult_safety"] = result
        ctx.detections["adultSafety"] = result

        ctx.add_warning("Adult safety ML unavailable")
        ctx.add_error("ml_adult_safety", str(exc), critical=False)

        ctx.set_debug_section(
            "ml_adult_safety",
            {
                "label": "unknown",
                "confidence": 0.0,
                "adultScore": 0.0,
                "riskLevel": "unknown",
                "isReliable": False,
                "error": str(exc),
            },
        )

    ctx.mark_step_done("ml_adult_safety")
    return ctx