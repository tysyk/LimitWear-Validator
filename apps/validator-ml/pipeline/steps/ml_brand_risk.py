from ml.brand_risk.inference import predict_brand_risk


def run(ctx):
    logo_presence = (ctx.ml or {}).get("logo_presence")

    if not logo_presence:
        ctx.detections["ml_brand_risk"] = {
            "skipped": True,
            "reason": "ml_logo_presence_missing",
        }
        return ctx

    label = logo_presence.get("label")
    confidence = float(logo_presence.get("confidence", 0.0) or 0.0)
    is_reliable = bool(logo_presence.get("isReliable", False))

    should_run = (
        label == "logo"
        or not is_reliable
        or confidence >= 0.4
    )

    if not should_run:
        ctx.detections["ml_brand_risk"] = {
            "skipped": True,
            "reason": "logo_not_likely",
            "logoPresence": logo_presence,
        }
        return ctx

    result = predict_brand_risk(ctx.bgr)

    ctx.detections["ml_brand_risk"] = {
        **result,
        "skipped": False,
        "trigger": {
            "logoPresenceLabel": label,
            "logoPresenceConfidence": confidence,
            "logoPresenceReliable": is_reliable,
        },
    }

    ctx.ml["brand_risk"] = ctx.detections["ml_brand_risk"]

    return ctx