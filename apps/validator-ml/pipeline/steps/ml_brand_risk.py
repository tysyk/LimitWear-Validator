from ml.brand_risk.inference import predict_brand_risk


def run(ctx):
    logo_presence = (ctx.ml or {}).get("logo_presence")
    apparel = (ctx.ml or {}).get("apparel", {})

    if not logo_presence:
        ctx.detections["ml_brand_risk"] = {
            "skipped": True,
            "reason": "ml_logo_presence_missing",
        }
        return ctx

    logo_label = logo_presence.get("label")
    logo_confidence = float(logo_presence.get("confidence", 0.0) or 0.0)
    logo_reliable = bool(logo_presence.get("isReliable", False))

    apparel_label = apparel.get("label")
    apparel_confidence = float(apparel.get("confidence", 0.0) or 0.0)
    apparel_reliable = bool(apparel.get("isReliable", False))

    is_reliable_apparel = (
        apparel_label == "apparel"
        and apparel_reliable
        and apparel_confidence >= 0.88
    )

    should_run = (
        is_reliable_apparel
        and logo_label == "logo"
        and logo_reliable
        and logo_confidence >= 0.75
    )

    if not should_run:
        ctx.detections["ml_brand_risk"] = {
            "skipped": True,
            "reason": "brand_risk_conditions_not_met",
            "trigger": {
                "logoPresenceLabel": logo_label,
                "logoPresenceConfidence": logo_confidence,
                "logoPresenceReliable": logo_reliable,
                "apparelLabel": apparel_label,
                "apparelConfidence": apparel_confidence,
                "apparelReliable": apparel_reliable,
            },
        }
        return ctx

    result = predict_brand_risk(ctx.bgr)

    ctx.detections["ml_brand_risk"] = {
        **result,
        "skipped": False,
        "trigger": {
            "logoPresenceLabel": logo_label,
            "logoPresenceConfidence": logo_confidence,
            "logoPresenceReliable": logo_reliable,
            "apparelLabel": apparel_label,
            "apparelConfidence": apparel_confidence,
            "apparelReliable": apparel_reliable,
        },
    }

    ctx.ml["brand_risk"] = ctx.detections["ml_brand_risk"]

    return ctx