from __future__ import annotations


def _set_need_review(ctx, reason: str, *, min_score: int = 55, max_score: int = 70) -> None:
    ctx.score = min(max(ctx.score, min_score), max_score)
    ctx.set_verdict("NEED_REVIEW")
    ctx.debug["need_review_reason"] = reason


def run(ctx) -> None:
    if ctx.verdict == "FAIL":
        ctx.score = 0
        ctx.mark_step_done("aggregate")
        return

    if ctx.verdict == "ERROR":
        ctx.score = 0
        ctx.mark_step_done("aggregate")
        return

    penalties = 0
    high_count = 0
    med_count = 0
    low_count = 0

    for rule in ctx.rule_results:
        if rule.get("passed", True):
            continue

        severity = str(rule.get("severity", "low")).lower()
        penalty = int(rule.get("penalty", 0) or 0)

        penalties += penalty
        if severity == "high":
            high_count += 1
        elif severity == "medium":
            med_count += 1
        else:
            low_count += 1

    moderation = ctx.moderation or {}
    ip = (ctx.detections or {}).get("ip") or {}
    scene = ctx.scene or {}
    apparel_ml = (ctx.ml or {}).get("apparel", {})

    scene_type = scene.get("type")
    is_apparel = bool(scene.get("is_apparel", True))
    apparel_confidence = float(scene.get("apparel_confidence", apparel_ml.get("confidence", 0.0)) or 0.0)
    apparel_source = scene.get("apparel_source", "unknown")

    quality_score = float(ctx.quality.get("quality_score", 1.0) or 1.0)
    blur_ok = bool(ctx.quality.get("passed_blur", True))
    resolution_ok = bool(ctx.quality.get("passed_resolution", True))

    score = 100 - penalties

    if not resolution_ok:
        score -= 20
    if not blur_ok:
        score -= 10
    if quality_score < 0.55:
        score -= 10

    score = max(0, min(100, score))
    ctx.score = int(score)

    scene_conflict = (
        scene_type in {"text_heavy_cover", "poster_like"}
        and apparel_source == "ml"
        and apparel_confidence < 0.60
    )

    ctx.set_debug_section(
        "aggregate",
        {
            "penalties": penalties,
            "highViolations": high_count,
            "mediumViolations": med_count,
            "lowViolations": low_count,
            "qualityScore": quality_score,
            "ipBlocked": ip.get("blocked", False),
            "ipNeedsReview": ip.get("needsReview", False),
            "sceneType": scene_type,
            "isApparel": is_apparel,
            "apparelConfidence": round(apparel_confidence, 4),
            "apparelSource": apparel_source,
            "sceneConflict": scene_conflict,
        },
    )

    if moderation.get("blocked"):
        ctx.score = 0
        ctx.set_verdict("FAIL")
        ctx.mark_step_done("aggregate")
        return

    if ip.get("blocked"):
        ctx.score = 0
        ctx.set_verdict("FAIL")
        ctx.mark_step_done("aggregate")
        return

    if not resolution_ok:
        _set_need_review(ctx, "Input resolution is below the minimum for reliable validation.")
        ctx.mark_step_done("aggregate")
        return

    if quality_score < 0.35 or (not blur_ok and quality_score < 0.60):
        _set_need_review(ctx, "Input quality is too low for reliable validation.")
        ctx.mark_step_done("aggregate")
        return

    if not is_apparel:
        review_reason = (
            "ML apparel classifier marked the image as non-apparel."
            if apparel_source == "ml"
            else "Scene heuristics marked the image as non-apparel."
        )
        _set_need_review(ctx, review_reason)
        ctx.mark_step_done("aggregate")
        return

    if scene_conflict:
        _set_need_review(ctx, "Scene heuristics disagree with a weak apparel prediction.")
        ctx.mark_step_done("aggregate")
        return

    if ip.get("needsReview"):
        _set_need_review(ctx, "Potential IP similarity needs manual review.")
        ctx.mark_step_done("aggregate")
        return

    if moderation.get("needsReview"):
        _set_need_review(ctx, "Moderation confidence is too low for a final automatic decision.")
        ctx.mark_step_done("aggregate")
        return

    if high_count >= 1 or ctx.score < 45:
        if high_count >= 1:
            ctx.score = min(ctx.score, 40)
        ctx.set_verdict("FAIL")
        ctx.mark_step_done("aggregate")
        return

    if med_count >= 1 or low_count >= 2 or ctx.score < 75:
        ctx.set_verdict("WARN")
        ctx.mark_step_done("aggregate")
        return

    ctx.set_verdict("PASS")
    ctx.mark_step_done("aggregate")
