from __future__ import annotations


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
    ip = ctx.detections.get("ip") or {}
    scene = ctx.scene or {}

    scene_type = scene.get("type")
    is_apparel = scene.get("is_apparel", True)

    quality_score = float(ctx.quality.get("quality_score", 1.0) or 1.0)
    blur_ok = bool(ctx.quality.get("passed_blur", True))
    resolution_ok = bool(ctx.quality.get("passed_resolution", True))

    score = 100 - penalties

    if not resolution_ok:
        score -= 20
    if not blur_ok:
        score -= 15
    if quality_score < 0.55:
        score -= 10

    score = max(0, min(100, score))
    ctx.score = int(score)

    ctx.debug["aggregate"] = {
        "penalties": penalties,
        "highViolations": high_count,
        "mediumViolations": med_count,
        "lowViolations": low_count,
        "qualityScore": quality_score,
        "ipBlocked": ip.get("blocked", False),
        "ipNeedsReview": ip.get("needsReview", False),
        "sceneType": scene_type,
        "isApparel": is_apparel,
    }

    # hard fail conditions
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

    # unreliable input quality
    if not resolution_ok or quality_score < 0.35:
        ctx.set_verdict("NEED_REVIEW")
        ctx.debug["need_review_reason"] = "Input quality is too low for reliable validation"
        ctx.mark_step_done("aggregate")
        return

    # non-apparel / text-heavy / poster-like content
    if not is_apparel or scene_type in ["text_heavy_cover", "poster_like"]:
        ctx.score = max(ctx.score, 60)
        ctx.set_verdict("NEED_REVIEW")
        ctx.debug["need_review_reason"] = "Non-apparel or text-heavy input"
        ctx.mark_step_done("aggregate")
        return

    # review-required signals
    if ip.get("needsReview"):
        ctx.set_verdict("NEED_REVIEW")
        ctx.mark_step_done("aggregate")
        return

    if moderation.get("needsReview"):
        ctx.set_verdict("NEED_REVIEW")
        ctx.mark_step_done("aggregate")
        return

    # fail conditions for apparel-like inputs
    if high_count >= 1 or ctx.score < 45:
        ctx.set_verdict("FAIL")
        ctx.mark_step_done("aggregate")
        return

    # warn conditions
    if med_count >= 1 or low_count >= 2 or ctx.score < 75:
        ctx.set_verdict("WARN")
        ctx.mark_step_done("aggregate")
        return

    ctx.set_verdict("PASS")
    ctx.mark_step_done("aggregate")