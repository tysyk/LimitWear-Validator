def run(ctx):
    qs = float(ctx.quality.get("quality_score", 1.0))

    ctx.score = max(0, min(100, ctx.score))

    if ctx.debug.get("need_review_reason"):
        ctx.verdict = "NEED_REVIEW"
        return

    if qs < 0.5:
        ctx.verdict = "NEED_REVIEW"
        return

    if any(v.get("ruleId") == "SCAN_SKEW_HIGH" for v in ctx.violations):
        ctx.verdict = "NEED_REVIEW"
        return

    moderation_verdict = ctx.moderation.get("verdict")
    if moderation_verdict in ("BLOCK", "RESTRICT"):
        ctx.verdict = "FAIL"
        ctx.score = min(ctx.score, 20)
        return

    has_high = any(v.get("severity") == "HIGH" for v in ctx.violations)

    if has_high or ctx.score < 60:
        ctx.verdict = "FAIL"
    elif ctx.score < 85:
        ctx.verdict = "WARN"
    elif len(ctx.violations) > 0:
        ctx.verdict = "PASS_WITH_WARNINGS"
    else:
        ctx.verdict = "PASS"

    ctx.score = max(0, min(100, ctx.score))