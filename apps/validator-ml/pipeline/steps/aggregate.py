def run(ctx):
    qs = ctx.quality.get("quality_score", 1.0)

    if ctx.debug.get("need_review_reason"):
        ctx.verdict = "NEED_REVIEW"
        return

    # Якщо якість погана — просимо перезняти/пересканити
    if qs < 0.5:
        ctx.verdict = "NEED_REVIEW"
        # score не валимо, бо це не помилка дизайну, а якість вхідного зображення
        return
    
    if any(v.get("ruleId") == "SCAN_SKEW_HIGH" for v in ctx.violations):
        ctx.verdict = "NEED_REVIEW"
        return

    # moderation
    if ctx.moderation.get("verdict") in ("BLOCK", "RESTRICT"):
        ctx.verdict = "FAIL"
        ctx.score = min(ctx.score, 20)
        return

    # violations severity logic (простий старт)
    has_high = any(v.get("severity") in ("HIGH",) for v in ctx.violations)

    if has_high or ctx.score < 60:
        ctx.verdict = "FAIL"
    elif ctx.score < 85 or len(ctx.violations) > 0:
        ctx.verdict = "WARN"
    else:
        ctx.verdict = "PASS"

    if ctx.score < 0:
        ctx.score = 0
