from __future__ import annotations

from core.config import (
    APPAREL_CONFIDENCE_THRESHOLD,
    NON_APPAREL_BLOCK_CONFIDENCE,
    QUALITY_CRITICAL_SCORE,
)


def _set_need_review(ctx, reason: str, *, min_score: int = 55, max_score: int = 75) -> None:
    ctx.score = min(max(ctx.score, min_score), max_score)
    ctx.set_verdict("NEED_REVIEW")
    ctx.debug["need_review_reason"] = reason


def _set_fail(ctx, reason: str) -> None:
    ctx.score = min(ctx.score, 40)
    ctx.set_verdict("FAIL")
    ctx.debug["fail_reason"] = reason


def run(ctx) -> None:
    if ctx.verdict == "ERROR":
        ctx.score = 0
        ctx.mark_step_done("aggregate")
        return

    penalties = 0
    high_count = 0
    med_count = 0
    low_count = 0
    blocking_failures = []
    review_failures = []

    for rule in ctx.rule_results:
        if rule.get("passed", True):
            continue

        severity = str(rule.get("severity", "low")).lower()
        penalty = int(rule.get("penalty", 0) or 0)
        meta = rule.get("meta") or {}

        penalties += penalty
        if severity == "high":
            high_count += 1
        elif severity == "medium":
            med_count += 1
        else:
            low_count += 1

        if meta.get("blocking"):
            blocking_failures.append(rule)
        if meta.get("needsReview"):
            review_failures.append(rule)

    moderation = ctx.moderation or {}
    ip = (ctx.detections or {}).get("ip") or {}
    scene = ctx.scene or {}
    apparel_ml = (ctx.ml or {}).get("apparel", {})

    scene_type = scene.get("type")
    is_apparel = bool(scene.get("is_apparel", True))
    apparel_confidence = float(scene.get("apparel_confidence", apparel_ml.get("confidence", 0.0)) or 0.0)
    apparel_source = scene.get("apparel_source", "unknown")
    apparel_label = str(apparel_ml.get("label", scene.get("apparel_label", "unknown")))

    quality_score = float(ctx.quality.get("quality_score", 1.0) or 1.0)
    blur_ok = bool(ctx.quality.get("passed_blur", True))
    resolution_ok = bool(ctx.quality.get("passed_resolution", True))

    score = 100 - penalties
    if not resolution_ok:
        score -= 12
    if not blur_ok:
        score -= 8
    if quality_score < 0.55:
        score -= 8

    ctx.score = int(max(0, min(100, score)))

    uncertain_apparel = (
        apparel_source == "ml"
        and (
            apparel_label not in {"apparel", "non_apparel"}
            or apparel_confidence < APPAREL_CONFIDENCE_THRESHOLD
        )
    )
    unknown_scene_from_weak_ml = scene_type == "unknown" and uncertain_apparel

    ctx.set_debug_section(
        "aggregate",
        {
            "penalties": penalties,
            "highViolations": high_count,
            "mediumViolations": med_count,
            "lowViolations": low_count,
            "blockingRuleIds": [rule.get("ruleId") for rule in blocking_failures],
            "reviewRuleIds": [rule.get("ruleId") for rule in review_failures],
            "qualityScore": quality_score,
            "ipBlocked": ip.get("blocked", False),
            "ipNeedsReview": ip.get("needsReview", False),
            "sceneType": scene_type,
            "isApparel": is_apparel,
            "apparelLabel": apparel_label,
            "apparelConfidence": round(apparel_confidence, 4),
            "apparelSource": apparel_source,
            "uncertainApparel": uncertain_apparel,
        },
    )

    if moderation.get("blocked"):
        ctx.score = 0
        ctx.set_verdict("FAIL")
        ctx.debug["fail_reason"] = "Модерація заблокувала заборонений контент."
        ctx.mark_step_done("aggregate")
        return

    if ip.get("blocked") or ip.get("exactHits"):
        _set_fail(ctx, "Підтверджений IP/brand ризик.")
        ctx.mark_step_done("aggregate")
        return

    if blocking_failures:
        _set_fail(ctx, f"Блокуюче правило: {blocking_failures[0].get('ruleId')}")
        ctx.mark_step_done("aggregate")
        return

    if quality_score <= QUALITY_CRITICAL_SCORE and (not resolution_ok or not blur_ok):
        _set_need_review(ctx, "Якість зображення занадто низька для надійної автоматичної перевірки.")
        ctx.mark_step_done("aggregate")
        return

    if not resolution_ok:
        _set_need_review(ctx, "Роздільна здатність нижча за мінімум для надійної перевірки.")
        ctx.mark_step_done("aggregate")
        return

    if not is_apparel:
        if apparel_source == "ml" and apparel_confidence >= NON_APPAREL_BLOCK_CONFIDENCE:
            _set_fail(ctx, "ML apparel classifier з високою впевненістю визначив non-apparel.")
        else:
            _set_need_review(ctx, "Тип зображення невпевнений або може бути non-apparel.")
        ctx.mark_step_done("aggregate")
        return

    if uncertain_apparel or unknown_scene_from_weak_ml:
        _set_need_review(ctx, "Потрібна ручна перевірка через низьку впевненість визначення типу зображення.")
        ctx.mark_step_done("aggregate")
        return

    if ip.get("needsReview"):
        _set_need_review(ctx, "Потенційна IP-схожість потребує ручної перевірки.")
        ctx.mark_step_done("aggregate")
        return

    if moderation.get("needsReview"):
        _set_need_review(ctx, "Moderation-сигнали потребують ручної перевірки.")
        ctx.mark_step_done("aggregate")
        return

    if review_failures:
        _set_need_review(ctx, f"Рекомендована ручна перевірка для {review_failures[0].get('ruleId')}.")
        ctx.mark_step_done("aggregate")
        return

    if med_count >= 1 or low_count >= 1 or ctx.score < 75:
        ctx.set_verdict("WARN")
        ctx.mark_step_done("aggregate")
        return

    ctx.set_verdict("PASS")
    ctx.mark_step_done("aggregate")
