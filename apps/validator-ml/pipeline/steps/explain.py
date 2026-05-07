from __future__ import annotations

from explain.annotate import create_annotated_artifact


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _finding_sort_key(item):
    severity = str(item.get("severity", "low")).lower()
    penalty = int(item.get("penalty", 0) or 0)
    title = str(item.get("title") or item.get("ruleId") or "")
    return (SEVERITY_ORDER.get(severity, 3), -penalty, title)


def _format_confidence(value) -> str | None:
    if value is None:
        return None
    return f"{float(value) * 100:.1f}%"


def _apparel_signal_line(ctx) -> str | None:
    scene = ctx.scene or {}
    apparel_ml = (ctx.ml or {}).get("apparel", {})

    label = apparel_ml.get("label")
    if label is None and "is_apparel" in scene:
        label = "apparel" if scene.get("is_apparel") else "non_apparel"

    if label is None:
        return None

    confidence = _format_confidence(apparel_ml.get("confidence", scene.get("apparel_confidence")))
    source = scene.get("apparel_source", "unknown")
    reliable = apparel_ml.get("isReliable")
    reliability = "надійний" if reliable else "невпевнений"

    if confidence:
        return f"ML apparel signal: {label} ({confidence}, {reliability}, source={source})."
    return f"Сигнал типу зображення: {label} ({reliability}, source={source})."


def _apparel_type_line(ctx) -> str | None:
    scene = ctx.scene or {}
    apparel_type_ml = (ctx.ml or {}).get("apparel_type", {})

    label = apparel_type_ml.get("label") or scene.get("apparel_type")
    if not label or label == "unknown":
        return None

    confidence = _format_confidence(
        apparel_type_ml.get("confidence", scene.get("apparel_type_confidence"))
    )

    if confidence:
        return f"Тип одягу: {label} ({confidence}, ML)."
    return f"Тип одягу: {label} (ML)."


def _logo_line(ctx):
    ml = (ctx.ml or {}).get("logo_presence", {})

    label = ml.get("label")
    conf = ml.get("confidence")

    if not label or label == "unknown":
        return None

    if conf:
        return f"Наявність логотипа: {label} ({round(conf*100,1)}%, ML)."

    return f"Наявність логотипа: {label} (ML)."


def _quality_line(ctx) -> str | None:
    quality = ctx.quality or {}
    if not quality:
        return None

    if not quality.get("passed_resolution", True):
        return "Якість вхідного зображення обмежує автоматичну перевірку: роздільна здатність нижча за мінімальну."

    if not quality.get("passed_blur", True):
        return "Якість вхідного зображення обмежує автоматичну перевірку: зображення розмите."

    return None


def _blocking_violations(ctx):
    return [
        violation
        for violation in ctx.violations or []
        if (violation.get("meta") or {}).get("blocking")
    ]


def run(ctx) -> None:
    ctx.explain = []

    verdict = ctx.verdict
    violations = sorted(ctx.violations or [], key=_finding_sort_key)
    review_reason = (ctx.debug or {}).get("need_review_reason")
    fail_reason = (ctx.debug or {}).get("fail_reason")

    if verdict == "PASS":
        ctx.add_explain("Автоматична перевірка пройдена. Блокуючих ризиків не виявлено.")
    elif verdict == "WARN":
        ctx.add_explain("Автоматична перевірка завершена з неблокуючими зауваженнями.")
    elif verdict == "FAIL":
        ctx.add_explain("Автоматична перевірка не пройдена, тому що підтверджено блокуючий ризик.")
        if fail_reason:
            ctx.add_explain(f"Блокуюча причина: {fail_reason}")
    elif verdict == "NEED_REVIEW":
        ctx.add_explain("Потрібна ручна перевірка перед фінальним рішенням.")
    elif verdict == "ERROR":
        ctx.add_explain("Під час аналізу сталася технічна помилка.")
        if ctx.errors:
            ctx.add_explain(f"Остання помилка: {ctx.errors[-1].get('message')}")
        ctx.mark_step_done("explain")
        return

    apparel_line = _apparel_signal_line(ctx)
    if apparel_line:
        ctx.add_explain(apparel_line)

    apparel_type_line = _apparel_type_line(ctx)
    if apparel_type_line:
        ctx.add_explain(apparel_type_line)

    logo_line = _logo_line(ctx)
    if logo_line:
        ctx.add_explain(logo_line)

    scene = ctx.scene or {}
    apparel_ml = (ctx.ml or {}).get("apparel", {})
    if scene.get("type") == "unknown" and not apparel_ml.get("isReliable", False):
        ctx.add_explain("Потрібна ручна перевірка через невпевнене визначення типу зображення.")

    if review_reason and verdict == "NEED_REVIEW":
        ctx.add_explain(f"Причина ручної перевірки: {review_reason}")

    visible_findings = _blocking_violations(ctx) if verdict == "FAIL" else violations
    for violation in visible_findings[:3]:
        title = violation.get("title", "Finding")
        message = violation.get("message", "")
        ctx.add_explain(f"{title}: {message}" if message else title)

    quality_line = _quality_line(ctx)
    if quality_line:
        ctx.add_explain(quality_line)

    if verdict == "PASS":
        ctx.add_explain("Зображення можна пропустити до наступного етапу перевірки або публікації.")
    elif verdict == "WARN":
        ctx.add_explain("Заявку можна продовжити, але зауваження варто перевірити перед публікацією.")
    elif verdict == "FAIL":
        ctx.add_explain("Потрібно усунути блокуючий ризик і повторити перевірку.")
    elif verdict == "NEED_REVIEW":
        ctx.add_explain("Це кейс для ручної ескалації, а не автоматичне відхилення.")

    try:
        create_annotated_artifact(ctx)
    except Exception as exc:
        ctx.add_warning(f"Annotated artifact was not created: {exc}")
        ctx.merge_debug_section("artifacts", {"annotated": "failed", "error": str(exc)})

    ctx.mark_step_done("explain")