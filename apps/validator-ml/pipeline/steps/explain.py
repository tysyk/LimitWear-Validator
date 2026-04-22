from __future__ import annotations


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

    if confidence:
        return f"ML-сигнал одягу: {label} ({confidence}, source={source})."
    return f"Сигнал типу зображення: {label} (source={source})."


def _quality_line(ctx) -> str | None:
    quality = ctx.quality or {}
    if not quality:
        return None

    if not quality.get("passed_resolution", True):
        return "Якість вхідного зображення обмежує надійність перевірки: роздільна здатність нижча за мінімальну."

    if not quality.get("passed_blur", True):
        return "Якість вхідного зображення обмежує надійність перевірки: зображення розмите."

    return None


def run(ctx) -> None:
    ctx.explain = []

    verdict = ctx.verdict
    violations = sorted(ctx.violations or [], key=_finding_sort_key)
    review_reason = (ctx.debug or {}).get("need_review_reason")

    if verdict == "PASS":
        ctx.add_explain("Автоматична перевірка пройдена, критичних порушень не виявлено.")
    elif verdict == "WARN":
        ctx.add_explain("Автоматична перевірка завершена із зауваженнями перед публікацією.")
    elif verdict == "FAIL":
        ctx.add_explain("Автоматична перевірка не пройдена через критичні або блокуючі порушення.")
    elif verdict == "NEED_REVIEW":
        ctx.add_explain("Автоматична перевірка не дала фінального рішення, потрібна ручна перевірка.")
    elif verdict == "ERROR":
        ctx.add_explain("Під час аналізу сталася технічна помилка.")
        if ctx.errors:
            ctx.add_explain(f"Остання помилка: {ctx.errors[-1].get('message')}")
        ctx.mark_step_done("explain")
        return

    apparel_line = _apparel_signal_line(ctx)
    if apparel_line:
        ctx.add_explain(apparel_line)

    if review_reason and verdict == "NEED_REVIEW":
        ctx.add_explain(f"Причина ескалації: {review_reason}")

    for violation in violations[:3]:
        title = violation.get("title", "Порушення")
        message = violation.get("message", "")
        ctx.add_explain(f"{title}: {message}" if message else title)

    quality_line = _quality_line(ctx)
    if quality_line:
        ctx.add_explain(quality_line)

    if verdict == "PASS":
        ctx.add_explain("Цей кейс можна показувати як позитивний приклад у демо.")
    elif verdict == "WARN":
        ctx.add_explain("Перед публікацією бажано виправити зауваження і повторити перевірку.")
    elif verdict == "FAIL":
        ctx.add_explain("Потрібно усунути блокуючі проблеми перед повторною подачею.")
    elif verdict == "NEED_REVIEW":
        ctx.add_explain("Для демо цей кейс підходить як приклад ручної ескалації та пояснюваного рішення.")

    ctx.mark_step_done("explain")
