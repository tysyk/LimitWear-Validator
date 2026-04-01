from __future__ import annotations


def run(ctx) -> None:
    verdict = ctx.verdict
    violations = ctx.violations or []

    if verdict == "PASS":
        ctx.add_explain("Дизайн пройшов автоматичну перевірку.")
        ctx.add_explain("Критичних порушень не виявлено.")
        ctx.mark_step_done("explain")
        return

    if verdict == "WARN":
        ctx.add_explain("Дизайн має зауваження, які бажано виправити перед публікацією.")
    elif verdict == "FAIL":
        ctx.add_explain("Дизайн не пройшов перевірку через критичні порушення.")
    elif verdict == "NEED_REVIEW":
        ctx.add_explain("Дизайн потребує ручної перевірки.")
    elif verdict == "ERROR":
        ctx.add_explain("Під час аналізу сталася помилка.")
        ctx.mark_step_done("explain")
        return

    for v in violations[:7]:
        title = v.get("title", "Порушення")
        message = v.get("message", "")

        if message:
            ctx.add_explain(f"{title}: {message}")
        else:
            ctx.add_explain(title)

    if verdict in {"FAIL", "WARN", "NEED_REVIEW"}:
        ctx.add_explain(
            "Рекомендується перевірити текст, IP-ризики, розміщення елементів і чистоту композиції."
        )

    ctx.mark_step_done("explain")