from __future__ import annotations

from explain.annotate import create_annotated_artifact
from core.messages import get_explain_message


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _finding_sort_key(item):
    severity = str(item.get("severity", "low")).lower()
    penalty = int(item.get("penalty", 0) or 0)
    title = str(item.get("title") or item.get("ruleId") or "")
    return (SEVERITY_ORDER.get(severity, 3), -penalty, title)


def _blocking_violations(ctx):
    return [
        violation
        for violation in ctx.violations or []
        if (violation.get("meta") or {}).get("blocking")
    ]


def _review_violations(ctx):
    return [
        violation
        for violation in ctx.violations or []
        if (violation.get("meta") or {}).get("needsReview")
    ]


def _quality_line(ctx) -> str | None:
    quality = ctx.quality or {}

    if not quality:
        return None

    if not quality.get("passed_resolution", True):
        return (
            "Якість зображення обмежує автоматичну перевірку: "
            "роздільна здатність нижча за мінімальну."
        )

    if not quality.get("passed_blur", True):
        return (
            "Якість зображення обмежує автоматичну перевірку: "
            "зображення розмите."
        )

    return None


def _add_findings(ctx, findings, limit: int = 3) -> None:
    for violation in findings[:limit]:
        title = violation.get("title", "Зауваження")
        message = violation.get("message", "")

        if message:
            ctx.add_explain(f"{title}: {message}")
        else:
            ctx.add_explain(title)


def run(ctx) -> None:
    ctx.explain = []

    verdict = ctx.verdict
    violations = sorted(ctx.violations or [], key=_finding_sort_key)
    review_reason = (ctx.debug or {}).get("need_review_reason")
    fail_reason = (ctx.debug or {}).get("fail_reason")

    if verdict == "PASS":
        ctx.add_explain(get_explain_message("pass_intro"))

    elif verdict == "WARN":
        ctx.add_explain(get_explain_message("warn_intro"))

    elif verdict == "FAIL":
        ctx.add_explain(get_explain_message("fail_intro"))

        if fail_reason:
            ctx.add_explain(f"Причина: {fail_reason}")

    elif verdict == "NEED_REVIEW":
        ctx.add_explain(get_explain_message("need_review_intro"))

        if review_reason:
            ctx.add_explain(f"Причина: {review_reason}")

    elif verdict == "ERROR":
        ctx.add_explain(get_explain_message("error_intro"))

        if ctx.errors:
            ctx.add_explain(
                "Остання технічна помилка збережена в debug-даних."
            )

        ctx.mark_step_done("explain")
        return

    quality_line = _quality_line(ctx)
    if quality_line:
        ctx.add_explain(quality_line)

    if verdict == "FAIL":
        visible_findings = sorted(_blocking_violations(ctx), key=_finding_sort_key)
        _add_findings(ctx, visible_findings or violations)

    elif verdict == "NEED_REVIEW":
        visible_findings = sorted(_review_violations(ctx), key=_finding_sort_key)
        _add_findings(ctx, visible_findings or violations)

    elif verdict == "WARN":
        _add_findings(ctx, violations)

    if verdict == "PASS":
        ctx.add_explain(get_explain_message("pass_next"))

    elif verdict == "WARN":
        ctx.add_explain(get_explain_message("warn_next"))

    elif verdict == "FAIL":
        ctx.add_explain(get_explain_message("fail_next"))

    elif verdict == "NEED_REVIEW":
        ctx.add_explain(get_explain_message("need_review_next"))

    try:
        create_annotated_artifact(ctx)
    except Exception as exc:
        ctx.add_warning(f"Annotated artifact was not created: {exc}")
        ctx.merge_debug_section(
            "artifacts",
            {
                "annotated": "failed",
                "error": str(exc),
            },
        )

    ctx.mark_step_done("explain")