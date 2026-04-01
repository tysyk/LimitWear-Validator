from __future__ import annotations

from rules.checks.size_rule import check as size_check
from rules.checks.safe_area_text_rule import check as safe_area_text_check
from rules.checks.too_much_text_rule import check as too_much_text_check


def run(ctx) -> None:
    image_info = {
        "width": ctx.width,
        "height": ctx.height,
    }

    ocr = ctx.detections.get("ocr", [])
    profile = ctx.profile_id  # або ctx.profile якщо потім додаси

    checks = [
        ("SIZE", size_check),
        ("SAFE_AREA_TEXT", safe_area_text_check),
        ("TOO_MUCH_TEXT", too_much_text_check),
    ]

    for rule_id, fn in checks:
        try:
            violations, penalty = fn(image_info, ocr, profile)

            for v in violations:
                ctx.add_rule_result(
                    rule_id=rule_id,
                    passed=False,
                    severity=v.get("severity", "low"),
                    penalty=int(penalty or 0),
                    title=v.get("title", rule_id),
                    message=v.get("message", ""),
                    bbox=v.get("bbox"),
                    meta=v,
                )

        except Exception as e:
            ctx.add_error("rules_engine", f"{rule_id} failed: {str(e)}", critical=False)

    ctx.mark_step_done("rules")