from __future__ import annotations

from rules.checks.size_rule import check as size_check
from rules.checks.safe_area_text_rule import check as safe_area_text_check
from rules.checks.too_much_text_rule import check as too_much_text_check
from rules.checks.logo_visual_rule import run as check_logo_visual


def run(ctx) -> None:
    image_info = {
        "width": ctx.width,
        "height": ctx.height,
    }

    ocr_payload = ctx.detections.get("ocr", {}) or {}
    ocr_items = ocr_payload.get("items", [])
    profile = ctx.profile_id

    checks = [
        ("SIZE", size_check),
        ("SAFE_AREA_TEXT", safe_area_text_check),
        ("TOO_MUCH_TEXT", too_much_text_check),
    ]

    for rule_id, fn in checks:
        try:
            violations, penalty = fn(image_info, ocr_items, profile)

            for v in violations:
                ctx.add_rule_result(
                    rule_id=rule_id,
                    passed=False,
                    severity=v.get("severity", "low"),
                    penalty=int(v.get("penalty", penalty or 0)),
                    title=v.get("title", rule_id),
                    message=v.get("message", ""),
                    bbox=v.get("bbox"),
                    meta=v,
                )

        except Exception as e:
            ctx.add_error("rules_engine", f"{rule_id} failed: {str(e)}", critical=False)

    try:
        visual_logo_results = check_logo_visual(ctx)

        for item in visual_logo_results:
            ctx.add_rule_result(
                rule_id=item.get("code", "VISUAL_LOGO"),
                passed=bool(item.get("passed", False)),
                severity=item.get("severity", "low"),
                penalty=int(item.get("penalty", 0)),
                title=item.get("code", "VISUAL_LOGO"),
                message=item.get("message", ""),
                bbox=(item.get("meta") or {}).get("top_bbox"),
                meta=item.get("meta", {}),
            )

    except Exception as e:
        ctx.add_error("rules_engine", f"VISUAL_LOGO failed: {str(e)}", critical=False)

    ctx.mark_step_done("rules")