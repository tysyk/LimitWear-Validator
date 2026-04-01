from __future__ import annotations

from moderation.moderation_service import moderate_image_and_text


def run(ctx) -> None:
    moderation_result = moderate_image_and_text(
        scene=ctx.scene,
        detections=ctx.detections,
        quality=ctx.quality,
    )

    ctx.moderation = moderation_result or {}

    if ctx.moderation.get("blocked"):
        for label in ctx.moderation.get("labels", []):
            if label.get("blocked"):
                label_name = str(label.get("label", "unknown")).upper()

                ctx.add_rule_result(
                    rule_id=f"MODERATION_{label_name}",
                    passed=False,
                    severity="high",
                    penalty=100,
                    title="Заборонений контент",
                    message=f"Спрацювала модерація: {label.get('label', 'unknown')}",
                    meta={"evidence": label.get("evidence", [])},
                )

        ctx.debug["need_review_reason"] = None
        ctx.stop_pipeline = True
        ctx.set_verdict("FAIL")
        ctx.mark_step_done("moderation")
        return

    if ctx.moderation.get("needsReview"):
        ctx.debug["need_review_reason"] = "Low moderation confidence because image quality is poor"

    ctx.mark_step_done("moderation")