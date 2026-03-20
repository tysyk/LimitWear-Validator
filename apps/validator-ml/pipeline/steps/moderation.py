from __future__ import annotations

from moderation.moderation_service import moderate_image_and_text


def run(ctx):
    moderation_result = moderate_image_and_text(
        scene=ctx.scene,
        detections=ctx.detections,
        quality=ctx.quality,
    )

    ctx.moderation = moderation_result

    if moderation_result.get("blocked"):
        for label in moderation_result.get("labels", []):
            if label.get("blocked"):
                ctx.add_rule_result(
                    rule_id=f"MODERATION_{label['label'].upper()}",
                    passed=False,
                    severity="high",
                    penalty=100,
                    title="Заборонений контент",
                    message=f"Спрацювала модерація: {label['label']}",
                    meta={"evidence": label.get("evidence", [])},
                )

        ctx.debug["need_review_reason"] = None
        ctx.stop_pipeline = True
        ctx.set_verdict("FAIL")
        return

    if moderation_result.get("needsReview"):
        ctx.debug["need_review_reason"] = "Low moderation confidence because image quality is poor"