from __future__ import annotations

from core.messages import get_aggregate_reason, get_rule_message
from moderation.moderation_service import moderate_image_and_text


def run(ctx) -> None:
    moderation_result = moderate_image_and_text(
        scene=ctx.scene,
        detections=ctx.detections,
        quality=ctx.quality,
    )

    ctx.moderation = moderation_result or {}

    if ctx.moderation.get("blocked"):
        message = get_rule_message("MODERATION_BLOCK")

        for label in ctx.moderation.get("labels", []) or []:
            if not label.get("blocked"):
                continue

            label_name = str(label.get("label", "unknown")).upper()

            ctx.add_rule_result(
                rule_id=f"MODERATION_{label_name}",
                passed=False,
                severity="high",
                penalty=100,
                title=message["title"],
                message=message["message"],
                meta={
                    "label": label.get("label", "unknown"),
                    "score": label.get("score"),
                    "evidence": label.get("evidence", []),
                    "blocking": True,
                    "needsReview": False,
                    "riskType": "moderation",
                },
            )

        ctx.debug["need_review_reason"] = None
        ctx.debug["fail_reason"] = get_aggregate_reason("moderation_blocked")
        ctx.stop_pipeline = True
        ctx.set_verdict("FAIL")
        ctx.mark_step_done("moderation")
        return

    if ctx.moderation.get("needsReview"):
        ctx.debug["need_review_reason"] = get_aggregate_reason("moderation_review")

    ctx.mark_step_done("moderation")