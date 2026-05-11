from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2

from core.config import (
    ANNOTATED_ARTIFACT_FORMAT,
    ARTIFACTS_DIR,
)


APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = APP_ROOT / ARTIFACTS_DIR


def _valid_bbox(value: Any, *, width: int, height: int) -> List[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def _items_with_boxes(
    items: Iterable[Dict[str, Any]],
    *,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    boxed = []

    for item in items:
        bbox = _valid_bbox(
            item.get("bbox"),
            width=width,
            height=height,
        )

        if bbox:
            boxed.append({**item, "bbox": bbox})

    return boxed


def _label_for_item(item: Dict[str, Any]) -> str:
    meta = item.get("meta") or {}

    if meta.get("blocking"):
        return "Blocked risk"

    if meta.get("needsReview"):
        return "Review needed"

    return "Finding"


def create_annotated_artifact(ctx) -> Dict[str, Any] | None:
    image = ctx.bgr_used if ctx.bgr_used is not None else ctx.bgr

    if image is None:
        ctx.merge_debug_section(
            "artifacts",
            {"annotated": "skipped_no_image"},
        )
        return None

    height, width = image.shape[:2]

    findings = _items_with_boxes(
        ctx.violations or [],
        width=width,
        height=height,
    )

    if not findings:
        ctx.merge_debug_section(
            "artifacts",
            {"annotated": "skipped_no_bbox_findings"},
        )
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    canvas = image.copy()

    for item in findings:
        x1, y1, x2, y2 = item["bbox"]
        severity = str(item.get("severity", "low")).lower()

        color = {
            "high": (35, 35, 220),
            "medium": (0, 150, 255),
            "low": (80, 180, 80),
        }.get(severity, (180, 180, 180))

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        label = _label_for_item(item)
        text_y = max(16, y1 - 8)

        cv2.putText(
            canvas,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    extension = ANNOTATED_ARTIFACT_FORMAT.lower().strip(".")
    output_path = OUTPUT_DIR / f"{ctx.image_id}_annotated.{extension}"

    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write annotated artifact: {output_path}")

    artifact = {
        "annotatedPath": str(output_path),
        "annotatedUrl": str(output_path),
        "bboxCount": len(findings),
    }

    ctx.artifacts.update(artifact)

    ctx.merge_debug_section(
        "artifacts",
        {
            "annotated": "created",
            "bboxCount": len(findings),
        },
    )

    return artifact