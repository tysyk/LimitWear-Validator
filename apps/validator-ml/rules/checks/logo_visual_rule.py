from __future__ import annotations

from typing import Any, Dict, List


def _result(
    code: str,
    passed: bool,
    severity: str,
    penalty: int,
    message: str,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "passed": passed,
        "severity": severity,
        "penalty": penalty,
        "message": message,
        "meta": meta or {},
    }


def run(ctx) -> List[Dict[str, Any]]:
    detections = getattr(ctx, "detections", {}) or {}
    marks = detections.get("visualLogoMarks", []) or []

    if not marks:
        return [
            _result(
                code="VISUAL_LOGO",
                passed=True,
                severity="low",
                penalty=0,
                message="No visual logo-like marks detected.",
                meta={"count": 0},
            )
        ]

    marks = sorted(
        marks,
        key=lambda x: (
            float(x.get("emblem_score", 0.0)),
            float(x.get("area_ratio", 0.0)),
        ),
        reverse=True,
    )

    strong_centered = []
    medium_marks = []

    for mark in marks:
        score = float(mark.get("emblem_score", 0.0))
        area_ratio = float(mark.get("area_ratio", 0.0))
        center_dist = float(mark.get("center_dist", 1.0))

        if score >= 0.62 and area_ratio >= 0.02 and center_dist <= 0.38:
            strong_centered.append(mark)

        if score >= 0.58 and area_ratio >= 0.006:
            medium_marks.append(mark)

    if strong_centered:
        top = strong_centered[0]
        return [
            _result(
                code="VISUAL_LOGO_CENTER",
                passed=False,
                severity="high",
                penalty=40,
                message="Large logo/emblem-like mark detected near the center.",
                meta={
                    "count": len(strong_centered),
                    "top_bbox": top.get("bbox"),
                    "top_score": top.get("emblem_score"),
                    "top_area_ratio": top.get("area_ratio"),
                    "top_center_dist": top.get("center_dist"),
                },
            )
        ]

    if len(medium_marks) >= 3:
        return [
            _result(
                code="VISUAL_LOGO_MULTIPLE",
                passed=False,
                severity="medium",
                penalty=20,
                message="Multiple logo/emblem-like marks detected.",
                meta={
                    "count": len(medium_marks),
                    "top_items": medium_marks[:5],
                },
            )
        ]

    top = marks[0]
    return [
        _result(
            code="VISUAL_LOGO_SUSPECT",
            passed=False,
            severity="medium",
            penalty=12,
            message="Suspicious logo/emblem-like mark detected.",
            meta={
                "count": len(marks),
                "top_bbox": top.get("bbox"),
                "top_score": top.get("emblem_score"),
                "top_area_ratio": top.get("area_ratio"),
                "top_center_dist": top.get("center_dist"),
            },
        )
    ]