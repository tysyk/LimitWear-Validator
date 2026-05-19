from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from pipeline.context import PipelineContext
from pipeline.runner import run_pipeline


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

EXPECTED_DATASET_VERDICTS = {
    "pass": "PASS",
    "need_review": "NEED_REVIEW",
    "fail": "FAIL",
}

EXPECTED_PIPELINE_STEPS = [
    "quality_gate",
    "ml_apparel",
    "ml_apparel_type",
    "ml_logo_presence",
    "roi_extract",
    "detectors",
    "ml_brand_crop_classifier",
    "ml_adult_safety",
    "scene_type",
    "moderation",
    "rules",
    "aggregate",
    "explain",
]


@dataclass(frozen=True)
class DatasetCase:
    path: Path
    folder: str
    expected_verdict: str

    @property
    def case_id(self) -> str:
        return f"{self.folder}/{self.path.name}"


def iter_dataset_cases(evaluation_root: Path) -> list[DatasetCase]:
    cases: list[DatasetCase] = []

    for folder, expected_verdict in EXPECTED_DATASET_VERDICTS.items():
        folder_path = evaluation_root / folder

        for image_path in sorted(folder_path.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            cases.append(
                DatasetCase(
                    path=image_path,
                    folder=folder,
                    expected_verdict=expected_verdict,
                )
            )

    return cases


def dataset_counts(evaluation_root: Path) -> dict[str, int]:
    return {
        folder: len(
            [
                item
                for item in (evaluation_root / folder).iterdir()
                if item.suffix.lower() in IMAGE_SUFFIXES
            ]
        )
        for folder in EXPECTED_DATASET_VERDICTS
    }


def load_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))

    if image is None:
        raise AssertionError(f"Cannot read test image: {path}")

    return image


def build_context(
    image: np.ndarray,
    *,
    image_id: str = "test-analysis",
    profile_id: str = "pytest",
) -> PipelineContext:
    height, width = image.shape[:2]

    return PipelineContext(
        image_id=image_id,
        profile_id=profile_id,
        bgr=image,
        width=width,
        height=height,
    )


def run_pipeline_on_image(
    image_path: Path,
    *,
    profile_id: str = "pytest",
) -> PipelineContext:
    image = load_bgr_image(image_path)
    ctx = build_context(
        image,
        image_id=image_path.stem,
        profile_id=profile_id,
    )

    return run_pipeline(ctx)


def make_blank_context(
    *,
    width: int = 512,
    height: int = 512,
    profile_id: str = "pytest",
) -> PipelineContext:
    image = np.full((height, width, 3), 240, dtype=np.uint8)
    return build_context(image, image_id="synthetic", profile_id=profile_id)


def minimal_quality() -> dict:
    return {
        "passed_resolution": True,
        "passed_blur": True,
        "quality_score": 1.0,
        "blur_score": 100.0,
    }


def apparel_scene(*, is_apparel: bool = True, confidence: float = 0.96) -> dict:
    return {
        "type": "apparel" if is_apparel else "apparel_candidate",
        "is_apparel": is_apparel,
        "confidence": confidence,
        "type_source": "ml_apparel" if is_apparel else "heuristic_metadata",
        "apparel_source": "ml",
        "apparel_confidence": confidence,
        "apparel_label": "apparel" if is_apparel else "non_apparel",
    }


def apparel_ml(*, label: str = "apparel", confidence: float = 0.96) -> dict:
    return {
        "apparel": {
            "label": label,
            "confidence": confidence,
            "isReliable": confidence >= 0.88,
            "source": "ml",
        }
    }


def empty_detections() -> dict:
    return {
        "ocr": [],
        "lines": [],
        "logoLikeMarks": [],
        "qrMarks": [],
        "watermarkMarks": [],
        "visualLogoMarks": [],
        "logoCandidates": [],
        "ip": {
            "exactHits": [],
            "suspiciousHits": [],
            "blocked": False,
            "needsReview": False,
        },
    }


def violation_ids(ctx: PipelineContext) -> set[str]:
    return {str(item.get("ruleId")) for item in ctx.violations}


def compact_result(ctx: PipelineContext) -> str:
    return (
        f"verdict={ctx.verdict}, score={ctx.score}, "
        f"scene={ctx.scene.get('type')}, "
        f"violations={sorted(violation_ids(ctx))}, "
        f"errors={ctx.errors}"
    )


def assert_response_contract(body: dict, expected_keys: Iterable[str]) -> None:
    missing = set(expected_keys) - set(body)
    assert not missing, f"Missing response keys: {sorted(missing)}"

    assert body["verdict"] in {"PASS", "NEED_REVIEW", "FAIL", "WARN", "ERROR"}
    assert isinstance(body["summary"], dict)
    assert body["summary"]["decision"] == body["verdict"]
    assert isinstance(body["ruleResults"], list)
    assert isinstance(body["violations"], list)
    assert isinstance(body["stepsCompleted"], list)
    assert isinstance(body["timings"], dict)


def assert_pipeline_completed(ctx: PipelineContext) -> None:
    missing_steps = [
        step
        for step in EXPECTED_PIPELINE_STEPS
        if step not in ctx.steps_completed
    ]

    assert not missing_steps, (
        f"Pipeline did not complete expected steps: {missing_steps}. "
        f"{compact_result(ctx)}"
    )


def assert_verdict(ctx: PipelineContext, expected: str, *, case_id: str) -> None:
    assert ctx.verdict == expected, (
        f"{case_id}: expected {expected}, got {ctx.verdict}. "
        f"{compact_result(ctx)}"
    )
