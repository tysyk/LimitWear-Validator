from __future__ import annotations

import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def build_ctx(*, width: int = 1200, height: int = 1200):
    from pipeline.context import PipelineContext

    return PipelineContext(
        image_id="test-analysis",
        profile_id="default",
        bgr=None,
        width=width,
        height=height,
    )


def base_quality():
    return {
        "passed_resolution": True,
        "passed_blur": True,
        "quality_score": 1.0,
        "blur_score": 100.0,
    }


def base_scene(*, is_apparel: bool = True, scene_type: str = "apparel_candidate", confidence: float = 0.9):
    return {
        "is_apparel": is_apparel,
        "type": scene_type,
        "apparel_source": "ml",
        "apparel_confidence": confidence,
    }


def base_ml(*, label: str = "apparel", confidence: float = 0.9):
    return {
        "apparel": {
            "label": label,
            "confidence": confidence,
            "isReliable": True,
            "source": "ml",
        }
    }


def empty_detections():
    return {
        "ocr": [],
        "lines": [],
        "ip": {"exactHits": [], "suspiciousHits": [], "blocked": False, "needsReview": False},
        "logoLikeMarks": [],
        "visualLogoMarks": [],
        "qrMarks": [],
        "watermarkMarks": [],
    }
