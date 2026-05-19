from __future__ import annotations

import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = APP_ROOT / "tests"
REPO_ROOT = APP_ROOT.parents[1]

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


@pytest.fixture(scope="session")
def app_root() -> Path:
    return APP_ROOT


@pytest.fixture(scope="session")
def evaluation_root(app_root: Path) -> Path:
    return app_root / "data" / "evaluation"


@pytest.fixture(scope="session")
def pass_image(evaluation_root: Path) -> Path:
    return evaluation_root / "pass" / "1.jpg"


@pytest.fixture(scope="session")
def need_review_image(evaluation_root: Path) -> Path:
    return evaluation_root / "need_review" / "10.png"


@pytest.fixture(scope="session")
def fail_image(evaluation_root: Path) -> Path:
    return evaluation_root / "fail" / "5.jpg"


@pytest.fixture(scope="session")
def response_contract_keys() -> set[str]:
    return {
        "analysisId",
        "profileId",
        "summary",
        "input",
        "quality",
        "scene",
        "roi",
        "moderation",
        "detections",
        "ml",
        "ruleResults",
        "score",
        "verdict",
        "violations",
        "explain",
        "artifacts",
        "debug",
        "warnings",
        "errors",
        "stepsCompleted",
        "timings",
    }
