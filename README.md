# LimitWear Validator

LimitWear Validator is a FastAPI-based prototype for validating clothing design submissions before publication.

The current pre-final version uses a hybrid pipeline: the ML model classifies `apparel` vs `non_apparel`, while heuristic detectors, moderation, IP checks, and business rules make the final publishing decision.

## Setup

Create and activate a virtual environment:

```powershell
cd apps/validator-ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run API

```powershell
cd apps/validator-ml
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Analyze an image:

```powershell
curl -X POST http://127.0.0.1:8000/analyze `
  -F "profile_id=default" `
  -F "file=@C:\path\to\image.png"
```

## Pipeline

The active pipeline is:

```text
quality_gate -> ml_apparel -> roi_extract -> detectors -> scene_type -> moderation -> rules -> aggregate -> explain
```

Main responsibilities:

- `ml_apparel`: classifies the image as apparel or non-apparel.
- `scene_type`: stores scene metadata and aligns `scene.type` with reliable ML apparel output.
- `detectors`: extracts OCR, lines/skew, QR, watermark-like, logo-like, and visual-emblem signals.
- `moderation`: checks prohibited text signals such as adult, hate, violence, and self-harm content.
- `rules`: converts detector signals into business-rule results.
- `aggregate`: produces the final verdict.
- `explain`: creates human-readable explanation and optional annotated artifacts.

## Verdicts

- `PASS`: no blocking or review-worthy issues were found.
- `WARN`: non-blocking issues were found; the design can continue but should be checked.
- `NEED_REVIEW`: the system is uncertain or found heuristic risk that needs manual review.
- `FAIL`: confirmed blocking risk, such as moderation block, confirmed IP/brand risk, or high-confidence non-apparel input.
- `ERROR`: a technical error prevented analysis.

## Current ML Scope

The ML model currently answers only one question: whether the submitted image looks like apparel or non-apparel.

The final decision is not made by the ML model alone. It is made by the hybrid pipeline using:

- ML apparel classification;
- OCR and IP text risk;
- visual logo-like shape detection;
- QR and watermark-like checks;
- moderation signals;
- business rules and aggregate policy.

Future work after this pre-final state:

- retrain or replace `ml/weights/apparel_resnet18.pth`;
- build the web interface or frontend integration;
- expand the curated test image set in `data/test_images/`.
