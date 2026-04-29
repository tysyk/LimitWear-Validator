from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import cv2

try:
    import torch
    from torchvision import transforms
except Exception as exc:
    torch = None
    transforms = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "weights" / "apparel_type"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
LABELS_PATH = WEIGHTS_DIR / "labels.json"


def _ensure_runtime() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(f"ML runtime is unavailable: {IMPORT_ERROR}") from IMPORT_ERROR


@lru_cache(maxsize=1)
def _get_labels() -> Dict[str, str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Apparel type labels were not found: {LABELS_PATH}")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _get_transform():
    _ensure_runtime()

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


@lru_cache(maxsize=1)
def _get_model():
    _ensure_runtime()

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Apparel type weights were not found: {WEIGHTS_PATH}")

    from ml.apparel_type.model import get_model

    labels = _get_labels()

    model = get_model(
        num_classes=len(labels),
        pretrained_backbone=False,
    )

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def predict_apparel_type(image_bgr) -> Dict[str, Any]:
    if image_bgr is None:
        raise ValueError("Input image is empty")

    labels = _get_labels()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = _get_transform()(image_rgb).unsqueeze(0).to(DEVICE)
    model = _get_model()

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    scores = {
        labels[str(index)]: round(float(prob.item()), 4)
        for index, prob in enumerate(probs)
    }

    pred_index = int(torch.argmax(probs).item())
    label = labels[str(pred_index)]
    confidence = float(probs[pred_index].item())

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": scores,
        "model": "resnet18_apparel_type_v1",
        "weights": "ml/weights/apparel_type/best.pt",
        "source": "ml",
    }