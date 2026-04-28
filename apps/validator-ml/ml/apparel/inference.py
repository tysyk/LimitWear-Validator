from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import cv2

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except Exception as exc:
    torch = None
    nn = None
    models = None
    transforms = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "../weights/apparel/best.pt")

CLASS_NAMES = ["apparel", "non_apparel"]


def _ensure_runtime() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(f"ML runtime is unavailable: {IMPORT_ERROR}") from IMPORT_ERROR


@lru_cache(maxsize=1)
def _get_transform():
    _ensure_runtime()
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


@lru_cache(maxsize=1)
def _get_model():
    _ensure_runtime()

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Apparel weights were not found: {WEIGHTS_PATH}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def predict_apparel(image_bgr) -> Dict[str, Any]:
    if image_bgr is None:
        raise ValueError("Input image is empty")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = _get_transform()(image_rgb).unsqueeze(0).to(DEVICE)
    model = _get_model()

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    score_map = {
        CLASS_NAMES[index]: round(float(prob.item()), 4)
        for index, prob in enumerate(probs)
    }

    pred_index = int(torch.argmax(probs).item())
    confidence = float(probs[pred_index].item())

    return {
        "label": CLASS_NAMES[pred_index],
        "confidence": round(confidence, 4),
        "scores": score_map,
        "model": "resnet18_apparel_v2",
        "weights": "ml/weights/apparel/best.pt",
    }