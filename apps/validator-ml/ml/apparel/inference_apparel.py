from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import cv2

try:
    import torch
    from torchvision import transforms

    from ml.common.models.resnet18_classifier import build_resnet18_classifier
except Exception as exc:
    torch = None
    transforms = None
    build_resnet18_classifier = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "apparel" / "best.pt"

CLASS_NAMES = ["apparel", "non_apparel"]

DEVICE = (
    "cuda"
    if torch is not None and torch.cuda.is_available()
    else "cpu"
)


def _ensure_runtime() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            f"ML runtime is unavailable: {IMPORT_ERROR}"
        ) from IMPORT_ERROR


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
        raise FileNotFoundError(
            f"Apparel weights were not found: {WEIGHTS_PATH}"
        )

    model = build_resnet18_classifier(
        num_classes=len(CLASS_NAMES),
        pretrained_backbone=False,
    )

    state_dict = torch.load(
        WEIGHTS_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def predict_apparel(image_bgr) -> Dict[str, Any]:
    if image_bgr is None:
        raise ValueError("Input image is empty")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    tensor = (
        _get_transform()(image_rgb)
        .unsqueeze(0)
        .to(DEVICE)
    )

    model = _get_model()

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    scores = {
        CLASS_NAMES[index]: round(float(probability.item()), 4)
        for index, probability in enumerate(probabilities)
    }

    pred_index = int(torch.argmax(probabilities).item())
    label = CLASS_NAMES[pred_index]
    confidence = float(probabilities[pred_index].item())

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": scores,
        "model": "resnet18_apparel",
        "weights": "weights/apparel/best.pt",
        "source": "ml",
    }