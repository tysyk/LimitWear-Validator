from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import cv2

import torch
import torch.nn as nn
from torchvision import models, transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "weights" / "logo_presence"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
LABELS_PATH = WEIGHTS_DIR / "labels.json"


@lru_cache(maxsize=1)
def _get_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _get_transform():
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
    labels = _get_labels()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(labels))

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def predict_logo_presence(image_bgr) -> Dict[str, Any]:
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
        labels[str(i)]: round(float(prob.item()), 4)
        for i, prob in enumerate(probs)
    }

    pred_index = int(torch.argmax(probs).item())
    label = labels[str(pred_index)]
    confidence = float(probs[pred_index].item())

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": scores,
        "model": "resnet18_logo_presence_v1",
        "weights": "ml/weights/logo_presence/best.pt",
        "source": "ml",
    }