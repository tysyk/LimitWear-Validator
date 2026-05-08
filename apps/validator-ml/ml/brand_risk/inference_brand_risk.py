from pathlib import Path
import json

import cv2
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]

WEIGHTS_DIR = BASE_DIR / "weights" / "brand_risk"
MODEL_PATH = WEIGHTS_DIR / "best.pt"
LABELS_PATH = WEIGHTS_DIR / "labels.json"

IMG_SIZE = 224

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.65

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_idx_to_label = None


def _load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {int(k): v for k, v in raw.items()}


def _build_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(
        MODEL_PATH,
        map_location=_device,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.to(_device)
    model.eval()

    return model


def _get_model():
    global _model, _idx_to_label

    if _model is None:
        _idx_to_label = _load_labels()
        _model = _build_model(len(_idx_to_label))

    return _model, _idx_to_label


def _build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def _bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGB")


def predict_brand_risk(bgr):
    model, idx_to_label = _get_model()

    image = _bgr_to_pil(bgr)
    tfm = _build_transform()

    x = tfm(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    confidence, pred_idx = torch.max(probs, dim=0)

    label = idx_to_label[int(pred_idx)]
    confidence = float(confidence.item())

    if confidence >= HIGH_CONFIDENCE:
        reliability = "high"
        is_reliable = True
    elif confidence >= MEDIUM_CONFIDENCE:
        reliability = "medium"
        is_reliable = False
    else:
        reliability = "low"
        is_reliable = False

    if label == "brand_logo":
        if confidence >= HIGH_CONFIDENCE:
            risk_level = "suspicious_logo"
        elif confidence >= MEDIUM_CONFIDENCE:
            risk_level = "possible_brand_logo"
        else:
            risk_level = "uncertain"
    else:
        if confidence >= HIGH_CONFIDENCE:
            risk_level = "no_brand"
        else:
            risk_level = "uncertain"

    return {
        "label": label,
        "confidence": confidence,
        "isReliable": is_reliable,
        "reliability": reliability,
        "riskLevel": risk_level,
        "thresholds": {
            "medium": MEDIUM_CONFIDENCE,
            "high": HIGH_CONFIDENCE,
        },
    }