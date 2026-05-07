from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR = Path.cwd()

MODEL_PATH = BASE_DIR / "ml" / "models" / "brand_classifier" / "brand_classifier.pt"

CONFIDENCE_THRESHOLD = 0.60
NO_BRAND_CLASS = "no_brand"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_class_names = None
_img_size = 224


def _load_model():
    global _model, _class_names, _img_size

    if _model is not None:
        return

    checkpoint = torch.load(MODEL_PATH, map_location=_device)

    _class_names = checkpoint["class_names"]
    _img_size = checkpoint.get("img_size", 224)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(_class_names),
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(_device)
    model.eval()

    _model = model


def _preprocess_bgr(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    transform = transforms.Compose([
        transforms.Resize((_img_size, _img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transform(image).unsqueeze(0)


def predict_brand_classifier(bgr):
    _load_model()

    x = _preprocess_bgr(bgr).to(_device)

    with torch.no_grad():
        outputs = _model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    confidence_tensor, pred_idx_tensor = torch.max(probs, dim=0)

    predicted_class = _class_names[pred_idx_tensor.item()]
    confidence = float(confidence_tensor.item())

    if predicted_class == NO_BRAND_CLASS:
        label = "no_brand"
        brand_label = "no_brand"
    elif confidence >= CONFIDENCE_THRESHOLD:
        label = "brand"
        brand_label = predicted_class
    else:
        label = "unknown"
        brand_label = predicted_class

    top_probs, top_indices = torch.topk(probs, k=3)

    top_predictions = [
        {
            "label": _class_names[idx.item()],
            "confidence": round(float(prob.item()), 4),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    return {
        "label": label,
        "brand_label": brand_label,
        "raw_label": predicted_class,
        "confidence": round(confidence, 4),
        "isReliable": confidence >= CONFIDENCE_THRESHOLD,
        "threshold": CONFIDENCE_THRESHOLD,
        "source": "ml_brand_classifier",
        "top_predictions": top_predictions,
    }