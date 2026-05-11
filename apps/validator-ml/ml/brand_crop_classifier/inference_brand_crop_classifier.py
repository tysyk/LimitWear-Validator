from pathlib import Path
import json

import cv2
import torch
from PIL import Image
from torchvision import transforms

from core.brand_keywords import KNOWN_BRAND_CLASSES, NON_BRAND_CLASSES
from core.config import (
    BRAND_CONFIDENCE_THRESHOLD,
    BRAND_SUSPECTED_THRESHOLD,
)
from ml.common.models.mobilenet_v3_classifier import (
    build_mobilenet_v3_classifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_DIR = PROJECT_ROOT / "weights" / "brand_crop_classifier"
MODEL_PATH = WEIGHTS_DIR / "best.pt"
LABELS_PATH = WEIGHTS_DIR / "labels.json"

IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_idx_to_label = None


def _load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        raw = json.load(file)

    return {int(index): label for index, label in raw.items()}


def _load_model():
    global _model
    global _idx_to_label

    if _model is not None:
        return _model, _idx_to_label

    _idx_to_label = _load_labels()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = build_mobilenet_v3_classifier(
        num_classes=len(_idx_to_label),
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    _model = model

    return _model, _idx_to_label


def _preprocess_bgr(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    return transform(image).unsqueeze(0)


def _find_suspected_known_brand(top_predictions):
    for item in top_predictions:
        label = str(item.get("label", "")).lower()
        confidence = float(item.get("confidence", 0.0) or 0.0)

        if (
            label in KNOWN_BRAND_CLASSES
            and confidence >= BRAND_SUSPECTED_THRESHOLD
        ):
            return {
                "label": label,
                "confidence": confidence,
            }

    return None


def predict_single_crop(crop):
    model, idx_to_label = _load_model()

    tensor = _preprocess_bgr(crop).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    confidence_tensor, pred_idx_tensor = torch.max(probs, dim=0)

    raw_label = idx_to_label[int(pred_idx_tensor.item())]
    confidence = float(confidence_tensor.item())

    top_probs, top_indices = torch.topk(
        probs,
        k=min(5, len(idx_to_label)),
    )

    top_predictions = [
        {
            "label": idx_to_label[int(index.item())],
            "confidence": round(float(prob.item()), 4),
        }
        for prob, index in zip(top_probs, top_indices)
    ]

    is_known_brand = (
        raw_label not in NON_BRAND_CLASSES
        and confidence >= BRAND_CONFIDENCE_THRESHOLD
    )

    suspected_known_brand = _find_suspected_known_brand(top_predictions)

    return {
        "label": "brand" if is_known_brand else raw_label,
        "brand_label": raw_label,
        "raw_label": raw_label,
        "confidence": round(confidence, 4),
        "isReliable": is_known_brand,
        "isKnownBrand": is_known_brand,
        "threshold": BRAND_CONFIDENCE_THRESHOLD,
        "top_predictions": top_predictions,
        "suspectedKnownBrand": suspected_known_brand is not None,
        "suspectedBrandLabel": (
            suspected_known_brand["label"] if suspected_known_brand else None
        ),
        "suspectedBrandConfidence": (
            suspected_known_brand["confidence"] if suspected_known_brand else 0.0
        ),
        "suspectedBrandThreshold": BRAND_SUSPECTED_THRESHOLD,
    }


def predict_brand_crop_classifier(logo_candidates):
    results = []

    for candidate in logo_candidates or []:
        crop = candidate.get("crop")

        if crop is None:
            continue

        prediction = predict_single_crop(crop)

        prediction.update({
            "candidate_id": candidate.get("id"),
            "crop_bbox": candidate.get("bbox"),
            "original_bbox": candidate.get("original_bbox"),
            "crop_source": candidate.get("source"),
            "detector_confidence": candidate.get("emblem_score"),
        })

        results.append(prediction)

    known_brand_results = [
        item for item in results
        if item.get("isKnownBrand")
    ]

    suspected_brand_results = [
        item for item in results
        if item.get("suspectedKnownBrand")
    ]

    if known_brand_results:
        best = max(
            known_brand_results,
            key=lambda item: item.get("confidence", 0.0),
        )
    elif suspected_brand_results:
        best = max(
            suspected_brand_results,
            key=lambda item: item.get("suspectedBrandConfidence", 0.0),
        )
    elif results:
        best = max(
            results,
            key=lambda item: item.get("confidence", 0.0),
        )
    else:
        best = None

    if best is None:
        return {
            "label": "unknown",
            "brand_label": "unknown",
            "raw_label": "unknown",
            "confidence": 0.0,
            "isReliable": False,
            "isKnownBrand": False,
            "threshold": BRAND_CONFIDENCE_THRESHOLD,
            "suspectedKnownBrand": False,
            "suspectedBrandLabel": None,
            "suspectedBrandConfidence": 0.0,
            "suspectedBrandThreshold": BRAND_SUSPECTED_THRESHOLD,
            "source": "ml_brand_crop_classifier",
            "reason": "no_logo_candidate_crops",
            "crop_results": [],
        }

    return {
        "label": best["label"],
        "brand_label": best["brand_label"],
        "raw_label": best["raw_label"],
        "confidence": best["confidence"],
        "isReliable": best["isReliable"],
        "isKnownBrand": best["isKnownBrand"],
        "threshold": BRAND_CONFIDENCE_THRESHOLD,
        "suspectedKnownBrand": best.get("suspectedKnownBrand", False),
        "suspectedBrandLabel": best.get("suspectedBrandLabel"),
        "suspectedBrandConfidence": best.get("suspectedBrandConfidence", 0.0),
        "suspectedBrandThreshold": best.get(
            "suspectedBrandThreshold",
            BRAND_SUSPECTED_THRESHOLD,
        ),
        "source": "ml_brand_crop_classifier",
        "crop_bbox": best.get("crop_bbox"),
        "original_bbox": best.get("original_bbox"),
        "crop_source": best.get("crop_source"),
        "detector_confidence": best.get("detector_confidence"),
        "top_predictions": best.get("top_predictions", []),
        "crop_results": results,
    }