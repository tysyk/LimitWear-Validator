from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from ml.brand_classifier.crop_candidates_brand_classifier import (
    extract_logo_crops,
    fallback_chest_crops,
)

from ml.common.models.mobilenet_v3_classifier import (
    build_mobilenet_v3_classifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "weights"
    / "brand_crop_classifier"
    / "brand_crop_classifier.pt"
)

CONFIDENCE_THRESHOLD = 0.75
NO_BRAND_CLASS = "no_brand"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_class_names = None
_img_size = 224


def _load_model():
    global _model
    global _class_names
    global _img_size

    if _model is not None:
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    _class_names = checkpoint["class_names"]
    _img_size = checkpoint.get("img_size", 224)

    model = build_mobilenet_v3_classifier(
        num_classes=len(_class_names),
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
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


def _predict_single_crop(crop_item):
    crop = crop_item["crop"]
    tensor = _preprocess_bgr(crop).to(DEVICE)

    with torch.no_grad():
        outputs = _model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    confidence_tensor, pred_idx_tensor = torch.max(probabilities, dim=0)

    raw_label = _class_names[pred_idx_tensor.item()]
    confidence = float(confidence_tensor.item())

    top_probs, top_indices = torch.topk(
        probabilities,
        k=min(3, len(_class_names)),
    )

    top_predictions = [
        {
            "label": _class_names[idx.item()],
            "confidence": round(float(prob.item()), 4),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    if raw_label == NO_BRAND_CLASS:
        label = "no_brand"
        brand_label = "no_brand"

    elif confidence >= CONFIDENCE_THRESHOLD:
        label = "brand"
        brand_label = raw_label

    else:
        label = "unknown"
        brand_label = raw_label

    return {
        "label": label,
        "brand_label": brand_label,
        "raw_label": raw_label,
        "confidence": round(confidence, 4),
        "isReliable": label == "brand",
        "threshold": CONFIDENCE_THRESHOLD,
        "crop_source": crop_item.get("source"),
        "crop_bbox": crop_item.get("bbox"),
        "detector_confidence": crop_item.get("confidence", 0.0),
        "top_predictions": top_predictions,
    }


def _select_best_result(results):
    if not results:
        return None

    reliable_brand = [
        item for item in results
        if item["label"] == "brand" and item["isReliable"]
    ]

    if reliable_brand:
        return max(reliable_brand, key=lambda item: item["confidence"])

    non_no_brand = [
        item for item in results
        if item["raw_label"] != NO_BRAND_CLASS
    ]

    if non_no_brand:
        return max(non_no_brand, key=lambda item: item["confidence"])

    return max(results, key=lambda item: item["confidence"])


def predict_brand_crop_classifier(
    bgr,
    logo_detections=None,
    use_fallback=True,
):
    _load_model()

    crops = extract_logo_crops(
        bgr=bgr,
        detections=logo_detections or [],
    )

    used_fallback = False

    if not crops and use_fallback:
        crops = fallback_chest_crops(bgr)
        used_fallback = True

    print(f"[DEBUG] crops found: {len(crops)}")
    print(f"[DEBUG] used fallback: {used_fallback}")

    DEBUG_CROPS_DIR = PROJECT_ROOT / "artifacts" / "debug_brand_crops"
    DEBUG_CROPS_DIR.mkdir(parents=True, exist_ok=True)


    for index, item in enumerate(crops):
        cv2.imwrite(
            str(DEBUG_CROPS_DIR / f"crop_{index}_{item.get('source', 'unknown')}.jpg"),
            item["crop"],
        )

    print(f"[DEBUG] crops found: {len(crops)}")
    print(f"[DEBUG] used fallback: {used_fallback}")

    if not crops:
        return {
            "label": "unknown",
            "brand_label": "unknown",
            "raw_label": "unknown",
            "confidence": 0.0,
            "isReliable": False,
            "threshold": CONFIDENCE_THRESHOLD,
            "source": "ml_brand_crop_classifier",
            "reason": "no_logo_candidate_crops",
            "used_fallback": used_fallback,
            "crop_results": [],
        }

    crop_results = [
        _predict_single_crop(item)
        for item in crops
    ]

    best = _select_best_result(crop_results)

    print("[DEBUG] best brand crop:", best)

    return {
        "label": best["label"],
        "brand_label": best["brand_label"],
        "raw_label": best["raw_label"],
        "confidence": best["confidence"],
        "isReliable": best["isReliable"],
        "threshold": CONFIDENCE_THRESHOLD,
        "source": "ml_brand_crop_classifier",
        "used_fallback": used_fallback,
        "crop_source": best["crop_source"],
        "crop_bbox": best["crop_bbox"],
        "detector_confidence": best["detector_confidence"],
        "top_predictions": best["top_predictions"],
        "crop_results": crop_results,
    }