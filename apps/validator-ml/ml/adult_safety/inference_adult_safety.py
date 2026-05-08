import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "Marqo/nsfw-image-detection-384"

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.60

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_processor = None
_model = None


def _get_model():
    global _processor, _model

    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()

    return _processor, _model


def _bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGB")


def predict_adult_safety(bgr):
    processor, model = _get_model()

    image = _bgr_to_pil(bgr)
    inputs = processor(images=image, return_tensors="pt")

    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    id2label = model.config.id2label

    scores = {
        id2label[i].lower(): float(probs[i].item())
        for i in range(len(probs))
    }

    adult_score = max(
        scores.get("nsfw", 0.0),
        scores.get("unsafe", 0.0),
        scores.get("adult", 0.0),
    )

    safe_score = max(
        scores.get("safe", 0.0),
        scores.get("sfw", 0.0),
    )

    if adult_score >= HIGH_CONFIDENCE:
        label = "adult_risk"
        risk_level = "block"
        is_reliable = True
        reliability = "high"
        confidence = adult_score
    elif adult_score >= MEDIUM_CONFIDENCE:
        label = "adult_risk"
        risk_level = "needs_review"
        is_reliable = False
        reliability = "medium"
        confidence = adult_score
    else:
        label = "safe"
        risk_level = "safe"
        is_reliable = safe_score >= HIGH_CONFIDENCE
        reliability = "high" if is_reliable else "medium"
        confidence = safe_score

    return {
        "label": label,
        "confidence": confidence,
        "adultScore": adult_score,
        "safeScore": safe_score,
        "scores": scores,
        "isReliable": is_reliable,
        "reliability": reliability,
        "riskLevel": risk_level,
        "model": MODEL_NAME,
        "thresholds": {
            "medium": MEDIUM_CONFIDENCE,
            "high": HIGH_CONFIDENCE,
        },
    }