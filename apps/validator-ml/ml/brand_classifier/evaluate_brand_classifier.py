from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "brand_classifier"
    / "processed"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "weights"
    / "brand_classifier"
    / "brand_classifier.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NO_BRAND_CLASS = "no_brand"


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(class_names),
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, class_names, img_size


def to_binary(label):
    return "no_brand" if label == NO_BRAND_CLASS else "brand"


def main():
    model, class_names, img_size = load_model()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    y_true_multi = []
    y_pred_multi = []

    y_true_binary = []
    y_pred_binary = []

    wrong_binary = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            confidences, preds = torch.max(probs, dim=1)

            for true_idx, pred_idx, conf in zip(labels, preds.cpu(), confidences.cpu()):
                true_label = dataset.classes[true_idx.item()]
                pred_label = class_names[pred_idx.item()]
                confidence = float(conf.item())

                true_binary = to_binary(true_label)
                pred_binary = to_binary(pred_label)

                y_true_multi.append(true_label)
                y_pred_multi.append(pred_label)

                y_true_binary.append(true_binary)
                y_pred_binary.append(pred_binary)

                if true_binary != pred_binary:
                    wrong_binary.append({
                        "true": true_label,
                        "pred": pred_label,
                        "confidence": round(confidence, 4),
                    })

    print("\n=== MULTI-CLASS REPORT ===")
    print(classification_report(y_true_multi, y_pred_multi, labels=class_names))

    print("\n=== MULTI-CLASS CONFUSION MATRIX ===")
    print(class_names)
    print(confusion_matrix(y_true_multi, y_pred_multi, labels=class_names))

    print("\n=== BINARY BRAND / NO_BRAND REPORT ===")
    print(classification_report(
        y_true_binary,
        y_pred_binary,
        labels=["brand", "no_brand"],
    ))

    print("\n=== BINARY CONFUSION MATRIX ===")
    print(["brand", "no_brand"])
    print(confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=["brand", "no_brand"],
    ))

    print("\n=== BINARY ERRORS ===")
    if not wrong_binary:
        print("No binary errors.")
    else:
        for item in wrong_binary[:50]:
            print(item)

        if len(wrong_binary) > 50:
            print(f"...and {len(wrong_binary) - 50} more errors")


if __name__ == "__main__":
    main()