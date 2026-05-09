from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from ml.common.models.mobilenet_v3_classifier import (
    build_mobilenet_v3_classifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "datasets" / "brand_crop_classifier" / "test"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "brand_crop_classifier"
MODEL_PATH = WEIGHTS_DIR / "best.pt"
LABELS_PATH = WEIGHTS_DIR / "labels.json"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        raw = json.load(file)

    return {int(index): label for index, label in raw.items()}


def load_model(class_names):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = build_mobilenet_v3_classifier(
        num_classes=len(class_names),
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    img_size = checkpoint.get("img_size", 224)

    return model, img_size


def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def main():
    print("Device:", DEVICE)

    idx_to_label = load_labels()
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    model, img_size = load_model(class_names)

    dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=get_transforms(img_size),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            for true_idx, pred_idx in zip(labels_batch.tolist(), predictions):
                true_label = dataset.classes[true_idx]
                pred_label = class_names[pred_idx]

                y_true.append(true_label)
                y_pred.append(pred_label)

    print("Classes:", class_names)
    print("Test images:", len(dataset))

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=class_names,
            zero_division=0,
        )
    )

    print("\n=== CONFUSION MATRIX ===")
    print(class_names)
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=class_names,
        )
    )


if __name__ == "__main__":
    main()