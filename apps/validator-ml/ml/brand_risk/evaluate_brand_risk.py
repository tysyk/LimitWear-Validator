from pathlib import Path
import json
from collections import defaultdict

import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "datasets" / "brand_risk" / "processed"
WEIGHTS_DIR = BASE_DIR / "weights" / "brand_risk"

MODEL_PATH = WEIGHTS_DIR / "best.pt"
LABELS_PATH = WEIGHTS_DIR / "labels.json"

BATCH_SIZE = 32
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {int(k): v for k, v in raw.items()}


def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    idx_to_label = load_labels()
    classes = [idx_to_label[i] for i in range(len(idx_to_label))]

    test_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=test_tfms)

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = build_model(len(classes))

    total = 0
    correct = 0

    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels_batch.size(0)
            correct += (preds == labels_batch).sum().item()

            for true_idx, pred_idx in zip(
                labels_batch.cpu().tolist(),
                preds.cpu().tolist(),
            ):
                true_label = classes[true_idx]
                per_class_total[true_label] += 1

                if true_idx == pred_idx:
                    per_class_correct[true_label] += 1

    print("Classes:", classes)
    print("Test images:", total)
    print(f"Test accuracy: {correct / total:.4f}")

    print("\nPer-class accuracy:")
    for class_name in classes:
        class_total = per_class_total[class_name]
        class_correct = per_class_correct[class_name]
        acc = class_correct / class_total if class_total else 0.0

        print(f"{class_name}: {acc:.4f} ({class_correct}/{class_total})")


if __name__ == "__main__":
    main()