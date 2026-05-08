from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "brand_classifier"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "weights"
    / "brand_classifier"
)

MODEL_PATH = (
    MODEL_DIR
    / "brand_classifier.pt"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 1e-4
IMG_SIZE = 224

BRAND_CLASSES = {
    "nike",
    "adidas",
    "jordan",
    "gucci",
    "calvin_klein",
    "puma",
    "supreme",
    "chanel",
    "dior",
    "lv",
}

NO_BRAND_CLASS = "no_brand"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def build_dataloaders():
    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        DATASET_DIR / "val",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_model(num_classes):
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model.to(device)


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct_multiclass = 0
    correct_binary = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            correct_multiclass += (preds == labels).sum().item()

            correct_binary += count_binary_correct(
                preds=preds.cpu(),
                labels=labels.cpu(),
                class_names=loader.dataset.classes,
            )

            total += labels.size(0)

    avg_loss = total_loss / total
    multiclass_acc = correct_multiclass / total
    binary_acc = correct_binary / total

    return avg_loss, multiclass_acc, binary_acc


def count_binary_correct(preds, labels, class_names):
    correct = 0

    for pred_idx, label_idx in zip(preds, labels):
        pred_name = class_names[pred_idx.item()]
        true_name = class_names[label_idx.item()]

        pred_binary = "no_brand" if pred_name == NO_BRAND_CLASS else "brand"
        true_binary = "no_brand" if true_name == NO_BRAND_CLASS else "brand"

        if pred_binary == true_binary:
            correct += 1

    return correct


def validate_classes(class_names):
    expected = BRAND_CLASSES | {NO_BRAND_CLASS}
    found = set(class_names)

    missing = expected - found
    extra = found - expected

    if missing:
        print(f"[WARN] Missing classes: {sorted(missing)}")

    if extra:
        print(f"[WARN] Extra classes found: {sorted(extra)}")

    if NO_BRAND_CLASS not in found:
        raise ValueError("no_brand class is required.")


def save_model(model, class_names, epoch, val_multiclass_acc, val_binary_acc):
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "img_size": IMG_SIZE,
            "architecture": "mobilenet_v3_small",
            "task": "multiclass_brand_training_binary_risk_inference",
            "brand_classes": sorted(list(BRAND_CLASSES)),
            "no_brand_class": NO_BRAND_CLASS,
            "recommended_threshold": 0.85,
            "epoch": epoch,
            "val_multiclass_acc": val_multiclass_acc,
            "val_binary_acc": val_binary_acc,
        },
        MODEL_PATH,
    )


def main():
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()

    class_names = train_dataset.classes
    validate_classes(class_names)

    print("\nDevice:", device)
    print("\nClasses:")
    print(class_names)

    print(f"\nTrain images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")

    model = build_model(num_classes=len(class_names))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_binary_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)

        val_loss, val_multiclass_acc, val_binary_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
            f"\ntrain_loss={train_loss:.4f}"
            f"\nval_loss={val_loss:.4f}"
            f"\nval_multiclass_acc={val_multiclass_acc:.4f}"
            f"\nval_binary_acc={val_binary_acc:.4f}"
        )

        if val_binary_acc > best_binary_acc:
            best_binary_acc = val_binary_acc

            save_model(
                model=model,
                class_names=class_names,
                epoch=epoch,
                val_multiclass_acc=val_multiclass_acc,
                val_binary_acc=val_binary_acc,
            )

            print("\n[SAVED] best model by val_binary_acc")

    print(f"\nBest validation binary accuracy: {best_binary_acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()