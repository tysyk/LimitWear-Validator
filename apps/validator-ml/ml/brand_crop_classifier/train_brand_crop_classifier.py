from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.common.models.mobilenet_v3_classifier import (
    build_mobilenet_v3_classifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets" / "brand_crop_classifier" / "processed"
MODEL_DIR = PROJECT_ROOT / "weights" / "brand_crop_classifier"
MODEL_PATH = MODEL_DIR / "brand_crop_classifier.pt"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in train_loader:
        images = images.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels_batch).sum().item()
        total += labels_batch.size(0)

    return total_loss / total if total else 0.0, correct / total if total else 0.0


def evaluate(model, val_loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels_batch)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

    return total_loss / total if total else 0.0, correct / total if total else 0.0


def save_best_model(model, class_names, epoch, val_accuracy):
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "img_size": IMG_SIZE,
            "architecture": "mobilenet_v3_small",
            "task": "binary_brand_crop_classification",
            "recommended_threshold": 0.75,
            "epoch": epoch,
            "val_acc": val_accuracy,
        },
        MODEL_PATH,
    )


def main():
    print("Device:", DEVICE)

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train",
        transform=get_train_transforms(),
    )

    val_dataset = datasets.ImageFolder(
        DATASET_DIR / "val",
        transform=get_val_transforms(),
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

    class_names = train_dataset.classes

    print("Classes:", class_names)
    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))

    if set(class_names) != {"brand", "no_brand"}:
        raise ValueError("Expected classes: brand, no_brand")

    model = build_mobilenet_v3_classifier(
        num_classes=len(class_names),
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            save_best_model(
                model=model,
                class_names=class_names,
                epoch=epoch,
                val_accuracy=val_accuracy,
            )

            print("Saved new best model")

    print("Training finished")
    print("Best val acc:", best_accuracy)
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()