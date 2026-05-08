from pathlib import Path
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.common.models.resnet18_classifier import build_resnet18_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "datasets" / "logo_presence"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "logo_presence"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def save_labels(class_names):
    labels = {
        str(index): class_name
        for index, class_name in enumerate(class_names)
    }

    with open(WEIGHTS_DIR / "labels.json", "w", encoding="utf-8") as file:
        json.dump(labels, file, indent=2, ensure_ascii=False)


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

        total_loss += loss.item()

        _, predictions = torch.max(outputs, 1)
        correct += (predictions == labels_batch).sum().item()
        total += labels_batch.size(0)

    accuracy = correct / total if total else 0.0

    return total_loss, accuracy


def evaluate(model, val_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

    return correct / total if total else 0.0


def main():
    print("Device:", DEVICE)

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=get_train_transforms(),
    )

    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Classes:", train_dataset.classes)
    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))

    save_labels(train_dataset.classes)

    model = build_resnet18_classifier(
        num_classes=len(train_dataset.classes),
        pretrained_backbone=True,
    ).to(DEVICE)

    class_weights = torch.tensor([1.3, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_accuracy = evaluate(
            model=model,
            val_loader=val_loader,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), WEIGHTS_DIR / "best.pt")
            print("Saved new best model")

    print("Training finished")
    print("Best val acc:", best_accuracy)
    print("Weights saved to:", WEIGHTS_DIR / "best.pt")


if __name__ == "__main__":
    main()