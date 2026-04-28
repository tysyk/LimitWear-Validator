import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from torch import nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = Path(__file__).resolve().parents[1] / "datasets" / "apparel"
WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "apparel" / "best.pt"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


test_dataset = datasets.ImageFolder(BASE / "test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

print("Classes:", test_dataset.classes)
print("Test images:", len(test_dataset))


# 🔥 ВАЖЛИВО: та сама архітектура що була при train
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))

model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.to(DEVICE)
model.eval()


correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)


acc = correct / total
print("\nTEST ACCURACY:", round(acc, 4))