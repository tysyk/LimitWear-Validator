from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ml.logo_presence.model import get_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datasets" / "logo_presence"
WEIGHTS = BASE_DIR / "weights" / "logo_presence" / "best.pt"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

print("Classes:", test_ds.classes)
print("Test images:", len(test_ds))


model = get_model(num_classes=len(test_ds.classes), pretrained_backbone=False)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.to(DEVICE)
model.eval()


correct = 0
total = 0

class_correct = {cls: 0 for cls in test_ds.classes}
class_total = {cls: 0 for cls in test_ds.classes}

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for label, pred in zip(labels, preds):
            true_cls = test_ds.classes[int(label)]
            class_total[true_cls] += 1

            if label == pred:
                class_correct[true_cls] += 1


acc = correct / total if total else 0

print("\nTEST ACCURACY:", round(acc, 4))

print("\nPer-class accuracy:")
for cls in test_ds.classes:
    total_cls = class_total[cls]
    correct_cls = class_correct[cls]
    cls_acc = correct_cls / total_cls if total_cls else 0
    print(f"{cls}: {round(cls_acc, 4)} ({correct_cls}/{total_cls})")