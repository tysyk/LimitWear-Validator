import torch.nn as nn
from torchvision import models


def get_model(num_classes: int = 2, pretrained_backbone: bool = False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
    model = models.resnet18(weights=weights)

    # freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model
