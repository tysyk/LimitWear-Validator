import torch.nn as nn
from torchvision import models


def get_model(num_classes: int, pretrained_backbone: bool = False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model