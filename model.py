import torch.nn as nn
from torchvision import models


def get_model(pretrained=True):
    """ResNet-50 with a custom two-class head (Benign, Melanoma).

    pretrained=True loads ImageNet weights for training (transfer learning).
    Use pretrained=False when loading the saved melanoma_model.pth instead.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )

    return model
