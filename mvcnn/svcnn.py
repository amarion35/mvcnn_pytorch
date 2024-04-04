"""Define the SVCNN class"""

from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

from .model import Model


class SVCNN(Model):
    """Single View Convolutional Neural Network (SVCNN) model"""

    _net: nn.Module
    _embedding_model: nn.Module
    _embedding_size: int
    _classifier: nn.Module

    def __init__(self, cnn_name: str, pretraining: bool, n_classes: int) -> None:
        super().__init__()

        if cnn_name == "resnet18":
            resnet = models.resnet18(pretrained=pretraining)
        elif cnn_name == "resnet34":
            resnet = models.resnet34(pretrained=pretraining)
        elif cnn_name == "resnet50":
            resnet = models.resnet50(pretrained=pretraining)

        layers = list(resnet.children())
        last_layer = layers.pop()
        layers.append(nn.Flatten())
        self._embedding_model = nn.Sequential(*layers)

        self._embedding_size = last_layer.in_features
        fc_layer = nn.Linear(self._embedding_size, n_classes)
        self._classifier = nn.Sequential(fc_layer)

        self._net = nn.Sequential(self._embedding_model, self._classifier)

    def save(self, path: Path) -> None:
        """Save the model to the specified path"""
        torch.save(self._net.state_dict(), str(path))

    def load(self, path: Path) -> None:
        """Load the model from the specified path"""
        self._net.load_state_dict(torch.load(str(path)))

    @property
    def embedding_model(self) -> nn.Module:
        """Return the embedding model"""
        return self._embedding_model

    @property
    def embedding_size(self) -> int:
        """Return the embedding size"""
        return self._embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self._net(x)
