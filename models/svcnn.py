"""Define the SVCNN class"""

import torch
import torch.nn as nn
import torchvision.models as models

from .model import Model
from .svcnn_settings import SVCNNSettings


class SVCNN(Model):
    """Single View Convolutional Neural Network (SVCNN) model"""

    _settings: SVCNNSettings
    _net: nn.Module
    _embedding_model: nn.Module
    _embedding_size: int
    _classifier: nn.Module

    def __init__(self, settings: SVCNNSettings) -> None:
        super(SVCNN, self).__init__(settings=settings)

        if self._settings.cnn_name == "resnet18":
            resnet = models.resnet18(pretrained=self._settings.pretraining)
        elif self._settings.cnn_name == "resnet34":
            resnet = models.resnet34(pretrained=self._settings.pretraining)
        elif self._settings.cnn_name == "resnet50":
            resnet = models.resnet50(pretrained=self._settings.pretraining)

        layers = list(resnet.children())
        last_layer = layers.pop()
        layers.append(nn.Flatten())
        self._embedding_model = nn.Sequential(*layers)

        self._embedding_size = last_layer.in_features
        fc_layer = nn.Linear(self._embedding_size, 40)
        self._classifier = nn.Sequential(fc_layer)

        self._net = nn.Sequential(self._embedding_model, self._classifier)

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
