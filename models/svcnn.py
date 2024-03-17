"""Define the SVCNN class"""

import torch
import torch.nn as nn
import torchvision.models as models

from .model import Model
from .svcnn_settings import SVCNNSettings


class SVCNN(Model):
    """Single View Convolutional Neural Network (SVCNN) model"""

    _settings: SVCNNSettings
    _use_resnet: bool
    _net: nn.Module

    def __init__(self, settings: SVCNNSettings) -> None:
        super(SVCNN, self).__init__(settings=settings)

        self._use_resnet = self._settings.cnn_name.startswith("resnet")

        if self._use_resnet:
            if self._settings.cnn_name == "resnet18":
                self._net = models.resnet18(pretrained=self._settings.pretraining)
                self._net.fc = nn.Linear(512, 40)
            elif self._settings.cnn_name == "resnet34":
                self._net = models.resnet34(pretrained=self._settings.pretraining)
                self._net.fc = nn.Linear(512, 40)
            elif self._settings.cnn_name == "resnet50":
                self._net = models.resnet50(pretrained=self._settings.pretraining)
                self._net.fc = nn.Linear(2048, 40)
        else:
            if self._settings.cnn_name == "alexnet":
                self._net_1 = models.alexnet(
                    pretrained=self._settings.pretraining
                ).features
                self._net_2 = models.alexnet(
                    pretrained=self._settings.pretraining
                ).classifier
            elif self._settings.cnn_name == "vgg11":
                self._net_1 = models.vgg11(
                    pretrained=self._settings.pretraining
                ).features
                self._net_2 = models.vgg11(
                    pretrained=self._settings.pretraining
                ).classifier
            elif self._settings.cnn_name == "vgg16":
                self._net_1 = models.vgg16(
                    pretrained=self._settings.pretraining
                ).features
                self._net_2 = models.vgg16(
                    pretrained=self._settings.pretraining
                ).classifier

            self._net_2._modules["6"] = nn.Linear(4096, 40)

    @property
    def use_resnet(self) -> bool:
        """Return whether the model uses a ResNet architecture"""
        return self._use_resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self._use_resnet:
            return self._net(x)
        else:
            y = self._net_1(x)
            return self._net_2(y.view(y.shape[0], -1))
