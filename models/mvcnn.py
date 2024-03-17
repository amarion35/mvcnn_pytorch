import torch
import torch.nn as nn

from .model import Model
from .svcnn import SVCNN
from .mvcnn_settings import MVCNNSettings


class MVCNN(Model):
    """Multi View Convolutional Neural Network (MVCNN) model"""

    _settings: MVCNNSettings
    _net_1: nn.Module
    _net_2: nn.Module

    def __init__(self, model: SVCNN, settings: MVCNNSettings) -> None:
        super(MVCNN, self).__init__(settings=settings)

        if model.use_resnet:
            self._net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self._net_2 = model.net.fc
        else:
            self._net_1 = model._net_1
            self._net_2 = model._net_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        y = self._net_1(x)
        y = y.view(
            (
                int(x.shape[0] / self._settings.num_views),
                self._settings.num_views,
                y.shape[-3],
                y.shape[-2],
                y.shape[-1],
            )
        )  # (8,12,512,7,7)
        return self._net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
