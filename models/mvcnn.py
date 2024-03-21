import torch
import torch.nn as nn

from .model import Model
from .svcnn import SVCNN
from .mvcnn_settings import MVCNNSettings


class MVCNN(Model):
    """Multi View Convolutional Neural Network (MVCNN) model"""

    _settings: MVCNNSettings
    _svcnn_embedding_model: nn.Module
    _classifier: nn.Module
    _net: nn.Module

    def __init__(self, svcnn: SVCNN, settings: MVCNNSettings) -> None:
        super(MVCNN, self).__init__(settings=settings)

        self._svcnn_embedding_model = svcnn.embedding_model

        fc_layer = nn.Linear(svcnn.embedding_size, 40)
        self._classifier = nn.Sequential(fc_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Compute the embedding of each image
        y = self._svcnn_embedding_model(x)
        # Reshape the tensor to have the shape (num_views, batch_size, num_features)
        y = y.view(
            (
                int(x.shape[0] / self._settings.num_views),
                self._settings.num_views,
                y.shape[-3],
                y.shape[-2],
                y.shape[-1],
            )
        )
        # Max pooling over the views
        y = torch.max(y, 1)[0]
        # Flatten the tensor
        y = y.view(y.shape[0], -1)
        # Classification
        return self._classifier(y)
