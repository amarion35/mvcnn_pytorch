import torch
import torch.nn as nn

from .model import Model
from .svcnn import SVCNN


class MVCNN(Model):
    """Multi View Convolutional Neural Network (MVCNN) model"""

    _svcnn_embedding_model: nn.Module
    _classifier: nn.Module
    _net: nn.Module

    def __init__(self, svcnn: SVCNN, n_classes: int) -> None:
        super().__init__()

        self._svcnn_embedding_model = svcnn.embedding_model

        fc_layer = nn.Linear(svcnn.embedding_size, n_classes)
        self._classifier = nn.Sequential(fc_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size, num_views, channels, height, width = x.shape
        # Reshape the tensor to have the shape (num_views * batch_size, channels, height, width)
        x = x.view(-1, channels, height, width)
        # Compute the embedding of each image
        y = self._svcnn_embedding_model(x)
        # Reshape the tensor to have the shape (num_views, batch_size, num_features)
        y = y.view(batch_size, num_views, -1)
        # Max pooling over the views
        y = torch.max(y, 1)[0]
        # Flatten the tensor
        y = y.view(y.shape[0], -1)
        # Classification
        return self._classifier(y)
