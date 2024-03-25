from pathlib import Path
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

    def save(self, path: Path) -> None:
        """Save the model to the specified path"""
        torch.save(
            {
                "svcnn_embedding_model": self._svcnn_embedding_model.state_dict(),
                "classifier": self._classifier.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Load the model from the specified path"""
        checkpoint = torch.load(path)
        self._svcnn_embedding_model.load_state_dict(checkpoint["svcnn_embedding_model"])
        self._classifier.load_state_dict(checkpoint["classifier"])

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
