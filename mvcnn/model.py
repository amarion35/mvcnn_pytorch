"""Defines the Model interface."""

import abc
from pathlib import Path
import torch.nn as nn


class Model(nn.Module, abc.ABC):
    """Model class"""

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Save the model to the specified path."""

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        """Load the model from the specified path."""
