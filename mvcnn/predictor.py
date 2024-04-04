"""Define the Predictor class"""

import logging
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

from .model import Model


class Predictor(object):
    """Predictor class"""

    # Inputs
    _model: Model
    _loader: torch.utils.data.DataLoader
    _device: torch.device

    # Local
    _logger: logging.Logger

    def __init__(
        self,
        model: Model,
        loader: torch.utils.data.DataLoader,
        device: str,
    ) -> None:

        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model
        self._loader = loader
        self._device = torch.device(device)

        self._model.to(self._device)

    def _log_step(
        self,
        step: int,
        n_step: int,
        mode: str,
    ) -> None:
        """Log the loss"""
        self._logger.info(
            "%s - %s - Step %i/%i(%.0f%%)",
            self._model.__class__.__name__,
            mode.title(),
            step + 1,
            n_step,
            (step + 1) / n_step * 100,
        )

    def _predict(self) -> np.ndarray:
        """Validate the model for one epoch"""
        self._logger.info("Prediction started")

        n_steps = len(self._loader)

        predictions: list[np.ndarray] = []

        self._model.eval()
        with torch.no_grad():
            for step, (class_indices, images) in enumerate(self._loader):

                images = Variable(images).to(self._device)
                class_indices = Variable(class_indices).to(self._device)

                outputs = self._model(images)

                predictions.append(outputs.cpu().numpy())

                self._log_step(
                    step=step,
                    n_step=n_steps,
                    mode="eval",
                )

        self._logger.info("Prediction completed")

        return np.array(predictions).reshape(-1, self._model.n_classes)

    def predict_class_indices(self) -> np.ndarray:
        """Predict the class indices"""
        predictions = self._predict()
        return np.argmax(predictions, axis=1)

    def predict_class_probabilities(self) -> np.ndarray:
        """Predict the class probabilities"""
        predictions = self._predict()
        return predictions
