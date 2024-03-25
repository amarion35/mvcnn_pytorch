"""Define the Trainer class"""

import logging
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from .model import Model


class Trainer(object):
    """Trainer class"""

    # Inputs
    _model: Model
    _train_loader: torch.utils.data.DataLoader
    _val_loader: torch.utils.data.DataLoader
    _loss: nn.Module
    _metrics: nn.Module
    _optimizer: torch.optim.Optimizer
    _log_dir: Path
    _steps_per_epoch: int
    _device: torch.device

    # Local
    _logger: logging.Logger
    _writer: SummaryWriter
    _checkpoints_dir: Path

    def __init__(
        self,
        model: Model,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss: nn.Module,
        metrics: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
        steps_per_epoch: int,
        device: str,
    ) -> None:

        self._logger = logging.getLogger(self.__class__.__name__)
        self._optimizer = optimizer
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._loss = loss
        self._metrics = metrics
        self._log_dir = log_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self._steps_per_epoch = steps_per_epoch
        self._device = torch.device(device)

        self._model.to(self._device)
        self._writer = SummaryWriter(str(self._log_dir))

        self._checkpoints_dir = self._log_dir / "checkpoints"
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _log_step(
        self,
        epoch: int,
        n_epochs: int,
        step: int,
        n_step: int,
        loss: float,
        metrics: dict[str, float],
        mode: str,
    ) -> None:
        """Log the loss"""
        self._logger.info(
            "%s - %s - Epoch %i/%i(%.0f%%) - Step %i/%i(%.0f%%) - Loss %.4g%s",
            self._model.__class__.__name__,
            mode.title(),
            1 + epoch,
            n_epochs,
            (epoch) / n_epochs * 100,
            step + 1,
            n_step,
            (step + 1) / n_step * 100,
            loss,
            "".join([f" - {k.title()} {v:.4g}" for k, v in metrics.items()]),
        )
        self._writer.add_scalar(
            f"{mode}_batch_loss", loss, epoch * len(self._train_loader) + step
        )
        for metric_name, metric_value in metrics.items():
            self._writer.add_scalar(
                f"{mode}_batch_{metric_name}",
                metric_value,
                epoch * len(self._train_loader) + step,
            )

    def _log_epoch(
        self,
        epoch: int,
        n_epochs: int,
        loss: float,
        metrics: dict[str, float],
        mode: str,
    ) -> None:
        """Log the loss"""
        self._logger.info(
            "%s - %s - Epoch %i/%i(%.0f%%) - Loss %.4g%s",
            self._model.__class__.__name__,
            mode.title(),
            epoch + 1,
            n_epochs,
            (epoch + 1) / n_epochs * 100,
            loss,
            "".join([f" - {k} {v:.4g}" for k, v in metrics.items()]),
        )
        self._writer.add_scalar(f"{mode}_loss", loss, epoch)
        for metric_name, metric_value in metrics.items():
            self._writer.add_scalar(f"{mode}_{metric_name}", metric_value, epoch)

    def _train_epoch(self, epoch: int, n_epochs: int) -> None:
        """Train the model for one epoch"""
        epoch_losses: list[float] = []
        epoch_metrics: dict[str, list[float]] = {}
        n_steps = min(len(self._train_loader), self._steps_per_epoch)

        self._model.train()
        with torch.enable_grad():
            for step, (class_indices, images) in enumerate(self._train_loader):

                if step >= n_steps:
                    break

                images = Variable(images).to(self._device)
                class_indices = Variable(class_indices).to(self._device)

                self._optimizer.zero_grad()
                outputs = self._model(images)
                loss = self._loss(outputs, class_indices)
                loss.backward()
                self._optimizer.step()
                metrics = self._metrics(outputs, class_indices)

                epoch_losses.append(loss.item())
                for metric_name, metric_value in metrics.items():
                    if metric_name not in epoch_metrics:
                        epoch_metrics[metric_name] = []
                    epoch_metrics[metric_name].append(metric_value)

                self._log_step(
                    epoch=epoch,
                    n_epochs=n_epochs,
                    step=step,
                    n_step=n_steps,
                    loss=epoch_losses[-1],
                    metrics=metrics,
                    mode="train",
                )

        self._log_epoch(
            epoch=epoch,
            n_epochs=n_epochs,
            loss=sum(epoch_losses) / len(epoch_losses),
            metrics={k: sum(v) / len(v) for k, v in epoch_metrics.items()},
            mode="train",
        )

    def _eval_epoch(self, epoch: int, n_epochs: int) -> None:
        """Validate the model for one epoch"""
        epoch_losses: list[float] = []
        epoch_metrics: dict[str, list[float]] = {}
        n_steps = min(len(self._val_loader), self._steps_per_epoch)

        self._model.eval()
        with torch.no_grad():
            for step, (class_indices, images) in enumerate(self._val_loader):

                if step >= n_steps:
                    break

                images = Variable(images).to(self._device)
                class_indices = Variable(class_indices).to(self._device)

                outputs = self._model(images)
                loss = self._loss(outputs, class_indices)
                metrics = self._metrics(outputs, class_indices)

                epoch_losses.append(loss.item())
                for metric_name, metric_value in metrics.items():
                    if metric_name not in epoch_metrics:
                        epoch_metrics[metric_name] = []
                    epoch_metrics[metric_name].append(metric_value)

                self._log_step(
                    epoch=epoch,
                    n_epochs=n_epochs,
                    step=step,
                    n_step=n_steps,
                    loss=epoch_losses[-1],
                    metrics=metrics,
                    mode="eval",
                )

        self._log_epoch(
            epoch=epoch,
            n_epochs=n_epochs,
            loss=sum(epoch_losses) / len(epoch_losses),
            metrics={k: sum(v) / len(v) for k, v in epoch_metrics.items()},
            mode="eval",
        )

    def _save_model_checkpoint(self, path: Path) -> None:
        """Save the model checkpoint"""
        self._logger.info("Saving model checkpoint to %s", path)
        self._model.save(path)

    def train(self, n_epochs: int) -> None:
        """Train the model"""

        self._logger.info("Training started")

        for epoch in range(n_epochs):
            self._train_epoch(epoch, n_epochs)
            self._eval_epoch(epoch, n_epochs)
            self._save_model_checkpoint(self._checkpoints_dir / f"epoch_{epoch+1}.pth")

        self._logger.info("Training completed")
