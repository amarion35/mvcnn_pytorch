"""Define the Trainer class"""

import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class Trainer(object):
    """Trainer class"""

    # Inputs
    _model: nn.Module
    _train_loader: torch.utils.data.DataLoader
    _val_loader: torch.utils.data.DataLoader
    _loss: nn.Module
    _metrics: nn.Module
    _optimizer: torch.optim.Optimizer
    _log_dir: Path
    _steps_per_epoch: int

    # Local
    _logger: logging.Logger
    _writer: SummaryWriter

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss: nn.Module,
        metrics: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: Path,
        steps_per_epoch: int,
    ) -> None:

        self._logger = logging.getLogger(self.__class__.__name__)
        self._optimizer = optimizer
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._loss = loss
        self._metrics = metrics
        self._log_dir = log_dir
        self._steps_per_epoch = steps_per_epoch

        self._model.cuda()
        self._writer = SummaryWriter(str(self._log_dir))

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
            "%s - Epoch %i/%i(%.0f%%) - Step %i/%i(%.0f%%) - Loss %.4g%s",
            mode,
            1 + epoch,
            n_epochs,
            (epoch) / n_epochs * 100,
            step + 1,
            n_step,
            (step + 1) / n_step * 100,
            loss,
            "".join([f" - {k} {v:.4g}" for k, v in metrics.items()]),
        )
        self._writer.add_scalar(
            "train_batch_loss", loss, epoch * len(self._train_loader) + step
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
            "%s - Epoch %i/%i(%.0f%%) - Loss %.4g%s",
            mode,
            epoch + 1,
            n_epochs,
            (epoch + 1) / n_epochs * 100,
            loss,
            "".join([f" - {k} {v:.4g}" for k, v in metrics.items()]),
        )
        self._writer.add_scalar(f"{mode}_loss", loss, epoch)

    def _train_epoch(self, epoch: int, n_epochs: int) -> None:
        """Train the model for one epoch"""
        epoch_losses: list[float] = []
        epoch_metrics: dict[str, list[float]] = {}
        n_steps = min(len(self._train_loader), self._steps_per_epoch)

        self._model.train()

        for step, (class_indices, images) in enumerate(self._train_loader):

            if step >= n_steps:
                break

            images = Variable(images).cuda()
            class_indices = Variable(class_indices).cuda()

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
                mode="Train",
            )

        self._model.eval()

        self._log_epoch(
            epoch=epoch,
            n_epochs=n_epochs,
            loss=sum(epoch_losses) / len(epoch_losses),
            metrics={k: sum(v) / len(v) for k, v in epoch_metrics.items()},
            mode="Train",
        )

    def _val_epoch(self, epoch: int, n_epochs: int) -> None:
        """Validate the model for one epoch"""
        epoch_losses: list[float] = []
        epoch_metrics: dict[str, list[float]] = {}
        n_steps = min(len(self._val_loader), self._steps_per_epoch)

        self._model.eval()

        for step, (class_indices, images) in enumerate(self._val_loader):

            if step >= n_steps:
                break

            images = Variable(images).cuda()
            class_indices = Variable(class_indices).cuda()

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
                mode="Validation",
            )

        self._log_epoch(
            epoch=epoch,
            n_epochs=n_epochs,
            loss=sum(epoch_losses) / len(epoch_losses),
            metrics={k: sum(v) / len(v) for k, v in epoch_metrics.items()},
            mode="Validation",
        )

    def train(self, n_epochs: int) -> None:
        """Train the model"""

        for epoch in range(n_epochs):
            self._train_epoch(epoch, n_epochs)
            self._val_epoch(epoch, n_epochs)
