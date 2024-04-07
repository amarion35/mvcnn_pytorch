"""Define the Metrics class"""

import torch


class Metrics(torch.nn.Module):
    """Metrics class"""

    def __init__(self, metrics: list[torch.nn.Module]) -> None:
        super(Metrics, self).__init__()
        self.metrics = metrics

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Compute the metrics"""
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric(predictions, targets))
        return metrics


class Accuracy(torch.nn.Module):
    """Accuracy metric"""

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Compute the metrics"""
        assert len(predictions) == len(targets)
        # Get the class with the highest probability
        if len(predictions.shape) > 1:
            predictions = predictions.argmax(dim=1)
        # Compute the accuracy
        accuracy = (predictions == targets).float().mean().item()
        return {"accuracy": accuracy}
