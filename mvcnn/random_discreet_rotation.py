"""Define the RandomDiscreetRotation class"""

import torch
import torchvision.transforms as transforms


class RandomDiscreetRotation(transforms.RandomRotation):
    """Random discreet degrees rotation"""

    degrees: list[float]

    def __init__(self, degrees: list[float]) -> None:
        """Initialize the class"""
        super().__init__(degrees=180)
        self.degrees = degrees

    @staticmethod
    def get_params(degrees: list[float]) -> float:
        """Get parameters for ``rotate`` for a random discreet rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        idx = int(torch.randint(low=0, high=len(degrees), size=(1,)).item())
        angle = degrees[idx]
        return angle
